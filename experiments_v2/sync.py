"""
Geometric Synchronization and Registration Module for experiments_v2.

In reference-based (semi-blind) watermarking, the detector legitimately holds the
ORIGINAL COVER (or a compact template) as side information, but MUST NOT use the
true secret watermark bits to estimate geometry.

This module implements and evaluates:
1. Two honest geometric estimators returning a 2x3 affine matrix M:
   (a) Multi-scale ECC registration (cv2.findTransformECC, MOTION_AFFINE)
       between the attacked image and the original cover.
   (b) Feature-based registration (ORB/AKAZE + RANSAC estimateAffinePartial2D)
       against the original cover.
2. Warping the attacked image back via inverse transform (INTER_LINEAR) and extracting.
3. Three evaluation scenarios:
   - none (no registration)
   - auto (estimator a: ECC, estimator b: Feature RANSAC)
   - oracle (known ground-truth inverse transform, diagnostic upper bound)
4. Clearly labelled oracle-dependent check:
   - extract_with_rotation_search() from watermark/core.py (maximizes NC against
     metadata.watermark_binary) to demonstrate why it is an unrealistic oracle.
5. Estimation error tracking (degrees and pixels) and registration failure rates.

Outputs:
- results_v2/sync_raw.csv
- results_v2/sync_estimation_error.csv
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import cv2
import pandas as pd

from watermark.metrics import (
    compute_psnr,
    compute_ssim,
    compute_ber,
    compute_bcr,
    compute_nc,
)
from watermark.core import CascadedDwtDctSvdWatermarker
from experiments_v2.schema import (
    SCHEMA_COLUMNS,
    ResultRow,
    init_results_csv,
    append_result_rows,
)
from experiments_v2.common import get_git_commit_hash
from experiments_v2.harness import (
    create_method,
    calibrate_alpha_to_psnr,
    fast_batched_extract,
    quantize_u8,
)
from experiments_v2.payloads import (
    generate_original_logo,
    generate_whitened_logo,
    generate_balanced_random_payload,
)
from experiments_v2.attacks_v2 import attack_jpeg
from experiments_v2.runner_v2 import select_benchmark_subset


# =============================================================================
# 1. AFFINE MATRIX DECOMPOSITION
# =============================================================================

def decompose_affine_matrix(
    M: np.ndarray,
    center: Tuple[float, float] = (256.0, 256.0),
) -> Tuple[float, float, float, float]:
    """
    Decomposes a 2x3 affine transformation matrix M into:
    - angle_deg: rotation angle in degrees [-180, 180]
    - scale: isotropic scaling factor
    - trans_x: center displacement along x (pixels)
    - trans_y: center displacement along y (pixels)
    """
    angle_rad = np.arctan2(M[0, 1], M[0, 0])
    angle_deg = float(np.degrees(angle_rad))
    scale = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))

    cx, cy = center
    center_att = M @ np.array([cx, cy, 1.0])
    dx = float(center_att[0] - cx)
    dy = float(center_att[1] - cy)
    return angle_deg, scale, dx, dy


# =============================================================================
# 2. HONEST GEOMETRIC ESTIMATORS (NO WATERMARK BITS LEAKAGE)
# =============================================================================

def estimate_geometry_ecc(
    attacked_img: np.ndarray,
    reference_img: np.ndarray,
    levels: int = 3,
    max_iter: int = 60,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Estimator (a): Multi-Scale ECC Registration (cv2.findTransformECC).
    Estimates 2x3 affine matrix M mapping reference_img coordinates to attacked_img coordinates.
    Uses Gaussian pyramid to ensure convergence across larger geometric displacements.
    """
    ref_u8 = np.clip(reference_img, 0.0, 255.0).astype(np.uint8)
    att_u8 = np.clip(attacked_img, 0.0, 255.0).astype(np.uint8)

    # Build Gaussian pyramids
    ref_pyr = [ref_u8]
    att_pyr = [att_u8]
    for _ in range(levels - 1):
        ref_pyr.append(cv2.pyrDown(ref_pyr[-1]))
        att_pyr.append(cv2.pyrDown(att_pyr[-1]))

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    success = False
    converged_level = -1

    # Coarse to fine
    for lvl in range(levels - 1, -1, -1):
        r_lvl = ref_pyr[lvl]
        a_lvl = att_pyr[lvl]
        if lvl != levels - 1:
            warp[0, 2] *= 2.0
            warp[1, 2] *= 2.0
        try:
            _, warp = cv2.findTransformECC(r_lvl, a_lvl, warp, cv2.MOTION_AFFINE, criteria)
            success = True
            converged_level = lvl
        except cv2.error:
            # If fine resolution diverges, preserve coarser estimate
            pass

    # Sanity checks on estimated matrix
    det = warp[0, 0] * warp[1, 1] - warp[0, 1] * warp[1, 0]
    if np.isnan(warp).any() or np.isinf(warp).any() or abs(det) < 0.1 or abs(det) > 10.0:
        success = False
        warp = np.eye(2, 3, dtype=np.float32)

    angle_deg, scale, dx, dy = decompose_affine_matrix(warp)
    meta = {
        "estimator": "auto_ecc",
        "converged_level": converged_level,
        "est_angle_deg": angle_deg,
        "est_scale": scale,
        "est_dx": dx,
        "est_dy": dy,
    }
    return warp.astype(np.float64), success, meta


def estimate_geometry_features(
    attacked_img: np.ndarray,
    reference_img: np.ndarray,
    max_features: int = 1500,
    ratio_thresh: float = 0.75,
    ransac_reproj: float = 3.0,
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Estimator (b): Feature-based (ORB / AKAZE) + RANSAC estimateAffinePartial2D.
    Estimates 2x3 affine matrix M mapping reference_img coordinates to attacked_img coordinates.
    """
    ref_u8 = np.clip(reference_img, 0.0, 255.0).astype(np.uint8)
    att_u8 = np.clip(attacked_img, 0.0, 255.0).astype(np.uint8)

    # 1. Primary detector: ORB
    detector = cv2.ORB_create(nfeatures=max_features)
    kp_ref, desc_ref = detector.detectAndCompute(ref_u8, None)
    kp_att, desc_att = detector.detectAndCompute(att_u8, None)

    # Fallback to AKAZE if ORB yields insufficient features
    if desc_ref is None or desc_att is None or len(kp_ref) < 8 or len(kp_att) < 8:
        detector = cv2.AKAZE_create()
        kp_ref, desc_ref = detector.detectAndCompute(ref_u8, None)
        kp_att, desc_att = detector.detectAndCompute(att_u8, None)

    if desc_ref is None or desc_att is None or len(kp_ref) < 4 or len(kp_att) < 4:
        M = np.eye(2, 3, dtype=np.float64)
        angle_deg, scale, dx, dy = decompose_affine_matrix(M)
        return M, False, {"estimator": "auto_feature", "inliers": 0, "est_angle_deg": angle_deg, "est_scale": scale, "est_dx": dx, "est_dy": dy}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc_att, desc_ref, k=2)

    good = []
    for m_pair in matches:
        if len(m_pair) == 2 and m_pair[0].distance < ratio_thresh * m_pair[1].distance:
            good.append(m_pair[0])

    if len(good) < 4:
        M = np.eye(2, 3, dtype=np.float64)
        angle_deg, scale, dx, dy = decompose_affine_matrix(M)
        return M, False, {"estimator": "auto_feature", "inliers": len(good), "est_angle_deg": angle_deg, "est_scale": scale, "est_dx": dx, "est_dy": dy}

    pts_att = np.float32([kp_att[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M_est, inliers = cv2.estimateAffinePartial2D(
        pts_ref, pts_att, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj
    )

    if M_est is None:
        M = np.eye(2, 3, dtype=np.float64)
        angle_deg, scale, dx, dy = decompose_affine_matrix(M)
        return M, False, {"estimator": "auto_feature", "inliers": 0, "est_angle_deg": angle_deg, "est_scale": scale, "est_dx": dx, "est_dy": dy}

    num_inliers = int(np.sum(inliers)) if inliers is not None else 0
    success = (num_inliers >= 6)

    angle_deg, scale, dx, dy = decompose_affine_matrix(M_est)
    meta = {
        "estimator": "auto_feature",
        "inliers": num_inliers,
        "est_angle_deg": angle_deg,
        "est_scale": scale,
        "est_dx": dx,
        "est_dy": dy,
    }
    return M_est.astype(np.float64), success, meta


# =============================================================================
# 3. WARPING AND WATERMARK EXTRACTION
# =============================================================================

def warp_back_and_extract(
    attacked_img: np.ndarray,
    M: np.ndarray,
    metadata: Any,
) -> np.ndarray:
    """
    Applies inverse affine warping to restore attacked image to original coordinate grid,
    then executes fast_batched_extract.
    """
    h, w = metadata.cover_shape
    # M maps reference -> attacked. To pull pixels back to reference grid:
    # Use cv2.WARP_INVERSE_MAP!
    recovered = cv2.warpAffine(
        attacked_img.astype(np.float32),
        M.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    recovered = quantize_u8(recovered)
    extracted_bits = fast_batched_extract(recovered, metadata)
    return extracted_bits


# =============================================================================
# 4. GEOMETRIC ATTACK SPECIFICATION
# =============================================================================

@dataclass(frozen=True)
class SyncAttackSpec:
    """Specification of geometric attack with ground-truth affine parameters."""
    name: str
    severity: str
    true_angle_deg: float
    true_scale: float
    true_dx: float
    true_dy: float
    true_matrix: np.ndarray
    apply_fn: Any


def _make_sync_attack_specs() -> List[SyncAttackSpec]:
    """Builds the 10 geometric attack specifications."""
    specs = []
    h, w = 512, 512
    cx, cy = w / 2.0, h / 2.0

    # 1. Rotations: 0.5, 1, 2, 5, 10 degrees
    for deg in [0.5, 1.0, 2.0, 5.0, 10.0]:
        M_rot = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
        def _apply_rot(img, m=M_rot):
            return cv2.warpAffine(img.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        specs.append(
            SyncAttackSpec(
                name="rotation",
                severity=f"{deg}deg",
                true_angle_deg=deg,
                true_scale=1.0,
                true_dx=0.0,
                true_dy=0.0,
                true_matrix=M_rot.astype(np.float64),
                apply_fn=_apply_rot,
            )
        )

    # 2. Scaling: 0.9x and 1.1x around center
    for sc in [0.9, 1.1]:
        M_sc = np.array([[sc, 0.0, cx * (1.0 - sc)], [0.0, sc, cy * (1.0 - sc)]], dtype=np.float32)
        def _apply_scale(img, m=M_sc):
            return cv2.warpAffine(img.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        specs.append(
            SyncAttackSpec(
                name="scaling",
                severity=f"{sc}x",
                true_angle_deg=0.0,
                true_scale=sc,
                true_dx=0.0,
                true_dy=0.0,
                true_matrix=M_sc.astype(np.float64),
                apply_fn=_apply_scale,
            )
        )

    # 3. Translation: 4 and 8 px (both axes)
    for shift in [4.0, 8.0]:
        M_tr = np.array([[1.0, 0.0, shift], [0.0, 1.0, shift]], dtype=np.float32)
        def _apply_trans(img, m=M_tr):
            return cv2.warpAffine(img.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        specs.append(
            SyncAttackSpec(
                name="translation",
                severity=f"{int(shift)}px",
                true_angle_deg=0.0,
                true_scale=1.0,
                true_dx=shift,
                true_dy=shift,
                true_matrix=M_tr.astype(np.float64),
                apply_fn=_apply_trans,
            )
        )

    # 4. Compound: rotate 2° + JPEG70
    M_rot2 = cv2.getRotationMatrix2D((cx, cy), 2.0, 1.0)
    def _apply_compound(img, m=M_rot2):
        rotated = cv2.warpAffine(img.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return attack_jpeg(rotated, severity=70)

    specs.append(
        SyncAttackSpec(
            name="compound",
            severity="rotate2°+JPEG70",
            true_angle_deg=2.0,
            true_scale=1.0,
            true_dx=0.0,
            true_dy=0.0,
            true_matrix=M_rot2.astype(np.float64),
            apply_fn=_apply_compound,
        )
    )

    return specs


# =============================================================================
# 5. CONDITION WORKER
# =============================================================================

def _worker_sync_condition(args: Tuple) -> Tuple[str, List[ResultRow], List[Dict[str, Any]]]:
    """
    Executes all geometric synchronization scenarios for one condition:
    (image_id, payload_id, method, subband).
    """
    (
        run_id,
        image_id,
        image_path,
        payload_id,
        payload,
        method_name,
        subband,
        target_psnr,
        git_hash,
    ) = args

    cover = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cover is None:
        raise FileNotFoundError(f"Cover image not found: {image_path}")
    cover = cover.astype(np.float64)

    # 1. Calibrate alpha to target PSNR (46 dB) with quantization enabled
    cal_alpha = calibrate_alpha_to_psnr(
        method=method_name,
        cover=cover,
        payload=payload,
        target_psnr=target_psnr,
        tol=0.08,
        subband=subband,
    )

    # 2. Embed once
    method_obj = create_method(method_name, subband=subband)
    wm_img, side_info = method_obj.embed(cover, payload, alpha=cal_alpha, quantize=True)
    alpha_eff_mean = float(side_info["alpha_eff_mean"])
    metadata = side_info["metadata"]

    psnr_clean = float(compute_psnr(cover, wm_img))
    ssim_clean = float(compute_ssim(cover, wm_img))

    raw_rows: List[ResultRow] = []
    error_records: List[Dict[str, Any]] = []

    attack_specs = _make_sync_attack_specs()

    for att in attack_specs:
        att_img = quantize_u8(att.apply_fn(wm_img))
        psnr_att = float(compute_psnr(cover, att_img))

        # ---------------------------------------------------------------------
        # Scenario 1: None (no registration)
        # ---------------------------------------------------------------------
        bits_none = fast_batched_extract(att_img, metadata)
        raw_rows.append(
            ResultRow(
                run_id=run_id,
                experiment="geometric_sync",
                image_id=image_id,
                payload_id=payload_id,
                method=method_name,
                subband=subband,
                wavelet="haar",
                block_size=8,
                alpha_nominal=round(cal_alpha, 4),
                alpha_eff_mean=round(alpha_eff_mean, 6),
                attack=f"{att.name}_none",
                severity=att.severity,
                seed=42,
                psnr_clean=round(psnr_clean, 4),
                ssim_clean=round(ssim_clean, 6),
                psnr_attacked=round(psnr_att, 4),
                ber=round(float(compute_ber(payload, bits_none)), 6),
                bcr=round(float(compute_bcr(payload, bits_none)), 4),
                nc=round(float(compute_nc(payload, bits_none)), 6),
                extra_json=json.dumps({"scenario": "none", "attack_family": att.name, "git_commit": git_hash}),
            )
        )

        # ---------------------------------------------------------------------
        # Scenario 2a: Auto ECC (honest reference-based, no watermark bits)
        # ---------------------------------------------------------------------
        M_ecc, succ_ecc, meta_ecc = estimate_geometry_ecc(att_img, cover)
        bits_ecc = warp_back_and_extract(att_img, M_ecc, metadata)
        err_angle_ecc = abs(meta_ecc["est_angle_deg"] - att.true_angle_deg)
        err_scale_ecc = abs(meta_ecc["est_scale"] - att.true_scale)
        err_trans_ecc = np.sqrt(
            (meta_ecc["est_dx"] - att.true_dx) ** 2 + (meta_ecc["est_dy"] - att.true_dy) ** 2
        )

        raw_rows.append(
            ResultRow(
                run_id=run_id,
                experiment="geometric_sync",
                image_id=image_id,
                payload_id=payload_id,
                method=method_name,
                subband=subband,
                wavelet="haar",
                block_size=8,
                alpha_nominal=round(cal_alpha, 4),
                alpha_eff_mean=round(alpha_eff_mean, 6),
                attack=f"{att.name}_auto_ecc",
                severity=att.severity,
                seed=42,
                psnr_clean=round(psnr_clean, 4),
                ssim_clean=round(ssim_clean, 6),
                psnr_attacked=round(psnr_att, 4),
                ber=round(float(compute_ber(payload, bits_ecc)), 6),
                bcr=round(float(compute_bcr(payload, bits_ecc)), 4),
                nc=round(float(compute_nc(payload, bits_ecc)), 6),
                extra_json=json.dumps({
                    "scenario": "auto_ecc",
                    "attack_family": att.name,
                    "reg_success": succ_ecc,
                    "error_deg": round(err_angle_ecc, 4),
                    "error_trans_px": round(err_trans_ecc, 4),
                    "git_commit": git_hash,
                }),
            )
        )
        error_records.append({
            "image_id": image_id,
            "payload_id": payload_id,
            "method": method_name,
            "subband": subband,
            "attack": att.name,
            "severity": att.severity,
            "estimator": "auto_ecc",
            "true_angle_deg": round(att.true_angle_deg, 4),
            "est_angle_deg": round(meta_ecc["est_angle_deg"], 4),
            "error_angle_deg": round(err_angle_ecc, 4),
            "true_scale": round(att.true_scale, 4),
            "est_scale": round(meta_ecc["est_scale"], 4),
            "error_scale": round(err_scale_ecc, 4),
            "true_dx": round(att.true_dx, 4),
            "true_dy": round(att.true_dy, 4),
            "est_dx": round(meta_ecc["est_dx"], 4),
            "est_dy": round(meta_ecc["est_dy"], 4),
            "error_trans_px": round(err_trans_ecc, 4),
            "reg_success": succ_ecc,
            "inliers": meta_ecc.get("converged_level", 0),
        })

        # ---------------------------------------------------------------------
        # Scenario 2b: Auto Feature (ORB/AKAZE + RANSAC, honest, no watermark bits)
        # ---------------------------------------------------------------------
        M_feat, succ_feat, meta_feat = estimate_geometry_features(att_img, cover)
        bits_feat = warp_back_and_extract(att_img, M_feat, metadata)
        err_angle_feat = abs(meta_feat["est_angle_deg"] - att.true_angle_deg)
        err_scale_feat = abs(meta_feat["est_scale"] - att.true_scale)
        err_trans_feat = np.sqrt(
            (meta_feat["est_dx"] - att.true_dx) ** 2 + (meta_feat["est_dy"] - att.true_dy) ** 2
        )

        raw_rows.append(
            ResultRow(
                run_id=run_id,
                experiment="geometric_sync",
                image_id=image_id,
                payload_id=payload_id,
                method=method_name,
                subband=subband,
                wavelet="haar",
                block_size=8,
                alpha_nominal=round(cal_alpha, 4),
                alpha_eff_mean=round(alpha_eff_mean, 6),
                attack=f"{att.name}_auto_feature",
                severity=att.severity,
                seed=42,
                psnr_clean=round(psnr_clean, 4),
                ssim_clean=round(ssim_clean, 6),
                psnr_attacked=round(psnr_att, 4),
                ber=round(float(compute_ber(payload, bits_feat)), 6),
                bcr=round(float(compute_bcr(payload, bits_feat)), 4),
                nc=round(float(compute_nc(payload, bits_feat)), 6),
                extra_json=json.dumps({
                    "scenario": "auto_feature",
                    "attack_family": att.name,
                    "reg_success": succ_feat,
                    "error_deg": round(err_angle_feat, 4),
                    "error_trans_px": round(err_trans_feat, 4),
                    "git_commit": git_hash,
                }),
            )
        )
        error_records.append({
            "image_id": image_id,
            "payload_id": payload_id,
            "method": method_name,
            "subband": subband,
            "attack": att.name,
            "severity": att.severity,
            "estimator": "auto_feature",
            "true_angle_deg": round(att.true_angle_deg, 4),
            "est_angle_deg": round(meta_feat["est_angle_deg"], 4),
            "error_angle_deg": round(err_angle_feat, 4),
            "true_scale": round(att.true_scale, 4),
            "est_scale": round(meta_feat["est_scale"], 4),
            "error_scale": round(err_scale_feat, 4),
            "true_dx": round(att.true_dx, 4),
            "true_dy": round(att.true_dy, 4),
            "est_dx": round(meta_feat["est_dx"], 4),
            "est_dy": round(meta_feat["est_dy"], 4),
            "error_trans_px": round(err_trans_feat, 4),
            "reg_success": succ_feat,
            "inliers": meta_feat.get("inliers", 0),
        })

        # ---------------------------------------------------------------------
        # Scenario 3: Oracle (known ground-truth inverse transform)
        # ---------------------------------------------------------------------
        bits_oracle = warp_back_and_extract(att_img, att.true_matrix, metadata)
        raw_rows.append(
            ResultRow(
                run_id=run_id,
                experiment="geometric_sync",
                image_id=image_id,
                payload_id=payload_id,
                method=method_name,
                subband=subband,
                wavelet="haar",
                block_size=8,
                alpha_nominal=round(cal_alpha, 4),
                alpha_eff_mean=round(alpha_eff_mean, 6),
                attack=f"{att.name}_oracle",
                severity=att.severity,
                seed=42,
                psnr_clean=round(psnr_clean, 4),
                ssim_clean=round(ssim_clean, 6),
                psnr_attacked=round(psnr_att, 4),
                ber=round(float(compute_ber(payload, bits_oracle)), 6),
                bcr=round(float(compute_bcr(payload, bits_oracle)), 4),
                nc=round(float(compute_nc(payload, bits_oracle)), 6),
                extra_json=json.dumps({"scenario": "oracle", "attack_family": att.name, "git_commit": git_hash}),
            )
        )

        # ---------------------------------------------------------------------
        # Clearly Labeled Oracle Check: extract_with_rotation_search()
        # ---------------------------------------------------------------------
        if att.name in ("rotation", "compound"):
            # Uses true watermark bits to find rotation angle (oracle-dependent)
            best_nc = -1.0
            best_angle = 0.0
            best_bits = None
            candidate_angles = np.arange(-10.0, 10.0 + 0.25, 0.5)
            for ang in candidate_angles:
                cand_bits = fast_batched_extract(att_img, metadata, deskew_angle=float(ang))
                nc_cand = float(compute_nc(metadata.watermark_binary, cand_bits))
                if nc_cand > best_nc:
                    best_nc = nc_cand
                    best_angle = float(ang)
                    best_bits = cand_bits
            bits_search = best_bits

            raw_rows.append(
                ResultRow(
                    run_id=run_id,
                    experiment="geometric_sync",
                    image_id=image_id,
                    payload_id=payload_id,
                    method=method_name,
                    subband=subband,
                    wavelet="haar",
                    block_size=8,
                    alpha_nominal=round(cal_alpha, 4),
                    alpha_eff_mean=round(alpha_eff_mean, 6),
                    attack=f"{att.name}_oracle_search",
                    severity=att.severity,
                    seed=42,
                    psnr_clean=round(psnr_clean, 4),
                    ssim_clean=round(ssim_clean, 6),
                    psnr_attacked=round(psnr_att, 4),
                    ber=round(float(compute_ber(payload, bits_search)), 6),
                    bcr=round(float(compute_bcr(payload, bits_search)), 4),
                    nc=round(float(compute_nc(payload, bits_search)), 6),
                    extra_json=json.dumps({
                        "scenario": "oracle_search",
                        "attack_family": att.name,
                        "detected_angle": best_angle,
                        "search_nc": round(best_nc, 6),
                        "oracle_dependent": True,
                        "rationale": "Directly accesses metadata.watermark_binary to maximize correlation",
                        "git_commit": git_hash,
                    }),
                )
            )

    cond_key = f"{image_id}_{payload_id}_{method_name}_{subband}"
    return cond_key, raw_rows, error_records


# =============================================================================
# 6. STUDY RUNNER AND REPORTING
# =============================================================================

def run_sync_study(
    num_images: int = 100,
    subset_seed: int = 0,
    target_psnr: float = 46.0,
    workers: int = 12,
    manifest_csv: str = "results_v2/dataset_manifest.csv",
    dataset_dir: str = "data/dataset_v2",
    sync_raw_csv: str = "results_v2/sync_raw.csv",
    error_csv: str = "results_v2/sync_estimation_error.csv",
) -> None:
    """Executes the full geometric synchronization study."""
    os.makedirs(os.path.dirname(os.path.abspath(sync_raw_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(error_csv)), exist_ok=True)
    init_results_csv(sync_raw_csv)

    # 1. Select 100 images
    selected_ids = select_benchmark_subset(
        manifest_csv=manifest_csv, num_images=num_images, seed=subset_seed
    )
    print(f"[*] Sync Study: Selected {len(selected_ids)} benchmark images (seed {subset_seed}).")

    # 2. Payloads: 2 per image (whitened logo and balanced_0)
    logo = generate_original_logo()
    whitened, _ = generate_whitened_logo(logo, seed=42)
    balanced_0 = generate_balanced_random_payload(payload_id=0, seed=500)
    payload_list = [("whitened_logo", whitened), ("balanced_0", balanced_0)]

    git_hash = get_git_commit_hash()
    run_id = f"sync_{int(time.time())}"

    # 3. Check completed conditions for resumability
    completed_keys: Set[Tuple[str, str, str, str]] = set()
    if os.path.exists(sync_raw_csv) and os.path.getsize(sync_raw_csv) > 0:
        df_prev = pd.read_csv(sync_raw_csv, usecols=["image_id", "payload_id", "method", "subband"])
        completed_keys = set(zip(df_prev["image_id"], df_prev["payload_id"], df_prev["method"], df_prev["subband"]))

    methods = ["proposed_fixed", "proposed_adaptive"]
    subbands = ["HL", "LL"]

    tasks = []
    for iid in selected_ids:
        img_path = os.path.join(dataset_dir, f"{iid}.png")
        for pid, p_arr in payload_list:
            for m in methods:
                for sb in subbands:
                    if (iid, pid, m, sb) in completed_keys:
                        continue
                    tasks.append(
                        (
                            run_id,
                            iid,
                            img_path,
                            pid,
                            p_arr,
                            m,
                            sb,
                            target_psnr,
                            git_hash,
                        )
                    )

    total_conditions = len(selected_ids) * len(payload_list) * len(methods) * len(subbands)
    print(f"[*] Sync Study: Total conditions: {total_conditions} | Completed: {len(completed_keys)} | Pending: {len(tasks)}")

    if not tasks:
        print("[*] All conditions already completed.")
        return

    # 4. Multiprocessing execution
    start_time = time.perf_counter()
    done_count = 0
    all_error_records: List[Dict[str, Any]] = []

    # Initialize error CSV headers if not exists
    if not os.path.exists(error_csv) or os.path.getsize(error_csv) == 0:
        pd.DataFrame(columns=[
            "image_id", "payload_id", "method", "subband", "attack", "severity",
            "estimator", "true_angle_deg", "est_angle_deg", "error_angle_deg",
            "true_scale", "est_scale", "error_scale", "true_dx", "true_dy",
            "est_dx", "est_dy", "error_trans_px", "reg_success", "inliers"
        ]).to_csv(error_csv, index=False)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_sync_condition, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                cond_key, r_rows, e_recs = fut.result()
                append_result_rows(sync_raw_csv, r_rows)
                if e_recs:
                    pd.DataFrame(e_recs).to_csv(error_csv, mode="a", header=False, index=False)
                    all_error_records.extend(e_recs)
                done_count += 1

                elapsed = time.perf_counter() - start_time
                rate = done_count / elapsed
                remaining = (len(tasks) - done_count) / rate if rate > 0 else 0
                print(
                    f"  [{done_count}/{len(tasks)}] Done: {cond_key} | {rate:.2f} cond/s | ETA: {remaining:.1f}s",
                    flush=True,
                )
            except Exception as e:
                t = futures[fut]
                print(f"[!] Error in sync condition {t[1]}_{t[3]}_{t[5]}_{t[6]}: {e}", file=sys.stderr)
                raise e

    print(f"\n[*] Sync study execution completed in {time.perf_counter() - start_time:.2f}s.")


def print_sync_analysis_tables(
    sync_raw_csv: str = "results_v2/sync_raw.csv",
    error_csv: str = "results_v2/sync_estimation_error.csv",
) -> None:
    """Generates and prints academic summary tables for registration and estimation error."""
    if not os.path.exists(sync_raw_csv) or not os.path.exists(error_csv):
        print("[!] Sync result files missing.")
        return

    df_raw = pd.read_csv(sync_raw_csv)
    df_err = pd.read_csv(error_csv)

    print("\n" + "=" * 120)
    print(" 1. GEOMETRIC ESTIMATION ERROR & REGISTRATION FAILURE RATE TABLE (IMAGE AS UNIT)")
    print("=" * 120)

    # Aggregate error per image first (Rule 4)
    img_err = (
        df_err.groupby(["attack", "severity", "estimator", "image_id"], as_index=False)[
            ["error_angle_deg", "error_trans_px", "error_scale", "reg_success"]
        ]
        .mean()
    )

    err_summary = []
    for (att, sev, est), grp in img_err.groupby(["attack", "severity", "estimator"]):
        mean_ang_err = grp["error_angle_deg"].mean()
        mean_trans_err = grp["error_trans_px"].mean()
        mean_scale_err = grp["error_scale"].mean()
        fail_rate = 1.0 - grp["reg_success"].mean()

        err_summary.append({
            "Attack": att,
            "Severity": sev,
            "Estimator": est,
            "Mean Angle Error (deg)": f"{mean_ang_err:.3f}°",
            "Mean Trans Error (px)": f"{mean_trans_err:.2f} px",
            "Mean Scale Error": f"{mean_scale_err:.4f}",
            "Failure Rate": f"{fail_rate * 100.0:.1f}%",
        })

    df_err_table = pd.DataFrame(err_summary)
    print(df_err_table.to_string(index=False))

    print("\n" + "=" * 120)
    print(" 2. WATERMARK EXTRACTION PERFORMANCE (BER) ACROSS REGISTRATION SCENARIOS (RULE 4)")
    print(" Comparing: None (No Reg) vs Auto ECC vs Auto Feature vs Oracle vs Oracle Search")
    print("=" * 120)

    # Extract scenario from attack string or extra_json
    def parse_scenario(att_str):
        for sc in ["oracle_search", "auto_feature", "auto_ecc", "oracle", "none"]:
            if att_str.endswith(sc):
                return sc
        return "none"

    def parse_attack_family(att_str):
        for sc in ["_oracle_search", "_auto_feature", "_auto_ecc", "_oracle", "_none"]:
            if att_str.endswith(sc):
                return att_str[:-len(sc)]
        return att_str

    df_raw["scenario"] = df_raw["attack"].apply(parse_scenario)
    df_raw["attack_family"] = df_raw["attack"].apply(parse_attack_family)

    # Average payloads per image first (Rule 4: image as unit)
    img_ber = (
        df_raw.groupby(
            ["method", "subband", "attack_family", "severity", "scenario", "image_id"], as_index=False
        )["ber"]
        .mean()
    )

    # Across images, compute median BER
    med_ber = (
        img_ber.groupby(
            ["method", "subband", "attack_family", "severity", "scenario"], as_index=False
        )["ber"]
        .median()
    )

    ber_records = []
    # Order of attacks
    att_order = [
        ("rotation", "0.5deg"), ("rotation", "1.0deg"), ("rotation", "2.0deg"),
        ("rotation", "5.0deg"), ("rotation", "10.0deg"),
        ("scaling", "0.9x"), ("scaling", "1.1x"),
        ("translation", "4px"), ("translation", "8px"),
        ("compound", "rotate2°+JPEG70"),
    ]

    for att_fam, sev in att_order:
        row = {"Attack": att_fam, "Severity": sev}
        for sb in ["HL", "LL"]:
            for sc in ["none", "auto_ecc", "auto_feature", "oracle", "oracle_search"]:
                sub = med_ber[
                    (med_ber["attack_family"] == att_fam)
                    & (med_ber["severity"] == sev)
                    & (med_ber["subband"] == sb)
                    & (med_ber["scenario"] == sc)
                ]
                if len(sub) > 0:
                    val = sub["ber"].iloc[0]
                    row[f"{sb}_{sc}"] = f"{val:.4f}"
                else:
                    row[f"{sb}_{sc}"] = "-"
        ber_records.append(row)

    df_ber_table = pd.DataFrame(ber_records)

    # Print HL and LL tables
    hl_cols = ["Attack", "Severity", "HL_none", "HL_auto_ecc", "HL_auto_feature", "HL_oracle", "HL_oracle_search"]
    print("\n--- SUBBAND HL: MEDIAN BER BY REGISTRATION SCENARIO ---")
    print(df_ber_table[[c for c in hl_cols if c in df_ber_table.columns]].to_string(index=False))

    ll_cols = ["Attack", "Severity", "LL_none", "LL_auto_ecc", "LL_auto_feature", "LL_oracle", "LL_oracle_search"]
    print("\n--- SUBBAND LL: MEDIAN BER BY REGISTRATION SCENARIO ---")
    print(df_ber_table[[c for c in ll_cols if c in df_ber_table.columns]].to_string(index=False))
    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Geometric Synchronization Study (experiments_v2)")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker processes")
    parser.add_argument("--num-images", type=int, default=100, help="Number of benchmark images")
    parser.add_argument("--subset-seed", type=int, default=0, help="Seed for random image subset")
    parser.add_argument("--target-psnr", type=float, default=46.0, help="Target clean PSNR in dB")
    parser.add_argument("--manifest-csv", type=str, default="results_v2/dataset_manifest.csv")
    parser.add_argument("--dataset-dir", type=str, default="data/dataset_v2")
    parser.add_argument("--sync-raw-csv", type=str, default="results_v2/sync_raw.csv")
    parser.add_argument("--error-csv", type=str, default="results_v2/sync_estimation_error.csv")
    args = parser.parse_args()

    run_sync_study(
        num_images=args.num_images,
        subset_seed=args.subset_seed,
        target_psnr=args.target_psnr,
        workers=args.workers,
        manifest_csv=args.manifest_csv,
        dataset_dir=args.dataset_dir,
        sync_raw_csv=args.sync_raw_csv,
        error_csv=args.error_csv,
    )

    print_sync_analysis_tables(sync_raw_csv=args.sync_raw_csv, error_csv=args.error_csv)


if __name__ == "__main__":
    main()
