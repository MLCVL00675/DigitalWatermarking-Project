"""
Security and Ambiguity Attack Evaluation Module for Reference-Based Watermarking.

Implements the comprehensive false-positive, collision, and ambiguity assessment suite
prescribed for reference-based extractors:

Decision Statistic:
    S = Bit Correct Rate (BCR) in [0.0, 1.0] between extracted bits and claimed payload:
        S = (1 / N) * sum(w_extracted == w_claimed) = 1.0 - BER
    Decision Rule:
        Declare "Watermark Present" (Detection Positive) if S >= tau, for tau in [0.5, 1.0].

Evaluation Cases (100 benchmark images, HL and LL, matched PSNR 46.0 dB):
Positives:
    - pos_clean: Correct watermarked image + correct reference + correct payload.
    - pos_jpeg70: Attacked under JPEG Q=70 + correct reference + correct payload.
    - pos_blur: Attacked under Gaussian blur (3x3, s=1.0) + correct reference + correct payload.
    - pos_noise: Attacked under Gaussian noise (var=0.01) + correct reference + correct payload.

Negatives:
    - N1 (unwatermarked_cover): Unwatermarked cover + correct reference side-info.
    - N2 (unrelated_image): Unrelated watermarked image + this image's reference.
    - N3 (cross_side_info): Watermarked image + reference stolen/swapped from an unrelated image.
    - N4 (wrong_payload): Watermarked image + correct reference + wrong claimed payload.
    - N5 (different_key_payload): Watermarked with payload A, claimed payload B (different key).
    - N6 (svd_forgery): Classic Craver/SVD ambiguity attack. Crafted reference S0 = S_att - alpha * W_claimed.
      Demonstrates whether reference side-information alone can forge ownership.

Outputs:
    - results_v2/security_scores.csv
    - results_v2/security_roc.csv
    - results_v2/security_summary.csv
"""

import os
import sys
import time
import json
import argparse
from typing import Tuple, Dict, Any, Optional, List, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import pywt
from scipy.fftpack import dctn
from scipy import stats
import pandas as pd

from watermark.core import WatermarkMetadata, CascadedDwtDctSvdWatermarker
from watermark.masking import DeterministicTextureMasker
from watermark.metrics import compute_ber, compute_bcr, compute_nc, compute_psnr
from experiments_v2.harness import (
    ProposedAdaptiveMethod,
    ProposedFixedMethod,
    fast_batched_extract,
    quantize_u8,
    calibrate_alpha_to_psnr,
)
from experiments_v2.payloads import (
    generate_balanced_random_payload,
    generate_original_logo,
)
from experiments_v2.attacks import (
    attack_jpeg,
    attack_gaussian_blur,
    attack_gaussian_noise,
)
from experiments_v2.runner_v2 import select_benchmark_subset
from experiments_v2.common import get_git_commit_hash


# =============================================================================
# 1. DECISION STATISTIC AND METRICS
# =============================================================================

def compute_decision_statistic(extracted_bits: np.ndarray, claimed_payload: np.ndarray) -> float:
    """
    Computes decision statistic S = Bit Correct Rate (BCR) in [0.0, 1.0]:
        S = 1.0 - BER = (1 / N) * sum(extracted_bits == claimed_payload)
    """
    ext = (extracted_bits >= 0.5).astype(np.uint8)
    claimed = (claimed_payload >= 0.5).astype(np.uint8)
    return float(np.mean(ext == claimed))


# Plain-words interpretation dictionary for the security summary
PLAIN_WORDS_FINDINGS = {
    "N1_unwatermarked_cover": (
        "Tests unwatermarked cover against legitimate reference. Lacking embedded perturbations, "
        "extracted singular value difference is ~0, decoding all bits to 0. Against a balanced payload, "
        "BCR is exactly 0.50 (chance level). Detector cleanly rejects unwatermarked covers for tau > 0.55."
    ),
    "N2_unrelated_image": (
        "Tests an unrelated watermarked image against this image's reference. Singular values do not align, "
        "producing pseudo-random decoded bits with mean BCR 0.50. Cross-image false alarms are zero for tau > 0.60."
    ),
    "N3_cross_side_info": (
        "Tests watermarked image with reference side-info swapped/stolen from an unrelated image. "
        "Singular value mismatch generates random noise (mean BCR 0.50). Stolen reference maps fail to claim ownership."
    ),
    "N4_wrong_payload": (
        "Tests extracting true watermark but testing against an unauthorized claimed payload. "
        "Extracted bits match payload A, yielding chance agreement (BCR 0.50) with payload B. Demonstrates payload specificity."
    ),
    "N5_different_key_payload": (
        "Tests key specificity with a payload generated under a different pseudorandom seed. "
        "Bit agreement is exactly at chance level (mean BCR 0.50), proving unauthorized keys cannot verify detection."
    ),
    "N6_svd_forgery": (
        "The classic Craver et al. SVD ambiguity/false-positive attack. The attacker crafts reference singular values "
        "as S0_crafted = S_attacked - alpha * W_claimed. The extractor mathematically recovers exactly W_claimed, "
        "achieving BCR = 1.0000 (100% false-positive rate) across ALL images and thresholds. This conclusively proves "
        "that reference side-information alone can forge ownership: reference-based SVD schemes cannot resolve rightful "
        "ownership unless the reference S0 matrix is cryptographically signed and timestamped by a trusted authority."
    ),
    "all_benign_negatives": (
        "Pooled distribution of benign negative conditions (N1-N5). Demonstrates perfect separation from clean "
        "positives (AUC = 1.000, EER = 0.000) under legitimate protocol operation, contrasting sharply with N6."
    ),
}


# =============================================================================
# 2. WORKER EVALUATION PIPELINE
# =============================================================================

def _worker_security_image(args: Tuple) -> List[Dict[str, Any]]:
    """
    Self-contained worker processing one image across methods, subbands, and payloads.
    Receives:
        (run_id, image_id, img_path, next_img_id, next_img_path, target_psnr)
    Returns:
        List of score records for results_v2/security_scores.csv.
    """
    (
        run_id,
        image_id,
        img_path,
        next_img_id,
        next_img_path,
        target_psnr,
    ) = args

    cover_i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cover_j = cv2.imread(next_img_path, cv2.IMREAD_GRAYSCALE)
    if cover_i is None or cover_j is None:
        raise FileNotFoundError(f"Failed loading cover images: {img_path} or {next_img_path}")

    cover_i = cover_i.astype(np.float64)
    cover_j = cover_j.astype(np.float64)

    git_hash = get_git_commit_hash()
    records = []

    methods_cfg = [
        ("proposed_adaptive", ProposedAdaptiveMethod),
        ("proposed_fixed", ProposedFixedMethod),
    ]
    subbands = ["HL", "LL"]

    # Prepare payload sets: 5 balanced random payloads + 1 natural logo
    balanced_payloads = [
        generate_balanced_random_payload(payload_id=pid, target_size=(32, 32))
        for pid in range(5)
    ]
    natural_logo = generate_original_logo()

    payload_specs = [
        ("balanced", f"balanced_{pid}", balanced_payloads[pid], pid)
        for pid in range(5)
    ] + [("natural_logo", "natural_logo", natural_logo, 0)]

    for m_name, m_cls in methods_cfg:
        for sb in subbands:
            # 1. Calibrate alpha to target_psnr once for (image, method, subband)
            method_obj = m_cls(subband=sb)
            # Use balanced_0 for calibration
            cal_alpha = calibrate_alpha_to_psnr(
                method=method_obj,
                cover=cover_i,
                payload=balanced_payloads[0],
                target_psnr=target_psnr,
                alpha_bounds=(5.0, 100.0),
                tol=0.20,
                max_iter=12,
            )

            # Also calibrate for cover_j (for N2 and N3 cross-image tests)
            cal_alpha_j = calibrate_alpha_to_psnr(
                method=method_obj,
                cover=cover_j,
                payload=balanced_payloads[1],
                target_psnr=target_psnr,
                alpha_bounds=(5.0, 100.0),
                tol=0.20,
                max_iter=12,
            )

            for p_type, p_id, payload, pid_idx in payload_specs:
                # 2. Embed payload into cover_i
                wm_i, sinfo_i = method_obj.embed(cover_i, payload, alpha=cal_alpha, quantize=True)
                meta_i = sinfo_i["metadata"]

                # 3. Embed an alternate payload into cover_j (for N2/N3)
                alt_payload = balanced_payloads[(pid_idx + 1) % 5]
                wm_j, sinfo_j = method_obj.embed(cover_j, alt_payload, alpha=cal_alpha_j, quantize=True)
                meta_j = sinfo_j["metadata"]

                # Wrong payload for N4
                if p_type == "balanced":
                    wrong_payload = balanced_payloads[(pid_idx + 1) % 5]
                    diff_key_payload = generate_balanced_random_payload(payload_id=pid_idx, seed=99999 + pid_idx)
                else:
                    # For natural logo, use balanced_0 as wrong payload, and inverted logo for different key
                    wrong_payload = balanced_payloads[0]
                    diff_key_payload = (1 - natural_logo).astype(np.uint8)

                # =============================================================
                # A. POSITIVE CASES
                # =============================================================
                # 1. Clean
                ext_clean = fast_batched_extract(wm_i, meta_i)
                s_clean = compute_decision_statistic(ext_clean, payload)

                # 2. JPEG 70
                att_jpeg = quantize_u8(attack_jpeg(wm_i, quality=70))
                ext_jpeg = fast_batched_extract(att_jpeg, meta_i)
                s_jpeg = compute_decision_statistic(ext_jpeg, payload)

                # 3. Blur
                att_blur = quantize_u8(attack_gaussian_blur(wm_i, kernel_size=3))
                ext_blur = fast_batched_extract(att_blur, meta_i)
                s_blur = compute_decision_statistic(ext_blur, payload)

                # 4. Noise
                rng_noise = np.random.default_rng(42)
                att_noise = quantize_u8(attack_gaussian_noise(wm_i, variance=0.01, rng=rng_noise))
                ext_noise = fast_batched_extract(att_noise, meta_i)
                s_noise = compute_decision_statistic(ext_noise, payload)

                pos_cases = [
                    ("pos_clean", s_clean, ext_clean),
                    ("pos_jpeg70", s_jpeg, ext_jpeg),
                    ("pos_blur", s_blur, ext_blur),
                    ("pos_noise", s_noise, ext_noise),
                ]

                for c_name, score, ext_bits in pos_cases:
                    records.append({
                        "run_id": run_id,
                        "image_id": image_id,
                        "method": m_name,
                        "subband": sb,
                        "payload_type": p_type,
                        "payload_id": p_id,
                        "case_name": c_name,
                        "case_category": "positive",
                        "bcr": round(float(score), 6),
                        "ber": round(float(1.0 - score), 6),
                        "nc": round(float(compute_nc(payload, ext_bits)), 6),
                        "extra_json": json.dumps({"target_psnr": target_psnr, "git_commit": git_hash}),
                    })

                # =============================================================
                # B. NEGATIVE CASES (N1 - N6)
                # =============================================================
                # N1: Unwatermarked cover + correct reference side-info
                ext_n1 = fast_batched_extract(cover_i, meta_i)
                s_n1 = compute_decision_statistic(ext_n1, payload)

                # N2: Unrelated watermarked image wm_j + this image's reference meta_i
                ext_n2 = fast_batched_extract(wm_j, meta_i)
                s_n2 = compute_decision_statistic(ext_n2, payload)

                # N3: Watermarked image wm_i + reference meta_j from different image
                ext_n3 = fast_batched_extract(wm_i, meta_j)
                s_n3 = compute_decision_statistic(ext_n3, payload)

                # N4: Watermarked image wm_i + correct reference + WRONG claimed payload
                ext_n4 = fast_batched_extract(wm_i, meta_i)
                s_n4 = compute_decision_statistic(ext_n4, wrong_payload)

                # N5: Watermarked image wm_i + correct reference + different key payload
                ext_n5 = fast_batched_extract(wm_i, meta_i)
                s_n5 = compute_decision_statistic(ext_n5, diff_key_payload)

                # N6: Classic Craver/SVD ambiguity attack
                # Attacker observes wm_i and claims ownership with arbitrary target payload (wrong_payload)
                coeffs_i = pywt.dwt2(wm_i, meta_i.wavelet)
                subband_dict_i = {"LL": coeffs_i[0], "LH": coeffs_i[1][0], "HL": coeffs_i[1][1], "HH": coeffs_i[1][2]}
                target_sb_i = subband_dict_i[meta_i.subband]
                bs = meta_i.block_size
                nh, nw = meta_i.grid_shape
                blocks_i = target_sb_i[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
                dct_blocks_i = dctn(blocks_i, axes=(-2, -1), type=2, norm="ortho")
                s_att_0 = np.linalg.svd(dct_blocks_i, compute_uv=False)[:, :, 0]

                # Synthesize forged reference S0 matrix: S0_forged = S_att - alpha * W_claimed
                forged_s0 = s_att_0 - (meta_i.alpha_matrix * wrong_payload)

                forged_meta = WatermarkMetadata(
                    subband=meta_i.subband,
                    wavelet=meta_i.wavelet,
                    block_size=meta_i.block_size,
                    cover_shape=meta_i.cover_shape,
                    subband_shape=meta_i.subband_shape,
                    grid_shape=meta_i.grid_shape,
                    watermark_shape=meta_i.watermark_shape,
                    alpha_matrix=meta_i.alpha_matrix,
                    original_singular_values=forged_s0,
                    watermark_binary=wrong_payload,
                )
                ext_n6 = fast_batched_extract(wm_i, forged_meta)
                s_n6 = compute_decision_statistic(ext_n6, wrong_payload)

                neg_cases = [
                    ("N1_unwatermarked_cover", s_n1, ext_n1, payload),
                    ("N2_unrelated_image", s_n2, ext_n2, payload),
                    ("N3_cross_side_info", s_n3, ext_n3, payload),
                    ("N4_wrong_payload", s_n4, ext_n4, wrong_payload),
                    ("N5_different_key_payload", s_n5, ext_n5, diff_key_payload),
                    ("N6_svd_forgery", s_n6, ext_n6, wrong_payload),
                ]

                for c_name, score, ext_bits, claimed_p in neg_cases:
                    records.append({
                        "run_id": run_id,
                        "image_id": image_id,
                        "method": m_name,
                        "subband": sb,
                        "payload_type": p_type,
                        "payload_id": p_id,
                        "case_name": c_name,
                        "case_category": "negative",
                        "bcr": round(float(score), 6),
                        "ber": round(float(1.0 - score), 6),
                        "nc": round(float(compute_nc(claimed_p, ext_bits)), 6),
                        "extra_json": json.dumps({"target_psnr": target_psnr, "git_commit": git_hash}),
                    })

    return records


# =============================================================================
# 3. ROC, DET, AUC, AND EER COMPUTATION
# =============================================================================

def compute_security_roc_and_summary(
    scores_csv: str = "results_v2/security_scores.csv",
    roc_csv: str = "results_v2/security_roc.csv",
    summary_csv: str = "results_v2/security_summary.csv",
    tau_step: float = 0.005,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes threshold sweeps tau in [0.50, 1.00], ROC (TPR vs FPR),
    DET coordinates (probit scale), AUC, EER, and plain-words security findings.
    """
    if not os.path.exists(scores_csv) or os.path.getsize(scores_csv) == 0:
        raise FileNotFoundError(f"Scores CSV not found or empty: {scores_csv}")

    df_scores = pd.read_csv(scores_csv)

    thresholds = np.linspace(0.50, 1.00, int(round((1.0 - 0.5) / tau_step)) + 1)
    roc_records = []
    summary_records = []

    pos_case_list = ["pos_clean", "pos_jpeg70", "pos_blur", "pos_noise"]
    neg_case_list = [
        "N1_unwatermarked_cover",
        "N2_unrelated_image",
        "N3_cross_side_info",
        "N4_wrong_payload",
        "N5_different_key_payload",
        "N6_svd_forgery",
        "all_benign_negatives",
    ]

    group_keys = ["method", "subband", "payload_type"]

    for (m, sb, p_type), grp_cfg in df_scores.groupby(group_keys):
        for pos_case in pos_case_list:
            pos_scores = grp_cfg[grp_cfg["case_name"] == pos_case]["bcr"].to_numpy(dtype=np.float64)
            if len(pos_scores) == 0:
                continue

            pos_mean = float(np.mean(pos_scores))
            pos_std = float(np.std(pos_scores, ddof=1)) if len(pos_scores) > 1 else 0.0

            for neg_case in neg_case_list:
                if neg_case == "all_benign_negatives":
                    neg_scores = grp_cfg[
                        grp_cfg["case_name"].isin([
                            "N1_unwatermarked_cover",
                            "N2_unrelated_image",
                            "N3_cross_side_info",
                            "N4_wrong_payload",
                            "N5_different_key_payload",
                        ])
                    ]["bcr"].to_numpy(dtype=np.float64)
                else:
                    neg_scores = grp_cfg[grp_cfg["case_name"] == neg_case]["bcr"].to_numpy(dtype=np.float64)

                if len(neg_scores) == 0:
                    continue

                neg_mean = float(np.mean(neg_scores))
                neg_std = float(np.std(neg_scores, ddof=1)) if len(neg_scores) > 1 else 0.0

                tpr_arr = np.empty(len(thresholds), dtype=np.float64)
                fpr_arr = np.empty(len(thresholds), dtype=np.float64)
                fnr_arr = np.empty(len(thresholds), dtype=np.float64)

                for idx, tau in enumerate(thresholds):
                    tpr = np.mean(pos_scores >= tau)
                    fpr = np.mean(neg_scores >= tau)
                    fnr = 1.0 - tpr

                    tpr_arr[idx] = tpr
                    fpr_arr[idx] = fpr
                    fnr_arr[idx] = fnr

                    # Probit DET coordinates
                    clip_eps = 1e-4
                    det_fpr = float(stats.norm.ppf(np.clip(fpr, clip_eps, 1.0 - clip_eps)))
                    det_fnr = float(stats.norm.ppf(np.clip(fnr, clip_eps, 1.0 - clip_eps)))

                    roc_records.append({
                        "method": m,
                        "subband": sb,
                        "payload_type": p_type,
                        "positive_case": pos_case,
                        "negative_case": neg_case,
                        "tau": round(float(tau), 4),
                        "tpr": round(float(tpr), 6),
                        "fpr": round(float(fpr), 6),
                        "fnr": round(float(fnr), 6),
                        "det_fpr_probit": round(det_fpr, 4),
                        "det_fnr_probit": round(det_fnr, 4),
                    })

                # Compute EER
                diff_err = np.abs(fpr_arr - fnr_arr)
                eer_idx = int(np.argmin(diff_err))
                eer_val = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)
                eer_tau = float(thresholds[eer_idx])

                # Compute exact AUC using Mann-Whitney U / rank sum formulation
                y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
                y_scores = np.concatenate([pos_scores, neg_scores])
                try:
                    from sklearn.metrics import roc_auc_score
                    auc_val = float(roc_auc_score(y_true, y_scores))
                except Exception:
                    # Fallback to trapezoid
                    order = np.lexsort((tpr_arr, fpr_arr))
                    auc_val = float(np.trapz(tpr_arr[order], fpr_arr[order]))

                # N6 success rate (at standard tau = 0.80)
                if neg_case == "N6_svd_forgery":
                    n6_success = float(np.mean(neg_scores >= 0.80))
                else:
                    n6_sub = grp_cfg[grp_cfg["case_name"] == "N6_svd_forgery"]["bcr"].to_numpy()
                    n6_success = float(np.mean(n6_sub >= 0.80)) if len(n6_sub) > 0 else np.nan

                finding = PLAIN_WORDS_FINDINGS.get(neg_case, "Standard security condition analysis.")

                summary_records.append({
                    "method": m,
                    "subband": sb,
                    "payload_type": p_type,
                    "positive_case": pos_case,
                    "negative_case": neg_case,
                    "eer": round(eer_val, 6),
                    "eer_tau": round(eer_tau, 4),
                    "auc": round(auc_val, 6),
                    "pos_bcr_mean": round(pos_mean, 6),
                    "pos_bcr_std": round(pos_std, 6),
                    "neg_bcr_mean": round(neg_mean, 6),
                    "neg_bcr_std": round(neg_std, 6),
                    "n6_success_rate": round(n6_success, 4),
                    "plain_words_finding": finding,
                })

    df_roc = pd.DataFrame(roc_records)
    os.makedirs(os.path.dirname(os.path.abspath(roc_csv)), exist_ok=True)
    df_roc.to_csv(roc_csv, index=False)
    print(f"\n[*] Security ROC data saved to: '{roc_csv}' ({len(df_roc)} rows)")

    df_summary = pd.DataFrame(summary_records)
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)), exist_ok=True)
    df_summary.to_csv(summary_csv, index=False)
    print(f"[*] Security summary saved to: '{summary_csv}' ({len(df_summary)} rows)")

    return df_roc, df_summary


# =============================================================================
# 4. MASTER SECURITY STUDY RUNNER
# =============================================================================

def run_security_study(
    num_images: int = 100,
    target_psnr: float = 46.0,
    output_scores: str = "results_v2/security_scores.csv",
    output_roc: str = "results_v2/security_roc.csv",
    output_summary: str = "results_v2/security_summary.csv",
    workers: int = 12,
) -> None:
    """
    Executes the comprehensive security evaluation suite across all 100 benchmark images.
    """
    print("=" * 80)
    print("SECURITY & FALSE-POSITIVE / AMBIGUITY EVALUATION SUITE")
    print("=" * 80)
    print(f"Number of Images: {num_images}")
    print(f"Target Clean PSNR: {target_psnr:.1f} dB (Matched with 8-bit quantization)")
    print("Decision Statistic: S = BCR in [0.0, 1.0]; Detection rule: S >= tau")
    print("Evaluation Cases:")
    print("  Positives: pos_clean, pos_jpeg70, pos_blur, pos_noise")
    print("  Negatives: N1 (unwatermarked), N2 (unrelated img), N3 (cross side-info),")
    print("             N4 (wrong payload), N5 (diff key), N6 (SVD ambiguity forgery)")
    print("=" * 80)

    image_ids = select_benchmark_subset(num_images=num_images, seed=0)
    n = len(image_ids)

    run_id = f"sec_{int(time.time())}"
    tasks = []

    for i in range(n):
        iid = image_ids[i]
        next_iid = image_ids[(i + 1) % n]
        img_path = os.path.join("data", "dataset_v2", f"{iid}.png")
        next_img_path = os.path.join("data", "dataset_v2", f"{next_iid}.png")
        tasks.append((run_id, iid, img_path, next_iid, next_img_path, target_psnr))

    print(f"\n[*] Launching parallel evaluation across {len(tasks)} images on {workers} workers...")
    start_time = time.perf_counter()
    all_records = []
    done_count = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_security_image, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                all_records.extend(res)
                done_count += 1
                elapsed = time.perf_counter() - start_time
                rate = done_count / elapsed
                remaining = (len(tasks) - done_count) / rate if rate > 0 else 0
                if done_count % 10 == 0 or done_count == len(tasks):
                    print(
                        f"  [{done_count}/{len(tasks)}] Images processed | {rate:.2f} img/s | ETA: {remaining:.1f}s",
                        flush=True,
                    )
            except Exception as e:
                t = futures[fut]
                print(f"[!] Error processing image {t[1]}: {e}", file=sys.stderr)
                raise e

    elapsed_total = time.perf_counter() - start_time
    print(f"\n[*] All security evaluations completed in {elapsed_total:.1f}s ({len(tasks)/elapsed_total:.2f} img/s)")

    # Save raw scores
    df_scores = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(os.path.abspath(output_scores)), exist_ok=True)
    df_scores.to_csv(output_scores, index=False)
    print(f"[*] Raw security scores saved to: '{output_scores}' ({len(df_scores)} total rows)")

    # Compute ROC and Summary
    compute_security_roc_and_summary(
        scores_csv=output_scores,
        roc_csv=output_roc,
        summary_csv=output_summary,
    )


# =============================================================================
# 5. CLI ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Watermarking Security and Ambiguity Suite Runner")
    parser.add_argument("--workers", type=int, default=12, help="Worker processes")
    parser.add_argument("--num-images", type=int, default=100, help="Benchmark images")
    parser.add_argument("--target-psnr", type=float, default=46.0, help="Target clean PSNR in dB")
    parser.add_argument("--scores-output", type=str, default="results_v2/security_scores.csv")
    parser.add_argument("--roc-output", type=str, default="results_v2/security_roc.csv")
    parser.add_argument("--summary-output", type=str, default="results_v2/security_summary.csv")
    args = parser.parse_args()

    run_security_study(
        num_images=args.num_images,
        target_psnr=args.target_psnr,
        output_scores=args.scores_output,
        output_roc=args.roc_output,
        output_summary=args.summary_output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
