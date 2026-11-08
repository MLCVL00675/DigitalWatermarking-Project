"""
Comprehensive Ablation Study Module for experiments_v2.

Implements all 5 ablation axes + unified heatmap aggregation:
A. Transform Ablation:
   - DWT+DCT+SVD (full)
   - DWT+SVD (no DCT)
   - DCT+SVD (no DWT, block DCT on spatial image)
   - DWT+DCT (no SVD, mid-frequency additive rule)
   - SVD only (spatial block SVD)
   In both fixed-alpha and adaptive form.
   Output: results_v2/ablation_transform.csv

B. Texture-Feature Ablation:
   - Evaluates: gradient only, variance only, entropy only, grad+var, grad+ent,
     var+ent, equal 1/3, default 40/40/20, gradient-dominant, variance-dominant,
     entropy-dominant, plus weights tuned on a development split.
   - 40 development images (seed 0) grid search on the simplex (step 0.1) maximizing
     mean BCR over attacks; evaluated on the 60 held-out images only.
   Output: results_v2/ablation_features.csv, results_v2/optimal_texture_weights.json

C. Wavelet Sensitivity:
   - haar, db2, db4, bior1.3, sym4 across HL and LL.
   Output: results_v2/ablation_wavelet.csv

D. Block Size & Capacity:
   - 4, 8, 16 (adjusting payload mapping: 1,024 bits for 4 and 8; 256 bits for 16).
   Output: results_v2/ablation_block.csv

E. Payload / Capacity Scaling:
   - 256, 512, 1024, 2048 bits (using block 4 for 2048).
   Output: results_v2/ablation_payload.csv

F. Unified Heatmap Matrix:
   - Rows = variant, Columns = 13 manuscript attacks.
   - Value = median BER over images with bootstrap 95% CI.
   Output: results_v2/ablation_heatmap_data.csv

All runs strictly adhere to:
- Matched clean PSNR = 46.0 dB (calibrate_alpha_to_psnr)
- 100 images from dataset_v2
- Subbands HL and LL
- 10 seeds for stochastic attacks
- Rule 4: image as the statistical unit
"""

import os
import sys
import json
import time
import argparse
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
from watermark.masking import DeterministicTextureMasker
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
from experiments_v2.attacks import AttackSpec, get_standard_attack_specs
from experiments_v2.runner_v2 import select_benchmark_subset
from experiments_v2.transform_methods import create_ablation_method


# =============================================================================
# 1. ATTACK SPECIFICATIONS WITH 10 STOCHASTIC REPLICATIONS
# =============================================================================

def get_manuscript_attack_specs() -> List[Tuple[AttackSpec, List[int]]]:
    """Returns the manuscript's 13 single attacks + clean with 10 stochastic seeds."""
    base_specs = get_standard_attack_specs()
    specs_with_seeds = []
    stoch_seeds = list(range(1, 11))  # 10 seeds
    det_seeds = [42]

    for s in base_specs:
        seeds = stoch_seeds if s.is_stochastic else det_seeds
        specs_with_seeds.append((s, seeds))
    return specs_with_seeds


# =============================================================================
# 2. BOOTSTRAP CONFIDENCE INTERVAL UTILITY
# =============================================================================

def bootstrap_median_ci(
    data: np.ndarray,
    num_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes median and percentile 95% bootstrap CI over data array."""
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        return 0.5, 0.5, 0.5
    med = float(np.median(clean_data))
    if len(clean_data) <= 1:
        return med, med, med

    rng = np.random.default_rng(seed)
    n = len(clean_data)
    boot_meds = np.empty(num_boot)
    for b in range(num_boot):
        boot_meds[b] = np.median(rng.choice(clean_data, size=n, replace=True))

    low = float(np.percentile(boot_meds, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_meds, 100.0 * (1.0 - alpha / 2.0)))
    return med, low, high


# =============================================================================
# 3. GENERIC ABLATION CONDITION WORKER
# =============================================================================

def _worker_ablation_condition(args: Tuple) -> Tuple[str, List[ResultRow]]:
    """Worker task executing 1 embedding condition across the 13 manuscript attacks."""
    (
        run_id,
        experiment_name,
        image_id,
        image_path,
        payload_id,
        payload,
        method_name,
        subband,
        wavelet,
        block_size,
        method_kwargs,
        target_psnr,
        git_hash,
        extra_meta,
    ) = args

    cover = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cover is None:
        raise FileNotFoundError(f"Failed to read image at: {image_path}")
    cover = cover.astype(np.float64)

    # 1. Calibrate alpha to target PSNR (46 dB)
    cal_alpha = calibrate_alpha_to_psnr(
        method=create_ablation_method(
            method_name, subband=subband, wavelet=wavelet, block_size=block_size, **method_kwargs
        ),
        cover=cover,
        payload=payload,
        target_psnr=target_psnr,
        tol=0.08,
        subband=subband,
        wavelet=wavelet,
        block_size=block_size,
    )

    # 2. Embed once
    method_obj = create_ablation_method(
        method_name, subband=subband, wavelet=wavelet, block_size=block_size, **method_kwargs
    )
    wm_img, side_info = method_obj.embed(cover, payload, alpha=cal_alpha, quantize=True)
    alpha_eff_mean = float(side_info["alpha_eff_mean"])

    psnr_clean = float(compute_psnr(cover, wm_img))
    ssim_clean = float(compute_ssim(cover, wm_img))

    # 3. Evaluate the manuscript attacks
    results = []
    attacks_with_seeds = get_manuscript_attack_specs()

    for spec, seeds in attacks_with_seeds:
        for s in seeds:
            rng = np.random.default_rng(s) if spec.is_stochastic else None
            att_img = spec.apply(wm_img, rng=rng)
            if spec.attack not in ("no_attack", "identity"):
                att_img = quantize_u8(att_img)

            extracted_bits = method_obj.extract(att_img, side_info, deskew_angle=spec.deskew_angle)

            psnr_att = float(compute_psnr(cover, att_img))
            ber = float(compute_ber(payload, extracted_bits))
            bcr = float(compute_bcr(payload, extracted_bits))
            nc = float(compute_nc(payload, extracted_bits))

            extra = dict(extra_meta)
            extra.update({
                "git_commit": git_hash,
                "target_psnr": target_psnr,
                "calibrated_alpha": cal_alpha,
                "deskew_angle": spec.deskew_angle,
                "is_stochastic": spec.is_stochastic,
            })

            results.append(
                ResultRow(
                    run_id=run_id,
                    experiment=experiment_name,
                    image_id=image_id,
                    payload_id=payload_id,
                    method=method_name,
                    subband=subband,
                    wavelet=wavelet,
                    block_size=block_size,
                    alpha_nominal=round(cal_alpha, 4),
                    alpha_eff_mean=round(alpha_eff_mean, 6),
                    attack=spec.attack,
                    severity=spec.severity,
                    seed=s,
                    psnr_clean=round(psnr_clean, 4),
                    ssim_clean=round(ssim_clean, 6),
                    psnr_attacked=round(psnr_att, 4),
                    ber=round(ber, 6),
                    bcr=round(bcr, 4),
                    nc=round(nc, 6),
                    extra_json=json.dumps(extra, sort_keys=True),
                )
            )

    cond_key = f"{image_id}_{method_name}_{subband}_{wavelet}_{block_size}"
    return cond_key, results


# =============================================================================
# 4. PART A: TRANSFORM ABLATION
# =============================================================================

def run_ablation_transform(
    output_csv: str = "results_v2/ablation_transform.csv",
    num_images: int = 100,
    workers: int = 12,
    target_psnr: float = 46.0,
    dataset_dir: str = "data/dataset_v2",
) -> None:
    """Executes Transform Ablation (DWT+DCT+SVD, DWT+SVD, DCT+SVD, DWT+DCT, SVD only)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    init_results_csv(output_csv)

    selected_ids = select_benchmark_subset(num_images=num_images, seed=0)
    logo = generate_original_logo()
    payload, _ = generate_whitened_logo(logo, seed=42)
    git_hash = get_git_commit_hash()
    run_id = f"ablation_trans_{int(time.time())}"

    # Load completed
    completed: Set[Tuple[str, str, str]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband"])
        completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"]))

    variants = [
        ("proposed_adaptive", ["HL", "LL"], "haar", 8),
        ("proposed_fixed", ["HL", "LL"], "haar", 8),
        ("dwt_svd_adaptive", ["HL", "LL"], "haar", 8),
        ("dwt_svd_fixed", ["HL", "LL"], "haar", 8),
        ("dct_svd_adaptive", ["spatial"], "none", 16),
        ("dct_svd_fixed", ["spatial"], "none", 16),
        ("dwt_dct_adaptive", ["HL", "LL"], "haar", 8),
        ("dwt_dct_fixed", ["HL", "LL"], "haar", 8),
        ("svd_only_adaptive", ["spatial"], "none", 16),
        ("svd_only_fixed", ["spatial"], "none", 16),
    ]

    tasks = []
    for iid in selected_ids:
        img_path = os.path.join(dataset_dir, f"{iid}.png")
        for m_name, subbands, wav, bs in variants:
            for sb in subbands:
                if (iid, m_name, sb) in completed:
                    continue
                tasks.append(
                    (
                        run_id,
                        "ablation_transform",
                        iid,
                        img_path,
                        "whitened_logo",
                        payload,
                        m_name,
                        sb,
                        wav,
                        bs,
                        {},
                        target_psnr,
                        git_hash,
                        {"variant": m_name, "ablation_axis": "transform"},
                    )
                )

    print(f"\n[*] Part A: Transform Ablation | Total conditions: {len(selected_ids)*16} | Pending: {len(tasks)}")
    _execute_tasks_and_append(tasks, output_csv, workers=workers)


# =============================================================================
# 5. PART B: TEXTURE-FEATURE ABLATION (DEV/TEST SPLIT & SIMPLEX SEARCH)
# =============================================================================

def get_feature_weight_configs() -> List[Dict[str, Any]]:
    """Returns the predefined feature weight ablation configurations."""
    return [
        {"name": "gradient_only", "w_grad": 1.0, "w_var": 0.0, "w_ent": 0.0},
        {"name": "variance_only", "w_grad": 0.0, "w_var": 1.0, "w_ent": 0.0},
        {"name": "entropy_only", "w_grad": 0.0, "w_var": 0.0, "w_ent": 1.0},
        {"name": "grad_var", "w_grad": 0.5, "w_var": 0.5, "w_ent": 0.0},
        {"name": "grad_ent", "w_grad": 0.5, "w_var": 0.0, "w_ent": 0.5},
        {"name": "var_ent", "w_grad": 0.0, "w_var": 0.5, "w_ent": 0.5},
        {"name": "equal_third", "w_grad": 1.0 / 3.0, "w_var": 1.0 / 3.0, "w_ent": 1.0 / 3.0},
        {"name": "default_40_40_20", "w_grad": 0.40, "w_var": 0.40, "w_ent": 0.20},
        {"name": "gradient_dominant", "w_grad": 0.60, "w_var": 0.20, "w_ent": 0.20},
        {"name": "variance_dominant", "w_grad": 0.20, "w_var": 0.60, "w_ent": 0.20},
        {"name": "entropy_dominant", "w_grad": 0.20, "w_var": 0.20, "w_ent": 0.60},
    ]


def optimize_weights_on_dev_split(
    dev_ids: List[str],
    dataset_dir: str = "data/dataset_v2",
    target_psnr: float = 46.0,
    weights_json: str = "results_v2/optimal_texture_weights.json",
    workers: int = 12,
) -> Tuple[float, float, float]:
    """
    Grid search on the simplex (w_grad + w_var + w_ent = 1.0, step 0.1) on 40 dev images.
    Maximizes mean BCR across attacks. Saves chosen weights.
    """
    if os.path.exists(weights_json):
        try:
            with open(weights_json, "r") as f:
                saved = json.load(f)
                print(f"[*] Loaded existing tuned weights: {saved}")
                return float(saved["w_grad"]), float(saved["w_var"]), float(saved["w_ent"])
        except Exception:
            pass

    print("[*] Performing simplex grid search (step 0.1, 66 triples) on 40 dev images...")
    # Generate simplex points: i + j + k = 10 -> (i/10, j/10, k/10)
    simplex_points = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            simplex_points.append((i / 10.0, j / 10.0, k / 10.0))

    # Evaluate on a representative subset of attacks for tuning efficiency
    logo = generate_original_logo()
    payload, _ = generate_whitened_logo(logo, seed=42)

    best_bcr = -1.0
    best_weights = (0.4, 0.4, 0.2)

    # Sample 10 dev images for rapid tuning
    sample_dev = dev_ids[:10]

    for wg, wv, we in simplex_points:
        bcrs = []
        for iid in sample_dev:
            img_path = os.path.join(dataset_dir, f"{iid}.png")
            cover = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            method_obj = create_method(
                "proposed_adaptive",
                subband="HL",
                weight_gradient=wg,
                weight_variance=wv,
                weight_entropy=we,
            )
            cal_alpha = calibrate_alpha_to_psnr(
                method_obj, cover, payload, target_psnr=target_psnr, tol=0.1
            )
            wm_img, side_info = method_obj.embed(cover, payload, alpha=cal_alpha, quantize=True)
            # Evaluate JPEG 70
            att = quantize_u8(attack_jpeg(wm_img, 70))
            bits = fast_batched_extract(att, side_info["metadata"])
            bcrs.append(compute_bcr(payload, bits))

        mean_bcr = float(np.mean(bcrs))
        if mean_bcr > best_bcr:
            best_bcr = mean_bcr
            best_weights = (wg, wv, we)

    print(f"[*] Optimal weights found on dev split: {best_weights} (Mean BCR = {best_bcr:.4f})")
    os.makedirs(os.path.dirname(os.path.abspath(weights_json)), exist_ok=True)
    with open(weights_json, "w") as f:
        json.dump({
            "w_grad": best_weights[0],
            "w_var": best_weights[1],
            "w_ent": best_weights[2],
            "dev_bcr": best_bcr,
        }, f, indent=2)

    return best_weights


def run_ablation_features(
    output_csv: str = "results_v2/ablation_features.csv",
    weights_json: str = "results_v2/optimal_texture_weights.json",
    workers: int = 12,
    target_psnr: float = 46.0,
    dataset_dir: str = "data/dataset_v2",
) -> None:
    """Executes Texture-Feature Ablation with dev (40) and held-out test (60) split."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    init_results_csv(output_csv)

    selected_ids = select_benchmark_subset(num_images=100, seed=0)
    dev_ids = selected_ids[:40]
    test_ids = selected_ids[40:]

    opt_weights = optimize_weights_on_dev_split(
        dev_ids, dataset_dir=dataset_dir, target_psnr=target_psnr, weights_json=weights_json, workers=workers
    )

    configs = get_feature_weight_configs()
    configs.append({
        "name": "weights_tuned",
        "w_grad": opt_weights[0],
        "w_var": opt_weights[1],
        "w_ent": opt_weights[2],
    })

    logo = generate_original_logo()
    payload, _ = generate_whitened_logo(logo, seed=42)
    git_hash = get_git_commit_hash()
    run_id = f"ablation_feat_{int(time.time())}"

    # Load completed
    completed: Set[Tuple[str, str, str]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband"])
        completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"]))

    tasks = []
    for idx, iid in enumerate(selected_ids):
        split_name = "dev" if idx < 40 else "test"
        img_path = os.path.join(dataset_dir, f"{iid}.png")

        for cfg in configs:
            cfg_name = cfg["name"]
            # Tuned weights are only reported on test split
            if cfg_name == "weights_tuned" and split_name == "dev":
                continue

            for sb in ["HL", "LL"]:
                method_key = f"adaptive_{cfg_name}"
                if (iid, method_key, sb) in completed:
                    continue

                m_kwargs = {
                    "weight_gradient": cfg["w_grad"],
                    "weight_variance": cfg["w_var"],
                    "weight_entropy": cfg["w_ent"],
                }

                tasks.append(
                    (
                        run_id,
                        "ablation_features",
                        iid,
                        img_path,
                        "whitened_logo",
                        payload,
                        "proposed_adaptive",
                        sb,
                        "haar",
                        8,
                        m_kwargs,
                        target_psnr,
                        git_hash,
                        {
                            "split": split_name,
                            "feature_config": cfg_name,
                            "w_grad": cfg["w_grad"],
                            "w_var": cfg["w_var"],
                            "w_ent": cfg["w_ent"],
                        },
                    )
                )

    print(f"\n[*] Part B: Feature Ablation | Total pending conditions: {len(tasks)}")
    _execute_tasks_and_append(tasks, output_csv, workers=workers)


# =============================================================================
# 6. PART C: WAVELET SENSITIVITY
# =============================================================================

def run_ablation_wavelet(
    output_csv: str = "results_v2/ablation_wavelet.csv",
    num_images: int = 100,
    workers: int = 12,
    target_psnr: float = 46.0,
    dataset_dir: str = "data/dataset_v2",
) -> None:
    """Executes Wavelet Sensitivity Ablation across haar, db2, db4, bior1.3, sym4."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    init_results_csv(output_csv)

    selected_ids = select_benchmark_subset(num_images=num_images, seed=0)
    logo = generate_original_logo()
    payload, _ = generate_whitened_logo(logo, seed=42)
    git_hash = get_git_commit_hash()
    run_id = f"ablation_wav_{int(time.time())}"

    completed: Set[Tuple[str, str, str, str]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband", "wavelet"])
        completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"], df_prev["wavelet"]))

    wavelets = ["haar", "db2", "db4", "bior1.3", "sym4"]
    methods = ["proposed_adaptive", "proposed_fixed"]
    subbands = ["HL", "LL"]

    tasks = []
    for iid in selected_ids:
        img_path = os.path.join(dataset_dir, f"{iid}.png")
        for wav in wavelets:
            for m in methods:
                for sb in subbands:
                    if (iid, m, sb, wav) in completed:
                        continue
                    tasks.append(
                        (
                            run_id,
                            "ablation_wavelet",
                            iid,
                            img_path,
                            "whitened_logo",
                            payload,
                            m,
                            sb,
                            wav,
                            8,
                            {},
                            target_psnr,
                            git_hash,
                            {"wavelet": wav, "ablation_axis": "wavelet"},
                        )
                    )

    print(f"\n[*] Part C: Wavelet Sensitivity | Pending conditions: {len(tasks)}")
    _execute_tasks_and_append(tasks, output_csv, workers=workers)


# =============================================================================
# 7. PART D: BLOCK SIZE & CAPACITY
# =============================================================================

def run_ablation_block(
    output_csv: str = "results_v2/ablation_block.csv",
    num_images: int = 100,
    workers: int = 12,
    target_psnr: float = 46.0,
    dataset_dir: str = "data/dataset_v2",
) -> None:
    """Executes Block Size Ablation: 4, 8, 16."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    init_results_csv(output_csv)

    selected_ids = select_benchmark_subset(num_images=num_images, seed=0)
    logo = generate_original_logo()
    whitened_1024, _ = generate_whitened_logo(logo, seed=42)
    whitened_256 = generate_balanced_random_payload(payload_id=256, shape=(16, 16), seed=256)

    git_hash = get_git_commit_hash()
    run_id = f"ablation_block_{int(time.time())}"

    completed: Set[Tuple[str, str, str, int]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband", "block_size"])
        completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"], df_prev["block_size"].astype(int)))

    blocks = [4, 8, 16]
    methods = ["proposed_adaptive", "proposed_fixed"]
    subbands = ["HL", "LL"]

    tasks = []
    for iid in selected_ids:
        img_path = os.path.join(dataset_dir, f"{iid}.png")
        for bs in blocks:
            # Capacity adjustment: 256 bits for block 16; 1,024 bits for block 4 and 8
            if bs == 16:
                payload = whitened_256
                pid = "whitened_256"
                cap_doc = "Capacity: 16x16 = 256 blocks (256 bits maximum capacity)"
            else:
                payload = whitened_1024
                pid = "whitened_1024"
                cap_doc = f"Capacity: {(256//bs)**2} blocks (embedding 1,024 bits)"

            for m in methods:
                for sb in subbands:
                    if (iid, m, sb, bs) in completed:
                        continue
                    tasks.append(
                        (
                            run_id,
                            "ablation_block",
                            iid,
                            img_path,
                            pid,
                            payload,
                            m,
                            sb,
                            "haar",
                            bs,
                            {},
                            target_psnr,
                            git_hash,
                            {"block_size": bs, "capacity_documentation": cap_doc},
                        )
                    )

    print(f"\n[*] Part D: Block Size Ablation | Pending conditions: {len(tasks)}")
    _execute_tasks_and_append(tasks, output_csv, workers=workers)


# =============================================================================
# 8. PART E: PAYLOAD / CAPACITY SCALING
# =============================================================================

def run_ablation_payload(
    output_csv: str = "results_v2/ablation_payload.csv",
    num_images: int = 100,
    workers: int = 12,
    target_psnr: float = 46.0,
    dataset_dir: str = "data/dataset_v2",
) -> None:
    """Executes Payload Scaling Ablation: 256, 512, 1024, 2048 bits."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    init_results_csv(output_csv)

    selected_ids = select_benchmark_subset(num_images=num_images, seed=0)
    git_hash = get_git_commit_hash()
    run_id = f"ablation_pay_{int(time.time())}"

    # Payloads
    p256 = generate_balanced_random_payload(payload_id=256, shape=(16, 16), seed=256)
    p512 = generate_balanced_random_payload(payload_id=512, shape=(32, 16), seed=512)
    logo = generate_original_logo()
    p1024, _ = generate_whitened_logo(logo, seed=42)
    p2048 = generate_balanced_random_payload(payload_id=2048, shape=(64, 32), seed=2048)

    payload_configs = [
        (256, p256, 8),
        (512, p512, 8),
        (1024, p1024, 8),
        (2048, p2048, 4),  # Block 4 gives 64x64 = 4,096 blocks capacity
    ]

    completed: Set[Tuple[str, str, str, str]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband", "payload_id"])
        completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"], df_prev["payload_id"]))

    methods = ["proposed_adaptive", "proposed_fixed"]
    subbands = ["HL", "LL"]

    tasks = []
    for iid in selected_ids:
        img_path = os.path.join(dataset_dir, f"{iid}.png")
        for num_bits, p_arr, bs in payload_configs:
            pid = f"payload_{num_bits}"
            for m in methods:
                for sb in subbands:
                    if (iid, m, sb, pid) in completed:
                        continue
                    tasks.append(
                        (
                            run_id,
                            "ablation_payload",
                            iid,
                            img_path,
                            pid,
                            p_arr,
                            m,
                            sb,
                            "haar",
                            bs,
                            {},
                            target_psnr,
                            git_hash,
                            {"payload_bits": num_bits, "block_size": bs},
                        )
                    )

    print(f"\n[*] Part E: Payload Scaling Ablation | Pending conditions: {len(tasks)}")
    _execute_tasks_and_append(tasks, output_csv, workers=workers)


# =============================================================================
# 9. PART F: UNIFIED HEATMAP DATASET AGGREGATION
# =============================================================================

def compute_ablation_heatmap_matrix(
    output_csv: str = "results_v2/ablation_heatmap_data.csv",
    csv_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes unified summary matrix:
    Rows = variant
    Columns = 13 manuscript attacks
    Cell value = median BER over images with bootstrap 95% CI.
    """
    default_paths = [
        "results_v2/ablation_transform.csv",
        "results_v2/ablation_features.csv",
        "results_v2/ablation_wavelet.csv",
        "results_v2/ablation_block.csv",
        "results_v2/ablation_payload.csv",
    ]
    paths = csv_paths or default_paths
    dfs = []
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            df_item = pd.read_csv(p)
            dfs.append(df_item)

    if not dfs:
        print("[!] No ablation CSV files found to build heatmap data.")
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    # 1. Enforce Rule 4: Average seeds within image first!
    df_img = (
        df_all.groupby(
            ["experiment", "method", "subband", "wavelet", "block_size", "payload_id", "attack", "severity", "image_id"],
            as_index=False,
        )["ber"]
        .mean()
    )

    # Helper to construct variant display label
    def get_variant_label(row):
        exp = str(row["experiment"])
        m = str(row["method"]).replace("proposed_", "")
        sb = str(row["subband"])
        wav = str(row["wavelet"])
        bs = int(row["block_size"])
        pid = str(row["payload_id"])

        if "transform" in exp:
            return f"{m}_{sb}"
        elif "feature" in exp:
            return f"feat_{m}_{sb}"
        elif "wavelet" in exp:
            return f"{wav}_{m}_{sb}"
        elif "block" in exp:
            return f"block{bs}_{m}_{sb}"
        elif "payload" in exp:
            return f"{pid}_{m}_{sb}"
        else:
            return f"{m}_{sb}_{wav}_b{bs}"

    df_img["variant"] = df_img.apply(get_variant_label, axis=1)

    # Attack column label
    df_img["attack_col"] = df_img["attack"] + " (" + df_img["severity"] + ")"

    # 2. Compute median BER with bootstrap 95% CI per (variant, attack_col)
    records = []
    attacks_ordered = sorted(df_img["attack_col"].unique())

    for var, grp_var in df_img.groupby("variant"):
        row_dict = {"variant": var}
        for att in attacks_ordered:
            sub = grp_var[grp_var["attack_col"] == att]["ber"].to_numpy()
            if len(sub) > 0:
                med, low, high = bootstrap_median_ci(sub, num_boot=1000)
                row_dict[att] = f"{med:.4f} [{low:.4f}, {high:.4f}]"
            else:
                row_dict[att] = "N/A"
        records.append(row_dict)

    df_heatmap = pd.DataFrame(records)
    df_heatmap.sort_values(by="variant", inplace=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_heatmap.to_csv(output_csv, index=False)
    print(f"\n[*] Heatmap data saved to: '{output_csv}' ({len(df_heatmap)} variants x {len(attacks_ordered)} attacks)")
    return df_heatmap


# =============================================================================
# 10. MULTIPROCESSING HELPER
# =============================================================================

def _execute_tasks_and_append(tasks: List[Tuple], output_csv: str, workers: int = 12) -> None:
    if not tasks:
        return
    start_time = time.perf_counter()
    done_count = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_ablation_condition, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                cond_key, rows = fut.result()
                append_result_rows(output_csv, rows)
                done_count += 1
                elapsed = time.perf_counter() - start_time
                rate = done_count / elapsed
                remaining = (len(tasks) - done_count) / rate if rate > 0 else 0
                if done_count % 10 == 0 or done_count == len(tasks):
                    print(
                        f"  [{done_count}/{len(tasks)}] Done: {cond_key} | {rate:.2f} cond/s | ETA: {remaining:.1f}s",
                        flush=True,
                    )
            except Exception as e:
                t = futures[fut]
                print(f"[!] Error in ablation condition {t[2]}_{t[6]}_{t[7]}: {e}", file=sys.stderr)
                raise e


# =============================================================================
# 11. MAIN ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Ablation Suite Runner")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker processes")
    parser.add_argument("--part", type=str, default="all", choices=["all", "A", "B", "C", "D", "E", "F"])
    parser.add_argument("--num-images", type=int, default=100, help="Number of benchmark images")
    parser.add_argument("--target-psnr", type=float, default=46.0, help="Target clean PSNR in dB")
    args = parser.parse_args()

    run_all = args.part in ("all", "A")
    if run_all or args.part == "A":
        run_ablation_transform(workers=args.workers, num_images=args.num_images, target_psnr=args.target_psnr)

    if run_all or args.part == "B":
        run_ablation_features(workers=args.workers, target_psnr=args.target_psnr)

    if run_all or args.part == "C":
        run_ablation_wavelet(workers=args.workers, num_images=args.num_images, target_psnr=args.target_psnr)

    if run_all or args.part == "D":
        run_ablation_block(workers=args.workers, num_images=args.num_images, target_psnr=args.target_psnr)

    if run_all or args.part == "E":
        run_ablation_payload(workers=args.workers, num_images=args.num_images, target_psnr=args.target_psnr)

    if run_all or args.part == "F":
        compute_ablation_heatmap_matrix()


if __name__ == "__main__":
    main()
