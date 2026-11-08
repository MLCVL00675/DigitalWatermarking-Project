"""
Matched-Distortion Study and Comprehensive Watermarking Benchmark Module
for experiments_v2.

Outputs:
1. results_v2/core_raw.csv: Dense alpha sweep across all dataset_v2 images, subbands, methods, and attacks.
2. results_v2/matched_psnr_raw.csv: Matched-PSNR comparison at target PSNR in {40, 42, 44, 46, 48, 50} dB.
3. results_v2/pareto_points.csv: Pareto trade-off curve points (mean PSNR, SSIM, BER, alpha_eff_mean).
4. results_v2/paired_diffs.csv: Per-image paired differences (adaptive - fixed, HL - LL) with Wilcoxon & t-tests.
"""

import os
import sys
import time
import uuid
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2

from watermark.metrics import compute_psnr, compute_ssim, compute_nc, compute_ber, compute_bcr
from experiments_v2.schema import (
    ResultRow,
    make_trial_key,
    load_completed_keys,
    append_result_rows,
    init_results_csv,
)
from experiments_v2.attacks import AttackSpec
from experiments_v2.common import get_git_commit_hash
from experiments_v2.harness import (
    quantize_u8,
    create_method,
    calibrate_alpha_to_psnr,
    fast_batched_extract,
)
from experiments_v2.payloads import (
    generate_original_logo,
    generate_whitened_logo,
    generate_balanced_random_payload,
)
from experiments_v2.stats import aggregate_by_image, aggregate_across_images


# Standard attacks for the manuscript evaluation
def get_study_attack_specs() -> List[Tuple[AttackSpec, List[int]]]:
    """
    Returns list of (AttackSpec, seeds_list).
    Deterministic attacks have 1 seed [42].
    Stochastic attacks have 20 seeds [1..20].
    """
    stoch_seeds = list(range(1, 21))
    return [
        (AttackSpec(attack="no_attack", severity="none", is_stochastic=False), [42]),
        (AttackSpec(attack="jpeg", severity="Q=90", is_stochastic=False), [42]),
        (AttackSpec(attack="jpeg", severity="Q=70", is_stochastic=False), [42]),
        (AttackSpec(attack="jpeg", severity="Q=50", is_stochastic=False), [42]),
        (AttackSpec(attack="gaussian_noise", severity="var=0.01", is_stochastic=True), stoch_seeds),
        (AttackSpec(attack="speckle_noise", severity="var=0.01", is_stochastic=True), stoch_seeds),
        (AttackSpec(attack="salt_and_pepper", severity="1%", is_stochastic=True), stoch_seeds),
        (AttackSpec(attack="median_filter", severity="3x3", is_stochastic=False), [42]),
        (AttackSpec(attack="gaussian_blur", severity="3x3, s=1.0", is_stochastic=False), [42]),
        (AttackSpec(attack="cropping", severity="10%", is_stochastic=False), [42]),
        (AttackSpec(attack="rotation", severity="2 deg [Unsync]", is_stochastic=False, deskew_angle=None), [42]),
        (AttackSpec(attack="rotation", severity="5 deg [Unsync]", is_stochastic=False, deskew_angle=None), [42]),
        (AttackSpec(attack="contrast", severity="x1.15", is_stochastic=False), [42]),
    ]


def load_dataset_v2_images(dataset_dir: str = "data/dataset_v2") -> Dict[str, np.ndarray]:
    """Loads all preprocessed 512x512 images from data/dataset_v2/."""
    images = {}
    for fn in sorted(os.listdir(dataset_dir)):
        if fn.endswith(".png"):
            path = os.path.join(dataset_dir, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                iid = os.path.splitext(fn)[0]
                images[iid] = img.astype(np.float64)
    return images


def get_payload_set(
    count_per_image: int = 2,
    images_dir: str = "data/internet_images",
) -> List[Tuple[str, np.ndarray]]:
    """
    Returns payload set.
    If count_per_image == 2: [whitened_logo, balanced_0]
    If count_per_image == 6: [whitened_logo, balanced_0, balanced_1, balanced_2, balanced_3, balanced_4]
    """
    logo = generate_original_logo(images_dir)
    whitened, _ = generate_whitened_logo(logo, seed=42)

    payloads = [("whitened_logo", whitened)]
    num_balanced = max(1, count_per_image - 1)
    for i in range(num_balanced):
        bal = generate_balanced_random_payload(payload_id=i, seed=i + 500)
        payloads.append((f"balanced_{i}", bal))

    return payloads[:count_per_image]


def _worker_execute_embedding_condition(args: Tuple) -> List[ResultRow]:
    """
    Worker task: embeds once for a condition, then runs all 70 attack evaluations.
    Returns list of ResultRow entries.
    """
    (
        run_id,
        experiment,
        image_id,
        cover,
        payload_id,
        payload,
        method_name,
        subband,
        alpha_nominal,
        git_hash,
    ) = args

    method_obj = create_method(method_name, subband=subband)

    # 1. Embed once with 8-bit integer quantization
    wm_img, side_info = method_obj.embed(cover, payload, alpha=alpha_nominal, quantize=True)
    alpha_eff_mean = float(side_info["alpha_eff_mean"])

    # 2. Clean Metrics
    psnr_clean = float(compute_psnr(cover, wm_img))
    ssim_clean = float(compute_ssim(cover, wm_img))

    results = []
    attacks_and_seeds = get_study_attack_specs()

    for spec, seeds in attacks_and_seeds:
        for s in seeds:
            rng = np.random.default_rng(s) if spec.is_stochastic else None
            att_img = spec.apply(wm_img, rng=rng)
            if spec.attack not in ("no_attack", "identity"):
                att_img = quantize_u8(att_img)

            # Fast vectorized extraction
            extracted_bits = fast_batched_extract(
                att_img, side_info["metadata"], deskew_angle=spec.deskew_angle
            )

            psnr_att = float(compute_psnr(cover, att_img))
            ber = float(compute_ber(payload, extracted_bits))
            bcr = float(compute_bcr(payload, extracted_bits))
            nc = float(compute_nc(payload, extracted_bits))

            extra = {
                "git_commit": git_hash,
                "config": {"method": method_name, "subband": subband, "alpha_nominal": alpha_nominal},
                "deskew_angle": spec.deskew_angle,
                "is_stochastic": spec.is_stochastic,
            }

            row = ResultRow(
                run_id=run_id,
                experiment=experiment,
                image_id=image_id,
                payload_id=payload_id,
                method=method_name,
                subband=subband,
                wavelet="haar",
                block_size=8,
                alpha_nominal=float(alpha_nominal),
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
            results.append(row)

    return results


# =============================================================================
# RUNTIME ESTIMATION (3 IMAGES)
# =============================================================================

def estimate_runtime(
    images_dir: str = "data/dataset_v2",
    sample_size: int = 3,
    workers: int = 12,
) -> Dict[str, Any]:
    """
    Executes a benchmark on 3 sample images to measure exact condition execution speed
    and project total runtime for 208 images under 6 vs 2 payloads.
    """
    all_images = load_dataset_v2_images(images_dir)
    sample_ids = list(all_images.keys())[:sample_size]
    sample_imgs = {k: all_images[k] for k in sample_ids}

    alphas = [10.0, 30.0]  # 2 alphas for timing test
    subbands = ["HL", "LL"]
    methods = ["proposed_fixed", "proposed_adaptive"]
    payloads = get_payload_set(count_per_image=1)

    tasks = []
    run_id = "timing_bench"
    git_hash = get_git_commit_hash()

    for iid, img in sample_imgs.items():
        for pid, pl in payloads:
            for m in methods:
                for sb in subbands:
                    for a in alphas:
                        tasks.append((run_id, "timing", iid, img, pid, pl, m, sb, a, git_hash))

    n_sample_conditions = len(tasks)
    total_sample_trials = n_sample_conditions * 70

    print(f"[*] Running empirical timing benchmark: {n_sample_conditions} conditions ({total_sample_trials} trials) across {workers} workers...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_worker_execute_embedding_condition, tasks))
    elapsed = time.time() - t0

    sec_per_condition = elapsed / n_sample_conditions
    sec_per_trial = elapsed / total_sample_trials
    trials_per_sec = total_sample_trials / elapsed

    total_images = len(all_images)
    full_alphas = 10  # [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
    full_subbands = 4  # HL, LL, LH, HH
    full_methods = 2   # fixed, adaptive

    conditions_per_payload = total_images * full_methods * full_subbands * full_alphas  # 208*2*4*10 = 16,640

    # Projection for 6 payloads
    # Real wall-clock time per condition across workers
    wall_sec_per_condition = elapsed / n_sample_conditions

    # Projection for 6 payloads (5 balanced + 1 whitened logo)
    cond_6 = conditions_per_payload * 6
    trials_6 = cond_6 * 70
    est_hours_6 = (cond_6 * wall_sec_per_condition) / 3600.0

    # Projection for 2 payloads (1 balanced + 1 whitened logo)
    cond_2 = conditions_per_payload * 2
    trials_2 = cond_2 * 70
    est_hours_2 = (cond_2 * wall_sec_per_condition) / 3600.0

    # Projection for 1 payload (whitened logo)
    cond_1 = conditions_per_payload * 1
    trials_1 = cond_1 * 70
    est_hours_1 = (cond_1 * wall_sec_per_condition) / 3600.0

    print("\n" + "=" * 80)
    print(" EMPIRICAL RUNTIME ESTIMATION REPORT (WALL-CLOCK PROJECTIONS)")
    print("=" * 80)
    print(f"  Benchmark execution time: {elapsed:.2f} s for {n_sample_conditions} conditions ({total_sample_trials} trials)")
    print(f"  Throughput: {trials_per_sec:.1f} trials/s ({wall_sec_per_condition*1000:.1f} ms wall-clock per condition with {workers} workers)")
    print(f"\n  --- PROJECTIONS FOR FULL DATASET (N={total_images} IMAGES, 4 SUBBANDS, 10 ALPHAS, 70 ATTACKS) ---")
    print(f"  Full 6 Payloads (5 balanced + 1 whitened logo):")
    print(f"    - Total conditions: {cond_6:,} | Total trials: {trials_6:,}")
    print(f"    - Estimated wall-clock time ({workers} workers): {est_hours_6:.2f} hours")
    print(f"  Reduced 2 Payloads (1 balanced + 1 whitened logo):")
    print(f"    - Total conditions: {cond_2:,} | Total trials: {trials_2:,}")
    print(f"    - Estimated wall-clock time ({workers} workers): {est_hours_2:.2f} hours")
    print(f"  Single Standard Payload (1 whitened logo):")
    print(f"    - Total conditions: {cond_1:,} | Total trials: {trials_1:,}")
    print(f"    - Estimated wall-clock time ({workers} workers): {est_hours_1:.2f} hours")
    print("=" * 80 + "\n")

    return {
        "sec_per_condition": wall_sec_per_condition,
        "sec_per_trial": sec_per_trial,
        "est_hours_6_payloads": est_hours_6,
        "est_hours_2_payloads": est_hours_2,
        "est_hours_1_payload": est_hours_1,
        "total_images": total_images,
    }


# =============================================================================
# 1. CORE DENSE ALPHA STUDY RUNNER
# =============================================================================

def run_core_study(
    images_dir: str = "data/dataset_v2",
    output_csv: str = "results_v2/core_raw.csv",
    payloads_count: int = 2,
    workers: int = 12,
    batch_size: int = 25,
    max_images: int = 0,
) -> pd.DataFrame:
    """
    Executes the dense alpha grid experiment:
    - Images: all dataset_v2 (or first max_images if max_images > 0)
    - Payloads: 2 payloads per image (whitened logo + balanced_0)
    - Methods: proposed_fixed, proposed_adaptive
    - Subbands: HL, LL, LH, HH
    - Alphas: [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
    - Attacks: 10 deterministic + 3 stochastic * 20 seeds = 70 trials per condition
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    init_results_csv(output_csv)
    completed_keys = load_completed_keys(output_csv)

    all_images = load_dataset_v2_images(images_dir)
    if max_images > 0:
        sel_keys = list(all_images.keys())[:max_images]
        all_images = {k: all_images[k] for k in sel_keys}
    payloads = get_payload_set(count_per_image=payloads_count)
    methods = ["proposed_fixed", "proposed_adaptive"]
    subbands = ["HL", "LL", "LH", "HH"]
    alphas = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0]
    run_id = f"core_study_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    git_hash = get_git_commit_hash()

    print(f"[*] Core study: {len(completed_keys)} completed rows found in '{output_csv}'")

    pending_tasks = []
    for iid, img in all_images.items():
        for pid, pl in payloads:
            for m in methods:
                for sb in subbands:
                    for a in alphas:
                        # Representative key check using no_attack trial
                        key = make_trial_key(
                            experiment="core_study",
                            image_id=iid,
                            payload_id=pid,
                            method=m,
                            subband=sb,
                            wavelet="haar",
                            block_size=8,
                            alpha_nominal=a,
                            attack="no_attack",
                            severity="none",
                            seed=42,
                        )
                        if key not in completed_keys:
                            pending_tasks.append((run_id, "core_study", iid, img, pid, pl, m, sb, a, git_hash))

    total_tasks = len(pending_tasks)
    print(f"[*] Pending conditions to run: {total_tasks} ({total_tasks * 70:,} trials)")

    if total_tasks > 0:
        actual_workers = max(1, min(workers, os.cpu_count() or 4))
        done_cond = 0
        buffer: List[ResultRow] = []

        with ProcessPoolExecutor(max_workers=actual_workers) as ex:
            futures = {ex.submit(_worker_execute_embedding_condition, t): t for t in pending_tasks}
            for fut in as_completed(futures):
                rows = fut.result()
                buffer.extend(rows)
                done_cond += 1

                if len(buffer) >= batch_size * 70:
                    append_result_rows(output_csv, buffer)
                    buffer.clear()
                    print(f"  -> Progress: {done_cond}/{total_tasks} conditions ({done_cond / total_tasks * 100:.1f}%)")

            if buffer:
                append_result_rows(output_csv, buffer)
                buffer.clear()

        print(f"[*] Core study execution complete: all {total_tasks} conditions saved.")

    return pd.read_csv(output_csv)


# =============================================================================
# 2. MATCHED-PSNR STUDY RUNNER
# =============================================================================

def _worker_execute_matched_psnr_condition(args: Tuple) -> List[ResultRow]:
    """Calibrates alpha to target PSNR, embeds, and runs all 70 attack trials."""
    (
        run_id,
        experiment,
        image_id,
        cover,
        payload_id,
        payload,
        method_name,
        subband,
        target_psnr,
        git_hash,
    ) = args

    # Calibrate alpha to exact target clean PSNR (after quantization)
    cal_alpha = calibrate_alpha_to_psnr(
        method=method_name,
        cover=cover,
        payload=payload,
        target_psnr=target_psnr,
        tol=0.08,
        subband=subband,
    )

    method_obj = create_method(method_name, subband=subband)
    wm_img, side_info = method_obj.embed(cover, payload, alpha=cal_alpha, quantize=True)
    alpha_eff_mean = float(side_info["alpha_eff_mean"])

    psnr_clean = float(compute_psnr(cover, wm_img))
    ssim_clean = float(compute_ssim(cover, wm_img))

    results = []
    attacks_and_seeds = get_study_attack_specs()

    for spec, seeds in attacks_and_seeds:
        for s in seeds:
            rng = np.random.default_rng(s) if spec.is_stochastic else None
            att_img = spec.apply(wm_img, rng=rng)
            if spec.attack not in ("no_attack", "identity"):
                att_img = quantize_u8(att_img)

            extracted_bits = fast_batched_extract(
                att_img, side_info["metadata"], deskew_angle=spec.deskew_angle
            )

            psnr_att = float(compute_psnr(cover, att_img))
            ber = float(compute_ber(payload, extracted_bits))
            bcr = float(compute_bcr(payload, extracted_bits))
            nc = float(compute_nc(payload, extracted_bits))

            extra = {
                "git_commit": git_hash,
                "target_psnr": target_psnr,
                "calibrated_alpha": cal_alpha,
                "config": {"method": method_name, "subband": subband, "target_psnr": target_psnr},
                "deskew_angle": spec.deskew_angle,
                "is_stochastic": spec.is_stochastic,
            }

            row = ResultRow(
                run_id=run_id,
                experiment=experiment,
                image_id=image_id,
                payload_id=payload_id,
                method=method_name,
                subband=subband,
                wavelet="haar",
                block_size=8,
                alpha_nominal=float(cal_alpha),
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
            results.append(row)

    return results


def run_matched_psnr_study(
    images_dir: str = "data/dataset_v2",
    output_csv: str = "results_v2/matched_psnr_raw.csv",
    target_psnrs: Optional[List[float]] = None,
    workers: int = 12,
    batch_size: int = 20,
    max_images: int = 0,
) -> pd.DataFrame:
    """
    Executes the matched-distortion benchmark:
    - Target PSNR in {40, 42, 44, 46, 48, 50} dB
    - Calibrate alpha to target PSNR per image, method, and subband
    - Run all 70 attack trials per matched operating point
    - Writes results_v2/matched_psnr_raw.csv
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    init_results_csv(output_csv)
    completed_keys = load_completed_keys(output_csv)

    targets = target_psnrs if target_psnrs is not None else [40.0, 42.0, 44.0, 46.0, 48.0, 50.0]
    all_images = load_dataset_v2_images(images_dir)
    if max_images > 0:
        sel_keys = list(all_images.keys())[:max_images]
        all_images = {k: all_images[k] for k in sel_keys}
    payloads = get_payload_set(count_per_image=1)  # Whitened logo standard payload
    methods = ["proposed_fixed", "proposed_adaptive"]
    subbands = ["HL", "LL"]  # Primary subbands for matched comparison
    run_id = f"matched_psnr_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    git_hash = get_git_commit_hash()

    print(f"[*] Matched-PSNR study: {len(completed_keys)} completed rows found in '{output_csv}'")

    pending_tasks = []
    for iid, img in all_images.items():
        for pid, pl in payloads:
            for m in methods:
                for sb in subbands:
                    for t_psnr in targets:
                        # Check if condition already completed (by checking representative no_attack key)
                        # We tag target_psnr in experiment or config
                        exp_name = f"matched_psnr_{int(t_psnr)}dB"
                        key = make_trial_key(
                            experiment=exp_name,
                            image_id=iid,
                            payload_id=pid,
                            method=m,
                            subband=sb,
                            wavelet="haar",
                            block_size=8,
                            alpha_nominal=t_psnr,  # tracked canonical key
                            attack="no_attack",
                            severity="none",
                            seed=42,
                        )
                        if key not in completed_keys:
                            pending_tasks.append((run_id, exp_name, iid, img, pid, pl, m, sb, t_psnr, git_hash))

    total_tasks = len(pending_tasks)
    print(f"[*] Pending Matched-PSNR conditions to run: {total_tasks} ({total_tasks * 70:,} trials)")

    if total_tasks > 0:
        actual_workers = max(1, min(workers, os.cpu_count() or 4))
        done_cond = 0
        buffer: List[ResultRow] = []

        with ProcessPoolExecutor(max_workers=actual_workers) as ex:
            futures = {ex.submit(_worker_execute_matched_psnr_condition, t): t for t in pending_tasks}
            for fut in as_completed(futures):
                rows = fut.result()
                buffer.extend(rows)
                done_cond += 1

                if len(buffer) >= batch_size * 70:
                    append_result_rows(output_csv, buffer)
                    buffer.clear()
                    print(f"  -> Matched-PSNR Progress: {done_cond}/{total_tasks} ({done_cond / total_tasks * 100:.1f}%)")

            if buffer:
                append_result_rows(output_csv, buffer)
                buffer.clear()

        print(f"[*] Matched-PSNR study complete: all {total_tasks} conditions saved.")

    return pd.read_csv(output_csv)


# =============================================================================
# 3. PARETO DATA COMPUTATION
# =============================================================================

def compute_pareto_points(
    core_raw_csv: str = "results_v2/core_raw.csv",
    output_csv: str = "results_v2/pareto_points.csv",
) -> pd.DataFrame:
    """
    Computes Pareto frontier data (fidelity vs robustness):
    Mean PSNR, SSIM, BER, BCR, NC, and alpha_eff_mean per (method, subband, alpha_nominal).
    Enforces Rule 4 (averages within image first).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_raw = pd.read_csv(core_raw_csv)

    # 1. Image-level aggregation (Rule 4: unit = IMAGE)
    df_image = aggregate_by_image(
        df_raw,
        metrics=["psnr_clean", "ssim_clean", "psnr_attacked", "ber", "bcr", "nc", "alpha_eff_mean"],
        group_cols=["method", "subband", "alpha_nominal", "attack", "severity"],
    )

    # 2. Overall robustness across all attacks per (method, subband, alpha_nominal, image_id)
    overall_by_img = (
        df_image.groupby(["method", "subband", "alpha_nominal", "image_id"], as_index=False)[
            ["psnr_clean", "ssim_clean", "ber", "bcr", "nc", "alpha_eff_mean"]
        ]
        .mean()
    )

    # 3. Across-image mean, std, and SEM
    records = []
    for keys, grp in overall_by_img.groupby(["method", "subband", "alpha_nominal"]):
        m, sb, a = keys
        n_img = len(grp)
        records.append({
            "method": m,
            "subband": sb,
            "alpha_nominal": float(a),
            "n_images": n_img,
            "alpha_eff_mean": round(float(grp["alpha_eff_mean"].mean()), 4),
            "psnr_clean_mean": round(float(grp["psnr_clean"].mean()), 4),
            "psnr_clean_sem": round(float(grp["psnr_clean"].std(ddof=1) / np.sqrt(n_img)), 4),
            "ssim_clean_mean": round(float(grp["ssim_clean"].mean()), 6),
            "ber_mean": round(float(grp["ber"].mean()), 6),
            "ber_sem": round(float(grp["ber"].std(ddof=1) / np.sqrt(n_img)), 6),
            "bcr_mean": round(float(grp["bcr"].mean()), 4),
            "nc_mean": round(float(grp["nc"].mean()), 6),
            "nc_sem": round(float(grp["nc"].std(ddof=1) / np.sqrt(n_img)), 6),
        })

    df_pareto = pd.DataFrame(records)
    df_pareto.sort_values(by=["method", "subband", "alpha_nominal"], inplace=True)
    df_pareto.to_csv(output_csv, index=False)
    print(f"[*] Pareto frontier data written to: '{output_csv}'")
    return df_pareto


# =============================================================================
# 5. PAIRED DIFFERENCES ANALYSIS (IMAGE UNIT)
# =============================================================================

def compute_paired_differences(
    matched_raw_csv: str = "results_v2/matched_psnr_raw.csv",
    output_csv: str = "results_v2/paired_diffs.csv",
) -> pd.DataFrame:
    """
    Computes per-image paired differences at each matched PSNR and attack:
    1. Adaptive - Fixed (matched on image_id, target_psnr, attack)
    2. HL - LL (matched on image_id, target_psnr, attack)
    Strictly enforces Rule 4 (averages seeds within image first, N = number of images).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_raw = pd.read_csv(matched_raw_csv)

    # Extract target_psnr from extra_json or experiment
    def get_target_psnr(row):
        try:
            extra = json.loads(row["extra_json"])
            return float(extra.get("target_psnr", row["alpha_nominal"]))
        except Exception:
            return float(row["alpha_nominal"])

    df_raw["target_psnr"] = df_raw.apply(get_target_psnr, axis=1)

    # 1. Average over seeds within image (Rule 4)
    img_df = (
        df_raw.groupby(["target_psnr", "method", "subband", "attack", "severity", "image_id"], as_index=False)[
            ["ber", "nc", "psnr_attacked"]
        ]
        .mean()
    )

    paired_records = []
    targets = sorted(img_df["target_psnr"].unique())
    attack_conds = img_df[["attack", "severity"]].drop_duplicates().sort_values(by=["attack", "severity"]).values.tolist()

    # Comparison 1: Adaptive vs Fixed (for HL subband)
    for t_psnr in targets:
        for att, sev in attack_conds:
            sub = img_df[(img_df["target_psnr"] == t_psnr) & (img_df["attack"] == att) & (img_df["severity"] == sev)]
            ada = sub[(sub["method"] == "proposed_adaptive") & (sub["subband"] == "HL")].set_index("image_id")
            fix = sub[(sub["method"] == "proposed_fixed") & (sub["subband"] == "HL")].set_index("image_id")

            common = sorted(list(set(ada.index).intersection(set(fix.index))))
            if len(common) >= 2:
                # Difference: Adaptive - Fixed (positive means adaptive has higher NC, lower BER is better)
                diff_ber = np.array([float(ada.loc[img, "ber"]) - float(fix.loc[img, "ber"]) for img in common])
                diff_nc = np.array([float(ada.loc[img, "nc"]) - float(fix.loc[img, "nc"]) for img in common])
                n = len(common)

                from scipy import stats
                t_ber, p_ber = stats.ttest_rel(ada.loc[common, "ber"], fix.loc[common, "ber"])
                t_nc, p_nc = stats.ttest_rel(ada.loc[common, "nc"], fix.loc[common, "nc"])
                
                try:
                    res_w_nc = stats.wilcoxon(ada.loc[common, "nc"], fix.loc[common, "nc"], zero_method="pratt")
                    w_nc, pw_nc = float(res_w_nc.statistic), float(res_w_nc.pvalue)
                except Exception:
                    w_nc, pw_nc = np.nan, np.nan

                paired_records.append({
                    "comparison": "Adaptive - Fixed",
                    "subband": "HL",
                    "target_psnr": t_psnr,
                    "attack": att,
                    "severity": sev,
                    "n_images": n,
                    "diff_ber_mean": round(float(np.mean(diff_ber)), 6),
                    "diff_ber_sem": round(float(np.std(diff_ber, ddof=1) / np.sqrt(n)), 6),
                    "p_ttest_ber": round(float(p_ber), 6),
                    "diff_nc_mean": round(float(np.mean(diff_nc)), 6),
                    "diff_nc_sem": round(float(np.std(diff_nc, ddof=1) / np.sqrt(n)), 6),
                    "t_stat_nc": round(float(t_nc), 4),
                    "p_ttest_nc": round(float(p_nc), 6),
                    "wilcoxon_stat_nc": round(float(w_nc), 2) if not np.isnan(w_nc) else np.nan,
                    "p_wilcoxon_nc": round(float(pw_nc), 6) if not np.isnan(pw_nc) else np.nan,
                })

    # Comparison 2: HL vs LL (for Adaptive method)
    for t_psnr in targets:
        for att, sev in attack_conds:
            sub = img_df[(img_df["target_psnr"] == t_psnr) & (img_df["attack"] == att) & (img_df["severity"] == sev)]
            hl = sub[(sub["method"] == "proposed_adaptive") & (sub["subband"] == "HL")].set_index("image_id")
            ll = sub[(sub["method"] == "proposed_adaptive") & (sub["subband"] == "LL")].set_index("image_id")

            common = sorted(list(set(hl.index).intersection(set(ll.index))))
            if len(common) >= 2:
                diff_ber = np.array([float(hl.loc[img, "ber"]) - float(ll.loc[img, "ber"]) for img in common])
                diff_nc = np.array([float(hl.loc[img, "nc"]) - float(ll.loc[img, "nc"]) for img in common])
                n = len(common)

                from scipy import stats
                t_ber, p_ber = stats.ttest_rel(hl.loc[common, "ber"], ll.loc[common, "ber"])
                t_nc, p_nc = stats.ttest_rel(hl.loc[common, "nc"], ll.loc[common, "nc"])
                try:
                    res_w_nc = stats.wilcoxon(hl.loc[common, "nc"], ll.loc[common, "nc"], zero_method="pratt")
                    w_nc, pw_nc = float(res_w_nc.statistic), float(res_w_nc.pvalue)
                except Exception:
                    w_nc, pw_nc = np.nan, np.nan

                paired_records.append({
                    "comparison": "HL - LL",
                    "subband": "Adaptive",
                    "target_psnr": t_psnr,
                    "attack": att,
                    "severity": sev,
                    "n_images": n,
                    "diff_ber_mean": round(float(np.mean(diff_ber)), 6),
                    "diff_ber_sem": round(float(np.std(diff_ber, ddof=1) / np.sqrt(n)), 6),
                    "p_ttest_ber": round(float(p_ber), 6),
                    "diff_nc_mean": round(float(np.mean(diff_nc)), 6),
                    "diff_nc_sem": round(float(np.std(diff_nc, ddof=1) / np.sqrt(n)), 6),
                    "t_stat_nc": round(float(t_nc), 4),
                    "p_ttest_nc": round(float(p_nc), 6),
                    "wilcoxon_stat_nc": round(float(w_nc), 2) if not np.isnan(w_nc) else np.nan,
                    "p_wilcoxon_nc": round(float(pw_nc), 6) if not np.isnan(pw_nc) else np.nan,
                })

    df_paired = pd.DataFrame(paired_records)
    df_paired.to_csv(output_csv, index=False)
    print(f"[*] Paired differences written to: '{output_csv}'")
    return df_paired


# =============================================================================
# CLI MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Matched-Distortion Study and Comprehensive Watermarking Benchmark.")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker processes.")
    parser.add_argument("--estimate-only", action="store_true", help="Only run runtime estimation on 3 images and exit.")
    parser.add_argument("--payloads", type=int, default=0, help="Explicitly specify payloads per image (e.g. 2 or 6).")
    parser.add_argument("--force-6-payloads", action="store_true", help="Force running 6 payloads instead of checking runtime.")
    parser.add_argument("--max-images", type=int, default=0, help="If > 0, restrict run to N images.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print(" MATCHED-DISTORTION & PARETO COMPREHENSIVE BENCHMARK")
    print(f" Workers: {args.workers}")
    print("=" * 80)

    # STEP 1: Runtime Estimation on 3 Images
    timing = estimate_runtime(sample_size=3, workers=args.workers)

    if args.estimate_only:
        return

    # Determine payload count based on 6-hour threshold rule
    if args.payloads > 0:
        payloads_count = args.payloads
        print(f"[CONFIG] Payloads per image explicitly set to: {payloads_count}\n")
    elif timing["est_hours_6_payloads"] > 6.0 and not args.force_6_payloads:
        payloads_count = 2
        print(f"[DECISION] Estimated 6-payload runtime ({timing['est_hours_6_payloads']:.2f} h) exceeds 6 hours.")
        print(f"           Reducing payloads to 2 per image (whitened logo + balanced_0).")
        print(f"           Projected runtime with 2 payloads: {timing['est_hours_2_payloads']:.2f} hours.\n")
    else:
        payloads_count = 6
        print(f"[DECISION] Running full 6 payloads per image. Estimated runtime: {timing['est_hours_6_payloads']:.2f} h.\n")

    # STEP 2: Core Dense Alpha Grid
    print("\n>>> [1/4] Running Dense Alpha Grid (results_v2/core_raw.csv)...")
    run_core_study(
        output_csv="results_v2/core_raw.csv",
        payloads_count=payloads_count,
        workers=args.workers,
        max_images=args.max_images,
    )

    # STEP 3: Matched-PSNR Study
    print("\n>>> [2/4] Running Matched-PSNR Study (results_v2/matched_psnr_raw.csv)...")
    run_matched_psnr_study(
        output_csv="results_v2/matched_psnr_raw.csv",
        target_psnrs=[40.0, 42.0, 44.0, 46.0, 48.0, 50.0],
        workers=args.workers,
        max_images=args.max_images,
    )

    # STEP 4: Pareto Frontier Data
    print("\n>>> [3/4] Computing Pareto Frontier Data (results_v2/pareto_points.csv)...")
    compute_pareto_points(
        core_raw_csv="results_v2/core_raw.csv",
        output_csv="results_v2/pareto_points.csv",
    )

    # STEP 5: Paired Differences Analysis
    print("\n>>> [4/4] Computing Paired Differences (results_v2/paired_diffs.csv)...")
    compute_paired_differences(
        matched_raw_csv="results_v2/matched_psnr_raw.csv",
        output_csv="results_v2/paired_diffs.csv",
    )

    print("\n" + "=" * 80)
    print(" ALL BENCHMARK STUDIES COMPLETED SUCCESSFULLY")
    print(" Outputs written to results_v2/:")
    print("   1. results_v2/core_raw.csv")
    print("   2. results_v2/matched_psnr_raw.csv")
    print("   3. results_v2/pareto_points.csv")
    print("   4. results_v2/paired_diffs.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
