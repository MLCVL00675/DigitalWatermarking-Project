"""
Matched-Distortion Extended Attack Sweep and Compound Chain Study Runner (v2)

Executes:
1. Fixed random subset of 100 benchmark images (seed 0).
2. Methods: proposed_fixed and proposed_adaptive.
3. Subbands: HL and LL.
4. Matched PSNR: 46 dB (calibrated per image/method/subband with QUANTIZE=True).
5. Canonical 32x32 whitened payload (balanced 50% ones).
6. Single-attack sweeps (>= 6 levels each) with 10 seeds for stochastic attacks.
   Saved to results_v2/severity_raw.csv.
7. Compound degradation chains in exact sequence.
   Saved to results_v2/compound_raw.csv.
8. Breakdown severity calculation (first severity where median BER > 0.10 and > 0.25)
   enforcing Rule 4 (image as statistical unit).
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
)
from experiments_v2.attacks_v2 import (
    AttackSpecV2,
    get_single_attack_specs,
    get_compound_chain_specs,
)


# Ordered attack progressions for breakdown severity determination
ORDERED_ATTACK_SEVERITIES = {
    "jpeg": ["Q=95", "Q=90", "Q=80", "Q=70", "Q=60", "Q=50", "Q=40", "Q=30"],
    "gaussian_noise": ["var=0.0005", "var=0.001", "var=0.0025", "var=0.005", "var=0.01", "var=0.02"],
    "salt_and_pepper": ["0.1%", "0.5%", "1%", "2%", "5%"],
    "gaussian_blur": ["s=0.5", "s=0.75", "s=1.0", "s=1.5", "s=2.0", "s=3.0"],
    "median_filter": ["3x3", "5x5", "7x7"],
    "centered_crop": ["2.5%", "5.0%", "10.0%", "15.0%", "20.0%", "30.0%"],
    "random_crop": ["2.5%", "5.0%", "10.0%", "15.0%", "20.0%", "30.0%"],
    "scaling": ["0.9x", "1.1x", "0.75x", "1.25x", "0.5x", "1.5x"],
    "translation": ["1px", "2px", "4px", "8px", "16px"],
    "shear": ["0.01", "0.02", "0.05"],
    "rotation": ["0.5deg", "1.0deg", "2.0deg", "5.0deg", "10.0deg"],
    "contrast": ["x0.9", "x1.1", "x0.8", "x1.2"],
    "gamma": ["g=0.9", "g=1.1", "g=0.8", "g=1.2"],
}


def select_benchmark_subset(
    manifest_csv: str = "results_v2/dataset_manifest.csv",
    num_images: int = 100,
    seed: int = 0,
) -> List[str]:
    """Deterministically selects a fixed random subset of images."""
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    df = pd.read_csv(manifest_csv)
    all_ids = sorted(df["image_id"].tolist())
    rng = np.random.default_rng(seed)
    selected = sorted(list(rng.choice(all_ids, size=num_images, replace=False)))
    return selected


def _load_completed_condition_keys(csv_path: str) -> Set[Tuple[str, str, str]]:
    """Returns set of (image_id, method, subband) completed in csv_path."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    df = pd.read_csv(csv_path, usecols=["image_id", "method", "subband"])
    completed = set(zip(df["image_id"], df["method"], df["subband"]))
    return completed


def _worker_condition_v2(args: Tuple) -> Tuple[str, List[ResultRow], List[ResultRow]]:
    """
    Worker task:
    1. Loads cover image.
    2. Calibrates alpha to 46 dB PSNR.
    3. Embeds watermark once with 8-bit integer quantization.
    4. Evaluates all single attacks (severity sweep).
    5. Evaluates all compound degradation chains.
    Returns (condition_key, severity_rows, compound_rows).
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
        raise FileNotFoundError(f"Failed to read image at: {image_path}")
    cover = cover.astype(np.float64)

    # 1. Monotonic Bisection Alpha Calibration to target clean PSNR (with quantize=True)
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

    # Base extra info
    base_extra = {
        "git_commit": git_hash,
        "target_psnr": target_psnr,
        "calibrated_alpha": cal_alpha,
        "config": {"method": method_name, "subband": subband, "target_psnr": target_psnr},
    }

    # 3. Single-attack severity sweep evaluations
    severity_rows: List[ResultRow] = []
    single_specs = get_single_attack_specs()

    for spec, seeds in single_specs:
        for s in seeds:
            rng = np.random.default_rng(s) if spec.is_stochastic else None
            att_img = spec.apply(wm_img, rng=rng)
            att_img = quantize_u8(att_img)

            extracted_bits = fast_batched_extract(
                att_img, metadata, deskew_angle=spec.deskew_angle
            )

            psnr_att = float(compute_psnr(cover, att_img))
            ber = float(compute_ber(payload, extracted_bits))
            bcr = float(compute_bcr(payload, extracted_bits))
            nc = float(compute_nc(payload, extracted_bits))

            extra = dict(base_extra)
            extra["is_stochastic"] = spec.is_stochastic
            extra["deskew_angle"] = spec.deskew_angle

            severity_rows.append(
                ResultRow(
                    run_id=run_id,
                    experiment="severity_sweep",
                    image_id=image_id,
                    payload_id=payload_id,
                    method=method_name,
                    subband=subband,
                    wavelet="haar",
                    block_size=8,
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

    # 4. Compound degradation chains
    compound_rows: List[ResultRow] = []
    compound_specs = get_compound_chain_specs()

    for spec, seeds in compound_specs:
        for s in seeds:
            rng = np.random.default_rng(s) if spec.is_stochastic else None
            att_img = spec.apply(wm_img, rng=rng)
            att_img = quantize_u8(att_img)

            extracted_bits = fast_batched_extract(
                att_img, metadata, deskew_angle=spec.deskew_angle
            )

            psnr_att = float(compute_psnr(cover, att_img))
            ber = float(compute_ber(payload, extracted_bits))
            bcr = float(compute_bcr(payload, extracted_bits))
            nc = float(compute_nc(payload, extracted_bits))

            extra = dict(base_extra)
            extra["is_stochastic"] = spec.is_stochastic
            extra["chain_name"] = spec.severity

            compound_rows.append(
                ResultRow(
                    run_id=run_id,
                    experiment="compound_chain",
                    image_id=image_id,
                    payload_id=payload_id,
                    method=method_name,
                    subband=subband,
                    wavelet="haar",
                    block_size=8,
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

    cond_key = f"{image_id}_{method_name}_{subband}"
    return cond_key, severity_rows, compound_rows


def run_extended_study(
    num_images: int = 100,
    subset_seed: int = 0,
    target_psnr: float = 46.0,
    workers: int = 12,
    manifest_csv: str = "results_v2/dataset_manifest.csv",
    dataset_dir: str = "data/dataset_v2",
    severity_csv: str = "results_v2/severity_raw.csv",
    compound_csv: str = "results_v2/compound_raw.csv",
) -> None:
    """Executes the full matched-distortion extended study."""
    os.makedirs(os.path.dirname(os.path.abspath(severity_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(compound_csv)), exist_ok=True)
    init_results_csv(severity_csv)
    init_results_csv(compound_csv)

    # 1. Select 100 images
    selected_ids = select_benchmark_subset(
        manifest_csv=manifest_csv, num_images=num_images, seed=subset_seed
    )
    print(f"[*] Selected {len(selected_ids)} benchmark images (seed {subset_seed}).")

    # 2. Canonical payload: whitened logo (50% ones)
    logo = generate_original_logo()
    payload, _ = generate_whitened_logo(logo, seed=42)
    payload_id = "whitened_logo"

    git_hash = get_git_commit_hash()
    run_id = f"v2_matched_{int(time.time())}"

    # 3. Check completed conditions
    completed_sev = _load_completed_condition_keys(severity_csv)
    completed_cmp = _load_completed_condition_keys(compound_csv)
    fully_completed = completed_sev.intersection(completed_cmp)

    methods = ["proposed_fixed", "proposed_adaptive"]
    subbands = ["HL", "LL"]

    tasks = []
    for iid in selected_ids:
        img_path = os.path.join(dataset_dir, f"{iid}.png")
        for m in methods:
            for sb in subbands:
                if (iid, m, sb) in fully_completed:
                    continue
                tasks.append(
                    (
                        run_id,
                        iid,
                        img_path,
                        payload_id,
                        payload,
                        m,
                        sb,
                        target_psnr,
                        git_hash,
                    )
                )

    total_conditions = len(selected_ids) * len(methods) * len(subbands)
    print(f"[*] Total conditions: {total_conditions} | Completed: {len(fully_completed)} | Pending: {len(tasks)}")

    if not tasks:
        print("[*] All conditions already completed.")
        return

    # 4. Multiprocessing execution
    start_time = time.perf_counter()
    done_count = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_condition_v2, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                cond_key, s_rows, c_rows = fut.result()
                append_result_rows(severity_csv, s_rows)
                append_result_rows(compound_csv, c_rows)
                done_count += 1

                elapsed = time.perf_counter() - start_time
                rate = done_count / elapsed
                remaining = (len(tasks) - done_count) / rate if rate > 0 else 0
                print(
                    f"  [{done_count}/{len(tasks)}] Done: {cond_key} "
                    f"| {rate:.2f} cond/s | ETA: {remaining:.1f}s",
                    flush=True,
                )
            except Exception as e:
                t = futures[fut]
                print(f"[!] Error in condition {t[1]}_{t[5]}_{t[6]}: {e}", file=sys.stderr)
                raise e

    total_time = time.perf_counter() - start_time
    print(f"\n[*] Execution completed successfully in {total_time:.2f}s.")


def compute_and_print_breakdown_severities(
    severity_csv: str = "results_v2/severity_raw.csv",
) -> pd.DataFrame:
    """
    Computes breakdown severity (first severity where median BER > 0.10 and > 0.25).
    Strictly adheres to Rule 4: image is the statistical unit.
    """
    if not os.path.exists(severity_csv):
        raise FileNotFoundError(f"Severity CSV not found: {severity_csv}")

    df_raw = pd.read_csv(severity_csv)

    # 1. Enforce Rule 4: Average seeds within image first!
    df_img = (
        df_raw.groupby(["method", "subband", "attack", "severity", "image_id"], as_index=False)["ber"]
        .mean()
    )

    # 2. Across 100 images, compute median BER
    df_med = (
        df_img.groupby(["method", "subband", "attack", "severity"], as_index=False)["ber"]
        .median()
    )

    # 3. Determine breakdown severity for each attack
    results = []

    for attack_name, ordered_sevs in ORDERED_ATTACK_SEVERITIES.items():
        row_data = {"attack": attack_name}

        for sb in ["HL", "LL"]:
            for m in ["proposed_adaptive", "proposed_fixed"]:
                m_short = "Adaptive" if "adaptive" in m else "Fixed"
                col_prefix = f"{sb}_{m_short}"

                sub_df = df_med[(df_med["attack"] == attack_name) & (df_med["subband"] == sb) & (df_med["method"] == m)]
                med_map = dict(zip(sub_df["severity"], sub_df["ber"]))

                bd_10 = "> max"
                bd_25 = "> max"

                for sev in ordered_sevs:
                    if sev in med_map:
                        med_val = med_map[sev]
                        if med_val > 0.10 and bd_10 == "> max":
                            bd_10 = f"{sev} ({med_val:.3f})"
                        if med_val > 0.25 and bd_25 == "> max":
                            bd_25 = f"{sev} ({med_val:.3f})"

                row_data[f"{col_prefix}_BER>0.10"] = bd_10
                row_data[f"{col_prefix}_BER>0.25"] = bd_25

        results.append(row_data)

    df_breakdown = pd.DataFrame(results)

    print("\n" + "=" * 120)
    print(" BREAKDOWN SEVERITY SUMMARY TABLE (MATCHED PSNR = 46.0 dB, N = 100 IMAGES, RULE 4 IMAGE-UNIT)")
    print(" First severity where median BER > 0.10 and > 0.25 across benchmark images")
    print("=" * 120)

    # Print nicely formatted sub-tables for readability
    print("\n--- SUBBAND HL: BREAKDOWN SEVERITIES ---")
    hl_cols = ["attack", "HL_Adaptive_BER>0.10", "HL_Fixed_BER>0.10", "HL_Adaptive_BER>0.25", "HL_Fixed_BER>0.25"]
    print(df_breakdown[hl_cols].to_string(index=False))

    print("\n--- SUBBAND LL: BREAKDOWN SEVERITIES ---")
    ll_cols = ["attack", "LL_Adaptive_BER>0.10", "LL_Fixed_BER>0.10", "LL_Adaptive_BER>0.25", "LL_Fixed_BER>0.25"]
    print(df_breakdown[ll_cols].to_string(index=False))

    print("=" * 120 + "\n")
    return df_breakdown


def compute_and_print_compound_summary(
    compound_csv: str = "results_v2/compound_raw.csv",
) -> pd.DataFrame:
    """Summarizes compound degradation chains adhering to Rule 4."""
    if not os.path.exists(compound_csv):
        raise FileNotFoundError(f"Compound CSV not found: {compound_csv}")

    df_raw = pd.read_csv(compound_csv)

    # 1. Enforce Rule 4: Average seeds within image first!
    df_img = (
        df_raw.groupby(["method", "subband", "severity", "image_id"], as_index=False)[
            ["ber", "nc", "bcr", "psnr_attacked"]
        ]
        .mean()
    )

    # 2. Across 100 images, compute median and IQR
    records = []
    chain_order = [
        "crop10%+JPEG70",
        "rotate2°+JPEG70",
        "downscale0.75+resize+JPEG70",
        "blur(s=1)+JPEG70",
        "noise(0.005)+JPEG70",
        "JPEG70+blur(s=1)",
        "crop10%+resize+JPEG50",
        "contrast+noise+JPEG70",
    ]

    for chain in chain_order:
        row = {"compound_chain": chain}
        for sb in ["HL", "LL"]:
            for m in ["proposed_adaptive", "proposed_fixed"]:
                m_short = "Adaptive" if "adaptive" in m else "Fixed"
                sub = df_img[(df_img["severity"] == chain) & (df_img["subband"] == sb) & (df_img["method"] == m)]
                if len(sub) > 0:
                    med_ber = float(sub["ber"].median())
                    iqr_ber = float(sub["ber"].quantile(0.75) - sub["ber"].quantile(0.25))
                    med_nc = float(sub["nc"].median())
                    row[f"{sb}_{m_short}_BER"] = f"{med_ber:.4f} (±{iqr_ber:.4f})"
                    row[f"{sb}_{m_short}_NC"] = f"{med_nc:.4f}"
                else:
                    row[f"{sb}_{m_short}_BER"] = "N/A"
                    row[f"{sb}_{m_short}_NC"] = "N/A"
        records.append(row)

    df_comp = pd.DataFrame(records)
    print("\n" + "=" * 120)
    print(" COMPOUND DEGRADATION CHAINS SUMMARY TABLE (MATCHED PSNR = 46.0 dB, N = 100 IMAGES)")
    print(" Median BER (±IQR) and Median NC across benchmark images")
    print("=" * 120)
    print(df_comp.to_string(index=False))
    print("=" * 120 + "\n")
    return df_comp


def main():
    parser = argparse.ArgumentParser(description="Extended Attack Sweep Study Runner (v2)")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker processes")
    parser.add_argument("--num-images", type=int, default=100, help="Number of benchmark images")
    parser.add_argument("--subset-seed", type=int, default=0, help="Seed for random image subset")
    parser.add_argument("--target-psnr", type=float, default=46.0, help="Target clean PSNR in dB")
    parser.add_argument("--manifest-csv", type=str, default="results_v2/dataset_manifest.csv")
    parser.add_argument("--dataset-dir", type=str, default="data/dataset_v2")
    parser.add_argument("--severity-csv", type=str, default="results_v2/severity_raw.csv")
    parser.add_argument("--compound-csv", type=str, default="results_v2/compound_raw.csv")
    args = parser.parse_args()

    run_extended_study(
        num_images=args.num_images,
        subset_seed=args.subset_seed,
        target_psnr=args.target_psnr,
        workers=args.workers,
        manifest_csv=args.manifest_csv,
        dataset_dir=args.dataset_dir,
        severity_csv=args.severity_csv,
        compound_csv=args.compound_csv,
    )

    compute_and_print_breakdown_severities(severity_csv=args.severity_csv)
    compute_and_print_compound_summary(compound_csv=args.compound_csv)


if __name__ == "__main__":
    main()
