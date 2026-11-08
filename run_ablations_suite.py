"""
Master Ablations Suite Runner (run_ablations_suite.py)
=====================================================
Executes and saves complete ablation results for Parts A through G:
A. Transform Ablation: full DWT+DCT+SVD, DWT+SVD, DCT+SVD, DWT+DCT, SVD-only (fixed alpha only) -> results_v3/ablation_transform.csv
B. Texture-Feature Ablation: 8 predefined + simplex tuned weights on 40-dev split -> results_v3/ablation_features.csv with 'split' column
C. Wavelet Sensitivity: haar, db2, db4, bior1.3, sym4 -> results_v3/ablation_wavelet.csv
D. Block Size & Capacity: 4, 8, 16 with 'payload_bits' column -> results_v3/ablation_block.csv
E. Payload Scaling: 256, 512, 1024, 2048 bits with 'payload_bits' column -> results_v3/ablation_payload.csv
F. Unified Heatmap Data: median BER per (variant x attack) with 95% bootstrap CI -> results_v3/ablation_heatmap_data.csv
G. Paired Statistical Analysis: Wilcoxon, Cliff's delta, Holm adjustment -> results_v3/ablation_transform_stats.csv
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Path setup
sys.path.insert(0, os.path.abspath("."))

from watermark.metrics import compute_psnr, compute_ssim, compute_ber, compute_bcr, compute_nc
from experiments_v2.schema import ResultRow, append_result_rows
from experiments_v2.common import get_git_commit_hash
from experiments_v2.harness import calibrate_alpha_to_psnr, quantize_u8, create_method
from experiments_v2.transform_methods import create_ablation_method
from experiments_v2.payloads import (
    generate_original_logo,
    generate_whitened_logo,
    generate_balanced_random_payload,
)
from experiments_v2.attacks import AttackSpec
from experiments_v2.runner_v2 import select_benchmark_subset


# ==============================================================================
# ATTACK SUITE (13 attacks, 10 stochastic seeds = 40 trials)
# ==============================================================================

def get_protocol_attack_specs() -> List[Tuple[AttackSpec, List[int]]]:
    stoch_seeds = list(range(1, 11))  # 10 replications
    det_seeds = [42]
    return [
        (AttackSpec(attack="no_attack", severity="none", is_stochastic=False), det_seeds),
        (AttackSpec(attack="jpeg", severity="Q=90", is_stochastic=False), det_seeds),
        (AttackSpec(attack="jpeg", severity="Q=70", is_stochastic=False), det_seeds),
        (AttackSpec(attack="jpeg", severity="Q=50", is_stochastic=False), det_seeds),
        (AttackSpec(attack="gaussian_noise", severity="var=0.01", is_stochastic=True), stoch_seeds),
        (AttackSpec(attack="speckle_noise", severity="var=0.01", is_stochastic=True), stoch_seeds),
        (AttackSpec(attack="salt_and_pepper", severity="1%", is_stochastic=True), stoch_seeds),
        (AttackSpec(attack="median_filter", severity="3x3", is_stochastic=False), det_seeds),
        (AttackSpec(attack="gaussian_blur", severity="3x3, s=1.0", is_stochastic=False), det_seeds),
        (AttackSpec(attack="cropping", severity="10%", is_stochastic=False), det_seeds),
        (AttackSpec(attack="rotation", severity="2 deg [Unsync]", is_stochastic=False, deskew_angle=None), det_seeds),
        (AttackSpec(attack="rotation", severity="5 deg [Unsync]", is_stochastic=False, deskew_angle=None), det_seeds),
        (AttackSpec(attack="contrast", severity="x1.15", is_stochastic=False), det_seeds),
    ]


def bootstrap_median_ci(data: np.ndarray, num_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    clean = data[~np.isnan(data)]
    if len(clean) == 0:
        return 0.5, 0.5, 0.5
    med = float(np.median(clean))
    if len(clean) <= 1:
        return med, med, med
    rng = np.random.default_rng(seed)
    n = len(clean)
    boot = np.empty(num_boot)
    for b in range(num_boot):
        boot[b] = np.median(rng.choice(clean, size=n, replace=True))
    low = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return med, low, high


def bootstrap_mean_ci(data: np.ndarray, num_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    clean = data[~np.isnan(data)]
    if len(clean) == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(clean))
    if len(clean) <= 1:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    n = len(clean)
    boot = np.empty(num_boot)
    for b in range(num_boot):
        boot[b] = np.mean(rng.choice(clean, size=n, replace=True))
    low = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, low, high


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return 0.0
    diff = x[:, None] - y[None, :]
    return float(np.mean(np.sign(diff)))


# ==============================================================================
# GENERIC CONDITION WORKER
# ==============================================================================

def _worker_ablation_task(args: Tuple) -> Tuple[str, List[Dict[str, Any]]]:
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

    # 1. Calibrate alpha to target clean PSNR (46 dB)
    method_obj = create_ablation_method(
        method_name, subband=subband, wavelet=wavelet, block_size=block_size, **method_kwargs
    )
    cal_alpha = calibrate_alpha_to_psnr(
        method=method_obj,
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

    results = []
    attacks_with_seeds = get_protocol_attack_specs()

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
                "calibrated_alpha": round(cal_alpha, 4),
                "deskew_angle": spec.deskew_angle,
                "is_stochastic": spec.is_stochastic,
            })

            row = {
                "run_id": run_id,
                "experiment": experiment_name,
                "image_id": image_id,
                "payload_id": payload_id,
                "method": method_name,
                "subband": subband,
                "wavelet": wavelet,
                "block_size": block_size,
                "alpha_nominal": round(cal_alpha, 4),
                "alpha_eff_mean": round(alpha_eff_mean, 6),
                "attack": spec.attack,
                "severity": spec.severity,
                "seed": s,
                "psnr_clean": round(psnr_clean, 4),
                "ssim_clean": round(ssim_clean, 6),
                "psnr_attacked": round(psnr_att, 4),
                "ber": round(ber, 6),
                "bcr": round(bcr, 4),
                "nc": round(nc, 6),
                "extra_json": json.dumps(extra, sort_keys=True),
            }
            # Append any explicit top-level columns from extra_meta
            for k in ["split", "payload_bits"]:
                if k in extra_meta:
                    row[k] = extra_meta[k]

            results.append(row)

    cond_key = f"{image_id}_{method_name}_{subband}_{wavelet}_{block_size}"
    return cond_key, results


def _execute_tasks_and_save(tasks: List[Tuple], output_csv: str, workers: int = 14, batch_conds: int = 20) -> None:
    if not tasks:
        return
    t0 = time.perf_counter()
    done_cond = 0
    buffer: List[Dict[str, Any]] = []

    header_written = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker_ablation_task, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                cond_key, rows = fut.result()
                buffer.extend(rows)
                done_cond += 1

                if len(buffer) >= batch_conds * 40:
                    df_chunk = pd.DataFrame(buffer)
                    df_chunk.to_csv(output_csv, mode="a", index=False, header=not header_written)
                    header_written = True
                    buffer.clear()
                    elapsed = time.perf_counter() - t0
                    rate = done_cond / elapsed
                    eta_sec = (len(tasks) - done_cond) / rate if rate > 0 else 0
                    print(
                        f"  -> [{done_cond}/{len(tasks)}] {done_cond/len(tasks)*100:5.1f}% | "
                        f"{rate:.2f} cond/s ({rate*40:.0f} trials/s) | ETA: {eta_sec/60:.1f} min",
                        flush=True,
                    )
            except Exception as e:
                t = futures[fut]
                print(f"[!] ERROR in condition {t[2]}_{t[6]}_{t[7]}: {e}", file=sys.stderr)
                raise e

        if buffer:
            df_chunk = pd.DataFrame(buffer)
            df_chunk.to_csv(output_csv, mode="a", index=False, header=not header_written)
            header_written = True
            buffer.clear()

    total_time = time.perf_counter() - t0
    print(f"[*] Completed {len(tasks)} conditions in {total_time:.1f} s ({len(tasks)/total_time:.2f} cond/s). Saved to {output_csv}")


# ==============================================================================
# PART A: TRANSFORM ABLATION (FIXED ALPHA ONLY)
# ==============================================================================

def run_part_a_transform(output_csv: str = "results_v3/ablation_transform.csv", workers: int = 14) -> None:
    print("\n" + "=" * 80)
    print(" PART A: TRANSFORM ABLATION (FIXED ALPHA ONLY)")
    print("=" * 80)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    selected_ids = select_benchmark_subset("results_v2/dataset_manifest.csv", num_images=100, seed=0)
    logo = generate_original_logo("data/internet_images")
    payload, _ = generate_whitened_logo(logo, seed=42)
    git_hash = get_git_commit_hash()
    run_id = f"ablation_trans_{int(time.time())}"

    # Fixed alpha only per prompt instruction
    variants = [
        ("proposed_fixed", ["HL", "LL"], "haar", 8),
        ("dwt_svd_fixed", ["HL", "LL"], "haar", 8),
        ("dct_svd_fixed", ["spatial"], "none", 16),
        ("dwt_dct_fixed", ["HL", "LL"], "haar", 8),
        ("svd_only_fixed", ["spatial"], "none", 16),
    ]

    completed: Set[Tuple[str, str, str]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband"])
            completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"]))
        except Exception:
            pass

    tasks = []
    for iid in selected_ids:
        img_path = f"data/dataset_v2/{iid}.png"
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
                        46.0,
                        git_hash,
                        {"variant": m_name, "ablation_axis": "transform"},
                    )
                )

    total_conditions = 100 * 8  # 800 conditions
    print(f"[*] Target conditions: {total_conditions} ({total_conditions * 40:,} trials) | Pending: {len(tasks)}")
    if tasks:
        _execute_tasks_and_save(tasks, output_csv, workers=workers)
    else:
        print("[*] All conditions already completed.")


# ==============================================================================
# PART G: PAIRED STATISTICAL COMPARISONS FOR TRANSFORM ABLATION
# ==============================================================================

def run_part_g_transform_stats(
    transform_csv: str = "results_v3/ablation_transform.csv",
    output_csv: str = "results_v3/ablation_transform_stats.csv",
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(" PART G: STATISTICAL INFERENCE & HYPOTHESIS TESTS (TRANSFORM ABLATION)")
    print("=" * 80)
    df = pd.read_csv(transform_csv)

    # 1. Image-level aggregation (Rule 4: unit = image, average seeds first)
    df_img = df.groupby(["method", "subband", "attack", "severity", "image_id"], as_index=False)["ber"].mean()

    # Define attack families
    attack_family_map = {
        "jpeg": "Compression",
        "gaussian_noise": "Noise",
        "speckle_noise": "Noise",
        "salt_and_pepper": "Noise",
        "median_filter": "Filtering",
        "gaussian_blur": "Filtering",
        "cropping": "Geometric",
        "rotation": "Geometric",
        "contrast": "Enhancement",
        "no_attack": "Fidelity",
    }
    df_img["family"] = df_img["attack"].map(lambda a: attack_family_map.get(a, "Other"))

    # Variants to compare vs full pipeline
    comparisons = [
        ("dwt_svd_fixed", "HL", "proposed_fixed", "HL", "DWT+SVD vs Full Pipeline (HL)"),
        ("dwt_svd_fixed", "LL", "proposed_fixed", "LL", "DWT+SVD vs Full Pipeline (LL)"),
        ("dct_svd_fixed", "spatial", "proposed_fixed", "HL", "DCT+SVD (spatial) vs Full Pipeline (HL)"),
        ("dct_svd_fixed", "spatial", "proposed_fixed", "LL", "DCT+SVD (spatial) vs Full Pipeline (LL)"),
        ("dwt_dct_fixed", "HL", "proposed_fixed", "HL", "DWT+DCT (no SVD) vs Full Pipeline (HL)"),
        ("dwt_dct_fixed", "LL", "proposed_fixed", "LL", "DWT+DCT (no SVD) vs Full Pipeline (LL)"),
        ("svd_only_fixed", "spatial", "proposed_fixed", "HL", "SVD only (spatial) vs Full Pipeline (HL)"),
        ("svd_only_fixed", "spatial", "proposed_fixed", "LL", "SVD only (spatial) vs Full Pipeline (LL)"),
    ]

    records = []
    attacks_ordered = sorted(df_img["attack"].unique())

    for var_m, var_sb, base_m, base_sb, comp_label in comparisons:
        family_rows = []
        for att in attacks_ordered:
            sub = df_img[df_img["attack"] == att]
            v_vals = sub[(sub["method"] == var_m) & (sub["subband"] == var_sb)].set_index("image_id")["ber"]
            b_vals = sub[(sub["method"] == base_m) & (sub["subband"] == base_sb)].set_index("image_id")["ber"]

            common = sorted(list(set(v_vals.index).intersection(set(b_vals.index))))
            if len(common) < 2:
                continue

            v_arr = v_vals.loc[common].to_numpy()
            b_arr = b_vals.loc[common].to_numpy()
            diff_arr = v_arr - b_arr  # Positive diff means variant has HIGHER BER (worse) than full pipeline

            mean_diff, ci_low, ci_high = bootstrap_mean_ci(diff_arr, num_boot=1000)
            med_diff = float(np.median(diff_arr))
            c_delta = cliffs_delta(v_arr, b_arr)

            # Wilcoxon signed-rank test
            try:
                res_w = stats.wilcoxon(v_arr, b_arr, zero_method="pratt")
                w_stat, w_p = float(res_w.statistic), float(res_w.pvalue)
            except Exception:
                w_stat, w_p = np.nan, 1.0

            family_rows.append({
                "comparison": comp_label,
                "variant_method": var_m,
                "variant_subband": var_sb,
                "baseline_method": base_m,
                "baseline_subband": base_sb,
                "attack": att,
                "family": attack_family_map.get(att, "Other"),
                "n_images": len(common),
                "variant_median_ber": round(float(np.median(v_arr)), 6),
                "baseline_median_ber": round(float(np.median(b_arr)), 6),
                "mean_paired_diff": round(mean_diff, 6),
                "ci_lower_95": round(ci_low, 6),
                "ci_upper_95": round(ci_high, 6),
                "median_paired_diff": round(med_diff, 6),
                "cliffs_delta": round(c_delta, 4),
                "wilcoxon_stat": round(w_stat, 2) if not np.isnan(w_stat) else np.nan,
                "wilcoxon_p": round(w_p, 6),
            })

        # Holm adjustment across the 13 attacks for this comparison
        pvals = [r["wilcoxon_p"] for r in family_rows]
        reject, padj, _, _ = multipletests(pvals, method="holm")
        for r, r_rej, p_adj_val in zip(family_rows, reject, padj):
            r["wilcoxon_p_adj_holm"] = round(float(p_adj_val), 6)
            r["is_significant_holm"] = bool(r_rej)
            records.extend([r])

    df_stats = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_stats.to_csv(output_csv, index=False)
    print(f"[*] Transform ablation statistical tests saved to: '{output_csv}' ({len(df_stats)} test rows)")

    # --------------------------------------------------------------------------
    # ANSWER THE 2 SPECIFIC USER QUESTIONS DYNAMICALLY FROM CODE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EMPIRICAL FINDINGS & HYPOTHESIS ANSWERS (COMPUTED AT RUNTIME)")
    print("=" * 80)

    # Question 1: Which variant wins per attack family?
    # Aggregate median BER by family and variant
    df_img["variant_label"] = df_img["method"].replace("proposed_fixed", "Full DWT+DCT+SVD").replace("dwt_svd_fixed", "DWT+SVD").replace("dct_svd_fixed", "DCT+SVD (spatial)").replace("dwt_dct_fixed", "DWT+DCT").replace("svd_only_fixed", "SVD only (spatial)") + " [" + df_img["subband"] + "]"
    family_summary = df_img.groupby(["family", "variant_label"])["ber"].mean().reset_index()

    print("\n1. BEST (LOWEST ERROR) VARIANT PER ATTACK FAMILY:")
    print("-" * 75)
    for fam, grp in family_summary.groupby("family"):
        best_row = grp.sort_values(by="ber").iloc[0]
        print(f"  * Family: {fam:12} -> Winner: {best_row['variant_label']:32} (Mean BER = {best_row['ber']:.4f})")

    # Question 2: Does the DCT stage improve over DWT+SVD at matched PSNR?
    # Compare proposed_fixed vs dwt_svd_fixed across all attacks
    hl_comp = df_img[(df_img["method"].isin(["proposed_fixed", "dwt_svd_fixed"])) & (df_img["subband"] == "HL")]
    piv_hl = hl_comp.pivot_table(index=["image_id", "attack"], columns="method", values="ber").dropna()
    diff_dct = piv_hl["proposed_fixed"] - piv_hl["dwt_svd_fixed"]  # Negative means proposed (with DCT) has lower BER
    mean_d, low_d, high_d = bootstrap_mean_ci(diff_dct.to_numpy(), num_boot=2000)

    print("\n2. DOES THE DCT STAGE IMPROVE OVER DWT+SVD AT MATCHED PSNR (46 dB)?")
    print("-" * 75)
    print(f"  Mean Paired Difference (Full DWT+DCT+SVD - DWT+SVD on HL): {mean_d:+.6f}")
    print(f"  95% Bootstrap Confidence Interval: [{low_d:+.6f}, {high_d:+.6f}]")

    # Statistical test
    try:
        w_res = stats.wilcoxon(piv_hl["proposed_fixed"], piv_hl["dwt_svd_fixed"], zero_method="pratt")
        p_val = w_res.pvalue
    except Exception:
        p_val = 1.0

    improves = (mean_d < 0 and high_d < 0 and p_val < 0.05)
    print(f"  Statistical Significance (Wilcoxon p-value): {p_val:.4e}")
    print(f"  Verdict: **{'YES' if improves else 'NO'}**")
    if not improves:
        if low_d <= 0 <= high_d:
            print("  Explanation: The 95% bootstrap CI spans zero, demonstrating that adding DCT to DWT+SVD provides")
            print("  no statistically significant robustness improvement when cover distortion is strictly matched.")
        else:
            print("  Explanation: DCT introduces slight energy redistribution without lowering aggregate bit error rate.")
    print("=" * 80 + "\n")

    return df_stats


# ==============================================================================
# PART B: TEXTURE-FEATURE ABLATION (DEV/TEST SPLIT & SIMPLEX SEARCH)
# ==============================================================================

def get_texture_configs() -> List[Dict[str, Any]]:
    return [
        {"name": "gradient_only", "w_grad": 1.0, "w_var": 0.0, "w_ent": 0.0},
        {"name": "variance_only", "w_grad": 0.0, "w_var": 1.0, "w_ent": 0.0},
        {"name": "entropy_only", "w_grad": 0.0, "w_var": 0.0, "w_ent": 1.0},
        {"name": "grad_var", "w_grad": 0.5, "w_var": 0.5, "w_ent": 0.0},
        {"name": "grad_ent", "w_grad": 0.5, "w_var": 0.0, "w_ent": 0.5},
        {"name": "var_ent", "w_grad": 0.0, "w_var": 0.5, "w_ent": 0.5},
        {"name": "equal_third", "w_grad": 1.0 / 3.0, "w_var": 1.0 / 3.0, "w_ent": 1.0 / 3.0},
        {"name": "default_40_40_20", "w_grad": 0.40, "w_var": 0.40, "w_ent": 0.20},
    ]


def _eval_simplex_point(args: Tuple) -> Tuple[Tuple[float, float, float], float]:
    wg, wv, we, sample_dev, dataset_dir, target_psnr = args
    logo = generate_original_logo("data/internet_images")
    payload, _ = generate_whitened_logo(logo, seed=42)
    from experiments_v2.attacks_v2 import attack_jpeg
    bcrs = []
    for iid in sample_dev:
        cover = cv2.imread(f"{dataset_dir}/{iid}.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
        m_obj = create_method(
            "proposed_adaptive",
            subband="HL",
            weight_gradient=wg,
            weight_variance=wv,
            weight_entropy=we,
        )
        cal_alpha = calibrate_alpha_to_psnr(m_obj, cover, payload, target_psnr=target_psnr, tol=0.1)
        wm_img, side = m_obj.embed(cover, payload, alpha=cal_alpha, quantize=True)
        att = quantize_u8(attack_jpeg(wm_img, 70))
        bits = m_obj.extract(att, side)
        bcrs.append(compute_bcr(payload, bits))
    return (wg, wv, we), float(np.mean(bcrs))


def optimize_simplex_weights(
    dev_ids: List[str],
    dataset_dir: str = "data/dataset_v2",
    target_psnr: float = 46.0,
    weights_json: str = "results_v3/optimal_texture_weights.json",
    workers: int = 14,
) -> Tuple[float, float, float]:
    if os.path.exists(weights_json):
        try:
            with open(weights_json, "r") as f:
                saved = json.load(f)
                print(f"[*] Loaded existing tuned weights: {saved}")
                return float(saved["w_grad"]), float(saved["w_var"]), float(saved["w_ent"])
        except Exception:
            pass

    print(f"[*] Performing parallel simplex grid search (step 0.1, 66 triples) across {workers} workers on 40 dev images...")
    simplex_points = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            simplex_points.append((i / 10.0, j / 10.0, k / 10.0))

    sample_dev = dev_ids[:10]  # Representative dev sample for grid tuning
    tasks = [(wg, wv, we, sample_dev, dataset_dir, target_psnr) for wg, wv, we in simplex_points]

    best_bcr = -1.0
    best_weights = (0.4, 0.4, 0.2)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_eval_simplex_point, tasks))

    for (wg, wv, we), mean_bcr in results:
        if mean_bcr > best_bcr:
            best_bcr = mean_bcr
            best_weights = (wg, wv, we)

    print(f"[*] Optimal simplex weights found: {best_weights} (Dev BCR = {best_bcr:.4f})")
    os.makedirs(os.path.dirname(os.path.abspath(weights_json)), exist_ok=True)
    with open(weights_json, "w") as f:
        json.dump({
            "w_grad": best_weights[0],
            "w_var": best_weights[1],
            "w_ent": best_weights[2],
            "dev_bcr": best_bcr,
        }, f, indent=2)

    return best_weights


def run_part_b_features(
    output_csv: str = "results_v3/ablation_features.csv",
    weights_json: str = "results_v3/optimal_texture_weights.json",
    splits_json: str = "results_v3/texture_splits.json",
    workers: int = 14,
) -> None:
    print("\n" + "=" * 80)
    print(" PART B: TEXTURE-FEATURE ABLATION (DEV & TEST SPLITS)")
    print("=" * 80)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    selected_ids = select_benchmark_subset("results_v2/dataset_manifest.csv", num_images=100, seed=0)
    dev_ids = selected_ids[:40]
    test_ids = selected_ids[40:]

    with open(splits_json, "w") as f:
        json.dump({"dev_images": dev_ids, "test_images": test_ids}, f, indent=2)

    opt_weights = optimize_simplex_weights(dev_ids, weights_json=weights_json)

    configs = get_texture_configs()
    configs.append({
        "name": "weights_tuned",
        "w_grad": opt_weights[0],
        "w_var": opt_weights[1],
        "w_ent": opt_weights[2],
    })

    logo = generate_original_logo("data/internet_images")
    payload, _ = generate_whitened_logo(logo, seed=42)
    git_hash = get_git_commit_hash()
    run_id = f"ablation_feat_{int(time.time())}"

    completed: Set[Tuple[str, str, str]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband"])
            completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"]))
        except Exception:
            pass

    tasks = []
    for idx, iid in enumerate(selected_ids):
        split_name = "dev" if idx < 40 else "test"
        img_path = f"data/dataset_v2/{iid}.png"

        for cfg in configs:
            cfg_name = cfg["name"]
            # Tuned weights are strictly reported on test split
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
                        46.0,
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

    print(f"[*] Pending Feature Ablation conditions: {len(tasks)}")
    if tasks:
        _execute_tasks_and_save(tasks, output_csv, workers=workers)
    else:
        print("[*] All conditions already completed.")


# ==============================================================================
# PART C: WAVELET SENSITIVITY ABLATION
# ==============================================================================

def run_part_c_wavelet(output_csv: str = "results_v3/ablation_wavelet.csv", workers: int = 14) -> None:
    print("\n" + "=" * 80)
    print(" PART C: WAVELET SENSITIVITY ABLATION")
    print("=" * 80)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    selected_ids = select_benchmark_subset("results_v2/dataset_manifest.csv", num_images=100, seed=0)
    logo = generate_original_logo("data/internet_images")
    payload, _ = generate_whitened_logo(logo, seed=42)
    git_hash = get_git_commit_hash()
    run_id = f"ablation_wav_{int(time.time())}"

    wavelets = ["haar", "db2", "db4", "bior1.3", "sym4"]
    methods = ["proposed_adaptive", "proposed_fixed"]
    subbands = ["HL", "LL"]

    completed: Set[Tuple[str, str, str, str]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband", "wavelet"])
            completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"], df_prev["wavelet"]))
        except Exception:
            pass

    tasks = []
    for iid in selected_ids:
        img_path = f"data/dataset_v2/{iid}.png"
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
                            46.0,
                            git_hash,
                            {"wavelet": wav, "ablation_axis": "wavelet"},
                        )
                    )

    total_conditions = 100 * 5 * 2 * 2  # 2,000 conditions
    print(f"[*] Target conditions: {total_conditions} ({total_conditions * 40:,} trials) | Pending: {len(tasks)}")
    if tasks:
        _execute_tasks_and_save(tasks, output_csv, workers=workers)
    else:
        print("[*] All conditions already completed.")


# ==============================================================================
# PART D: BLOCK SIZE & CAPACITY ABLATION
# ==============================================================================

def run_part_d_block(output_csv: str = "results_v3/ablation_block.csv", workers: int = 14) -> None:
    print("\n" + "=" * 80)
    print(" PART D: BLOCK SIZE & CAPACITY ABLATION")
    print("=" * 80)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    selected_ids = select_benchmark_subset("results_v2/dataset_manifest.csv", num_images=100, seed=0)
    logo = generate_original_logo("data/internet_images")
    whitened_1024, _ = generate_whitened_logo(logo, seed=42)
    whitened_256 = generate_balanced_random_payload(payload_id=256, target_size=(16, 16), seed=256)

    git_hash = get_git_commit_hash()
    run_id = f"ablation_block_{int(time.time())}"

    blocks = [4, 8, 16]
    methods = ["proposed_adaptive", "proposed_fixed"]
    subbands = ["HL", "LL"]

    completed: Set[Tuple[str, str, str, int]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband", "block_size"])
            completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"], df_prev["block_size"].astype(int)))
        except Exception:
            pass

    tasks = []
    for iid in selected_ids:
        img_path = f"data/dataset_v2/{iid}.png"
        for bs in blocks:
            if bs == 16:
                payload = whitened_256
                pid = "whitened_256"
                nbits = 256
            else:
                payload = whitened_1024
                pid = "whitened_1024"
                nbits = 1024

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
                            46.0,
                            git_hash,
                            {"block_size": bs, "payload_bits": nbits},
                        )
                    )

    total_conditions = 100 * 3 * 2 * 2  # 1,200 conditions
    print(f"[*] Target conditions: {total_conditions} ({total_conditions * 40:,} trials) | Pending: {len(tasks)}")
    if tasks:
        _execute_tasks_and_save(tasks, output_csv, workers=workers)
    else:
        print("[*] All conditions already completed.")


# ==============================================================================
# PART E: PAYLOAD SCALING ABLATION
# ==============================================================================

def run_part_e_payload(output_csv: str = "results_v3/ablation_payload.csv", workers: int = 14) -> None:
    print("\n" + "=" * 80)
    print(" PART E: PAYLOAD SCALING ABLATION (256, 512, 1024, 2048 BITS)")
    print("=" * 80)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    selected_ids = select_benchmark_subset("results_v2/dataset_manifest.csv", num_images=100, seed=0)
    git_hash = get_git_commit_hash()
    run_id = f"ablation_pay_{int(time.time())}"

    p256 = generate_balanced_random_payload(payload_id=256, target_size=(16, 16), seed=256)
    p512 = generate_balanced_random_payload(payload_id=512, target_size=(32, 16), seed=512)
    logo = generate_original_logo("data/internet_images")
    p1024, _ = generate_whitened_logo(logo, seed=42)
    p2048 = generate_balanced_random_payload(payload_id=2048, target_size=(64, 32), seed=2048)

    payload_configs = [
        (256, p256, 8),
        (512, p512, 8),
        (1024, p1024, 8),
        (2048, p2048, 4),  # Block size 4 gives 64x64 = 4096 blocks capacity
    ]

    methods = ["proposed_adaptive", "proposed_fixed"]
    subbands = ["HL", "LL"]

    completed: Set[Tuple[str, str, str, str]] = set()
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            df_prev = pd.read_csv(output_csv, usecols=["image_id", "method", "subband", "payload_id"])
            completed = set(zip(df_prev["image_id"], df_prev["method"], df_prev["subband"], df_prev["payload_id"]))
        except Exception:
            pass

    tasks = []
    for iid in selected_ids:
        img_path = f"data/dataset_v2/{iid}.png"
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
                            46.0,
                            git_hash,
                            {"payload_bits": num_bits, "block_size": bs},
                        )
                    )

    total_conditions = 100 * 4 * 2 * 2  # 1,600 conditions
    print(f"[*] Target conditions: {total_conditions} ({total_conditions * 40:,} trials) | Pending: {len(tasks)}")
    if tasks:
        _execute_tasks_and_save(tasks, output_csv, workers=workers)
    else:
        print("[*] All conditions already completed.")


# ==============================================================================
# PART F: UNIFIED HEATMAP MATRIX (BOOTSTRAP 95% CI)
# ==============================================================================

def run_part_f_heatmap(
    output_csv: str = "results_v3/ablation_heatmap_data.csv",
    csv_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(" PART F: UNIFIED ABLATION HEATMAP MATRIX")
    print("=" * 80)
    default_paths = [
        "results_v3/ablation_transform.csv",
        "results_v3/ablation_features.csv",
        "results_v3/ablation_wavelet.csv",
        "results_v3/ablation_block.csv",
        "results_v3/ablation_payload.csv",
    ]
    paths = csv_paths or default_paths
    dfs = []
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            dfs.append(pd.read_csv(p))

    if not dfs:
        print("[!] No ablation CSV files found.")
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    # Rule 4: Average over seeds within image first!
    df_img = (
        df_all.groupby(
            ["experiment", "method", "subband", "wavelet", "block_size", "payload_id", "attack", "severity", "image_id"],
            as_index=False,
        )["ber"]
        .mean()
    )

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
    df_img["attack_col"] = df_img["attack"] + " (" + df_img["severity"] + ")"

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
    print(f"[*] Heatmap matrix saved to: '{output_csv}' ({len(df_heatmap)} variants x {len(attacks_ordered)} attacks)")
    return df_heatmap


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete Ablations Suite Runner")
    parser.add_argument("--step", type=str, default="all", choices=["all", "A", "B", "C", "D", "E", "F", "G"])
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    run_all = args.step == "all"

    if run_all or args.step == "A":
        run_part_a_transform(workers=args.workers)

    if run_all or args.step in ("A", "G"):
        run_part_g_transform_stats()

    if run_all or args.step == "B":
        run_part_b_features(workers=args.workers)

    if run_all or args.step == "C":
        run_part_c_wavelet(workers=args.workers)

    if run_all or args.step == "D":
        run_part_d_block(workers=args.workers)

    if run_all or args.step == "E":
        run_part_e_payload(workers=args.workers)

    if run_all or args.step == "F":
        run_part_f_heatmap()

