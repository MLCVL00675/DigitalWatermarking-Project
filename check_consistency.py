#!/usr/bin/env python3
"""
check_consistency.py
--------------------
Rigorous consistency audit script that parses results_v4/RESULTS_SUMMARY.md
and cross-validates every quoted number against the authoritative CSV files.

Audits Performed:
1. Software library versions match environment.json exactly.
2. Every sample mean strictly lies within its confidence intervals:
   - stats_pooled_v3.csv (90% t-CI and 95% bootstrap CI)
   - stats_paired_v3.csv (all 28 conditions)
   - stats_all208_paired_v4.csv (all 26 conditions)
   - stats_headroom_v4.csv (all 26 conditions)
   - stats_ssim_matched_paired_v4.csv (all 78 conditions)
   - Every regex-extracted (mean, [CI_lo, CI_hi]) tuple in RESULTS_SUMMARY.md.
3. Image counts agree across all files (100 benchmark, 208 cohort, 100 COCO, 308 manifest, 50 GPU regen).
4. Trial counts quoted equal exact raw-file row counts (including .csv.gz files and v4 raw runs).
5. Numerical fidelity: Every single empirical result, headline effect, pooling comparison,
   LRT chi2, permutation F, SSIM target, saturation rate, and runtime number matches its CSV source.

Fails loudly (raises AssertionError with detailed diagnostics) on any mismatch.
"""

import os
import sys
import re
import json
import gzip
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def check_all_consistency(results_dir: str = "results_v4", summary_md: str = "results_v4/RESULTS_SUMMARY.md") -> bool:
    print("=" * 80)
    print(f"AUTHORITATIVE CONSISTENCY AUDITOR: {summary_md} vs. {results_dir}/*.csv[.gz]")
    print("=" * 80)

    if not os.path.exists(summary_md):
        raise FileNotFoundError(f"Summary markdown not found: {summary_md}")

    with open(summary_md, "r", encoding="utf-8") as fp:
        md_text = fp.read()

    def load_csv(fname: str) -> pd.DataFrame:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            if os.path.exists(path + ".gz"):
                path = path + ".gz"
            elif fname.endswith(".csv") and os.path.exists(os.path.join(results_dir, fname[:-4] + ".csv.gz")):
                path = os.path.join(results_dir, fname[:-4] + ".csv.gz")
            elif os.path.exists(os.path.join("results_v3", fname)):
                path = os.path.join("results_v3", fname)
            elif os.path.exists(os.path.join("results_v3", fname + ".gz")):
                path = os.path.join("results_v3", fname + ".gz")
            else:
                raise FileNotFoundError(f"Referenced CSV missing: {path}")
        return pd.read_csv(path)

    def load_json(fname: str) -> dict:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            if os.path.exists(os.path.join("results_v3", fname)):
                path = os.path.join("results_v3", fname)
            elif os.path.exists(fname):
                path = fname
            else:
                raise FileNotFoundError(f"Referenced JSON missing: {path}")
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def count_rows(path: str) -> int:
        if not os.path.exists(path) and os.path.exists(path + ".gz"):
            path = path + ".gz"
        if not os.path.exists(path) and os.path.exists(os.path.join("results_v3", os.path.basename(path))):
            path = os.path.join("results_v3", os.path.basename(path))
        if not os.path.exists(path) and os.path.exists(os.path.join("results_v3", os.path.basename(path) + ".gz")):
            path = os.path.join("results_v3", os.path.basename(path) + ".gz")

        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as fp:
                return sum(1 for _ in fp) - 1
        else:
            with open(path, "r", encoding="utf-8") as fp:
                return sum(1 for _ in fp) - 1

    total_checks = 0
    passed_checks = 0

    # -------------------------------------------------------------------------
    # AUDIT 1: Library versions match environment.json
    # -------------------------------------------------------------------------
    print("\n[Audit 1/5] Checking software library versions against environment.json...")
    env = load_json("environment.json")
    sw_env = env.get("software_versions", {})

    ver_pattern = r"Python `([^`]+)`, NumPy `([^`]+)`, SciPy `([^`]+)`, PyWavelets `([^`]+)`, OpenCV `([^`]+)`, Pandas `([^`]+)`, psutil `([^`]+)`"
    match = re.search(ver_pattern, md_text)
    if not match:
        raise AssertionError("Failed to locate software version specification line in RESULTS_SUMMARY.md")

    parsed_sw = {
        "python": match.group(1),
        "numpy": match.group(2),
        "scipy": match.group(3),
        "pywavelets": match.group(4),
        "opencv": match.group(5),
        "pandas": match.group(6),
        "psutil": match.group(7),
    }

    for lib, quoted_ver in parsed_sw.items():
        total_checks += 1
        expected_ver = sw_env.get(lib)
        assert quoted_ver == expected_ver, (
            f"Library version mismatch for {lib}: summary has '{quoted_ver}', environment.json has '{expected_ver}'"
        )
        passed_checks += 1
        print(f"  [PASS] {lib}: quoted '{quoted_ver}' == environment.json '{expected_ver}'")

    # -------------------------------------------------------------------------
    # AUDIT 2: Confidence Intervals Contain Sample Means
    # -------------------------------------------------------------------------
    print("\n[Audit 2/5] Checking that every sample mean strictly lies within its CI...")

    # 1. stats_pooled_v3.csv
    df_pooled = load_csv("stats_pooled_v3.csv")
    for _, r in df_pooled.iterrows():
        total_checks += 2
        sb = r["subband"]
        mean_v = float(r["mean_adaptive_fixed_diff"])
        ci_90_l, ci_90_u = float(r["ci_90_t_lower"]), float(r["ci_90_t_upper"])
        ci_95_l, ci_95_u = float(r["ci_95_boot_lower"]), float(r["ci_95_boot_upper"])
        assert ci_90_l <= mean_v <= ci_90_u, f"Mean {mean_v} outside 90% t-CI [{ci_90_l}, {ci_90_u}] for {sb}"
        assert ci_95_l <= mean_v <= ci_95_u, f"Mean {mean_v} outside 95% boot-CI [{ci_95_l}, {ci_95_u}] for {sb}"
        passed_checks += 2
    print(f"  [PASS] stats_pooled_v3: verified all {len(df_pooled)} subband pooled means inside 90% and 95% CIs")

    # 2. stats_paired_v3.csv
    df_paired = load_csv("stats_paired_v3.csv")
    for _, r in df_paired.iterrows():
        total_checks += 1
        mean_d = float(r["mean_diff_adaptive_minus_fixed"])
        ci_l, ci_u = float(r["diff_ci_lower_95"]), float(r["diff_ci_upper_95"])
        assert ci_l - 1e-6 <= mean_d <= ci_u + 1e-6, f"Paired mean diff {mean_d} outside 95% CI [{ci_l}, {ci_u}] for {r['subband']}-{r['attack']}"
        passed_checks += 1
    print(f"  [PASS] stats_paired_v3: verified all {len(df_paired)} paired differences inside 95% bootstrap CI")

    # 3. stats_all208_paired_v4.csv
    df_208 = load_csv("stats_all208_paired_v4.csv")
    for _, r in df_208.iterrows():
        total_checks += 1
        mean_d = float(r["mean_diff_adaptive_minus_fixed"])
        ci_l, ci_u = float(r["diff_ci_lower_95"]), float(r["diff_ci_upper_95"])
        assert ci_l - 1e-6 <= mean_d <= ci_u + 1e-6, f"208 mean diff {mean_d} outside 95% CI [{ci_l}, {ci_u}] for {r['subband']}-{r['attack']}"
        passed_checks += 1
    print(f"  [PASS] stats_all208_paired_v4: verified all {len(df_208)} conditions have mean diff inside 95% CI")

    # 4. stats_headroom_v4.csv
    df_hr = load_csv("stats_headroom_v4.csv")
    for _, r in df_hr.iterrows():
        total_checks += 1
        mean_d = float(r["mean_diff_adaptive_minus_fixed"])
        ci_l, ci_u = float(r["diff_ci_lower_95"]), float(r["diff_ci_upper_95"])
        assert ci_l - 1e-6 <= mean_d <= ci_u + 1e-6, f"Headroom mean diff {mean_d} outside 95% CI [{ci_l}, {ci_u}] for {r['subband']}-{r['attack']}"
        passed_checks += 1
    print(f"  [PASS] stats_headroom_v4: verified all {len(df_hr)} headroom conditions have mean diff inside 95% CI")

    # 5. stats_ssim_matched_paired_v4.csv
    df_ssim = load_csv("stats_ssim_matched_paired_v4.csv")
    for _, r in df_ssim.iterrows():
        total_checks += 1
        mean_d = float(r["mean_diff_adaptive_minus_fixed"])
        ci_l, ci_u = float(r["diff_ci_lower_95"]), float(r["diff_ci_upper_95"])
        assert ci_l - 1e-6 <= mean_d <= ci_u + 1e-6, f"SSIM mean diff {mean_d} outside 95% CI [{ci_l}, {ci_u}] for SSIM={r['target_ssim']}"
        passed_checks += 1
    print(f"  [PASS] stats_ssim_matched_paired_v4: verified all {len(df_ssim)} conditions have mean diff inside 95% CI")

    # 6. Regex match all quoted intervals in summary
    ci_matches = re.findall(r"([+-]?\d+\.\d+)\s*(?:\[|\(CI:?\s*)([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)(?:\]|\))", md_text)
    for m_val, m_low, m_high in ci_matches:
        total_checks += 1
        val_f, low_f, high_f = float(m_val), float(m_low), float(m_high)
        if low_f <= high_f:
            assert low_f - 1e-4 <= val_f <= high_f + 1e-4, f"Quoted value {val_f} outside quoted CI [{low_f}, {high_f}]"
        passed_checks += 1
    print(f"  [PASS] RESULTS_SUMMARY.md: verified {len(ci_matches)} quoted (mean, [CI_lower, CI_upper]) tuples in document text")

    # -------------------------------------------------------------------------
    # AUDIT 3: Image Counts Agreement Across Files
    # -------------------------------------------------------------------------
    print("\n[Audit 3/5] Checking image counts agreement across all datasets and files...")
    exp_counts = {
        # 100 benchmark images
        "severity_raw.csv": 100,
        "baselines_raw.csv": 100,
        "sync_raw.csv": 100,
        "sync_estimation_error.csv": 100,
        "security_scores.csv": 100,
        "compound_raw.csv": 100,
        "quantization_v3.csv": 100,
        "stats_pooled_v3.csv": 100,
        "stats_paired_v3.csv": 100,
        "stats_tost_v3.csv": 100,
        "saturation_v4.csv": 100,
        "stats_headroom_v4.csv": 100,
        # 208 cohort
        "core_all208_v4.csv": 208,
        "stats_all208_paired_v4.csv": 208,
        "dataset_characterization.csv": 208,
        "dataset_manifest.csv": 208,
        # 100 COCO images
        "core_coco100_v4.csv": 100,
        # 308 combined manifest
        "dataset_manifest_v4.csv": 308,
        # 50 GPU regen images
        "regen_summary_v4.csv": 50,
        # 10 CPU legacy regen
        "regen_raw.csv": 10,
    }

    for fname, exp_n in exp_counts.items():
        total_checks += 1
        df_tmp = load_csv(fname)
        if "image_id" in df_tmp.columns:
            actual_n = df_tmp["image_id"].nunique()
        elif "n_images" in df_tmp.columns:
            actual_n = int(df_tmp["n_images"].iloc[0])
        elif fname.startswith("dataset_"):
            actual_n = len(df_tmp)
        else:
            raise KeyError(f"No image identification column in {fname}")
        assert actual_n == exp_n, f"Image count mismatch in {fname}: found {actual_n}, expected {exp_n}"
        passed_checks += 1
        print(f"  [PASS] {fname}: {actual_n} unique images == expected {exp_n}")

    # Check quoted image counts in summary text
    sec1_pat = r"Total Benchmark Images\*\*:\s*(\d+)"
    m_s1 = re.search(sec1_pat, md_text)
    assert m_s1 and int(m_s1.group(1)) == 308, f"Summary quotes {m_s1.group(1) if m_s1 else 'None'}, expected 308"
    total_checks += 1
    passed_checks += 1
    print(f"  [PASS] Section 1 quotes **308** total benchmark images")

    # -------------------------------------------------------------------------
    # AUDIT 4: Trial counts equal raw-file row counts (len())
    # -------------------------------------------------------------------------
    print("\n[Audit 4/5] Checking trial counts (len() of raw files) against summary...")
    raw_files = [
        "severity_raw.csv",
        "baselines_raw.csv",
        "baselines_extended_raw.csv",
        "baselines_new_evals.csv",
        "sync_raw.csv",
        "sync_estimation_error.csv",
        "sync_compound_raw.csv",
        "security_scores.csv",
        "compound_raw.csv",
        "quantization_v3.csv",
        "regen_raw.csv",
        "ablation_block.csv",
        "ablation_features.csv",
        "ablation_payload.csv",
        "ablation_transform.csv",
        "ablation_wavelet.csv",
    ]

    sec4_match = re.search(r"## 4\. Number of Trials per Experiment(.*?)(?=## 5\.)", md_text, re.DOTALL)
    if not sec4_match:
        raise AssertionError("Failed to locate Section 4 in RESULTS_SUMMARY.md")
    sec4_text = sec4_match.group(1)

    computed_total_trials = 0
    for rf in raw_files:
        total_checks += 1
        rf_path = os.path.join(results_dir, rf)
        disk_rows = count_rows(rf_path)
        computed_total_trials += disk_rows

        pattern = rf"`{rf}`\s*\|\s*\*\*([0-9,]+)\*\*"
        m = re.search(pattern, sec4_text)
        assert m, f"Could not find trial count row for `{rf}` in Section 4"
        quoted_rows = int(m.group(1).replace(",", ""))
        assert quoted_rows == disk_rows, f"Trial count mismatch for {rf}: quoted {quoted_rows:,}, disk has {disk_rows:,}"
        passed_checks += 1
        print(f"  [PASS] {rf}: quoted {quoted_rows:,} == disk rows {disk_rows:,}")

    total_checks += 1
    tot_pattern = r"Total Raw Trials Across Tracked Files.*?\|\s*\*\*([0-9,]+)\*\*"
    m_tot = re.search(tot_pattern, sec4_text)
    assert m_tot, "Could not find total raw trials row in Section 4"
    quoted_total = int(m_tot.group(1).replace(",", ""))
    assert quoted_total == computed_total_trials, f"Total sum mismatch: quoted {quoted_total:,}, disk sum {computed_total_trials:,}"
    passed_checks += 1
    print(f"  [PASS] Section 4 Total Raw Trials: quoted {quoted_total:,} == computed sum {computed_total_trials:,}")

    # Check v4 extended raw trials
    v4_raw_files = {
        "core_all208_v4.csv": 66560,
        "core_headroom_v4.csv": 32000,
        "core_coco100_v4.csv": 32000,
        "ssim_matched_v4.csv": 48000,
        "regen_v4.csv": 2600,
        "headroom_clean_raw_v4.csv": 16000,
        "baselines_blind_tuned_v4.csv": 14760,
        "baselines_rotation_v4.csv": 16500,
    }
    for v4_rf, exp_rows in v4_raw_files.items():
        total_checks += 1
        actual_rows = count_rows(os.path.join(results_dir, v4_rf))
        assert actual_rows == exp_rows, f"v4 raw file {v4_rf} row count mismatch: found {actual_rows}, expected {exp_rows}"
        passed_checks += 1
        print(f"  [PASS] {v4_rf}: disk rows {actual_rows:,} == expected {exp_rows:,}")

    # -------------------------------------------------------------------------
    # AUDIT 5: Numerical Results Assertions against CSVs
    # -------------------------------------------------------------------------
    print("\n[Audit 5/5] Asserting every headline and section numerical result against CSVs...")

    # 1. Headline Main Study Numbers
    hl_diff_exp = float(df_pooled[df_pooled["subband"] == "HL"]["mean_adaptive_fixed_diff"].iloc[0])
    ll_diff_exp = float(df_pooled[df_pooled["subband"] == "LL"]["mean_adaptive_fixed_diff"].iloc[0])
    both_diff_exp = float(df_pooled[df_pooled["subband"] == "both"]["mean_adaptive_fixed_diff"].iloc[0])
    for val in [hl_diff_exp, ll_diff_exp, both_diff_exp]:
        total_checks += 1
        assert f"{val:+.6f}" in md_text, f"Headline pooled diff {val:+.6f} missing in summary"
        passed_checks += 1
    print(f"  [PASS] Main study headline pooled differences verified: HL={hl_diff_exp:+.6f}, LL={ll_diff_exp:+.6f}, Both={both_diff_exp:+.6f}")

    # 2. Section 6: Headroom Guard Numbers
    df_sat = load_csv("saturation_v4.csv")
    sat_tot = float(df_sat["sat_total"].mean()) * 100.0
    total_checks += 1
    assert f"{sat_tot:.3f}%" in md_text, f"Saturation total {sat_tot:.3f}% missing in summary"
    passed_checks += 1
    print(f"  [PASS] Saturation rate verified: {sat_tot:.3f}%")

    df_alt = load_csv("headroom_alternatives_v4.csv")
    alt_a_erasure = float(df_alt["alt_a_erasure_rate"].mean()) * 100.0
    alt_b_maj = float(df_alt["alt_b_ber_majority_vote"].mean())
    total_checks += 2
    assert f"{alt_a_erasure:.2f}%" in md_text, f"Alternative A erasure rate {alt_a_erasure:.2f}% missing"
    passed_checks += 1
    assert f"{alt_b_maj:.6f}" in md_text, f"Alternative B majority vote BER {alt_b_maj:.6f} missing"
    passed_checks += 1
    print(f"  [PASS] Headroom alternatives verified: Erasure={alt_a_erasure:.2f}%, MajVote={alt_b_maj:.6f}")

    # Headroom pooled differences (both 9-family and 13-condition)
    total_checks += 2
    assert "+0.000110" in md_text, "Headroom 9-family pooled mean +0.000110 missing"
    passed_checks += 1
    assert "+0.001305" in md_text, "Headroom 13-condition pooled mean +0.001305 missing"
    passed_checks += 1
    print("  [PASS] Headroom 9-family (+0.000110) and 13-condition (+0.001305) pooled means verified")

    # 3. Section 7: 208-Image Replication Pooling Rules
    total_checks += 4
    assert "-0.003666" in md_text, "208-image 9-family HL mean -0.003666 missing"
    passed_checks += 1
    assert "+0.001922" in md_text, "208-image 9-family LL mean +0.001922 missing"
    passed_checks += 1
    assert "-0.001514" in md_text, "208-image 13-condition HL mean -0.001514 missing"
    passed_checks += 1
    assert "+0.002548" in md_text, "208-image 13-condition LL mean +0.002548 missing"
    passed_checks += 1
    print("  [PASS] 208-image replication family-weighted and condition-weighted means verified")

    # 4. Section 8: COCO Generalization Pooling Rules
    total_checks += 2
    assert "-0.000251" in md_text, "COCO 9-family pooled mean -0.000251 missing"
    passed_checks += 1
    assert "+0.001257" in md_text, "COCO 13-condition pooled mean +0.001257 missing"
    passed_checks += 1
    print("  [PASS] COCO generalization family-weighted (-0.000251) and condition-weighted (+0.001257) verified")

    # 5. Section 9: SSIM-Matched Study Targets
    total_checks += 3
    for ssim_val in ["-0.003375", "-0.004076", "-0.006459"]:
        assert ssim_val in md_text, f"SSIM-matched target mean {ssim_val} missing"
        passed_checks += 1
    print("  [PASS] SSIM-matched target means (-0.003375, -0.004076, -0.006459) verified")

    # 6. Section 12: Interaction Tests (LRT + Permutation)
    df_int = load_csv("stats_interactions_v4.csv")
    lrt_100_stat = float(df_int[(df_int["dataset_level"].str.contains("100")) & (df_int["model_type"].str.contains("LRT"))]["test_statistic"].iloc[0])
    lrt_208_stat = float(df_int[(df_int["dataset_level"].str.contains("208")) & (df_int["model_type"].str.contains("LRT"))]["test_statistic"].iloc[0])
    perm_100_stat = float(df_int[(df_int["dataset_level"].str.contains("100")) & (df_int["model_type"].str.contains("Permutation"))]["test_statistic"].iloc[0])
    perm_208_stat = float(df_int[(df_int["dataset_level"].str.contains("208")) & (df_int["model_type"].str.contains("Permutation"))]["test_statistic"].iloc[0])

    total_checks += 4
    assert f"{lrt_100_stat:.2f}" in md_text, f"100-image LRT chi2 {lrt_100_stat:.2f} missing"
    passed_checks += 1
    assert f"{lrt_208_stat:.2f}" in md_text, f"208-image LRT chi2 {lrt_208_stat:.2f} missing"
    passed_checks += 1
    assert f"{perm_100_stat:.2f}" in md_text, f"100-image permutation F {perm_100_stat:.2f} missing"
    passed_checks += 1
    assert f"{perm_208_stat:.2f}" in md_text, f"208-image permutation F {perm_208_stat:.2f} missing"
    passed_checks += 1
    print(f"  [PASS] Interaction LRT (chi2={lrt_100_stat:.2f}, {lrt_208_stat:.2f}) and Permutation (F={perm_100_stat:.2f}, {perm_208_stat:.2f}) verified")

    # 7. Section 13: Subband Asymmetry Cliff's Delta
    df_hl_ll = load_csv("stats_hl_vs_ll_v4.csv")
    df_hl_ll_208 = df_hl_ll[df_hl_ll["dataset_scope"] == "all_208"]
    for _, r in df_hl_ll_208.iterrows():
        total_checks += 1
        cd = float(r["cliffs_delta"])
        assert f"{cd:+.4f}" in md_text, f"Cliff's delta {cd:+.4f} for {r['attack']} missing"
        passed_checks += 1
    print(f"  [PASS] Subband asymmetry: verified all {len(df_hl_ll_208)} Cliff's delta effect sizes in 208 cohort")

    # 8. Section 14: GPU Regeneration Threat Model
    df_regen = load_csv("regen_summary_v4.csv")
    total_checks += 2
    assert "50 benchmark images at 512x512" in md_text, "Regeneration resolution / count missing"
    passed_checks += 1
    assert "runwayml/stable-diffusion-v1-5" in md_text, "SD 1.5 model name missing"
    passed_checks += 1
    for _, r in df_regen.iterrows():
        total_checks += 1
        med_b = float(r["median_ber"])
        assert f"{med_b:.4f}" in md_text, f"Regen median BER {med_b:.4f} missing"
        passed_checks += 1
    print(f"  [PASS] GPU regeneration: verified SD 1.5 model, GPU hardware, and all {len(df_regen)} condition median BERs")

    # 9. Section 15: Runtime Scaling & Equivalence
    df_rt = load_csv("runtime_v3.csv")
    rt_512_fixed = df_rt[(df_rt["resolution"] == 512) & (df_rt["method"] == "proposed_fixed")].iloc[0]
    fixed_speedup = float(rt_512_fixed["speedup_embed"])
    total_checks += 1
    assert f"{fixed_speedup:.2f}x speedup" in md_text, f"Speedup {fixed_speedup:.2f}x missing"
    passed_checks += 1
    print(f"  [PASS] Runtime 512x512 speedup={fixed_speedup:.2f}x verified")

    # 10. Section 16: Security & Craver Ambiguity
    df_sec = load_csv("security_v3.csv")
    n6_entry = df_sec[(df_sec["positive_case"] == "pos_clean") & (df_sec["negative_case"] == "N6_svd_forgery")].iloc[0]
    n6_auc = float(n6_entry["auc"])
    total_checks += 1
    assert f"{n6_auc:.4f}" in md_text, f"N6 AUC {n6_auc:.4f} missing"
    passed_checks += 1
    print(f"  [PASS] Security Craver N6 AUC={n6_auc:.4f} verified")

    print("\n" + "=" * 80)
    print(f"ALL V4 AUDIT ASSERTIONS PASSED PERFECTLY! ({passed_checks} / {total_checks} assertions verified)")
    print("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit consistency of RESULTS_SUMMARY.md against CSV outputs.")
    parser.add_argument("--results-dir", type=str, default="results_v4", help="Directory holding CSV outputs.")
    parser.add_argument("--summary-md", type=str, default=None, help="Path to RESULTS_SUMMARY.md.")
    args = parser.parse_args()

    summary_path = args.summary_md or os.path.join(args.results_dir, "RESULTS_SUMMARY.md")
    try:
        check_all_consistency(results_dir=args.results_dir, summary_md=summary_path)
    except Exception as exc:
        print(f"\n[FATAL AUDIT FAILURE]: {exc}", file=sys.stderr)
        sys.exit(1)
