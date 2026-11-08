"""
Core Study Pipeline Orchestrator (v3)

Runs:
1. Matched-PSNR study (128,000 rows) -> results_v3/matched_psnr_raw.csv
2. Paired differences (104 rows) -> results_v3/paired_diffs.csv
3. Core alpha sweep (160,000 rows) -> results_v3/core_raw.csv
4. Pareto frontier points (40 rows) -> results_v3/pareto_points.csv
5. Full factor verification -> results_v3/RAW_FILE_CHECK.md
6. Manifest refresh -> results_v3/MANIFEST.json
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.abspath("."))

from run_missing_experiments import (
    run_generate_matched_psnr,
    run_generate_paired_diffs,
    run_generate_core_raw,
    run_generate_pareto_points,
    verify_and_report_files,
    compute_sha256,
)


def refresh_manifest(results_dir: str = "results_v3"):
    print(f"\n[*] Refreshing {results_dir}/MANIFEST.json...")
    manifest_path = os.path.join(results_dir, "MANIFEST.json")
    entries = {}
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f == "MANIFEST.json":
                continue
            fpath = os.path.join(root, f)
            rel_path = os.path.relpath(fpath, results_dir).replace("\\", "/")
            size_b = os.path.getsize(fpath)
            sha = compute_sha256(fpath)
            row_count = None
            if f.endswith(".csv"):
                try:
                    with open(fpath, "r", encoding="utf-8") as fp:
                        row_count = max(0, sum(1 for _ in fp) - 1)
                except Exception:
                    pass
            entries[rel_path] = {
                "sha256": sha,
                "size_bytes": size_b,
                "size_mb": round(size_b / 1024 / 1024, 3),
                "rows": row_count,
            }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(entries, fp, indent=2, sort_keys=True)
    print(f"[OK] MANIFEST.json updated with {len(entries)} tracked files.")


def main():
    workers = 14
    t_start = time.perf_counter()

    print("================================================================================")
    print(" CORE STUDY REGENERATION PIPELINE (RESULTS_V3)")
    print(f" Workers: {workers} | Target PSNRs: 40, 44, 46, 48 dB | Images: 100")
    print("================================================================================")

    # 1. Matched-PSNR Benchmark
    t0 = time.perf_counter()
    print("\n[STEP 1/4] Starting Matched-PSNR Study (Target: 128,000 rows)...")
    run_generate_matched_psnr(output_csv="results_v3/matched_psnr_raw.csv", workers=workers)
    print(f"  -> Matched-PSNR raw generation done in {time.perf_counter() - t0:.1f} s")

    # 2. Paired Differences
    t0 = time.perf_counter()
    print("\n[STEP 2/4] Computing Paired Differences (Target: 104 rows)...")
    run_generate_paired_diffs(
        matched_csv="results_v3/matched_psnr_raw.csv",
        output_csv="results_v3/paired_diffs.csv",
    )
    print(f"  -> Paired differences computed in {time.perf_counter() - t0:.1f} s")

    # 3. Core Alpha Sweep
    t0 = time.perf_counter()
    print("\n[STEP 3/4] Starting Core Alpha Sweep Study (Target: 160,000 rows)...")
    run_generate_core_raw(output_csv="results_v3/core_raw.csv", workers=workers)
    print(f"  -> Core alpha sweep raw generation done in {time.perf_counter() - t0:.1f} s")

    # 4. Pareto Points
    t0 = time.perf_counter()
    print("\n[STEP 4/4] Computing Pareto Frontier Points (Target: 40 rows)...")
    run_generate_pareto_points(
        core_csv="results_v3/core_raw.csv",
        output_csv="results_v3/pareto_points.csv",
    )
    print(f"  -> Pareto points computed in {time.perf_counter() - t0:.1f} s")

    # Final Verification
    print("\n[*] Running Factor Verification on All Benchmark & Ablation Files...")
    verify_and_report_files(results_dir="results_v3")

    # Refresh Manifest
    refresh_manifest(results_dir="results_v3")

    elapsed_total = time.perf_counter() - t_start
    print(f"\n[OK] ALL CORE STUDY FILES REGENERATED AND VERIFIED IN {elapsed_total/60:.2f} MINUTES!")


if __name__ == "__main__":
    main()

