"""
Experiment 2: Texture Weight Ablation Analysis (V2).

Evaluates the sensitivity and contribution of Gradient, Variance, and Entropy features.
Strictly adheres to Rules 1-6:
- Output to results_v2/
- Deterministic seeding with np.random.default_rng(seed)
- 20-column long-format schema
- Resumable execution
- Multiprocessing with --workers flag
- Image-level statistical aggregation
- Terminal summary table
"""

import os
import sys
import time
import uuid
import argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments_v2.schema import (
    ResultRow,
    init_results_csv,
    load_completed_keys,
    append_result_rows,
)
from experiments_v2.attacks import AttackSpec, get_standard_attack_specs
from experiments_v2.common import (
    TrialTask,
    _worker_execute_task,
    load_standard_test_images,
    load_standard_watermark,
    get_git_commit_hash,
)
from experiments_v2.stats import (
    aggregate_by_image,
    aggregate_across_images,
    compute_paired_image_test,
    format_summary_table,
)


def get_weight_configurations() -> List[Dict[str, Any]]:
    """Returns the standard 6 texture weighting configurations."""
    return [
        {"id": "A", "name": "Proposed (40/40/20)", "w_grad": 0.40, "w_var": 0.40, "w_ent": 0.20},
        {"id": "B", "name": "Equal (33/33/34)", "w_grad": 0.3333, "w_var": 0.3333, "w_ent": 0.3334},
        {"id": "C", "name": "Gradient-Dominant (60/20/20)", "w_grad": 0.60, "w_var": 0.20, "w_ent": 0.20},
        {"id": "D", "name": "Variance-Dominant (20/60/20)", "w_grad": 0.20, "w_var": 0.60, "w_ent": 0.20},
        {"id": "E", "name": "Entropy-Dominant (20/20/60)", "w_grad": 0.20, "w_var": 0.20, "w_ent": 0.60},
        {"id": "F", "name": "Grad+Var Only (50/50/0)", "w_grad": 0.50, "w_var": 0.50, "w_ent": 0.00},
    ]


def run_weight_ablation(
    images_dir: str = "data/internet_images",
    output_dir: str = "results_v2/weight_ablation",
    base_alpha: float = 30.0,
    subband: str = "HL",
    seeds: Optional[List[int]] = None,
    wavelet: str = "haar",
    block_size: int = 8,
    workers: int = 4,
    batch_size: int = 20,
    attacks: Optional[List[AttackSpec]] = None,
    configs: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Executes the resumable Weight Ablation experiment across the 6 weight configurations.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "weight_ablation_raw.csv")
    init_results_csv(csv_path)

    weight_configs = configs if configs is not None else get_weight_configurations()
    s_list = seeds if seeds is not None else [1, 2, 3, 4, 5]
    att_specs = attacks if attacks is not None else get_standard_attack_specs()

    images = load_standard_test_images(images_dir)
    payload_id, watermark = load_standard_watermark(images_dir)
    git_hash = get_git_commit_hash()
    run_id = f"weight_ablation_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    completed_keys = load_completed_keys(csv_path)
    print(f"[*] Checking existing results: found {len(completed_keys)} completed rows in '{csv_path}'")

    pending_tasks: List[Tuple[TrialTask, np.ndarray, np.ndarray]] = []

    for w_cfg in weight_configs:
        method_name = f"Adaptive_{w_cfg['id']}"
        for img_id, cover in images.items():
            for spec in att_specs:
                trial_seeds = s_list if spec.is_stochastic else [42]
                for s in trial_seeds:
                    task = TrialTask(
                        run_id=run_id,
                        experiment="weight_ablation",
                        image_id=img_id,
                        payload_id=payload_id,
                        method=method_name,
                        subband=subband,
                        wavelet=wavelet,
                        block_size=block_size,
                        alpha_nominal=base_alpha,
                        attack_spec=spec,
                        seed=s,
                        config={
                            "weight_id": w_cfg["id"],
                            "weight_name": w_cfg["name"],
                            "weight_gradient": w_cfg["w_grad"],
                            "weight_variance": w_cfg["w_var"],
                            "weight_entropy": w_cfg["w_ent"],
                            "base_alpha": base_alpha,
                        },
                        git_commit=git_hash,
                    )
                    if task.key() not in completed_keys:
                        pending_tasks.append((task, cover, watermark))

    total_tasks = len(pending_tasks)
    print(f"[*] Total planned trials: {len(completed_keys) + total_tasks} | Already done: {len(completed_keys)} | To run: {total_tasks}")

    if total_tasks > 0:
        actual_workers = max(1, min(workers, os.cpu_count() or 4))
        print(f"[*] Starting execution across {actual_workers} workers...")
        buffer: List[ResultRow] = []
        done_count = 0

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_to_task = {executor.submit(_worker_execute_task, item): item[0] for item in pending_tasks}
            for future in as_completed(future_to_task):
                row = future.result()
                buffer.append(row)
                done_count += 1

                if len(buffer) >= batch_size:
                    append_result_rows(csv_path, buffer)
                    buffer.clear()
                    print(f"  -> Progress: {done_count}/{total_tasks} ({done_count / total_tasks * 100:.1f}%)")

            if buffer:
                append_result_rows(csv_path, buffer)
                buffer.clear()

        print(f"[*] All {total_tasks} trials completed and saved.")

    # Image-Level Statistical Aggregation (Rule 4)
    df_raw = pd.read_csv(csv_path)
    df_image = aggregate_by_image(df_raw)
    df_summary = aggregate_across_images(df_image)

    by_img_path = os.path.join(output_dir, "weight_ablation_by_image.csv")
    summary_path = os.path.join(output_dir, "weight_ablation_summary.csv")
    df_image.to_csv(by_img_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    # Paired Image-Level Tests against Baseline Config A
    paired_tests = []
    baseline_method = "Adaptive_A"
    for w_cfg in weight_configs:
        comp_method = f"Adaptive_{w_cfg['id']}"
        if comp_method == baseline_method:
            continue
        test_res = compute_paired_image_test(
            df_raw=df_raw,
            condition_a={"method": baseline_method, "attack": "no_attack"},
            condition_b={"method": comp_method, "attack": "no_attack"},
            metric="psnr_clean",
        )
        paired_tests.append({
            "comparison": f"{baseline_method} vs {comp_method}",
            "metric": "psnr_clean",
            "n_images": test_res["n_images"],
            "mean_diff": test_res["mean_diff"],
            "t_stat": test_res["t_stat"],
            "p_ttest": test_res["p_ttest"],
            "wilcoxon_stat": test_res["wilcoxon_stat"],
            "p_wilcoxon": test_res["p_wilcoxon"],
        })

    df_paired = pd.DataFrame(paired_tests)
    paired_path = os.path.join(output_dir, "weight_ablation_paired_tests.csv")
    df_paired.to_csv(paired_path, index=False)

    # Print short summary at the end (Rule 6)
    print("\n" + "=" * 80)
    print(" EXPERIMENT 2 (WEIGHT ABLATION) SUMMARY (EXPERIMENTAL UNIT: IMAGE)")
    print("=" * 80)
    print(format_summary_table(df_summary.head(18)))
    if len(df_paired) > 0:
        print("\n--- Paired Image-Level Tests vs Proposed Config A (No Attack PSNR) ---")
        print(df_paired.to_string(index=False))
    print("=" * 80 + "\n")

    return df_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Weight Ablation (V2) with Resumability and Multiprocessing.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument("--base-alpha", type=float, default=30.0, help="Base nominal embedding alpha.")
    parser.add_argument("--subband", type=str, default="HL", help="Wavelet subband (HL or LL).")
    parser.add_argument("--images-dir", type=str, default="data/internet_images", help="Path to test images.")
    parser.add_argument("--output-dir", type=str, default="results_v2/weight_ablation", help="Output directory.")
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke test with reduced grid.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        run_weight_ablation(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            base_alpha=args.base_alpha,
            subband=args.subband,
            seeds=[1, 2],
            workers=args.workers,
            attacks=[
                AttackSpec(attack="no_attack", severity="none", is_stochastic=False),
                AttackSpec(attack="jpeg", severity="Q=70", is_stochastic=False),
            ],
            configs=get_weight_configurations()[:2],
        )
    else:
        run_weight_ablation(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            base_alpha=args.base_alpha,
            subband=args.subband,
            workers=args.workers,
        )
