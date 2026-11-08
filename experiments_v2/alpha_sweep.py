"""
Experiment 1: Alpha Sweep & Pareto Trade-Off Analysis (V2).

Compares Fixed Embedding vs. Texture-Adaptive Embedding across matched operating points.
Strictly adheres to Rules 1-6:
- Output to results_v2/
- Deterministic seeding with np.random.default_rng(seed)
- 20-column long-format schema
- Resumable trial execution
- Multiprocessing with --workers flag
- Image-level statistical aggregation
- Terminal summary table
"""

import os
import sys
import time
import uuid
import argparse
from typing import List, Dict, Any, Optional
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
    format_summary_table,
)


def run_alpha_sweep(
    images_dir: str = "data/internet_images",
    output_dir: str = "results_v2/alpha_sweep",
    fixed_alphas: Optional[List[float]] = None,
    adaptive_alphas: Optional[List[float]] = None,
    subbands: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    wavelet: str = "haar",
    block_size: int = 8,
    workers: int = 4,
    batch_size: int = 20,
    attacks: Optional[List[AttackSpec]] = None,
) -> pd.DataFrame:
    """
    Executes the resumable Alpha Sweep benchmark across methods, subbands, and attacks.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "alpha_sweep_raw.csv")
    init_results_csv(csv_path)

    f_alphas = fixed_alphas if fixed_alphas is not None else [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
    a_alphas = adaptive_alphas if adaptive_alphas is not None else [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0]
    s_bands = subbands if subbands is not None else ["HL", "LL"]
    s_list = seeds if seeds is not None else [1, 2, 3, 4, 5]
    att_specs = attacks if attacks is not None else get_standard_attack_specs()

    images = load_standard_test_images(images_dir)
    payload_id, watermark = load_standard_watermark(images_dir)
    git_hash = get_git_commit_hash()
    run_id = f"alpha_sweep_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    completed_keys = load_completed_keys(csv_path)
    print(f"[*] Checking existing results: found {len(completed_keys)} completed rows in '{csv_path}'")

    # Build queue of pending tasks
    pending_tasks: List[Tuple[TrialTask, np.ndarray, np.ndarray]] = []

    for sb in s_bands:
        # Fixed conditions
        for alpha in f_alphas:
            for img_id, cover in images.items():
                for spec in att_specs:
                    trial_seeds = s_list if spec.is_stochastic else [42]
                    for s in trial_seeds:
                        task = TrialTask(
                            run_id=run_id,
                            experiment="alpha_sweep",
                            image_id=img_id,
                            payload_id=payload_id,
                            method="Fixed",
                            subband=sb,
                            wavelet=wavelet,
                            block_size=block_size,
                            alpha_nominal=alpha,
                            attack_spec=spec,
                            seed=s,
                            config={"type": "fixed", "alpha": alpha},
                            git_commit=git_hash,
                        )
                        if task.key() not in completed_keys:
                            pending_tasks.append((task, cover, watermark))

        # Adaptive conditions
        for alpha in a_alphas:
            for img_id, cover in images.items():
                for spec in att_specs:
                    trial_seeds = s_list if spec.is_stochastic else [42]
                    for s in trial_seeds:
                        task = TrialTask(
                            run_id=run_id,
                            experiment="alpha_sweep",
                            image_id=img_id,
                            payload_id=payload_id,
                            method="Adaptive",
                            subband=sb,
                            wavelet=wavelet,
                            block_size=block_size,
                            alpha_nominal=alpha,
                            attack_spec=spec,
                            seed=s,
                            config={
                                "type": "adaptive",
                                "base_alpha": alpha,
                                "weight_gradient": 0.40,
                                "weight_variance": 0.40,
                                "weight_entropy": 0.20,
                            },
                            git_commit=git_hash,
                        )
                        if task.key() not in completed_keys:
                            pending_tasks.append((task, cover, watermark))

    total_tasks = len(pending_tasks)
    print(f"[*] Total planned trials: {len(completed_keys) + total_tasks} | Already done: {len(completed_keys)} | To run: {total_tasks}")

    # Multiprocessing execution
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

    # Generate Statistical Aggregations (Rule 4: IMAGE is the experimental unit)
    df_raw = pd.read_csv(csv_path)
    df_image = aggregate_by_image(df_raw)
    df_summary = aggregate_across_images(df_image)

    by_img_path = os.path.join(output_dir, "alpha_sweep_by_image.csv")
    summary_path = os.path.join(output_dir, "alpha_sweep_summary.csv")
    df_image.to_csv(by_img_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    # Print short summary at the end (Rule 6)
    print("\n" + "=" * 80)
    print(" EXPERIMENT 1 (ALPHA SWEEP) SUMMARY (EXPERIMENTAL UNIT: IMAGE)")
    print("=" * 80)
    # Highlight No Attack and representative attack for quick review
    quick_view = df_summary[df_summary["attack"].isin(["no_attack", "jpeg", "gaussian_noise"])]
    if len(quick_view) > 0:
        print(format_summary_table(quick_view.head(20)))
    else:
        print(format_summary_table(df_summary.head(20)))
    print("=" * 80 + "\n")

    return df_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Alpha Sweep (V2) with Resumability and Multiprocessing.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument("--images-dir", type=str, default="data/internet_images", help="Path to test images.")
    parser.add_argument("--output-dir", type=str, default="results_v2/alpha_sweep", help="Directory for output results.")
    parser.add_argument("--quick", action="store_true", help="Run a quick smoke test with reduced grid.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        run_alpha_sweep(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            fixed_alphas=[30.0],
            adaptive_alphas=[30.0],
            subbands=["HL"],
            seeds=[1, 2],
            workers=args.workers,
            attacks=[
                AttackSpec(attack="no_attack", severity="none", is_stochastic=False),
                AttackSpec(attack="jpeg", severity="Q=70", is_stochastic=False),
                AttackSpec(attack="gaussian_noise", severity="var=0.01", is_stochastic=True),
            ],
        )
    else:
        run_alpha_sweep(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            workers=args.workers,
        )
