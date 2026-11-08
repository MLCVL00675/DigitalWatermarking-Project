"""
Master Runner for Experiments V2.
Orchestrates Experiment 1 (Alpha Sweep) and Experiment 2 (Weight Ablation) sequentially
with resumability, multiprocessing, and rigorous image-level statistics.
"""

import os
import sys
import time
import argparse

from experiments_v2.alpha_sweep import run_alpha_sweep
from experiments_v2.weight_ablation import run_weight_ablation


def main():
    parser = argparse.ArgumentParser(description="Master Runner for Experiments V2.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes.")
    parser.add_argument("--images-dir", type=str, default="data/internet_images", help="Image directory.")
    parser.add_argument("--results-root", type=str, default="results_v2", help="Root results directory.")
    parser.add_argument("--quick", action="store_true", help="Run quick test sweep.")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 80)
    print(" INITIATING EXPERIMENTS V2 MASTER SUITE")
    print(f" Workers: {args.workers} | Results Root: {args.results_root} | Quick Mode: {args.quick}")
    print("=" * 80)

    # 1. Alpha Sweep
    print("\n>>> [STEP 1/2] Running Alpha Sweep & Pareto Trade-Off Analysis...")
    alpha_out = os.path.join(args.results_root, "alpha_sweep")
    if args.quick:
        run_alpha_sweep(
            images_dir=args.images_dir,
            output_dir=alpha_out,
            fixed_alphas=[30.0],
            adaptive_alphas=[30.0],
            subbands=["HL"],
            seeds=[1, 2],
            workers=args.workers,
        )
    else:
        run_alpha_sweep(
            images_dir=args.images_dir,
            output_dir=alpha_out,
            workers=args.workers,
        )

    # 2. Weight Ablation
    print("\n>>> [STEP 2/2] Running Texture Weight Ablation...")
    weight_out = os.path.join(args.results_root, "weight_ablation")
    if args.quick:
        run_weight_ablation(
            images_dir=args.images_dir,
            output_dir=weight_out,
            base_alpha=30.0,
            subband="HL",
            seeds=[1, 2],
            workers=args.workers,
        )
    else:
        run_weight_ablation(
            images_dir=args.images_dir,
            output_dir=weight_out,
            base_alpha=30.0,
            subband="HL",
            workers=args.workers,
        )

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f" EXPERIMENTS V2 MASTER SUITE FINISHED IN {elapsed:.1f}s")
    print(f" Outputs written to: {os.path.abspath(args.results_root)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
