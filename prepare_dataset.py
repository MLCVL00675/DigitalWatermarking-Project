"""
Dataset Preparation CLI Script for experiments_v2.

Executes:
1. Ingestion of >=200 images across BSDS500, Kodak, USC-SIPI, and Legacy5.
2. Deterministic preprocessing: grayscale -> center-crop to shorter side -> resize to 512x512.
3. Writing results_v2/dataset_manifest.csv.
4. Feature extraction and texture stratification (terciles of T) -> results_v2/dataset_characterization.csv.
5. Overview plotting -> results_v2/figs/dataset_overview.pdf.
6. Summary paper table printed to terminal.
"""

import os
import sys
import argparse
import time

from experiments_v2.dataset import prepare_dataset_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare, preprocess, and characterize 512x512 benchmark dataset.")
    parser.add_argument("--output-dir", type=str, default="data/dataset_v2", help="Target directory for processed PNGs.")
    parser.add_argument("--manifest-csv", type=str, default="results_v2/dataset_manifest.csv", help="Manifest CSV path.")
    parser.add_argument("--characterization-csv", type=str, default="results_v2/dataset_characterization.csv", help="Characterization CSV path.")
    parser.add_argument("--plot-pdf", type=str, default="results_v2/figs/dataset_overview.pdf", help="Dataset overview figure PDF.")
    parser.add_argument("--legacy-dir", type=str, default="data/internet_images", help="Directory with original 5 benchmark images.")
    parser.add_argument("--num-bsds", type=int, default=140, help="Number of BSDS500 test images to download (total target >= 200).")
    parser.add_argument("--workers", type=int, default=6, help="Worker threads for parallel download.")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    print("=" * 80)
    print(" PREPARING PUBLIC BENCHMARK DATASET (>= 200 IMAGES, 512x512 PNG)")
    print(f" Output Directory: {args.output_dir}")
    print(f" Manifest: {args.manifest_csv}")
    print(f" Characterization: {args.characterization_csv}")
    print(f" Overview Plot: {args.plot_pdf}")
    print("=" * 80)

    try:
        df_manifest, df_char = prepare_dataset_pipeline(
            output_dir=args.output_dir,
            manifest_csv=args.manifest_csv,
            characterization_csv=args.characterization_csv,
            plot_pdf=args.plot_pdf,
            legacy_dir=args.legacy_dir,
            num_bsds=args.num_bsds,
            workers=args.workers,
        )
        elapsed = time.time() - t_start
        print(f"[*] Dataset pipeline completed successfully in {elapsed:.1f}s.")
        print(f"[*] Processed {len(df_manifest)} images stored in '{args.output_dir}'.")
    except Exception as e:
        print(f"\n[!] ERROR occurred during dataset preparation: {e}")
        print("\nManual Fetch Instructions:")
        print("  - If Kodak download was blocked: Download kodim01.png..kodim24.png from http://r0k.us/graphics/kodak/ and place in data/kodak_raw/")
        print("  - If USC-SIPI was blocked: Download misc.zip from https://sipi.usc.edu/database/misc.zip and place in data/usc_raw/")
        print("  - If BSDS500 was blocked: Download BSR_bsds500.tgz from http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz")
        sys.exit(1)


if __name__ == "__main__":
    main()
