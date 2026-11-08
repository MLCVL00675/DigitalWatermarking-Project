"""
Ablation Experiment: Controlled Evaluation of Fixed vs. Texture-Adaptive Embedding
in Cascaded DWT-DCT-SVD Digital Image Watermarking.

Configurations:
  Configuration A (Fixed): DWT -> DCT -> SVD -> Constant embedding strength (alpha = 30.0 for every block)
  Configuration B (Adaptive): DWT -> DCT -> SVD -> Texture analysis -> Adaptive embedding strength (alpha_k from texture score)

Fairness Controls:
  - Identical cover images (Lena, Baboon, Fruits, Building, Butterfly; 512x512 grayscale)
  - Identical watermark payload (32x32 binary emblem, 1024 bits)
  - Identical DWT (Haar, 1-level 2D DWT)
  - Evaluated separately on HL and LL subbands
  - Identical 8x8 block DCT and SVD decomposition
  - Identical extraction logic and decision threshold (0.5)
  - Identical attack parameters & synchronized random seeds for stochastic attacks
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from watermark.core import CascadedDwtDctSvdWatermarker, WatermarkResult, ExtractionResult
from watermark.masking import DeterministicTextureMasker
from watermark.attacks import AttackSuite
from watermark.metrics import compute_psnr, compute_ssim, compute_nc, compute_ber, compute_bcr, compute_mse


class ConstantAlphaMasker(DeterministicTextureMasker):
    """
    Fixed Alpha Masker for Configuration A.
    Returns a strictly uniform alpha matrix of value `base_alpha` across all blocks.
    """
    def generate_block_alpha_matrix(
        self,
        image: np.ndarray,
        target_subband_shape: Tuple[int, int],
        block_size: int = 8,
        base_alpha: float = 30.0,
    ) -> np.ndarray:
        sub_h, sub_w = target_subband_shape
        num_blocks_h = sub_h // block_size
        num_blocks_w = sub_w // block_size
        alpha_val = float(base_alpha if base_alpha is not None else self.base_alpha)
        return np.full((num_blocks_h, num_blocks_w), fill_value=alpha_val, dtype=np.float64)


def load_cover_image(path: str, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Loads grayscale cover image resized to target dimensions."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cover image not found: {path}")
    if img.shape != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float64)


def load_watermark_image(path: str, target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Loads binary watermark logo resized to target dimensions."""
    wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Watermark not found: {path}")
    wm = cv2.resize(wm, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    thresh = (np.max(wm) + np.min(wm)) / 2.0
    return (wm >= thresh).astype(np.uint8)


def get_standard_attacks() -> Dict[str, Any]:
    """Returns all 14 standard benchmark attacks."""
    return {
        "No Attack": {"fn": AttackSuite.no_attack, "is_random": False, "deskew": None},
        "JPEG Compression (Q=90)": {"fn": lambda img: AttackSuite.jpeg_compression(img, quality=90), "is_random": False, "deskew": None},
        "JPEG Compression (Q=70)": {"fn": lambda img: AttackSuite.jpeg_compression(img, quality=70), "is_random": False, "deskew": None},
        "JPEG Compression (Q=50)": {"fn": lambda img: AttackSuite.jpeg_compression(img, quality=50), "is_random": False, "deskew": None},
        "Gaussian Noise (var=0.01)": {"fn": lambda img: AttackSuite.gaussian_noise(img, variance=0.01), "is_random": True, "deskew": None},
        "Speckle Noise (var=0.01)": {"fn": lambda img: AttackSuite.speckle_noise(img, variance=0.01), "is_random": True, "deskew": None},
        "Salt & Pepper Noise (1%)": {"fn": lambda img: AttackSuite.salt_and_pepper_noise(img, amount=0.01), "is_random": True, "deskew": None},
        "Median Filter (3x3)": {"fn": lambda img: AttackSuite.median_filter(img, kernel_size=3), "is_random": False, "deskew": None},
        "Gaussian Blur (3x3, s=1.0)": {"fn": lambda img: AttackSuite.gaussian_blur(img, kernel_size=3, sigma=1.0), "is_random": False, "deskew": None},
        "Border Cropping (10%)": {"fn": lambda img: AttackSuite.cropping(img, crop_ratio=0.10, fill_mode="black"), "is_random": False, "deskew": None},
        "Rotation (2 deg) [Unsync]": {"fn": lambda img: AttackSuite.rotation(img, angle_deg=2.0), "is_random": False, "deskew": None},
        "Rotation (2 deg) [Sync Deskew]": {"fn": lambda img: AttackSuite.rotation(img, angle_deg=2.0), "is_random": False, "deskew": 2.0},
        "Rotation (5 deg) [Sync Deskew]": {"fn": lambda img: AttackSuite.rotation(img, angle_deg=5.0), "is_random": False, "deskew": 5.0},
        "Contrast Adjust (x1.15)": {"fn": lambda img: AttackSuite.contrast_adjustment(img, alpha=1.15, beta=5.0), "is_random": False, "deskew": None},
    }


def run_controlled_ablation(
    images_dir: str = "data/internet_images",
    output_dir: str = "results/ablation",
    base_alpha: float = 30.0,
    wavelet: str = "haar",
    block_size: int = 8,
):
    """Executes the full controlled ablation experiment."""
    os.makedirs(output_dir, exist_ok=True)

    test_images = [
        {"name": "Lena", "file": "lena.png"},
        {"name": "Baboon", "file": "baboon.jpg"},
        {"name": "Fruits", "file": "fruits.jpg"},
        {"name": "Building", "file": "building.jpg"},
        {"name": "Butterfly", "file": "butterfly.jpg"},
    ]

    watermark_path = os.path.join(images_dir, "watermark_logo.png")
    watermark = load_watermark_image(watermark_path, target_size=(32, 32))
    print(f"[*] Watermark loaded: 32x32 ({watermark.size} bits). Positive bits: {np.sum(watermark == 1)}")

    # Initialize Watermarkers
    fixed_masker = ConstantAlphaMasker(base_alpha=base_alpha)
    adaptive_masker = DeterministicTextureMasker(
        weight_gradient=0.40,
        weight_variance=0.40,
        weight_entropy=0.20,
        base_alpha=base_alpha,
    )

    attacks = get_standard_attacks()
    subbands = ["HL", "LL"]

    records = []
    alpha_maps_cache = {}

    for img_info in test_images:
        img_name = img_info["name"]
        img_path = os.path.join(images_dir, img_info["file"])
        cover = load_cover_image(img_path, target_size=(512, 512))

        print(f"\n=======================================================")
        print(f" Processing Image: {img_name} ({img_info['file']})")
        print(f"=======================================================")

        for sb in subbands:
            # Watermarkers for this subband
            wm_fixed = CascadedDwtDctSvdWatermarker(
                wavelet=wavelet,
                target_subband=sb,
                block_size=block_size,
                base_alpha=base_alpha,
                texture_masker=fixed_masker,
            )
            wm_adaptive = CascadedDwtDctSvdWatermarker(
                wavelet=wavelet,
                target_subband=sb,
                block_size=block_size,
                base_alpha=base_alpha,
                texture_masker=adaptive_masker,
            )

            # Embed Fixed
            res_fixed = wm_fixed.embed(cover, watermark)
            # Embed Adaptive
            res_adaptive = wm_adaptive.embed(cover, watermark)

            # Validation Checks
            fixed_alphas = res_fixed.alpha_matrix
            adaptive_alphas = res_adaptive.alpha_matrix

            assert np.allclose(fixed_alphas, base_alpha), f"Fixed alpha failed: min={fixed_alphas.min()}, max={fixed_alphas.max()}"
            assert np.std(adaptive_alphas) > 0.0, f"Adaptive alpha failed to vary for {img_name}!"
            assert np.array_equal(res_fixed.metadata.watermark_binary, res_adaptive.metadata.watermark_binary), "Watermark bits mismatch!"

            # Cache alpha maps for visualization
            alpha_maps_cache[(img_name, sb)] = {
                "cover": cover,
                "spatial_mask": res_adaptive.spatial_activity_mask,
                "fixed_alpha": fixed_alphas,
                "adaptive_alpha": adaptive_alphas,
            }

            # Imperceptibility Metrics (Cover vs Watermarked)
            psnr_fix = compute_psnr(cover, res_fixed.watermarked_image)
            ssim_fix = compute_ssim(cover, res_fixed.watermarked_image)
            mse_fix = compute_mse(cover, res_fixed.watermarked_image)

            psnr_ada = compute_psnr(cover, res_adaptive.watermarked_image)
            ssim_ada = compute_ssim(cover, res_adaptive.watermarked_image)
            mse_ada = compute_mse(cover, res_adaptive.watermarked_image)

            # Run Attacks
            for att_idx, (att_name, att_spec) in enumerate(attacks.items()):
                att_fn = att_spec["fn"]
                is_rand = att_spec["is_random"]
                deskew = att_spec["deskew"]

                # Synchronized Random Seed for Fairness
                seed = 42000 + (att_idx * 100) + (hash(img_name) % 100)
                if is_rand:
                    np.random.seed(seed)
                att_fixed_img = att_fn(res_fixed.watermarked_image)

                if is_rand:
                    np.random.seed(seed)
                att_ada_img = att_fn(res_adaptive.watermarked_image)

                # Extraction (Identical extraction threshold = 0.5)
                ext_fix = wm_fixed.extract(att_fixed_img, res_fixed.metadata, threshold=0.5, rotation_sync_angle=deskew)
                ext_ada = wm_adaptive.extract(att_ada_img, res_adaptive.metadata, threshold=0.5, rotation_sync_angle=deskew)

                # Robustness Metrics (Original vs Extracted)
                nc_fix = compute_nc(watermark, ext_fix.extracted_binary)
                ber_fix = compute_ber(watermark, ext_fix.extracted_binary)
                bcr_fix = compute_bcr(watermark, ext_fix.extracted_binary)

                nc_ada = compute_nc(watermark, ext_ada.extracted_binary)
                ber_ada = compute_ber(watermark, ext_ada.extracted_binary)
                bcr_ada = compute_bcr(watermark, ext_ada.extracted_binary)

                # Distortion of attacked image
                psnr_att_fix = compute_psnr(cover, att_fixed_img)
                ssim_att_fix = compute_ssim(cover, att_fixed_img)
                psnr_att_ada = compute_psnr(cover, att_ada_img)
                ssim_att_ada = compute_ssim(cover, att_ada_img)

                records.append({
                    "image": img_name,
                    "subband": sb,
                    "attack": att_name,
                    # Imperceptibility (Clean)
                    "fixed_psnr_clean": psnr_fix,
                    "adaptive_psnr_clean": psnr_ada,
                    "diff_psnr_clean": psnr_ada - psnr_fix,
                    "fixed_ssim_clean": ssim_fix,
                    "adaptive_ssim_clean": ssim_ada,
                    "diff_ssim_clean": ssim_ada - ssim_fix,
                    # Robustness Under Attack
                    "fixed_nc": nc_fix,
                    "adaptive_nc": nc_ada,
                    "diff_nc": nc_ada - nc_fix,  # positive means adaptive is better
                    "fixed_ber": ber_fix,
                    "adaptive_ber": ber_ada,
                    "diff_ber": ber_fix - ber_ada,  # positive means adaptive is better (lower BER)
                    "fixed_bcr": bcr_fix,
                    "adaptive_bcr": bcr_ada,
                    # Attacked visual quality
                    "fixed_psnr_attacked": psnr_att_fix,
                    "adaptive_psnr_attacked": psnr_att_ada,
                    "fixed_ssim_attacked": ssim_att_fix,
                    "adaptive_ssim_attacked": ssim_att_ada,
                    # Alpha statistics
                    "fixed_alpha_mean": float(np.mean(fixed_alphas)),
                    "adaptive_alpha_mean": float(np.mean(adaptive_alphas)),
                    "adaptive_alpha_min": float(np.min(adaptive_alphas)),
                    "adaptive_alpha_max": float(np.max(adaptive_alphas)),
                    "adaptive_alpha_std": float(np.std(adaptive_alphas)),
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "comparison_results_raw.csv"), index=False)
    print(f"[*] Saved raw experimental results: {len(df)} records.")

    # Save Fixed and Adaptive split CSVs
    fixed_cols = [c for c in df.columns if not c.startswith("adaptive_") and not c.startswith("diff_")]
    adaptive_cols = [c for c in df.columns if not c.startswith("fixed_") and not c.startswith("diff_")]

    df_fixed = df[["image", "subband", "attack", "fixed_psnr_clean", "fixed_ssim_clean", "fixed_nc", "fixed_ber", "fixed_bcr", "fixed_psnr_attacked", "fixed_ssim_attacked", "fixed_alpha_mean"]].copy()
    df_fixed.columns = [c.replace("fixed_", "") for c in df_fixed.columns]
    df_fixed.to_csv(os.path.join(output_dir, "fixed_results.csv"), index=False)

    df_adaptive = df[["image", "subband", "attack", "adaptive_psnr_clean", "adaptive_ssim_clean", "adaptive_nc", "adaptive_ber", "adaptive_bcr", "adaptive_psnr_attacked", "adaptive_ssim_attacked", "adaptive_alpha_mean", "adaptive_alpha_min", "adaptive_alpha_max", "adaptive_alpha_std"]].copy()
    df_adaptive.columns = [c.replace("adaptive_", "") for c in df_adaptive.columns]
    df_adaptive.to_csv(os.path.join(output_dir, "adaptive_results.csv"), index=False)

    # Save curated comparison CSV
    df_comp = df[["image", "subband", "attack", "fixed_psnr_clean", "adaptive_psnr_clean", "diff_psnr_clean", "fixed_ssim_clean", "adaptive_ssim_clean", "diff_ssim_clean", "fixed_nc", "adaptive_nc", "diff_nc", "fixed_ber", "adaptive_ber", "diff_ber"]].copy()
    df_comp.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)

    # Build Aggregate Tables and Visualizations
    build_analysis_and_figures(df, alpha_maps_cache, output_dir)

    return df


def build_analysis_and_figures(df: pd.DataFrame, alpha_maps_cache: dict, output_dir: str):
    """Generates all required summary tables, statistical analyses, and plots."""

    # 1. Imperceptibility Summary (Clean / No Attack)
    clean_df = df[df["attack"] == "No Attack"].drop_duplicates(subset=["image", "subband"])
    clean_summary = clean_df[["image", "subband", "fixed_psnr_clean", "adaptive_psnr_clean", "diff_psnr_clean", "fixed_ssim_clean", "adaptive_ssim_clean", "diff_ssim_clean"]].copy()
    clean_summary.to_csv(os.path.join(output_dir, "table1_imperceptibility.csv"), index=False)

    # 2. Attack-wise Robustness Aggregates across the 5 images
    agg_funcs = {
        "fixed_nc": ["mean", "std"],
        "adaptive_nc": ["mean", "std"],
        "diff_nc": ["mean", "std"],
        "fixed_ber": ["mean", "std"],
        "adaptive_ber": ["mean", "std"],
        "diff_ber": ["mean", "std"],
        "fixed_bcr": ["mean", "std"],
        "adaptive_bcr": ["mean", "std"],
    }

    attack_agg = df.groupby(["subband", "attack"]).agg(agg_funcs).reset_index()
    # Flatten MultiIndex columns
    attack_agg.columns = ["_".join(col).strip("_") for col in attack_agg.columns.values]
    attack_agg.to_csv(os.path.join(output_dir, "table2_attack_aggregates.csv"), index=False)

    # 3. Overall Grand Aggregates
    grand_agg = df.groupby(["subband"]).agg({
        "fixed_psnr_clean": "mean",
        "adaptive_psnr_clean": "mean",
        "fixed_ssim_clean": "mean",
        "adaptive_ssim_clean": "mean",
        "fixed_nc": "mean",
        "adaptive_nc": "mean",
        "fixed_ber": "mean",
        "adaptive_ber": "mean",
        "fixed_bcr": "mean",
        "adaptive_bcr": "mean",
    }).reset_index()
    grand_agg.to_csv(os.path.join(output_dir, "table3_grand_aggregates.csv"), index=False)

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "fixed": "#1f77b4",       # Steel Blue
        "adaptive": "#d62728",    # Crimson Red
        "fixed_alt": "#2ca02c",   # Green
        "adaptive_alt": "#ff7f0e",# Orange
    }

    attacks_order = [
        "No Attack",
        "JPEG Compression (Q=90)",
        "JPEG Compression (Q=70)",
        "JPEG Compression (Q=50)",
        "Gaussian Noise (var=0.01)",
        "Speckle Noise (var=0.01)",
        "Salt & Pepper Noise (1%)",
        "Median Filter (3x3)",
        "Gaussian Blur (3x3, s=1.0)",
        "Border Cropping (10%)",
        "Rotation (2 deg) [Unsync]",
        "Rotation (2 deg) [Sync Deskew]",
        "Rotation (5 deg) [Sync Deskew]",
        "Contrast Adjust (x1.15)",
    ]

    short_attack_labels = [
        "No Attack",
        "JPEG Q=90",
        "JPEG Q=70",
        "JPEG Q=50",
        "Gauss Noise",
        "Speckle Noise",
        "S&P Noise",
        "Median 3x3",
        "Gauss Blur",
        "Crop 10%",
        "Rot 2° Unsync",
        "Rot 2° Sync",
        "Rot 5° Sync",
        "Contrast 1.15x",
    ]

    # FIGURE 1: Mean BER Comparison across Attacks (HL & LL)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for idx, sb in enumerate(["HL", "LL"]):
        sb_df = attack_agg[attack_agg["subband"] == sb].set_index("attack").reindex(attacks_order).reset_index()
        x = np.arange(len(attacks_order))
        width = 0.35

        axes[idx].bar(x - width/2, sb_df["fixed_ber_mean"], width, yerr=sb_df["fixed_ber_std"], label="Fixed Strength (α = 30.0)", color=colors["fixed"], alpha=0.9, capsize=3)
        axes[idx].bar(x + width/2, sb_df["adaptive_ber_mean"], width, yerr=sb_df["adaptive_ber_std"], label="Texture-Adaptive Strength", color=colors["adaptive"], alpha=0.9, capsize=3)
        axes[idx].set_ylabel("Mean Bit Error Rate (BER)", fontsize=11, fontweight="bold")
        axes[idx].set_title(f"Ablation Comparison: Watermark Bit Error Rate (BER) — Subband {sb} (Lower is Better)", fontsize=12, fontweight="bold")
        axes[idx].set_ylim(0.0, 0.55)
        axes[idx].legend(loc="upper right", frameon=True, fontsize=10)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.7)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_attack_labels, rotation=35, ha="right", fontsize=10, fontweight="bold")
    plt.suptitle("Controlled Ablation Study: Fixed vs. Texture-Adaptive Embedding (Mean BER)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_ber_comparison.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # FIGURE 2: Mean NC Comparison across Attacks (HL & LL)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for idx, sb in enumerate(["HL", "LL"]):
        sb_df = attack_agg[attack_agg["subband"] == sb].set_index("attack").reindex(attacks_order).reset_index()
        x = np.arange(len(attacks_order))
        width = 0.35

        axes[idx].bar(x - width/2, sb_df["fixed_nc_mean"], width, yerr=sb_df["fixed_nc_std"], label="Fixed Strength (α = 30.0)", color=colors["fixed"], alpha=0.9, capsize=3)
        axes[idx].bar(x + width/2, sb_df["adaptive_nc_mean"], width, yerr=sb_df["adaptive_nc_std"], label="Texture-Adaptive Strength", color=colors["adaptive"], alpha=0.9, capsize=3)
        axes[idx].set_ylabel("Normalized Cross-Correlation (NC)", fontsize=11, fontweight="bold")
        axes[idx].set_title(f"Ablation Comparison: Watermark Correlation (NC) — Subband {sb} (Higher is Better)", fontsize=12, fontweight="bold")
        axes[idx].set_ylim(0.0, 1.08)
        axes[idx].legend(loc="lower right", frameon=True, fontsize=10)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.7)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_attack_labels, rotation=35, ha="right", fontsize=10, fontweight="bold")
    plt.suptitle("Controlled Ablation Study: Fixed vs. Texture-Adaptive Embedding (Mean NC)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_nc_comparison.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # FIGURE 3: Imperceptibility Comparison (PSNR & SSIM on Clean Images)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    img_names = ["Lena", "Baboon", "Fruits", "Building", "Butterfly"]
    x = np.arange(len(img_names))
    width = 0.35

    for idx, sb in enumerate(["HL", "LL"]):
        sb_clean = clean_df[clean_df["subband"] == sb].set_index("image").reindex(img_names).reset_index()

        # PSNR
        axes[idx, 0].bar(x - width/2, sb_clean["fixed_psnr_clean"], width, label="Fixed (α=30)", color=colors["fixed"])
        axes[idx, 0].bar(x + width/2, sb_clean["adaptive_psnr_clean"], width, label="Adaptive", color=colors["adaptive"])
        axes[idx, 0].set_ylabel("PSNR (dB)", fontweight="bold")
        axes[idx, 0].set_title(f"Watermarked PSNR ({sb})", fontweight="bold")
        axes[idx, 0].set_xticks(x)
        axes[idx, 0].set_xticklabels(img_names, fontweight="bold")
        axes[idx, 0].set_ylim(40, 60)
        axes[idx, 0].legend(loc="lower left")
        axes[idx, 0].grid(axis="y", linestyle="--", alpha=0.7)

        for i in range(len(img_names)):
            diff_p = sb_clean["diff_psnr_clean"].iloc[i]
            sign = "+" if diff_p >= 0 else ""
            axes[idx, 0].text(i, max(sb_clean["fixed_psnr_clean"].iloc[i], sb_clean["adaptive_psnr_clean"].iloc[i]) + 0.8, f"Δ{sign}{diff_p:.1f}dB", ha="center", fontsize=8.5, fontweight="bold")

        # SSIM
        axes[idx, 1].bar(x - width/2, sb_clean["fixed_ssim_clean"], width, label="Fixed (α=30)", color=colors["fixed_alt"])
        axes[idx, 1].bar(x + width/2, sb_clean["adaptive_ssim_clean"], width, label="Adaptive", color=colors["adaptive_alt"])
        axes[idx, 1].set_ylabel("SSIM", fontweight="bold")
        axes[idx, 1].set_title(f"Watermarked SSIM ({sb})", fontweight="bold")
        axes[idx, 1].set_xticks(x)
        axes[idx, 1].set_xticklabels(img_names, fontweight="bold")
        axes[idx, 1].set_ylim(0.990, 1.001)
        axes[idx, 1].legend(loc="lower left")
        axes[idx, 1].grid(axis="y", linestyle="--", alpha=0.7)

        for i in range(len(img_names)):
            diff_s = sb_clean["diff_ssim_clean"].iloc[i]
            sign = "+" if diff_s >= 0 else ""
            axes[idx, 1].text(i, 0.9998, f"Δ{sign}{diff_s:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Controlled Ablation: Imperceptibility Metrics across 5 Test Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_imperceptibility_comparison.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # FIGURE 4: Spatial Alpha Map vs Fixed Alpha Map Diagnostics
    fig, axes = plt.subplots(len(img_names), 4, figsize=(16, 3.2 * len(img_names)))
    for row_idx, img_name in enumerate(img_names):
        data = alpha_maps_cache[(img_name, "HL")]
        cover_img = data["cover"]
        spatial_mask = data["spatial_mask"]
        fixed_alpha = data["fixed_alpha"]
        ada_alpha = data["adaptive_alpha"]

        # 1. Cover
        axes[row_idx, 0].imshow(cover_img, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].set_title(f"{img_name} (Cover)", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 2. Spatial Activity Mask
        im1 = axes[row_idx, 1].imshow(spatial_mask, cmap="viridis", vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"Spatial Activity Mask (0-1)", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # 3. Fixed Alpha Map
        im2 = axes[row_idx, 2].imshow(fixed_alpha, cmap="magma", vmin=10, vmax=50)
        axes[row_idx, 2].set_title(f"Fixed α Map (Uniform α=30.0)", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # 4. Adaptive Alpha Map
        im3 = axes[row_idx, 3].imshow(ada_alpha, cmap="magma", vmin=10, vmax=50)
        axes[row_idx, 3].set_title(f"Adaptive α Map [min={ada_alpha.min():.1f}, max={ada_alpha.max():.1f}]", fontsize=10, fontweight="bold")
        axes[row_idx, 3].axis("off")

    # Add colorbar for Alpha maps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im3, cax=cbar_ax, label="Embedding Strength (α)")

    plt.suptitle("Visual Verification: Fixed vs. Texture-Adaptive Embedding Strength Maps", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 0.91, 0.98])
    plt.savefig(os.path.join(output_dir, "fig4_alpha_maps_verification.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # FIGURE 5: Metric Difference Profiles (Delta NC and Delta BER)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for idx, sb in enumerate(["HL", "LL"]):
        sb_df = attack_agg[attack_agg["subband"] == sb].set_index("attack").reindex(attacks_order).reset_index()
        x = np.arange(len(attacks_order))

        # NC difference: Adaptive NC - Fixed NC (positive = adaptive better)
        axes[idx].axhline(0, color="black", linestyle="--", linewidth=1.0)
        bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in sb_df["diff_nc_mean"]]
        axes[idx].bar(x, sb_df["diff_nc_mean"], yerr=sb_df["diff_nc_std"], color=bar_colors, width=0.55, capsize=3, alpha=0.85)
        axes[idx].set_ylabel("Δ NC (Adaptive - Fixed)", fontweight="bold")
        axes[idx].set_title(f"Subband {sb}: Net Robustness Advantage (Δ NC > 0 = Adaptive Superior, Δ NC < 0 = Fixed Superior)", fontweight="bold")
        axes[idx].grid(axis="y", linestyle="--", alpha=0.7)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_attack_labels, rotation=35, ha="right", fontsize=10, fontweight="bold")
    plt.suptitle("Attack-by-Attack Performance Differential (NC Advantage)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_delta_nc_profile.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"[*] Successfully built all ablation figures in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Controlled Ablation Experiment")
    parser.add_argument("--images_dir", type=str, default="data/internet_images")
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    parser.add_argument("--alpha", type=float, default=30.0)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--block_size", type=int, default=8)
    args = parser.parse_args()

    run_controlled_ablation(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        base_alpha=args.alpha,
        wavelet=args.wavelet,
        block_size=args.block_size,
    )
