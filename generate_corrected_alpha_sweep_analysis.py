"""
Corrected Visualization and Analysis Suite for Alpha Sweep & Pareto Experiments.
Strictly uses existing experimental CSV data as source of truth.
No watermarking code is modified, and no experiments are rerun.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def generate_corrected_visualizations(
    results_dir: str = "results/alpha_sweep",
    output_dir: str = "results/alpha_sweep",
):
    """Regenerates all figures with precise scientific rigor, matched-PSNR focus, and no axis truncation."""
    os.makedirs(output_dir, exist_ok=True)
    raw_csv = os.path.join(results_dir, "alpha_sweep_raw.csv")
    by_att_csv = os.path.join(results_dir, "alpha_sweep_by_attack.csv")
    summary_csv = os.path.join(results_dir, "alpha_sweep_summary.csv")

    if not os.path.isfile(raw_csv) or not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"Required CSV files not found in {results_dir}")

    df_raw = pd.read_csv(raw_csv)
    df_summary = pd.read_csv(summary_csv)
    df_by_att = pd.read_csv(by_att_csv)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # Style mapping
    styles = {
        ("Fixed", "HL"): {"color": "#1f77b4", "linestyle": "-", "marker": "o", "label": "Fixed HL (α ∈ [10, 50])"},
        ("Fixed", "LL"): {"color": "#2ca02c", "linestyle": "-", "marker": "s", "label": "Fixed LL (α ∈ [10, 50])"},
        ("Adaptive", "HL"): {"color": "#d62728", "linestyle": "--", "marker": "^", "label": "Adaptive HL (Base α ∈ [10, 60])"},
        ("Adaptive", "LL"): {"color": "#ff7f0e", "linestyle": "--", "marker": "D", "label": "Adaptive LL (Base α ∈ [10, 60])"},
    }

    # =========================================================================
    # 1. MATCHED-PSNR PARETO PLOTS: PSNR vs NC & PSNR vs BER
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: PSNR vs NC
    ax_nc = axes[0]
    for (method, sb), st in styles.items():
        sub_df = df_summary[(df_summary["method"] == method) & (df_summary["subband"] == sb)].sort_values("psnr_clean_mean")
        ax_nc.plot(
            sub_df["psnr_clean_mean"],
            sub_df["nc_mean"],
            label=st["label"],
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=6.5,
            alpha=0.9,
        )

    ax_nc.set_xlabel("Imperceptibility: Watermarked PSNR (dB) [Higher = Better Quality]", fontsize=11, fontweight="bold")
    ax_nc.set_ylabel("Robustness: Mean NC Across 13 Attacks [Higher = Better]", fontsize=11, fontweight="bold")
    ax_nc.set_title("Pareto Frontier: Imperceptibility (PSNR) vs. Robustness (NC)", fontsize=12, fontweight="bold")
    ax_nc.set_ylim(0.0, 1.0)
    ax_nc.set_xlim(40, 63)
    ax_nc.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax_nc.grid(True, linestyle="--", alpha=0.6)

    # Panel B: PSNR vs BER
    ax_ber = axes[1]
    for (method, sb), st in styles.items():
        sub_df = df_summary[(df_summary["method"] == method) & (df_summary["subband"] == sb)].sort_values("psnr_clean_mean")
        ax_ber.plot(
            sub_df["psnr_clean_mean"],
            sub_df["ber_mean"],
            label=st["label"],
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=6.5,
            alpha=0.9,
        )

    ax_ber.set_xlabel("Imperceptibility: Watermarked PSNR (dB) [Higher = Better Quality]", fontsize=11, fontweight="bold")
    ax_ber.set_ylabel("Error Rate: Mean BER Across 13 Attacks [Lower = Better]", fontsize=11, fontweight="bold")
    ax_ber.set_title("Pareto Frontier: Imperceptibility (PSNR) vs. Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    ax_ber.set_ylim(0.0, 0.5)
    ax_ber.set_xlim(40, 63)
    ax_ber.legend(loc="upper left", fontsize=9.5, frameon=True)
    ax_ber.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Matched-Distortion Pareto Analysis: Fixed vs. Texture-Adaptive (N=5 Images)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_matched_psnr_tradeoff.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 2. SINGLE-PANEL ALPHA vs PSNR (Eliminating Duplication)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fix_hl = df_summary[(df_summary["method"] == "Fixed") & (df_summary["subband"] == "HL")].sort_values("alpha_setting")
    ada_hl = df_summary[(df_summary["method"] == "Adaptive") & (df_summary["subband"] == "HL")].sort_values("alpha_setting")

    ax.plot(
        fix_hl["alpha_setting"],
        fix_hl["psnr_clean_mean"],
        label="Fixed Strength: Nominal α = Effective α (α ∈ [10, 50])",
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=2.2,
        markersize=7,
    )
    ax.plot(
        ada_hl["alpha_setting"],
        ada_hl["psnr_clean_mean"],
        label="Adaptive Strength: Nominal Base α (Base α ∈ [10, 60], Eff α ≈ 0.60×Base)",
        color="#d62728",
        linestyle="--",
        marker="^",
        linewidth=2.2,
        markersize=7,
    )
    # Also plot Adaptive vs its Effective Alpha to prove matched curve!
    ax.plot(
        ada_hl["eff_alpha_mean_mean"],
        ada_hl["psnr_clean_mean"],
        label="Adaptive Strength plotted vs. Mean Effective α (Eff α ∈ [6.0, 36.2])",
        color="#9467bd",
        linestyle=":",
        marker="x",
        linewidth=2.0,
        markersize=7,
    )

    ax.set_xlabel("Alpha Embedding Strength Parameter (Nominal or Effective)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Watermarked Image Fidelity: PSNR (dB)", fontsize=11, fontweight="bold")
    ax.set_title("Visual Imperceptibility vs. Alpha Parameter (HL & LL Subbands Identical, N=5 Images)", fontsize=12, fontweight="bold")
    ax.set_ylim(40, 63)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curve_alpha_vs_psnr_corrected.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 3. NOMINAL & EFFECTIVE ALPHA vs ROBUSTNESS (NC & BER, y-axis at 0)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    # Row 1: Alpha vs NC (HL and LL)
    for idx, sb in enumerate(["HL", "LL"]):
        ax = axes[0, idx]
        for method in ["Fixed", "Adaptive"]:
            st = styles[(method, sb)]
            sub_df = df_summary[(df_summary["method"] == method) & (df_summary["subband"] == sb)].sort_values("alpha_setting")
            ax.plot(
                sub_df["alpha_setting"],
                sub_df["nc_mean"],
                label=f"{method} {sb} (Nominal α)",
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=2.0,
                markersize=6,
            )
        ax.set_title(f"Nominal Alpha vs. Mean NC — Subband {sb} (N=5)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Nominal Alpha Parameter", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Robustness (NC)", fontsize=10, fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right", fontsize=9.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    # Row 2: Alpha vs BER (HL and LL)
    for idx, sb in enumerate(["HL", "LL"]):
        ax = axes[1, idx]
        for method in ["Fixed", "Adaptive"]:
            st = styles[(method, sb)]
            sub_df = df_summary[(df_summary["method"] == method) & (df_summary["subband"] == sb)].sort_values("alpha_setting")
            ax.plot(
                sub_df["alpha_setting"],
                sub_df["ber_mean"],
                label=f"{method} {sb} (Nominal α)",
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=2.0,
                markersize=6,
            )
        ax.set_title(f"Nominal Alpha vs. Mean BER — Subband {sb} (N=5)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Nominal Alpha Parameter", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Bit Error Rate (BER)", fontsize=10, fontweight="bold")
        ax.set_ylim(0.0, 0.5)
        ax.legend(loc="upper right", fontsize=9.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Robustness Metrics across Nominal Alpha Parameter (Fixed α ∈ [10, 50], Adaptive Base α ∈ [10, 60])", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curve_nominal_alpha_vs_robustness_corrected.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 4. DEDICATED FIXED LL TRANSITION PLOT
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    fix_ll = df_summary[(df_summary["method"] == "Fixed") & (df_summary["subband"] == "LL")].sort_values("alpha_setting")

    color_nc = "#2ca02c"
    color_psnr = "#1f77b4"
    color_ber = "#d62728"

    # Highlight transition region alpha in [15, 20]
    ax1.axvspan(15, 20, color="#ffe082", alpha=0.35, label="Empirical Transition Zone (α ≈ 15–20)")

    ax1.plot(fix_ll["alpha_setting"], fix_ll["nc_mean"], color=color_nc, linestyle="-", marker="s", linewidth=2.4, label="Robustness: Mean NC (Left Axis)")
    ax1.plot(fix_ll["alpha_setting"], fix_ll["ber_mean"], color=color_ber, linestyle="--", marker="v", linewidth=2.0, label="Error Rate: Mean BER (Left Axis)")
    ax1.set_xlabel("Fixed Embedding Strength (α)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("NC / BER Score", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Right axis: PSNR
    ax2 = ax1.twinx()
    ax2.plot(fix_ll["alpha_setting"], fix_ll["psnr_clean_mean"], color=color_psnr, linestyle=":", marker="o", linewidth=2.0, label="Fidelity: PSNR (dB, Right Axis)")
    ax2.set_ylabel("Watermarked PSNR (dB)", fontsize=11, fontweight="bold", color=color_psnr)
    ax2.set_ylim(40, 60)
    ax2.grid(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9.5, frameon=True)

    plt.title("Fixed LL Subband: Alpha Transition & Operational Trade-Off (N=5 Images)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fixed_ll_alpha_transition.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 5. PER-ATTACK COMPARISON: HIGHLIGHTING HL/LL COMPLEMENTARITY
    # =========================================================================
    attacks_all = [
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

    short_names = [
        "JPEG Q90", "JPEG Q70", "JPEG Q50",
        "Gauss Noise", "Speckle Noise", "S&P Noise",
        "Median 3x3", "Gauss Blur", "Crop 10%",
        "Rot 2° Unsync", "Rot 2° Sync", "Rot 5° Sync", "Contrast 1.15x"
    ]

    # Evaluate at nominal alpha = 30 for direct per-attack bar comparison
    att_30 = df_by_att[df_by_att["alpha_setting"] == 30.0]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Panel A: Mean NC across attacks
    x = np.arange(len(attacks_all))
    width = 0.18

    # Fixed HL, Adaptive HL, Fixed LL, Adaptive LL
    series_info = [
        ("Fixed", "HL", -1.5, styles[("Fixed", "HL")], ""),
        ("Adaptive", "HL", -0.5, styles[("Adaptive", "HL")], "//"),
        ("Fixed", "LL", 0.5, styles[("Fixed", "LL")], ""),
        ("Adaptive", "LL", 1.5, styles[("Adaptive", "LL")], "//"),
    ]

    for method, sb, offset_mult, st, hatch in series_info:
        sub_att = att_30[(att_30["method"] == method) & (att_30["subband"] == sb)].set_index("attack").reindex(attacks_all).reset_index()
        pos = x + offset_mult * width
        axes[0].bar(pos, sub_att["nc_mean"], width, label=f"{method} {sb} (α=30)", color=st["color"], alpha=0.85, hatch=hatch)

    axes[0].set_ylabel("Mean Robustness (NC)", fontsize=11, fontweight="bold")
    axes[0].set_title("Per-Attack Watermark Recovery (NC) at Nominal α=30 (HL vs. LL Complementarity)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="upper right", ncol=2, fontsize=9, frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Panel B: Mean BER across attacks
    for method, sb, offset_mult, st, hatch in series_info:
        sub_att = att_30[(att_30["method"] == method) & (att_30["subband"] == sb)].set_index("attack").reindex(attacks_all).reset_index()
        pos = x + offset_mult * width
        axes[1].bar(pos, sub_att["ber_mean"], width, label=f"{method} {sb} (α=30)", color=st["color"], alpha=0.85, hatch=hatch)

    axes[1].set_ylabel("Mean Bit Error Rate (BER)", fontsize=11, fontweight="bold")
    axes[1].set_title("Per-Attack Watermark Error Rate (BER) at Nominal α=30 (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0.0, 0.65)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=30, ha="right", fontsize=9.5, fontweight="bold")
    axes[1].legend(loc="upper right", ncol=2, fontsize=9, frameon=True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.suptitle("Attack-Specific Robustness Profile (N=5 Images, Seeds 1..5 for Stochastic Attacks)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_attack_robustness_complementarity.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 6. ADAPTIVE EFFECTIVE ALPHA RATIO TABLE & FIGURE
    # =========================================================================
    ada_summary = df_summary[df_summary["method"] == "Adaptive"].copy()
    ada_summary["eff_ratio"] = ada_summary["eff_alpha_mean_mean"] / ada_summary["alpha_setting"]
    eff_table = ada_summary[["subband", "alpha_setting", "eff_alpha_mean_mean", "eff_alpha_min_mean", "eff_alpha_max_mean", "eff_ratio", "psnr_clean_mean", "nc_mean", "ber_mean"]].copy()
    eff_table.columns = ["Subband", "Base Alpha", "Mean Eff Alpha", "Min Eff Alpha", "Max Eff Alpha", "Eff/Base Ratio", "Clean PSNR (dB)", "Mean NC", "Mean BER"]
    eff_table.to_csv(os.path.join(output_dir, "adaptive_effective_alpha_mapping.csv"), index=False)

    print(f"[*] Successfully regenerated all corrected visualizations in: {output_dir}")
    return eff_table


if __name__ == "__main__":
    generate_corrected_visualizations()
