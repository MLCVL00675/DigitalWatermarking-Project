# Methodological Deviations, Missing Artifacts & Empirical Audit Report

**Repository Git Commit Hash**: `dd81f8184517449d1adb62cc90cc4381d73c1cc1`  
**Audit Timestamp**: 2026-10-30 06:07:24 UTC  
**Governing Standard**: Strict Rule 1-5 Compliance (Zero Hand-Entered Values, Verified SHA-256 Hashing, Image-Level Analysis).

---

## 1. Executive Summary of Audit Findings

An exhaustive forensic comparison between the published narrative summary (`results_v2/RESULTS_SUMMARY.md`), the pipeline orchestrator (`run_all_v2.py`), and the underlying raw and summary CSV files in `results_v2/` revealed two principal categories of defect:

1. **Missing Raw Experimental and Ablation Files**: Several intermediate raw dataset files called for by `run_all_v2.py` (specifically Step 2 `core_raw.csv` and Step 5 `ablation_variants_raw.csv` along with modular ablation outputs) were never written to `results_v2/`.
2. **Fabricated / Unsynchronized Text Numbers**: The summary generator in `run_all_v2.py` contained hardcoded strings containing synthetic or legacy v1 numbers (e.g., baseline error rates, runtime benchmarks, texture correlations, generative channel metrics) that did not match the corresponding empirical numbers in `results_v2/*.csv`.

Per **Rule 3**, missing experimental files cannot be invented, backfilled, or guessed. All numbers in `results_v3/` are computed dynamically from the existing verified CSV files, and all discrepancies are cataloged below.

---

## 2. Missing Raw & Ablation Files Analysis

| Expected File | Generating Script / Step | Status in `results_v2/` | Root Cause & Resolution |
| :--- | :--- | :--- | :--- |
| `core_raw.csv`<br>`matched_psnr_raw.csv`<br>`pareto_points.csv`<br>`paired_diffs.csv` | `experiments_v2.study`<br>(Pipeline Step 2) | **MISSING in v2; REGENERATED in v3** | In `results_v2/`, Step 2 core benchmark outputs were omitted. In `results_v3/`, all four files were systematically regenerated under the strict protocol: 100 benchmark images (matching `image_subsets.csv`), 2 payloads (`whitened_logo`, `balanced_0`), matched clean PSNR in {40, 44, 46, 48} dB and 10 nominal alphas {5..60}, subbands HL and LL, and 10 seeds for stochastic attacks (40 trials/condition). All row counts match factor formulas (`matched_psnr_raw.csv`: 128,000; `paired_diffs.csv`: 104; `core_raw.csv`: 320,000; `pareto_points.csv`: 40). |
| `ablation_variants_raw.csv`<br>`ablation_transform.csv`<br>`ablation_payload.csv`<br>`ablation_block.csv`<br>`ablation_features.csv`<br>`ablation_wavelet.csv` | `experiments_v2.ablations`<br>(Pipeline Step 5) | **MISSING in v2; REGENERATED in v3** | In `results_v2/`, Step 5 modular ablation outputs were omitted. In `results_v3/`, all modular ablation datasets (`ablation_transform.csv`, `ablation_features.csv`, `ablation_wavelet.csv`, `ablation_block.csv`, `ablation_payload.csv`, `ablation_heatmap_data.csv`, and `ablation_transform_stats.csv`) were systematically regenerated under the strict protocol: matched clean PSNR 46 dB (calibrated alpha per image, after 8-bit integer quantization), subbands HL and LL, 100 benchmark images, and 10 seeds for stochastic attacks. |
| `weight_ablation_raw.csv`<br>`alpha_sweep_raw.csv` | Legacy v1 scripts (`experiments/`) | **Present only in legacy `results/`** | These legacy v1 outputs were performed on earlier subsets (5-40 images) and were not part of the standardized 100-image `results_v2` suite. |

---

## 3. Forensic Comparison: v2 Narrative Claims vs. Verified CSV Data

The following table explicitly contrasts every claim in `results_v2/RESULTS_SUMMARY.md` against the exact value computed from the corresponding CSV in `results_v2/`:

| Section / Metric | Value Claimed in v2 `RESULTS_SUMMARY.md` | Actual Empirical Value in `results_v2/*.csv` | Source CSV File | Finding & Forensic Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Total Tracked Trials** | `281,200+ trials` (incl. ~241,200 core study) | **257,600 trials** | Sum of all raw CSV files | v2 claimed 281,200+ trials based on an unexecuted/missing core grid. The true sum of all raw trial records across all 8 raw CSVs is exactly 257,600. |
| **Paired BER Diff 95% CI** | `[-0.00021, +0.00132]` | **Empirical CIs range from -0.011 to +0.013** per attack | `stats_paired.csv` | v2 hardcoded a single synthetic interval. In reality, paired CIs are calculated per attack condition in `stats_paired.csv`. |
| **TOST Equivalence p-value** | `$p_{\text{TOST}} = 9.51 \times 10^{-91}$` | Pooled HL: **$9.51 \times 10^{-91}$**;<br>Per-attack: **$1.95 \times 10^{-21}$ to $2.33 \times 10^{-93}$** | `stats_tost.csv` | v2 quoted the pooled HL subband row while presenting it as the overall result. All individual attack p-values in `stats_tost.csv` reject non-equivalence. |
| **Texture Rank Correlation $\rho$** | `$\rho = +0.012, p = 0.908$` | **$\rho = +0.3031, p = 0.00217$** (pooled)<br>$\rho = +0.0231, p = 0.8986$ (mid stratum) | `stats_texture.csv` | v2 reported $\rho = +0.012, p = 0.908$ which does not exist in `stats_texture.csv`. The true pooled Spearman correlation is $+0.3031$ ($p = 0.00217$). |
| **Lai-Tsai Blur (s=1.0) BER** | `0.0000` | **0.0264** | `baselines_summary.csv` | v2 narrative claimed zero error; actual empirical median BER is 0.0264. |
| **Lai-Tsai Contrast (x1.15) BER** | `0.4892` | **0.5000** | `baselines_summary.csv` | v2 narrative reported 0.4892; actual median is 0.5000 (complete loss of synchronization). |
| **DCT-SVD Blur (s=1.0) BER** | `0.4120` | **0.0366** | `baselines_summary.csv` | v2 narrative reported 0.4120; actual empirical median BER is 0.0366. |
| **DCT-SVD Contrast (x1.15) BER** | `0.0000` | **0.5000** | `baselines_summary.csv` | v2 narrative claimed zero error; actual median is 0.5000. |
| **Standard QIM JPEG 70 BER** | `0.1450` | **0.4229** | `baselines_summary.csv` | v2 narrative claimed 0.1450; actual empirical median BER is 0.4229. |
| **Adaptive QIM JPEG 70 BER** | `0.0820` | **0.4097** | `baselines_summary.csv` | v2 narrative claimed 0.0820; actual empirical median BER is 0.4097. |
| **QIM Adaptation Gain** | `0.063` BER reduction | **0.0132** (0.4229 - 0.4097) | `baselines_summary.csv` | v2 narrative claimed a 0.063 gain; the actual measured reduction at Q=70 is 0.0132. |
| **Adaptive Runtime (512x512)** | `48.2 ms` | **238.80 ms** | `runtime_summary.csv` | v2 narrative claimed 48.2 ms; actual measured median embedding runtime is 238.80 ms. |
| **Fixed Runtime (512x512)** | `21.5 ms` | **151.28 ms** | `runtime_summary.csv` | v2 narrative claimed 21.5 ms; actual measured median embedding runtime is 151.28 ms. |
| **Peak Memory Allocation** | `42.1 MB` / `38.6 MB` | **13.50 MB** / **12.50 MB** | `runtime_summary.csv` (`embed_peak_kb`) | v2 narrative claimed ~40 MB RSS; actual peak tracked is 13.50 MB (13,826.91 KB). |
| **Vectorization Speedup** | `6.8x speedup` | **4.53x speedup** | `runtime_summary.csv` (`block_speedup`) | v2 narrative claimed 6.8x; actual vectorized block speedup is 4.53x. |
| **VAE HL Reconstruction PSNR** | `23.63 dB`, SSIM `0.615` | **24.98 dB**, SSIM **0.629** | `regen_raw.csv` | v2 narrative reported 23.63 dB; actual average stego quality is 24.98 dB. |
| **VAE LL Fixed BER** | `0.3721` | **0.3760** | `regen_raw.csv` | v2 narrative reported 0.3721; actual median BER is 0.3760. |
| **VAE LL Adaptive BER** | `0.3901` | **0.4023** | `regen_raw.csv` | v2 narrative reported 0.3901; actual median BER is 0.4023. |
| **Geometric Auto-Registration BER** | `< 0.0200` BER | `auto_ecc`: **0.0781**;<br>`auto_feature`: **0.1045** | `sync_raw.csv` | v2 narrative claimed <0.0200; actual median BER across all geometric attacks is 0.0781 (auto_ecc) and 0.1045 (auto_feature). |
| **Registration Convergence** | `94.0%` convergence | **100.0%** recorded convergence | `sync_estimation_error.csv` | `reg_success` is 100% (16,000 / 16,000 fits) in `sync_estimation_error.csv`. |

---

## 4. Pipeline Bug in `make_figures.py` Table 5 Generation

In `experiments_v2/make_figures.py`, line 1575:
```python
"Median JPEG 70 BER": f"{grp[grp['attack'] == 'jpeg']['median_ber'].iloc[0]:.4f}"
```
Because the group `grp` is sorted by severity and the severity list for JPEG compression is `['Q=50', 'Q=70', 'Q=90']`, `iloc[0]` selected the **Q=50** value (`0.0020` for Lai-Tsai and DCT-SVD, `0.4619` for Adaptive QIM, `0.4653` for Blind QIM), while labeling the column as "Median JPEG 70 BER".

In `results_v3/tables/table5_baseline_comparisons`, this bug is resolved by explicitly filtering for `severity == 'Q=70'` to ensure numerical consistency with the column header.

---

## 5. Strict Compliance Certification

Under Rules 1 through 5:
1. `results_v3/RESULTS_SUMMARY.md` is strictly generated by executing `generate_v3_package.py`, ensuring that every cited value is dynamically loaded and computed from `results_v2/*.csv`.
2. No numbers have been manually adjusted, guessed, or smoothed.
3. Every unchanged file copied from `results_v2/` is cryptographically validated with its SHA-256 digest in `results_v3/MANIFEST.json`.
4. Image units of analysis are explicitly recorded in `results_v3/image_subsets.csv`.


---

## 6. Rotation Benchmark Boundary Mode Discrepancy & Resolution (v4 vs. v5)

An empirical investigation of the discrepancy between `sync_raw.csv.gz` (median BER $\approx 0.025$ at $2.0^\circ$, 200 trials) and `baselines_rotation_v4.csv` (median BER $\approx 0.061$, 100 trials) identified the root cause:
- **Boundary Extrapolation in Warping**: `experiments_v2/sync.py` uses `borderMode=cv2.BORDER_REFLECT` during rotation attack, whereas `scratch/run_baselines_rotation_v4.py` used `borderMode=cv2.BORDER_CONSTANT` with `borderValue=0.0`.
- **Mechanism of Distortion**: Constant black borders introduce artificial zero-padding triangles at the four corners. When inverted during deskew, these black triangles corrupt peripheral 8x8 DWT blocks (giving oracle BER $\approx 0.045$) and bias multi-scale ECC gradient descent (increasing registration angle error from $0.0014^\circ$ to $0.0322^\circ$).
- **Resolution in v5**: `scratch/run_baselines_rotation_v5.py` aligns the shared rotation function with `sync.py` using `BORDER_REFLECT`. Under `baselines_rotation_v5.csv`, `proposed_fixed` LL median BER returns to $0.024903$, perfectly matching `sync_raw.csv.gz`.
- **Equivalence Status**: Across all 100 images, `lai_tsai_dwt_svd`, `hybrid_precedent_fixed`, and `proposed_fixed` (LL) achieve identical bit error rates ($d = 0.000000$, $p_{\text{TOST}} < 10^{-15}$, Cliff's $\delta = 0.0000$).


---

## 7. SSIM-Matched Benchmark Minimum-Strength Floor & Feasibility Audit (v4 vs. v5)

An empirical audit of `results_v4/ssim_matched_v4.csv.gz` revealed clean-extraction failures (BER $> 0.01$) across 9-14% of benchmark images in the LL subband at target SSIM $\in \{0.97, 0.98, 0.99\}$.
- **Root Cause**: To reach SSIM $\in [0.97, 0.99]$ in the LL subband, calibration was either pushed to very large $\alpha$ ($> 150$), inducing severe boundary clipping ($> 5,000$ pixels clipped), or in images with low dynamic range, $\alpha$ fell below the 8-bit quantization floor.
- **Resolution in v5**: A strict minimum-strength floor ($\alpha_{\text{base}} \ge 10.0$ for `proposed_fixed`, $\alpha_{\text{base}} \ge 25.0$ for `proposed_adaptive`) with headroom guard $h=1$ was enforced. Any image where target SSIM cannot be met above the floor with clean BER $\le 0.001$ is explicitly classified as **infeasible** ($N=24$ to $28$ images in LL, $N=2$ to $10$ in HL).
- **Paired Statistics**: Paired adaptive-vs-fixed statistics were computed exclusively on mutually feasible image pairs ($N_{\text{feas}} = 72$ to $93$ images). Under the primary 9-family-weighted pooling rule, `proposed_adaptive` demonstrates statistically significant BER reductions in HL across all feasible SSIM targets ($\Delta = -0.009138$ to $-0.011872$, $p < 10^{-15}$), and a modest reduction in LL at SSIM 0.98 ($\Delta = -0.000846, p = 2.86 \times 10^{-6}$) and 0.99 ($\Delta = -0.003049, p = 6.04 \times 10^{-13}$).

---

## 8. Package Availability for Perceptual Metrics (MS-SSIM / LPIPS)

Per Section 4 requirements, an environment inspection was conducted for perceptual distortion metrics:
- **LPIPS**: The `lpips` package and its prerequisite `torch` runtime are not installed in the execution environment (`ModuleNotFoundError: No module named 'lpips'`).
- **MS-SSIM**: The PyTorch-based `pytorch_msssim` package is likewise unavailable. Standard multi-scale SSIM was not computed because only single-scale `structural_similarity` is provided by `scikit-image` in the standard Python environment without deep learning dependencies.
- Consequently, perceptual matching is strictly evaluated via standardized single-scale SSIM (`skimage.metrics.structural_similarity`).
