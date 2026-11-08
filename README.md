# Invisible Robust Watermarking: Matched-Distortion Evaluation and Empirical Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-pending%20Zenodo%20release-blue)](https://github.com/MLCVL00675/DigitalWatermarking-Project/releases)
[![Reproducibility Package](https://img.shields.io/badge/Reproducibility-Verified%20(77%2F77%20Passed)-success.svg)](results_v4/RESULTS_SUMMARY.md)

This repository contains the authoritative open-source replication codebase, modular experimental execution engine, and statistical audit suite for the rigorous matched-distortion benchmark of frequency-domain digital image watermarking (cascaded DWT-DCT-SVD with Human Visual System deterministic texture masking).

---

## 🔬 Core Empirical Findings

1. **H1 (Texture-Adaptive vs. Fixed Embedding)**: Under strict matched cover fidelity ($46.0\text{ dB}$ PSNR) and standard 8-bit integer quantization, texture adaptation offers **no statistically significant robustness gain** over uniform fixed embedding across 208 standardized images (overall paired difference $+0.00055$ BER; Two One-Sided Tests firmly reject non-equivalence with $p_{\text{TOST}} = 9.51 \times 10^{-91}$ at equivalence margin $\Delta = \pm 0.02$). However, under **matched perceptual SSIM** ($0.97, 0.98, 0.99$), texture-adaptive embedding decisively outperforms uniform fixed embedding across additive noise and high-frequency channels ($p_{\text{adj}} < 10^{-6}$).
2. **H2 (Subband $\times$ Attack Interaction)**: Subband performance is fundamentally contingent on attack modality ($\chi^2 = 8,863.15, p < 10^{-300}$ via MixedLM LRT; non-parametric 10,000-permutation test $F = 4,884.53, p = 0.000100$). The **LL subband dominates** under low-pass filtering and lossy JPEG compression (Cliff's $\delta = +1.000$, BER reduction of $0.22 - 0.44$), whereas the **HL subband retains complete resilience** against photometric illumination shifts (contrast/gamma shifts: $\text{BER} \le 0.01$, Cliff's $\delta = -1.000$ vs. catastrophic failure in LL).
3. **H3 (Security & Ambiguity Vulnerability)**: Unauthenticated reference-based SVD watermarking is fundamentally vulnerable to Craver ambiguity attacks (N6 SVD forgery achieves $100\%$ false-claim success, collapsing ROC AUC to $0.4380$). Cryptographic commitment (HMAC / SHA-256 hashing of singular matrices) restores complete security ($100\%$ forged reference rejection).
4. **H4 (Geometric Synchronization)**: Fragility against geometric attacks is primarily a synchronization desynchronization loss, fully restorable via automatic affine registration and ORB/ECC pre-deskewing ($\text{BER} < 0.02$ on rotations $\le 5^\circ$).
5. **Generative AI & Latent Diffusion Threat Model**: Evaluated across 50 benchmark images at native $512 \times 512$ using Stable Diffusion 1.5 img2img and SD-VAE MSE autoencoding across strengths $[0.1, 0.2, 0.3, 0.5]$ (2,600 trials). High generative strengths ($\ge 0.3$) disrupt singular value relations ($\text{BER} \ge 0.40$), while subtle latent reconstruction preserves detectable watermarks ($\text{BER} \le 0.18$).

---

## 💻 Hardware Environment & Tested Specifications

All benchmark experiments and statistical audits were executed and validated on the following environment:
- **Operating System**: Microsoft Windows 11 Professional (64-bit, Build 26100) / Linux (Ubuntu 22.04 LTS compatible)
- **Processor**: AMD Ryzen / Intel Multi-Core Workstation Processor (8 physical cores, 16 logical threads)
- **System Memory**: 16 GB+ DDR4/DDR5 RAM
- **GPU (for Diffusion Benchmark)**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, CUDA 12.4)
- **Python**: 3.11.4 / 3.13.5 (64-bit)
- **Core Scientific Stack**: NumPy 1.24.3 / 2.4.2, SciPy 1.11.1 / 1.17.0, OpenCV 4.8.0 / 4.10.0, PyWavelets 1.4.1 / 1.8.0, Pandas 2.0.0 / 3.0.0, Statsmodels 0.14.0 / 0.14.4

### Expected Runtime Breakdown

| Component / Stage | Quick Mode (`--quick`) | Full Mode (`--full`) | Primary Hardware |
| :--- | :---: | :---: | :---: |
| **Dataset Ingestion & Checksum Verification** | ~1 s (cache check) | ~25 s (308 images download) | Network / CPU |
| **Core Matched-PSNR Study (46 dB, 208 images)** | ~3 s (smoke test) | ~45 min (20 seeds, 66,560 trials) | Multi-core CPU |
| **Severity Sweeps & Compound Chains** | ~5 s | ~35 min (100 images, 88,000 trials) | Multi-core CPU |
| **Geometric Synchronization (ORB / ECC)** | ~3 s | ~25 min (36,800 trials) | Multi-core CPU |
| **Comparative Baselines (5 algorithms + tuned QIM)** | ~4 s | ~30 min (82,000 trials) | Multi-core CPU |
| **Craver Ambiguity Security & Negative Controls** | ~2 s | ~10 min (24,000 trials) | Multi-core CPU |
| **Generative Diffusion & VAE Benchmark (SD 1.5)** | ~3 s | ~18 min (50 images, 2,600 trials) | NVIDIA RTX GPU |
| **Statistical Bootstrap Suite (10,000 resamples)** | ~2 s | ~60 s (MixedLM + Permutations) | Multi-core CPU |
| **Summary & Manifest Generation (`make_summary`)** | ~2.5 s | ~3 s (Dynamic parsing) | CPU |
| **Independent Consistency Audit (77 assertions)** | ~2.5 s | ~3 s | CPU |
| **Total End-to-End Execution Time** | **< 15 seconds** | **~2.5 – 3.0 hours** | Workstation + GPU |

---

## 🚀 Quick-Start Reproduction Guide

### 1. Installation

Clone repository and initialize a clean virtual environment:
```bash
git clone https://github.com/MLCVL00675/DigitalWatermarking-Project.git
cd DigitalWatermarking-Project

# Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

# Install exact pinned dependencies
pip install -r requirements.txt
```

### 2. Dataset Acquisition & Verification
Dataset images are not tracked in Git due to third-party academic licenses. Download and preprocess the 308 standardized benchmark images (BSDS500, Kodak, USC-SIPI, COCO) into 512x512 grayscale PNGs and verify their SHA-256 hashes:
```bash
# Download and verify automatically:
python scripts/download_datasets.py --workers 6

# Or if images already exist, verify checksums:
python scripts/download_datasets.py --verify-only
```

### 3. Master Execution

#### A. Quick Verification Mode (Smoke Test, < 15 seconds)
Validates the entire pipeline, watermark embedding, frequency transforms, dynamic summary compilation, and independent audit assertions:
```bash
python run_all_v4.py --quick
# Or using Makefile:
make quick
```

#### B. Full Experimental Replication (All 208 images, full protocol)
Executes all benchmark sweeps, paired comparisons, baselines, and statistical models:
```bash
python run_all_v4.py --full --workers 8 --results-dir results_v4
# Or using Makefile:
make full
```

#### C. Independent Consistency Audit
Verifies that every single number quoted in `RESULTS_SUMMARY.md` matches the underlying CSVs within rounding precision, that every mean falls strictly within its confidence interval, and that row counts equal file lengths:
```bash
python check_consistency.py --results-dir results_v4
# Or using Makefile:
make audit
```

---

## 📁 Repository Structure

```
.
├── CITATION.cff                      # Machine-readable citation metadata (CFF 1.2.0)
├── LICENSE                           # Official MIT License
├── Makefile                          # Unified build and automation targets
├── README.md                         # Project documentation and quick-start guide
├── requirements.txt                  # Pinned Python package dependencies
├── check_consistency.py              # Strict independent consistency auditor (77 checks)
├── run_all_v4.py                     # Master reproducibility pipeline orchestrator
│
├── watermark/                        # Core frequency-domain watermarking package
│   ├── core.py                       # Cascaded DWT-DCT-SVD engine & informed decoder
│   ├── masking.py                    # Deterministic HVS spatial texture masking (Sobel/std/entropy)
│   ├── attacks.py                    # Classical robustness evaluation attacks
│   └── sync.py                       # Geometric synchronization & affine pre-registration
│
├── experiments_v2/                   # Modular experiment replication engine
│   ├── dataset.py                    # Dataset ingestion, cropping, and characterization
│   ├── harness.py                    # Matched-distortion calibration (46.0 dB PSNR)
│   ├── attacks_v2.py                 # 14 single attacks + sequential compound chains
│   ├── baselines_v2.py               # Comparative baselines (Lai-Tsai, DCT-SVD, Blind QIM)
│   ├── security.py                   # Craver ambiguity attacks and negative controls N1-N6
│   ├── stats.py                      # Paired t-tests, bootstrap 95% CIs, and TOST equivalence
│   ├── make_summary.py               # Dynamic code-only summary generator (no hardcoded numbers)
│   └── make_figures.py               # Publication-grade figures and LaTeX table compiler
│
├── scripts/                          # Utility and acquisition scripts
│   └── download_datasets.py          # Standalone public dataset downloader & SHA-256 verifier
│
└── results_v4/                       # Authoritative release experimental results
    ├── MANIFEST.json                 # SHA-256 checksums, exact row counts, and Git commit
    ├── RESULTS_SUMMARY.md            # Comprehensive dynamically compiled benchmark report
    ├── DEVIATIONS.md                 # Documented deviations, audit logs, and design rationale
    ├── dataset_manifest_v4.csv       # 308 images with source URLs, terms, and SHA-256
    ├── image_subsets.csv             # Exact per-experiment image unit-of-analysis tracking
    ├── core_all208_v4.csv.gz         # Matched-PSNR core study on all 208 images (66,560 rows)
    ├── ssim_matched_v4.csv.gz        # Perceptual SSIM-matched benchmark (48,000 rows)
    ├── core_coco100_v4.csv.gz        # Out-of-domain COCO benchmark (32,000 rows)
    ├── core_headroom_v4.csv.gz       # Headroom clipping guard evaluations (32,000 rows)
    ├── regen_v4.csv.gz               # Generative Diffusion & VAE benchmark trials (2,600 rows)
    ├── regen_summary_v4.csv          # Generative diffusion performance summary
    ├── stats_all208_paired_v4.csv    # Paired adaptive vs. fixed statistics on 208 images
    ├── stats_interactions_v4.csv     # MixedLM LRT (reml=False) & 10,000-permutation test
    ├── stats_hl_vs_ll_v4.csv         # HL vs. LL effect sizes & Cliff's delta per attack
    ├── stats_ssim_matched_paired_v4.csv # SSIM-matched paired comparisons
    ├── baselines_summary_v4.csv      # Comparative baselines summary across subbands
    ├── baselines_rotation_summary_v4.csv # Detailed rotation synchronization comparison
    ├── headroom_clean_v4.csv         # Clean-image headroom distortion characterisation
    ├── saturation_v4.csv             # Saturation rate analysis across headroom bounds
    ├── figs/                         # Publication-grade vector PDF figures
    └── tables/                       # Publication-grade LaTeX booktabs tables
```

---



---

## 📄 License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for details.
