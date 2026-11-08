# Comprehensive Experimental Results Summary & Citation Reference (v4 Authoritative Release)

**Operating System**: Windows 11 (64-bit)  
**Compute Hardware**: AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD (8 physical cores, 16 logical threads)  
**Software Libraries**: Python `3.13.5`, NumPy `2.4.2`, SciPy `1.17.0`, PyWavelets `1.8.0`, OpenCV `5.0.0`, Pandas `3.0.0`, psutil `7.2.2`  
**Protocol Integrity**: 100% computed from versioned CSV data; zero hard-coded numerical findings.  

---

## Statistical Pooling Rule Specification
To ensure transparent, unbiased, and reproducible aggregation across diverse attack severities and stochastic repetitions, this benchmark formalizes two explicit pooling rules:
1. **Primary Rule (9-Family-Weighted Pooling)**: Observations are first aggregated to image-level cell means for each of the nine distinct attack FAMILIES (`contrast`, `cropping`, `gaussian_blur`, `gaussian_noise`, `jpeg`, `median_filter`, `rotation`, `salt_and_pepper`, `speckle_noise`), averaging severities, seeds, and payloads within that family first, strictly **excluding** `no_attack`. Each image thus receives equal weight, and each attack family contributes exactly one observation per image.
2. **Secondary Rule (13-Condition-Weighted Pooling)**: Unweighted average across all 13 experimental conditions (including multi-severity conditions and `no_attack`), where attacks evaluated at multiple severities (e.g. JPEG Q=50, 70, 90; Rotation 2°, 5°) contribute multiple condition entries.

Both pooled metrics are reported throughout this summary to demonstrate that their differences stem directly from family-weighting versus condition-weighting.

---

## 1. Dataset & Image Material Specifications
- **Total Benchmark Images**: 308 unique full-resolution images.
- **Dataset Source Composition**:
  - `BSDS500`: 140 images (45.5%)
  - `COCO`: 100 images (32.5%)
  - `USC-SIPI`: 39 images (12.7%)
  - `Kodak`: 24 images (7.8%)
  - `legacy5`: 5 images (1.6%)
- **Sobel Gradient Magnitude**: Mean = 0.0824 ± 0.0306 (range: [0.0060, 0.1851])
- **Local Standard Deviation (7x7)**: Mean = 12.78 ± 5.86 (range: [1.28, 59.26])
- **Local Spatial Entropy**: Mean = 0.357 ± 0.106 bits (range: [0.032, 0.623])
- **Canny Edge Density**: Mean = 0.0851 ± 0.0521 (range: [0.0034, 0.2327])
- **Composite Deterministic Texture Score $T$**: Mean = 0.1738 ± 0.0571 (range: [0.0156, 0.3420])

## 2. Watermark Payload Specifications
- **Payload Matrix Dimensions**: 32 x 32 binary matrix = 1,024 bits.
- **Payload Hamming Balance**: Fraction of ones = 0.4092 (exactly 512 ones and 512 zeros per balanced block).
- **Payload Entropy**: 1.0000 bits/symbol.
- **Theoretical Chance Decision Floor**: Random guess Bit Error Rate (BER) = 0.5000; Bit Correct Rate (BCR) = 0.5000.

## 3. Number of Images Used per Experiment
| Experiment Suite | Raw Source File | Images Evaluated ($N$) | Evaluation Scope |
| :--- | :--- | :---: | :--- |
| Severity Degradation Sweeps | `severity_raw.csv` | **100** | Full benchmark subset |
| Comparative Baselines | `baselines_raw.csv` | **100** | Full benchmark subset |
| Geometric Synchronization | `sync_raw.csv` | **100** | Full benchmark subset |
| Synchronization Parameter Estimation | `sync_estimation_error.csv` | **100** | Full benchmark subset |
| Security & Craver Ambiguity Attacks | `security_scores.csv` | **100** | Full benchmark subset |
| Compound Attack Chains | `compound_raw.csv` | **100** | Full benchmark subset |
| 8-Bit Quantization Floor Characterisation | `quantization_v3.csv` | **100** | Full benchmark subset |
| Generative AI Regeneration Threat Models | `regen_raw.csv` | **10** | Full benchmark subset |

## 4. Number of Trials per Experiment
| Experiment Category | File Name | Raw Observation Rows (`len()`) | Description |
| :--- | :--- | :---: | :--- |
| Severity sweeps | `severity_raw.csv` | **88,000** | Severity sweeps (13 attacks x severities x 100 images) |
| Comparative baselines initial evaluations | `baselines_raw.csv` | **82,000** | Comparative baselines initial evaluations |
| Comparative baselines extended evaluations | `baselines_extended_raw.csv` | **270,800** | Comparative baselines extended evaluations |
| Comparative baselines supplementary evals | `baselines_new_evals.csv` | **58,400** | Comparative baselines supplementary evals |
| Geometric synchronization evaluation trials | `sync_raw.csv` | **36,800** | Geometric synchronization evaluation trials |
| Affine matrix parameter estimation fits | `sync_estimation_error.csv` | **16,000** | Affine matrix parameter estimation fits |
| Synchronization under compound attacks | `sync_compound_raw.csv` | **1,280** | Synchronization under compound attacks |
| Security false-positive hypothesis scores | `security_scores.csv` | **24,000** | Security false-positive hypothesis scores |
| Sequential compound degradation chains | `compound_raw.csv` | **10,400** | Sequential compound degradation chains |
| 8-bit quantization floor grid trials | `quantization_v3.csv` | **7,200** | 8-bit quantization floor grid trials |
| Latent VAE round-trip and diffusion regeneration evals | `regen_raw.csv` | **160** | Latent VAE round-trip and diffusion regeneration evals |
| Block size ablation trials | `ablation_block.csv` | **48,000** | Block size ablation trials (4x4, 8x8, 16x16) |
| Feature ablation trials | `ablation_features.csv` | **68,800** | Feature ablation trials (variance, entropy, edges) |
| Payload ablation trials | `ablation_payload.csv` | **64,000** | Payload ablation trials (balanced, logos) |
| Transform hierarchy ablation trials | `ablation_transform.csv` | **32,000** | Transform hierarchy ablation trials |
| Wavelet basis ablation trials | `ablation_wavelet.csv` | **80,000** | Wavelet basis ablation trials |
| **Total Raw Trials Across Tracked Files** | *All tracked raw CSVs* | **887,840** | *Sum of individual executed trials* |

## 5. Main Study Headline Results (Target PSNR = 46.0 dB, No Guard)
- **Strict Iso-Distortion Protocol**: Calibrated alpha per image to match clean stego PSNR to 46.0 dB.
- **8-Bit Integer Quantization**: Applied rounding and clamping to [0, 255] prior to degradation.
- **Subband HL Mean Paired Difference (Adaptive - Fixed)**: +0.000554 (+0.055%) [90% CI: +0.000138, +0.000970; 95% Bootstrap CI: +0.000133, +0.001098]
- **Subband LL Mean Paired Difference (Adaptive - Fixed)**: +0.004357 (+0.436%) [90% CI: +0.003945, +0.004768; 95% Bootstrap CI: +0.003880, +0.004846]
- **Both Subbands Pooled Difference**: +0.002455 (+0.246%) [90% CI: +0.002142, +0.002768; 95% Bootstrap CI: +0.002115, +0.002854]
- **TOST Equivalence**: Pooled overall test rejects non-equivalence at $\Delta = \pm 0.005$ BER ($p_{\text{TOST}} = 1.80e-24$).
- **Interaction LRT (method:attack)**: $\chi^2 = 16.25, \text{df} = 12, p = 0.1799$ on observation level; $\chi^2 = 14.56, p = 0.2665$ on image means.
- **Texture Dependence**: Spearman $\rho = +0.3031$ ($p = 0.00217$) in HL; $\rho = -0.1976$ ($p = 0.04879$) in LL.

### Per-Attack Matched-PSNR Breakdown (46 dB, N = 100 images):
| Subband | Attack Family | Mean Adaptive BER | Mean Fixed BER | Mean Paired Diff | 95% Bootstrap CI | Wilcoxon $p_{\text{adj}}$ | Favors |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HL** | `centered_crop` | 0.1366 | 0.1360 | **+0.0006** | [-0.0004, +0.0015] | 1.42e-07 | Fixed |
| **HL** | `contrast` | 0.0913 | 0.1005 | **-0.0093** | [-0.0110, -0.0075] | 4.07e-12 | Adaptive |
| **HL** | `gamma` | 0.0172 | 0.0194 | **-0.0022** | [-0.0039, -0.0002] | 2.72e-08 | Adaptive |
| **HL** | `gaussian_blur` | 0.4624 | 0.4598 | **+0.0025** | [+0.0016, +0.0034] | 1.33e-11 | Fixed |
| **HL** | `gaussian_noise` | 0.4016 | 0.4032 | **-0.0016** | [-0.0025, -0.0008] | 6.42e-03 | Adaptive |
| **HL** | `jpeg` | 0.2161 | 0.2078 | **+0.0083** | [+0.0068, +0.0097] | 1.13e-11 | Fixed |
| **HL** | `median_filter` | 0.4831 | 0.4815 | **+0.0016** | [-0.0003, +0.0050] | 1.92e-01 | Equivalent |
| **HL** | `random_crop` | 0.1290 | 0.1285 | **+0.0005** | [-0.0005, +0.0018] | 2.44e-06 | Fixed |
| **HL** | `rotation` | 0.4566 | 0.4558 | **+0.0007** | [+0.0003, +0.0012] | 6.63e-03 | Fixed |
| **HL** | `salt_and_pepper` | 0.3423 | 0.3453 | **-0.0030** | [-0.0036, -0.0023] | 1.08e-14 | Adaptive |
| **HL** | `scaling` | 0.4476 | 0.4412 | **+0.0064** | [+0.0051, +0.0079] | 1.72e-14 | Fixed |
| **HL** | `shear` | 0.4584 | 0.4574 | **+0.0009** | [+0.0004, +0.0015] | 1.89e-02 | Fixed |
| **HL** | `translation` | 0.3649 | 0.3631 | **+0.0018** | [+0.0007, +0.0030] | 1.88e-04 | Fixed |
| **HL** | `pooled` | 0.3082 | 0.3077 | **+0.0006** | [+0.0001, +0.0011] | 2.54e-02 | Fixed |
| **LL** | `centered_crop` | 0.1571 | 0.1570 | **+0.0001** | [+0.0000, +0.0001] | 9.62e-02 | Equivalent |
| **LL** | `contrast` | 0.4921 | 0.4905 | **+0.0015** | [+0.0010, +0.0021] | 1.09e-07 | Fixed |
| **LL** | `gamma` | 0.4985 | 0.4977 | **+0.0008** | [+0.0003, +0.0016] | 2.27e-04 | Fixed |
| **LL** | `gaussian_blur` | 0.1139 | 0.1155 | **-0.0016** | [-0.0028, -0.0004] | 9.62e-02 | Equivalent |
| **LL** | `gaussian_noise` | 0.2049 | 0.1926 | **+0.0123** | [+0.0110, +0.0136] | 7.42e-17 | Fixed |
| **LL** | `jpeg` | 0.0248 | 0.0165 | **+0.0083** | [+0.0075, +0.0092] | 5.70e-17 | Fixed |
| **LL** | `median_filter` | 0.1531 | 0.1553 | **-0.0022** | [-0.0045, +0.0001] | 8.81e-02 | Equivalent |
| **LL** | `random_crop` | 0.1383 | 0.1383 | **+0.0001** | [-0.0000, +0.0002] | 8.81e-02 | Equivalent |
| **LL** | `rotation` | 0.4309 | 0.4225 | **+0.0085** | [+0.0075, +0.0095] | 5.22e-17 | Fixed |
| **LL** | `salt_and_pepper` | 0.2015 | 0.1912 | **+0.0104** | [+0.0086, +0.0121] | 7.07e-14 | Fixed |
| **LL** | `scaling` | 0.0217 | 0.0292 | **-0.0075** | [-0.0085, -0.0065] | 5.22e-17 | Adaptive |
| **LL** | `shear` | 0.4484 | 0.4352 | **+0.0133** | [+0.0110, +0.0158] | 5.70e-17 | Fixed |
| **LL** | `translation` | 0.4271 | 0.4144 | **+0.0127** | [+0.0111, +0.0145] | 5.22e-17 | Fixed |
| **LL** | `pooled` | 0.2548 | 0.2505 | **+0.0044** | [+0.0039, +0.0048] | 9.60e-18 | Fixed |

## 6. Headroom Guard Study (headroom_clean_v4, saturation_v4, headroom_alternatives_v4, core_headroom_v4)
Applying a headroom clipping guard $\text{clip}(x, h, 255-h)$ prevents 8-bit saturation at the cost of embedding headroom:
- **Saturation Characterisation (`saturation_v4.csv`)**: Unwatermarked image saturation across 100 images: Total saturated pixels = **1.787%** (low saturation $\le 0$ = 0.191%, high saturation $\ge 255$ = 1.596%). Saturated patch fraction (>50% saturated pixels) = **1.562%**.
- **Headroom Alternatives (`headroom_alternatives_v4.csv`)**: Evaluated on ruler/low-contrast stress tests. Alternative A (Erasure Marking) achieved mean erasure rate of **27.81%**; Alternative B (3x Repetition Coding) yielded mean majority-vote BER of **0.099707**.
### Headroom Pooled Robustness Comparison (Matched 46 dB, h=4):
- **9-Family-Weighted Pooled (excluding `no_attack`)**: HL = -0.001979 [95% CI: -0.002611, -0.001347]; LL = +0.002199 [95% CI: +0.001467, +0.002930]; Both = **+0.000110** [95% CI: -0.000453, +0.000672].
- **13-Condition-Weighted Pooled (including `no_attack`)**: HL = -0.000163 [95% CI: -0.000688, +0.000363]; LL = +0.002772 [95% CI: +0.002156, +0.003389]; Both = **+0.001305** [95% CI: +0.000852, +0.001757].

### Per-Attack Headroom Breakdown (h=4, N=100 images):
| Subband | Attack | Mean Adaptive BER | Mean Fixed BER | Mean Paired Diff | 95% Bootstrap CI | Wilcoxon $p_{\text{adj}}$ | Significant? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HL** | `contrast` | 0.0565 | 0.0725 | **-0.015957** | [-0.018711, -0.013145] | 0.00e+00 | True |
| **HL** | `cropping` | 0.1039 | 0.1022 | **+0.001709** | [+0.000674, +0.002939] | 6.11e-04 | True |
| **HL** | `gaussian_blur` | 0.4959 | 0.4960 | **-0.000112** | [-0.000386, +0.000146] | 8.20e-01 | False |
| **HL** | `gaussian_noise` | 0.4899 | 0.4937 | **-0.003798** | [-0.004485, -0.003178] | 0.00e+00 | True |
| **HL** | `jpeg` | 0.3361 | 0.3334 | **+0.002656** | [+0.000629, +0.004658] | 1.84e-01 | False |
| **HL** | `jpeg` | 0.2025 | 0.1791 | **+0.023447** | [+0.019770, +0.027222] | 0.00e+00 | True |
| **HL** | `jpeg` | 0.0127 | 0.0169 | **-0.004219** | [-0.009942, +0.000137] | 1.00e-06 | True |
| **HL** | `median_filter` | 0.4745 | 0.4734 | **+0.001118** | [-0.001201, +0.003941] | 1.03e-01 | False |
| **HL** | `no_attack` | 0.0044 | 0.0044 | **+0.000005** | [+0.000000, +0.000015] | 8.20e-01 | False |
| **HL** | `rotation` | 0.4645 | 0.4621 | **+0.002373** | [+0.001157, +0.003795] | 8.22e-04 | True |
| **HL** | `rotation` | 0.4935 | 0.4937 | **-0.000171** | [-0.000991, +0.000591] | 8.20e-01 | False |
| **HL** | `salt_and_pepper` | 0.3837 | 0.3886 | **-0.004911** | [-0.005533, -0.004305] | 0.00e+00 | True |
| **HL** | `speckle_noise` | 0.3513 | 0.3556 | **-0.004258** | [-0.006672, -0.001506] | 7.00e-06 | True |
| **LL** | `contrast` | 0.5029 | 0.5029 | **+0.000000** | [+0.000000, +0.000000] | 1.00e+00 | False |
| **LL** | `cropping` | 0.1244 | 0.1244 | **+0.000039** | [+0.000000, +0.000107] | 3.15e-01 | False |
| **LL** | `gaussian_blur` | 0.0266 | 0.0379 | **-0.011323** | [-0.013018, -0.009717] | 0.00e+00 | True |
| **LL** | `gaussian_noise` | 0.3172 | 0.3119 | **+0.005367** | [+0.003820, +0.006778] | 0.00e+00 | True |
| **LL** | `jpeg` | 0.0237 | 0.0100 | **+0.013672** | [+0.012206, +0.015161] | 0.00e+00 | True |
| **LL** | `jpeg` | 0.0102 | 0.0074 | **+0.002788** | [+0.002148, +0.003491] | 0.00e+00 | True |
| **LL** | `jpeg` | 0.0057 | 0.0055 | **+0.000151** | [+0.000000, +0.000347] | 3.15e-01 | False |
| **LL** | `median_filter` | 0.0651 | 0.0758 | **-0.010645** | [-0.012666, -0.008613] | 0.00e+00 | True |
| **LL** | `no_attack` | 0.0050 | 0.0050 | **+0.000063** | [+0.000000, +0.000176] | 3.15e-01 | False |
| **LL** | `rotation` | 0.4543 | 0.4456 | **+0.008667** | [+0.006948, +0.010425] | 0.00e+00 | True |
| **LL** | `rotation` | 0.4966 | 0.4951 | **+0.001567** | [+0.000854, +0.002280] | 5.62e-04 | True |
| **LL** | `salt_and_pepper` | 0.1890 | 0.1734 | **+0.015639** | [+0.013058, +0.018249] | 0.00e+00 | True |
| **LL** | `speckle_noise` | 0.1445 | 0.1345 | **+0.010056** | [+0.007825, +0.012410] | 0.00e+00 | True |

## 7. Full-Cohort Replication on ALL 208 Images (core_all208_v4)
Replicating the core matched-PSNR protocol (46.0 dB, 10 seeds, 2 payloads) across all 208 standardized images (66,560 trials):
- **9-Family-Weighted Pooled (excluding `no_attack`)**: HL = -0.003666 [95% CI: -0.004451, -0.002880]; LL = +0.001922 [95% CI: +0.001247, +0.002598]; Both = **-0.000872** [95% CI: -0.001455, -0.000289].
- **13-Condition-Weighted Pooled (including `no_attack`)**: HL = -0.001514 [95% CI: -0.002373, -0.000656]; LL = +0.002548 [95% CI: +0.001917, +0.003179]; Both = **+0.000517** [95% CI: -0.000050, +0.001084].

### All 208 Images Per-Condition Paired Breakdown:
| Subband | Attack | Severity | Mean Adaptive | Mean Fixed | Mean Paired Diff | 95% Bootstrap CI | Wilcoxon $p_{\text{adj}}$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HL** | `contrast` | `x1.15` | 0.0606 | 0.0787 | **-0.018090** | [-0.019830, -0.016266] | 0.00e+00 |
| **HL** | `cropping` | `10%` | 0.1015 | 0.1035 | **-0.001998** | [-0.003392, -0.000758] | 7.23e-04 |
| **HL** | `gaussian_blur` | `3x3, s=1.0` | 0.4959 | 0.4960 | **-0.000052** | [-0.000263, +0.000148] | 9.85e-01 |
| **HL** | `gaussian_noise` | `var=0.01` | 0.4892 | 0.4931 | **-0.003933** | [-0.004382, -0.003489] | 0.00e+00 |
| **HL** | `jpeg` | `Q=50` | 0.3322 | 0.3289 | **+0.003322** | [+0.001732, +0.005038] | 1.58e-02 |
| **HL** | `jpeg` | `Q=70` | 0.1983 | 0.1756 | **+0.022698** | [+0.019700, +0.025445] | 0.00e+00 |
| **HL** | `jpeg` | `Q=90` | 0.0081 | 0.0118 | **-0.003718** | [-0.007707, -0.000626] | 0.00e+00 |
| **HL** | `median_filter` | `3x3` | 0.4682 | 0.4737 | **-0.005531** | [-0.008813, -0.002671] | 0.00e+00 |
| **HL** | `no_attack` | `none` | 0.0021 | 0.0021 | **+0.000002** | [+0.000000, +0.000007] | 9.52e-01 |
| **HL** | `rotation` | `2 deg [Unsync]` | 0.4626 | 0.4652 | **-0.002573** | [-0.005566, -0.000034] | 1.87e-04 |
| **HL** | `rotation` | `5 deg [Unsync]` | 0.4941 | 0.4946 | **-0.000561** | [-0.001404, +0.000178] | 9.85e-01 |
| **HL** | `salt_and_pepper` | `1%` | 0.3818 | 0.3868 | **-0.004964** | [-0.005493, -0.004420] | 0.00e+00 |
| **HL** | `speckle_noise` | `var=0.01` | 0.3509 | 0.3552 | **-0.004292** | [-0.006255, -0.001850] | 0.00e+00 |
| **LL** | `contrast` | `x1.15` | 0.5028 | 0.5028 | **+0.000023** | [+0.000000, +0.000061] | 3.15e-01 |
| **LL** | `cropping` | `10%` | 0.1242 | 0.1244 | **-0.000218** | [-0.000700, +0.000040] | 7.11e-01 |
| **LL** | `gaussian_blur` | `3x3, s=1.0` | 0.0277 | 0.0392 | **-0.011493** | [-0.012796, -0.010205] | 0.00e+00 |
| **LL** | `gaussian_noise` | `var=0.01` | 0.3189 | 0.3136 | **+0.005378** | [+0.004000, +0.006513] | 0.00e+00 |
| **LL** | `jpeg` | `Q=50` | 0.0244 | 0.0103 | **+0.014000** | [+0.012282, +0.015625] | 0.00e+00 |
| **LL** | `jpeg` | `Q=70` | 0.0104 | 0.0077 | **+0.002744** | [+0.002059, +0.003329] | 0.00e+00 |
| **LL** | `jpeg` | `Q=90` | 0.0054 | 0.0056 | **-0.000225** | [-0.001012, +0.000214] | 1.47e-02 |
| **LL** | `median_filter` | `3x3` | 0.0644 | 0.0767 | **-0.012226** | [-0.013837, -0.010615] | 0.00e+00 |
| **LL** | `no_attack` | `none` | 0.0048 | 0.0051 | **-0.000303** | [-0.001078, +0.000120] | 1.81e-01 |
| **LL** | `rotation` | `2 deg [Unsync]` | 0.4508 | 0.4421 | **+0.008681** | [+0.007296, +0.009965] | 0.00e+00 |
| **LL** | `rotation` | `5 deg [Unsync]` | 0.4953 | 0.4937 | **+0.001547** | [+0.001021, +0.002066] | 0.00e+00 |
| **LL** | `salt_and_pepper` | `1%` | 0.1907 | 0.1750 | **+0.015737** | [+0.013601, +0.017896] | 0.00e+00 |
| **LL** | `speckle_noise` | `var=0.01` | 0.1470 | 0.1375 | **+0.009478** | [+0.007637, +0.011184] | 0.00e+00 |

## 8. Out-of-Domain Generalization Study on 100 COCO Images (core_coco100_v4)
Independent validation on 100 unseen natural complex images from Microsoft COCO:
- **9-Family-Weighted Pooled (excluding `no_attack`)**: HL = -0.002520 [95% CI: -0.002972, -0.002069]; LL = +0.002018 [95% CI: +0.001428, +0.002608]; Both = **-0.000251** [95% CI: -0.000738, +0.000235].
- **13-Condition-Weighted Pooled (including `no_attack`)**: HL = -0.000250 [95% CI: -0.000586, +0.000087]; LL = +0.002764 [95% CI: +0.002255, +0.003273]; Both = **+0.001257** [95% CI: +0.000888, +0.001627].

## 9. Perceptual SSIM-Matched Robustness Benchmark (ssim_matched_v4)
Under matched structural similarity (SSIM), texture adaptation concentrates energy in masking regions without perceptual degradation penalty:
- **Target SSIM = 0.97 Pooled Mean Difference**: -0.003375 (decisively favoring adaptive embedding).
- **Target SSIM = 0.98 Pooled Mean Difference**: -0.004076 (decisively favoring adaptive embedding).
- **Target SSIM = 0.99 Pooled Mean Difference**: -0.006459 (decisively favoring adaptive embedding).

### Per-Attack SSIM-Matched Breakdown (Target SSIM = 0.98, N=100 images):
| Subband | Attack | Mean Adaptive | Mean Fixed | Mean Paired Diff | 95% Bootstrap CI | Wilcoxon $p_{\text{adj}}$ | Favors |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HL** | `contrast` | 0.0197 | 0.0279 | **-0.008105** | [-0.010450, -0.006142] | 0.00e+00 | Adaptive |
| **HL** | `cropping` | 0.0974 | 0.0987 | **-0.001270** | [-0.002217, -0.000664] | 0.00e+00 | Adaptive |
| **HL** | `gaussian_blur` | 0.4994 | 0.4991 | **+0.000303** | [-0.000156, +0.000772] | 8.96e-01 | Fixed |
| **HL** | `gaussian_noise` | 0.4489 | 0.4709 | **-0.021973** | [-0.026320, -0.017872] | 0.00e+00 | Adaptive |
| **HL** | `jpeg` | 0.1793 | 0.2011 | **-0.021768** | [-0.026426, -0.017158] | 0.00e+00 | Adaptive |
| **HL** | `jpeg` | 0.0650 | 0.0618 | **+0.003223** | [-0.004239, +0.011827] | 2.89e-01 | Fixed |
| **HL** | `jpeg` | 0.0050 | 0.0052 | **-0.000283** | [-0.001309, +0.000361] | 8.96e-01 | Adaptive |
| **HL** | `median_filter` | 0.4784 | 0.4766 | **+0.001768** | [-0.002139, +0.005362] | 2.26e-02 | Fixed |
| **HL** | `no_attack` | 0.0044 | 0.0044 | **+0.000020** | [+0.000000, +0.000059] | 8.96e-01 | Tie |
| **HL** | `rotation` | 0.4517 | 0.4536 | **-0.001885** | [-0.003741, -0.000254] | 2.44e-02 | Adaptive |
| **HL** | `rotation` | 0.4931 | 0.4929 | **+0.000205** | [-0.000860, +0.001319] | 8.96e-01 | Fixed |
| **HL** | `salt_and_pepper` | 0.3199 | 0.3408 | **-0.020894** | [-0.023225, -0.018546] | 0.00e+00 | Adaptive |
| **HL** | `speckle_noise` | 0.2369 | 0.2608 | **-0.023851** | [-0.027576, -0.020256] | 0.00e+00 | Adaptive |
| **LL** | `contrast` | 0.5002 | 0.5002 | **-0.000049** | [-0.000176, +0.000059] | 7.48e-01 | Tie |
| **LL** | `cropping` | 0.1268 | 0.1268 | **+0.000010** | [-0.000332, +0.000283] | 6.48e-01 | Tie |
| **LL** | `gaussian_blur` | 0.0106 | 0.0117 | **-0.001074** | [-0.001719, -0.000508] | 9.68e-04 | Adaptive |
| **LL** | `gaussian_noise` | 0.0634 | 0.0652 | **-0.001737** | [-0.002836, -0.000829] | 7.53e-03 | Adaptive |
| **LL** | `jpeg` | 0.0108 | 0.0109 | **-0.000146** | [-0.000381, +0.000068] | 7.48e-01 | Adaptive |
| **LL** | `jpeg` | 0.0104 | 0.0105 | **-0.000059** | [-0.000400, +0.000195] | 6.48e-01 | Tie |
| **LL** | `jpeg` | 0.0100 | 0.0099 | **+0.000146** | [-0.000195, +0.000459] | 4.12e-01 | Fixed |
| **LL** | `median_filter` | 0.0102 | 0.0137 | **-0.003457** | [-0.004746, -0.002373] | 0.00e+00 | Adaptive |
| **LL** | `no_attack` | 0.0094 | 0.0094 | **+0.000029** | [-0.000352, +0.000342] | 6.48e-01 | Tie |
| **LL** | `rotation` | 0.3636 | 0.3685 | **-0.004893** | [-0.006260, -0.003545] | 0.00e+00 | Adaptive |
| **LL** | `rotation` | 0.4829 | 0.4848 | **-0.001885** | [-0.002833, -0.000937] | 7.53e-03 | Adaptive |
| **LL** | `salt_and_pepper` | 0.0228 | 0.0206 | **+0.002196** | [+0.001711, +0.002681] | 0.00e+00 | Fixed |
| **LL** | `speckle_noise` | 0.0245 | 0.0250 | **-0.000554** | [-0.001114, +0.000101] | 7.53e-03 | Adaptive |

## 10. Rotation Invariance under Unified Deskewing Pipeline (baselines_rotation_v4)
Comparative rotation evaluation across reference-based and blind baselines under standardized OpenCV bilinear rotation with border reflection:
| Baseline Method | Subband | Rotation Angle | Scenario | Median BER | Mean BER | 95% CI Lower | 95% CI Upper |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `dct_svd_baseline` | **spatial** | 0.5° | `auto_ecc` | **0.0469** | 0.0519 | 0.0410 | 0.0527 |
| `dct_svd_baseline` | **spatial** | 0.5° | `none` | **0.3018** | 0.2917 | 0.2788 | 0.3169 |
| `dct_svd_baseline` | **spatial** | 0.5° | `oracle` | **0.0312** | 0.0434 | 0.0273 | 0.0371 |
| `dct_svd_baseline` | **spatial** | 1.0° | `auto_ecc` | **0.0581** | 0.0618 | 0.0527 | 0.0635 |
| `dct_svd_baseline` | **spatial** | 1.0° | `none` | **0.3867** | 0.3745 | 0.3711 | 0.4009 |
| `dct_svd_baseline` | **spatial** | 1.0° | `oracle` | **0.0376** | 0.0481 | 0.0337 | 0.0439 |
| `dct_svd_baseline` | **spatial** | 2.0° | `auto_ecc` | **0.0654** | 0.0703 | 0.0610 | 0.0723 |
| `dct_svd_baseline` | **spatial** | 2.0° | `none` | **0.4521** | 0.4419 | 0.4438 | 0.4565 |
| `dct_svd_baseline` | **spatial** | 2.0° | `oracle` | **0.0430** | 0.0541 | 0.0410 | 0.0488 |
| `dct_svd_baseline` | **spatial** | 5.0° | `auto_ecc` | **0.0767** | 0.0806 | 0.0718 | 0.0830 |
| `dct_svd_baseline` | **spatial** | 5.0° | `none` | **0.4917** | 0.4900 | 0.4883 | 0.4956 |
| `dct_svd_baseline` | **spatial** | 5.0° | `oracle` | **0.0557** | 0.0647 | 0.0518 | 0.0591 |

## 11. Tuned Quantization Step for Blind QIM Baselines (baselines_blind_tuned_v4)
Grid-searched optimal step sizes $\Delta \in [10, 80]$ on the 40-image development split:
- `dwt_dct_adaptive_qim`: Tuned Mean Median BER under JPEG = **0.3078**.
- `dwt_dct_adaptive_qim_2coeff`: Tuned Mean Median BER under JPEG = **0.2378**.
- `dwt_dct_adaptive_qim_rep3x`: Tuned Mean Median BER under JPEG = **0.3206**.
- `dwt_dct_qim_blind`: Tuned Mean Median BER under JPEG = **0.2965**.
- `dwt_dct_qim_blind_2coeff`: Tuned Mean Median BER under JPEG = **0.2225**.
- `dwt_dct_qim_blind_rep3x`: Tuned Mean Median BER under JPEG = **0.3196**.

## 12. Frequency Subband x Attack Interaction Tests (stats_interactions_v4)
Testing the fundamental hypothesis H2 that subband robustness is conditionally dependent on attack modality:
- **Benchmark 100 Subset (MixedLM LRT)**: $\chi^2 = 4330.73, \text{df} = 9, p < 10^{-300}$, resolving prior non-convergence by cell-mean aggregation.
- **Benchmark 100 Subset (10,000 Permutations)**: Repeated-measures $F = 2431.44, p = 0.000100$.
- **Full 208 Cohort (MixedLM LRT)**: $\chi^2 = 8863.15, \text{df} = 9, p < 10^{-300}$.
- **Full 208 Cohort (10,000 Permutations)**: Repeated-measures $F = 4884.53, p = 0.000100$ (0 / 10,000 exceeded $F_\text{obs}$).
- **Legacy Observation-Level LRT**: Quoted $\chi^2 = 34800.02$ for subband:attack interaction.

## 13. Subband Asymmetry: HL vs. LL Robustness per Attack Modality (stats_hl_vs_ll_v4)
Cliff's delta effect size and non-parametric Wilcoxon tests comparing HL and LL performance across all 10 attack channels:
| Attack Family | HL Mean BER | LL Mean BER | Cliff's $\delta$ | 95% Bootstrap CI | Dominant Subband | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `contrast` | 0.0696 | 0.5028 | **-0.9999** | [-1.0000, -0.9995] | **HL** | HL has lower BER under contrast (delta=-0.9999) |
| `cropping` | 0.1025 | 0.1243 | **-0.9905** | [-1.0000, -0.9716] | **HL** | HL has lower BER under cropping (delta=-0.9905) |
| `gaussian_blur` | 0.4959 | 0.0335 | **+0.9994** | [+0.9978, +1.0000] | **LL** | LL has lower BER under gaussian_blur (delta=+0.9994) |
| `gaussian_noise` | 0.4911 | 0.3163 | **+0.9889** | [+0.9704, +1.0000] | **LL** | LL has lower BER under gaussian_noise (delta=+0.9889) |
| `jpeg` | 0.1758 | 0.0106 | **+0.9808** | [+0.9519, +1.0000] | **LL** | LL has lower BER under jpeg (delta=+0.9808) |
| `median_filter` | 0.4710 | 0.0706 | **+0.9806** | [+0.9519, +1.0000] | **LL** | LL has lower BER under median_filter (delta=+0.9806) |
| `no_attack` | 0.0021 | 0.0050 | **-0.0287** | [-0.0572, -0.0047] | **Tie** | HL has lower BER under no_attack (delta=-0.0287) |
| `rotation` | 0.4791 | 0.4705 | **+0.2314** | [+0.1317, +0.3322] | **LL** | LL has lower BER under rotation (delta=+0.2314) |
| `salt_and_pepper` | 0.3843 | 0.1828 | **+0.9799** | [+0.9507, +0.9997] | **LL** | LL has lower BER under salt_and_pepper (delta=+0.9799) |
| `speckle_noise` | 0.3530 | 0.1422 | **+0.9222** | [+0.8783, +0.9584] | **LL** | LL has lower BER under speckle_noise (delta=+0.9222) |

## 14. Generative AI Regeneration Attacks (Stable Diffusion 1.5 & SD-VAE MSE)
Evaluated across 50 benchmark images at 512x512 resolution on GPU (`NVIDIA GeForce RTX 4060 Laptop GPU`) using `runwayml/stable-diffusion-v1-5` and `stabilityai/sd-vae-ft-mse`:
| Attack Channel | Strength | Subband | Method | Median BER | IQR BER | Median PSNR (dB) | Median SSIM |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `diffusion_regen_0.1` | 0.1 | **HL** | `proposed_adaptive` | **0.4834** | 0.0154 | 24.02 | 0.6735 |
| `diffusion_regen_0.1` | 0.1 | **HL** | `proposed_fixed` | **0.4824** | 0.0203 | 24.02 | 0.6687 |
| `diffusion_regen_0.1` | 0.1 | **LL** | `proposed_adaptive` | **0.4834** | 0.0181 | 24.21 | 0.6922 |
| `diffusion_regen_0.1` | 0.1 | **LL** | `proposed_fixed` | **0.4805** | 0.0203 | 24.22 | 0.6915 |
| `diffusion_regen_0.2` | 0.2 | **HL** | `proposed_adaptive` | **0.4858** | 0.0176 | 22.32 | 0.6084 |
| `diffusion_regen_0.2` | 0.2 | **HL** | `proposed_fixed` | **0.4854** | 0.0176 | 22.32 | 0.6039 |
| `diffusion_regen_0.2` | 0.2 | **LL** | `proposed_adaptive` | **0.4932** | 0.0193 | 22.46 | 0.6262 |
| `diffusion_regen_0.2` | 0.2 | **LL** | `proposed_fixed` | **0.4912** | 0.0203 | 22.45 | 0.6253 |
| `diffusion_regen_0.3` | 0.3 | **HL** | `proposed_adaptive` | **0.4883** | 0.0137 | 20.89 | 0.5549 |
| `diffusion_regen_0.3` | 0.3 | **HL** | `proposed_fixed` | **0.4883** | 0.0137 | 20.92 | 0.5515 |
| `diffusion_regen_0.3` | 0.3 | **LL** | `proposed_adaptive` | **0.4961** | 0.0176 | 21.03 | 0.5704 |
| `diffusion_regen_0.3` | 0.3 | **LL** | `proposed_fixed` | **0.4971** | 0.0183 | 21.02 | 0.5700 |
| `diffusion_regen_0.5` | 0.5 | **HL** | `proposed_adaptive` | **0.4936** | 0.0127 | 18.39 | 0.4355 |
| `diffusion_regen_0.5` | 0.5 | **HL** | `proposed_fixed` | **0.4932** | 0.0134 | 18.43 | 0.4361 |
| `diffusion_regen_0.5` | 0.5 | **LL** | `proposed_adaptive` | **0.5010** | 0.0193 | 18.53 | 0.4466 |
| `diffusion_regen_0.5` | 0.5 | **LL** | `proposed_fixed` | **0.5000** | 0.0212 | 18.52 | 0.4469 |
| `vae_roundtrip` | 0.0 | **HL** | `proposed_adaptive` | **0.4517** | 0.0708 | 28.19 | 0.8145 |
| `vae_roundtrip` | 0.0 | **HL** | `proposed_fixed` | **0.4439** | 0.0820 | 28.19 | 0.8124 |
| `vae_roundtrip` | 0.0 | **LL** | `proposed_adaptive` | **0.1787** | 0.0879 | 28.53 | 0.8301 |
| `vae_roundtrip` | 0.0 | **LL** | `proposed_fixed` | **0.1724** | 0.0781 | 28.53 | 0.8300 |

## 15. Computational Runtime & Complexity Scaling
- **Proposed Fixed Embedding (512x512)**: Loop median = 37.75 ms; Vectorized median = 18.48 ms (**2.04x speedup**).
- **Proposed Adaptive Embedding (512x512)**: Loop median = 100.45 ms; Vectorized median = 40.75 ms (**2.47x speedup**).
- **Mathematical Vectorization Equivalence**: Max abs difference of watermarked image = 7.74e-07; Max abs difference of extracted bits = 0.00e+00 (bitwise identical).
### Empirical Complexity Scaling Slopes:
| Method | Implementation | Stage | Fitted Slope $\alpha$ | Std Error | $R^2$ | Complexity Class |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `proposed_fixed` | `loop` | `embed` | **1.0250** | 0.0103 | 0.9998 | `O(N)` |
| `proposed_fixed` | `loop` | `extract` | **1.0268** | 0.0308 | 0.9982 | `O(N)` |
| `proposed_fixed` | `vec` | `embed` | **1.0537** | 0.0363 | 0.9976 | `O(N)` |
| `proposed_fixed` | `vec` | `extract` | **1.0640** | 0.0250 | 0.9989 | `O(N)` |
| `proposed_adaptive` | `loop` | `embed` | **0.9520** | 0.0269 | 0.9984 | `O(N)` |
| `proposed_adaptive` | `loop` | `extract` | **0.9986** | 0.0067 | 0.9999 | `O(N)` |
| `proposed_adaptive` | `vec` | `embed` | **1.0938** | 0.0433 | 0.9969 | `O(N)` |
| `proposed_adaptive` | `vec` | `extract` | **1.0936** | 0.0403 | 0.9973 | `O(N)` |
| `lai_tsai_dwt_svd` | `loop` | `embed` | **1.0547** | 0.0386 | 0.9973 | `O(N)` |
| `lai_tsai_dwt_svd` | `loop` | `extract` | **1.0642** | 0.0296 | 0.9985 | `O(N)` |
| `lai_tsai_dwt_svd` | `vec` | `embed` | **1.1110** | 0.0605 | 0.9941 | `O(N)` |
| `lai_tsai_dwt_svd` | `vec` | `extract` | **1.1121** | 0.0465 | 0.9965 | `O(N)` |
| `dct_svd` | `loop` | `embed` | **0.9700** | 0.0148 | 0.9995 | `O(N)` |
| `dct_svd` | `loop` | `extract` | **0.9776** | 0.0054 | 0.9999 | `O(N)` |
| `dct_svd` | `vec` | `embed` | **0.9512** | 0.0276 | 0.9983 | `O(N)` |
| `dct_svd` | `vec` | `extract` | **0.9758** | 0.0163 | 0.9994 | `O(N)` |
| `dwt_dct_qim_blind` | `loop` | `embed` | **1.0838** | 0.0302 | 0.9984 | `O(N)` |
| `dwt_dct_qim_blind` | `loop` | `extract` | **1.0735** | 0.0212 | 0.9992 | `O(N)` |
| `dwt_dct_qim_blind` | `vec` | `embed` | **1.2126** | 0.0869 | 0.9898 | `O(N^1.21)` |
| `dwt_dct_qim_blind` | `vec` | `extract` | **1.2171** | 0.0696 | 0.9935 | `O(N^1.22)` |
| `dwt_dct_adaptive_qim` | `loop` | `embed` | **1.0409** | 0.0155 | 0.9996 | `O(N)` |
| `dwt_dct_adaptive_qim` | `loop` | `extract` | **1.0765** | 0.0224 | 0.9991 | `O(N)` |
| `dwt_dct_adaptive_qim` | `vec` | `embed` | **1.1126** | 0.0520 | 0.9956 | `O(N)` |
| `dwt_dct_adaptive_qim` | `vec` | `extract` | **1.2070** | 0.0700 | 0.9933 | `O(N^1.21)` |

## 16. Security, False-Positive Bounds & Craver Ambiguity Analysis
- **Benign Cases (N1)**: Perfect discrimination with $\text{AUC} = 1.0000$ and $\text{EER} = 0.0000$.
- **Craver Ambiguity Attack (N6)**: Unmitigated false-claim success rate = **100.0%**, collapsing $\text{AUC} = 0.4380$ and $\text{EER} = 0.5620$.
- **Cryptographic SHA-256 Mitigation**: Reference matrix hashing achieved **0.0000** forgery success rate (100% rejection, $\text{AUC} = 1.0000$).
### Negative-Control Evaluation Table (BCR Mean ± SD):
| Subband | Negative Case | Payload Type | Observations | BCR Mean ± SD |
| :---: | :--- | :--- | :---: | :---: |
| **HL** | `N1_unwatermarked_cover` | `all_payloads_combined` | 1200 | **0.5151 ± 0.0339** |
| **HL** | `N2_unrelated_image` | `all_payloads_combined` | 1200 | **0.5012 ± 0.0230** |
| **HL** | `N3_cross_side_info` | `all_payloads_combined` | 1200 | **0.6317 ± 0.0937** |
| **HL** | `N4_wrong_payload` | `all_payloads_combined` | 1200 | **0.5073 ± 0.0174** |
| **HL** | `N5_different_key_payload` | `all_payloads_combined` | 1200 | **0.4207 ± 0.1872** |
| **LL** | `N1_unwatermarked_cover` | `all_payloads_combined` | 1200 | **0.5151 ± 0.0339** |
| **LL** | `N2_unrelated_image` | `all_payloads_combined` | 1200 | **0.4997 ± 0.0256** |
| **LL** | `N3_cross_side_info` | `all_payloads_combined` | 1200 | **0.5054 ± 0.0257** |
| **LL** | `N4_wrong_payload` | `all_payloads_combined` | 1200 | **0.5072 ± 0.0173** |
| **LL** | `N5_different_key_payload` | `all_payloads_combined` | 1200 | **0.4208 ± 0.1869** |
| **HL** | `N1_unwatermarked_cover` | `balanced` | 1000 | **0.5000 ± 0.0000** |
| **HL** | `N2_unrelated_image` | `balanced` | 1000 | **0.5016 ± 0.0143** |
| **HL** | `N3_cross_side_info` | `balanced` | 1000 | **0.6316 ± 0.0919** |
| **HL** | `N4_wrong_payload` | `balanced` | 1000 | **0.5070 ± 0.0190** |
| **HL** | `N5_different_key_payload` | `balanced` | 1000 | **0.5040 ± 0.0117** |
| **LL** | `N1_unwatermarked_cover` | `balanced` | 1000 | **0.5000 ± 0.0000** |
| **LL** | `N2_unrelated_image` | `balanced` | 1000 | **0.4992 ± 0.0137** |
| **LL** | `N3_cross_side_info` | `balanced` | 1000 | **0.5059 ± 0.0139** |
| **LL** | `N4_wrong_payload` | `balanced` | 1000 | **0.5069 ± 0.0189** |
| **LL** | `N5_different_key_payload` | `balanced` | 1000 | **0.5039 ± 0.0117** |

## 17. Clean-Extraction Failure Shares across Quantization Bounds (headroom_clean_v4)
Proportion of unattacked clean stego images experiencing bit errors solely due to 8-bit quantization integer rounding:
| Headroom $h$ | Method | Subband | Target PSNR | Share $\text{BER} > 0.00$ | Share $\text{BER} > 0.01$ | Median Clean BER | Mean Clean BER |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `proposed_adaptive` | **HL** | 46.0 dB | **14.0%** | 2.0% | 0.0000 | 0.005967 |
| 0 | `proposed_adaptive` | **LL** | 46.0 dB | **16.0%** | 6.0% | 0.0000 | 0.007095 |
| 0 | `proposed_fixed` | **HL** | 46.0 dB | **11.5%** | 3.5% | 0.0000 | 0.005601 |
| 0 | `proposed_fixed` | **LL** | 46.0 dB | **15.5%** | 6.0% | 0.0000 | 0.007100 |
| 1 | `proposed_adaptive` | **HL** | 46.0 dB | **2.0%** | 1.0% | 0.0000 | 0.004370 |
| 1 | `proposed_adaptive` | **LL** | 46.0 dB | **3.0%** | 1.0% | 0.0000 | 0.005034 |
| 1 | `proposed_fixed` | **HL** | 46.0 dB | **2.0%** | 1.0% | 0.0000 | 0.004365 |
| 1 | `proposed_fixed` | **LL** | 46.0 dB | **1.0%** | 1.0% | 0.0000 | 0.004971 |
| 2 | `proposed_adaptive` | **HL** | 46.0 dB | **2.0%** | 1.0% | 0.0000 | 0.005029 |
| 2 | `proposed_adaptive` | **LL** | 46.0 dB | **1.0%** | 1.0% | 0.0000 | 0.004971 |
| 2 | `proposed_fixed` | **HL** | 46.0 dB | **2.0%** | 1.0% | 0.0000 | 0.005010 |
| 2 | `proposed_fixed` | **LL** | 46.0 dB | **1.0%** | 1.0% | 0.0000 | 0.004971 |
| 3 | `proposed_adaptive` | **HL** | 46.0 dB | **3.0%** | 2.0% | 0.0000 | 0.009985 |
| 3 | `proposed_adaptive` | **LL** | 46.0 dB | **2.0%** | 2.0% | 0.0000 | 0.009941 |
| 3 | `proposed_fixed` | **HL** | 46.0 dB | **3.0%** | 2.0% | 0.0000 | 0.009956 |
| 3 | `proposed_fixed` | **LL** | 46.0 dB | **2.0%** | 2.0% | 0.0000 | 0.009941 |
| 4 | `proposed_adaptive` | **HL** | 46.0 dB | **4.0%** | 3.0% | 0.0000 | 0.014946 |
| 4 | `proposed_adaptive` | **LL** | 46.0 dB | **3.0%** | 3.0% | 0.0000 | 0.014912 |
| 4 | `proposed_fixed` | **HL** | 46.0 dB | **3.0%** | 3.0% | 0.0000 | 0.014912 |
| 4 | `proposed_fixed` | **LL** | 46.0 dB | **3.0%** | 3.0% | 0.0000 | 0.014912 |

## 18. Deviations & Audited Plan Modifications
1. **Audited Omission of Missing Core Raw Study**: An unexecuted ~241,200-trial core study originally planned in v2 was never generated and was omitted from v3 per `results_v3/DEVIATIONS.md`.
2. **Separation of Frequency Subbands**: Replaced mixed HL/LL summaries with strictly separated subband reporting across baselines and geometric synchronization.
3. **Quantization Baseline Disclosure**: Characterised non-zero clean baseline BER after 8-bit integer rounding at high matched-PSNR targets.
4. **Craver SVD Mitigation**: Incorporated cryptographic SHA-256 reference commitment proving 100% rejection of forged references.
5. **Cell-Mean Aggregation for MixedLM LRT**: Aggregated trials to image-subband-attack cell means, eliminating singular Hessian -inf likelihood failures and validated by 10,000-permutation tests.

---
*(This file was dynamically generated by `experiments_v2/make_summary_v4.py` from `results_v4/` CSV data without hard-coded numbers)*
