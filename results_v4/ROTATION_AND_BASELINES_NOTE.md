# Methodological Analysis: Rotation Discrepancies and Baseline Equivalence

**Generated:** 2026-11-02  
**Associated Datasets:** `baselines_rotation_v4.csv` (16,500 trials), `baselines_rotation_summary_v4.csv` (165 conditions).

---

## 1. Discrepancies in Rotation Application & Extraction Across Codebases

A comprehensive audit of `experiments_v2/baselines.py`, `experiments_v2/study.py`, `experiments_v2/attacks.py`, and `watermark/core.py` revealed several critical structural differences in how rotation was previously applied and extracted:

| Feature / Property | `proposed_fixed` & `proposed_adaptive` (in `study.py`) | `lai_tsai_dwt_svd` & `dct_svd_baseline` (in `baselines.py`) | `dwt_dct_qim_blind` & `dwt_dct_adaptive_qim` | `extract_with_rotation_search` (in `watermark/core.py`) |
|:---|:---|:---|:---|:---|
| **Angle Definition & Sign** | Counter-clockwise ($+\theta$) around center `(w/2, h/2)` | Counter-clockwise ($+\theta$) around center `(w/2, h/2)` | Counter-clockwise ($+\theta$) around center `(w/2, h/2)` | Counter-clockwise ($+\theta$) around center `(w/2, h/2)` |
| **Attack Border Mode** | `cv2.BORDER_CONSTANT` (0.0 black borders) | `cv2.BORDER_CONSTANT` (0.0 black borders) | `cv2.BORDER_CONSTANT` (0.0 black borders) | `cv2.BORDER_CONSTANT` (0.0 black borders) |
| **Deskew Border Mode** | N/A (no deskew was called in `study.py`) | `cv2.BORDER_REFLECT` (reflects boundary pixels) | `cv2.BORDER_REFLECT` (reflects boundary pixels) | `cv2.BORDER_REFLECT` (reflects boundary pixels) |
| **Deskew / Registration Applied?** | **None** (`deskew_angle=None`, unsynchronized) | **Oracle Ground Truth** (`deskew_angle = 2.0 / 5.0`) in `[Sync]`; none in `[Unsync]` | **Oracle Ground Truth** (`deskew_angle = 2.0 / 5.0`) in `[Sync]`; none in `[Unsync]` | Blind brute-force grid search over $[-10^\circ, +10^\circ]$ at $0.5^\circ$ step |
| **Use of True Secret Watermark?** | **No** (never accesses ground-truth watermark) | **No** (receives external oracle angle) | **No** (receives external oracle angle) | **YES (CHEATING ORACLE)**: Maximizes NC against `metadata.watermark_binary` |

---

## 2. Which Old Numbers Were Wrong and Why

### A. The False Baseline Advantage in `baselines_summary.csv`
In the legacy `baselines_summary.csv`, the reported rotation rows compared **incommensurate scenarios**:
- `lai_tsai_dwt_svd` (LL) and `dct_svd_baseline` were reported under `2 deg [Sync]` and `5 deg [Sync]`, where an **oracle** handed the decoder the exact inverse angle ($2.0^\circ$ and $5.0^\circ$), yielding near-zero median BER ($\approx 0.0195 - 0.0312$).
- Conversely, `proposed_fixed` and `proposed_adaptive` were only evaluated under `2 deg [Unsync]` and `5 deg [Unsync]` (zero registration), yielding median BER $\approx 0.452 - 0.489$.
- **Correction**: When evaluated under the identical scenario (see `baselines_rotation_summary_v4.csv`):
  - Under `none` (unsynchronized), `lai_tsai_dwt_svd` (LL), `proposed_fixed` (LL), and `hybrid_precedent_fixed` (LL) fail identically ($\text{BER} = 0.450684$).
  - Under `oracle` deskewing, all three achieve identical recovery ($\text{BER} = 0.040527$ at $2.0^\circ$).
  - Under honest `auto_ecc` registration, all three achieve identical recovery ($\text{BER} = 0.061035$ at $2.0^\circ$).

### B. Subband Confounding in `sync_raw.csv`
Legacy sync experiments pooled `HL` and `LL` together into single aggregate metrics. This masked the fundamental physical behavior of 2D wavelet subbands:
- In `LL` (approximation subband), geometric registration restores the watermark cleanly ($\text{BER} \approx 0.02 - 0.06$).
- In `HL` (high-frequency detail subband), interpolation round-trip loss from image rotation destroys high-frequency coefficients even under an **exact oracle** ($\text{BER} \approx 0.420 - 0.424$). Pooling HL and LL obscured this distinction.

### C. Oracle Watermark Leakage in `watermark/core.py`
The legacy `extract_with_rotation_search` method evaluated correlation against `metadata.watermark_binary` to pick the best rotation angle. Because a blind/semi-blind decoder in a production environment cannot possess the secret payload before extracting it, this constituted an invalid oracle check. Honest registration must rely on spatial image alignment (`auto_ecc` or feature-based RANSAC against the cover) without watermark leakage.

---

## 3. Mathematical and Empirical Equivalence: `lai_tsai_dwt_svd` vs `proposed_fixed`

### Theoretical Equivalence Proof
Let an $8 \times 8$ block of the DWT subband be $B \in \mathbb{R}^{8 \times 8}$.
1. **Lai-Tsai DWT-SVD**:
   Computes SVD directly on $B$:
   $$B = U_B \, S_B \, V_B^T$$
   Modifies the principal singular value: $S_{B, 0}' = S_{B, 0} + \alpha \, w$.
   Reconstructs the block: $B_{\text{lai}}' = U_B \, \text{diag}(S_B') \, V_B^T$.

2. **Proposed Fixed (Cascaded DWT-DCT-SVD)**:
   Computes 2D DCT on $B$. Since the 2D DCT is orthonormal, $D = C \, B \, C^T$ where $C \in \mathbb{R}^{8 \times 8}$ is an orthogonal matrix ($C^T C = C C^T = I$).
   Computes SVD on $D$:
   $$D = U_D \, S_D \, V_D^T$$
   Substituting $D = C \, B \, C^T$:
   $$B = C^T D C = (C^T U_D) \, S_D \, (V_D^T C)$$
   Because $C^T U_D$ and $C^T V_D$ remain orthonormal matrices, this is a valid Singular Value Decomposition of $B$. By uniqueness of singular values:
   $$S_D \equiv S_B$$
   Modifying $S_{D, 0}' = S_{D, 0} + \alpha \, w$ and inverting via IDCT:
   $$B_{\text{prop}}' = C^T (U_D \, \text{diag}(S_D') \, V_D^T) C = (C^T U_D) \, \text{diag}(S_B') \, (V_D^T C) \equiv B_{\text{lai}}'$$

**Conclusion**: For fixed embedding strength $\alpha$, the cascaded DWT-DCT-SVD scheme and the direct DWT-SVD scheme produce **mathematically identical watermarked images and identical extracted singular values**.

### Why Previous Summaries Showed Small Discrepancies ($|\Delta \text{BER}| \le 0.02$)
When tested under controlled identical settings (same image, same payload `balanced_0`, same calibrated $\alpha$), the BER difference between `lai_tsai_dwt_svd` and `proposed_fixed` is **0.000000 across all attacks**. Previous non-zero differences arose from:
1. **Payload Discrepancies**: `study.py` averaged across 2 payloads (`whitened_logo` and `balanced_0`), whereas `baselines.py` evaluated only `balanced_0`.
2. **Calibration Bisection Grids**: Separate bisection solver iterations, tolerances (`tol=0.08` vs `tol=0.10`), and bracket expansions produced slight variations in nominal $\alpha$.
3. **Stochastic Noise Seeds**: Independent random noise realizations were drawn when PRNG seed streams differed across modules.

