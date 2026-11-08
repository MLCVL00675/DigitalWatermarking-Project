### Table 3: Comprehensive Hyperparameter and Reproducibility Specification

| Component | Parameter | Value / Setting | Justification |
| --- | --- | --- | --- |
| Wavelet Decomposition | Wavelet Family | Haar (db1) | Standard benchmark orthogonality, compact support |
| Wavelet Decomposition | Subband Selection | HL (Mid-Freq) & LL (Low-Freq) | Orthogonal robustness trade-off evaluation |
| Block Transformation | Block Dimensions | $8 \times 8$ pixels | Standard DCT block size, 1,024 blocks in $256\times 256$ subband |
| SVD Modification | Singular Value Index | $S(0, 0)$ (Principal singular value) | Maximum energy compaction and stability |
| Texture Masking | Weight Vector $(w_g, w_v, w_e)$ | $(0.4, 0.4, 0.2)$ | Simplex grid search dev split (40 images) |
| Cover Fidelity | Target Matched PSNR | 46.0 dB (sweep: 40, 44, 46, 48 dB) | Strict iso-distortion comparison across methods |
| Cover Quantization | Integer Clipping | quantize_u8 (round + clip [0, 255]) | Real image format compliance before attacks |
| Payload Specification | Payload Dimensions | $32 \times 32 = 1,024$ bits | Balanced random (512 ones) and whitened logo |
| Verification Channel | Decision Threshold $\tau$ | 0.70 BCR ($S = 1.0 - \mathrm{BER}$) | EER optimization from security analysis |
