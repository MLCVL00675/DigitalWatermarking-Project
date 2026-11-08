# RAW FILE FACTOR VERIFICATION & INTEGRITY CHECK

Generated at: 2026-10-31 08:25:34 UTC

All numbers in this verification report are computed dynamically from the output CSVs.

## 1. Summary of Regenerated Raw and Ablation Files

| File | Status | Actual Rows | Expected Rows | Unique Images | Unique Methods | Unique Attacks | Size (MB) | GZIP Archive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `matched_psnr_raw.csv` | **MATCH** | 128,000 | 128,000 | 100 | 2 | 10 | 42.89 | Yes (.csv.gz: 2.46 MB) |
| `paired_diffs.csv` | **MATCH** | 104 | 104 | 100 | 2 | 10 | 0.01 | No |
| `core_raw.csv` | **MATCH** | 320,000 | 320,000 | 100 | 2 | 10 | 114.59 | Yes (.csv.gz: 6.01 MB) |
| `pareto_points.csv` | **MATCH** | 40 | 40 | 100 | 2 | 0 | 0.00 | No |
| `ablation_transform.csv` | **MATCH** | 32,000 | 32,000 | 100 | 5 | 10 | 12.85 | No |
| `ablation_transform_stats.csv` | **MATCH** | 80 | 80 | 100 | 8 | 10 | 0.01 | No |
| `ablation_features.csv` | **MATCH** | 68,800 | 68,800 | 100 | 1 | 10 | 31.07 | No |
| `ablation_wavelet.csv` | **MATCH** | 80,000 | 80,000 | 100 | 2 | 10 | 31.12 | No |
| `ablation_block.csv` | **MATCH** | 48,000 | 48,000 | 100 | 2 | 10 | 18.30 | No |
| `ablation_payload.csv` | **MATCH** | 64,000 | 64,000 | 100 | 2 | 10 | 24.27 | No |
| `ablation_heatmap_data.csv` | **MATCH** | 58 | 58 | 0 | 58 | 13 | 0.02 | No |

## 2. Factor Decomposition & Derivation Formulas

- **`matched_psnr_raw.csv`**: `100 images x 2 payloads x 2 methods x 2 subbands x 4 PSNRs x 40 trials` -> **128,000 rows**.
  - SHA-256: `03821d88db1a5da1571dbb69ce6c5cec1de755e8969882ebff2c11a80c88c4bb`
  - GZIP SHA-256: `fa6f9bae35286f535ee47f35731ff5d8de75a8ea66e5231d5224f7a94d125f22`
- **`paired_diffs.csv`**: `2 comparisons x 4 PSNRs x 13 attacks` -> **104 rows**.
  - SHA-256: `78ca5e269fd4b9748e8c21b475618985e0334d4046b86ef690cd274e2d3c715a`
- **`core_raw.csv`**: `100 images x 2 payloads x 2 methods x 2 subbands x 10 alphas x 40 trials` -> **320,000 rows**.
  - SHA-256: `709ae9219c1947c8c7a05e88bcd87571442cb3c60b01016fd3af848a615f3221`
  - GZIP SHA-256: `5ba345714e8b587d644aa41fa89687549d7aaef870b204d07b5a8e4af9710088`
- **`pareto_points.csv`**: `2 methods x 2 subbands x 10 alphas` -> **40 rows**.
  - SHA-256: `abfe2322be931bf78114eab66dc13a80495f03e8aef86777f9706be5fc79b429`
- **`ablation_transform.csv`**: `100 images x 1 payload x 8 configs (fixed alpha) x 40 trials` -> **32,000 rows**.
  - SHA-256: `67967b2b70509920dde1df3f031fdbc054247b7346b37d24b0d3ac4b64f63edf`
- **`ablation_transform_stats.csv`**: `8 comparisons x 10 attacks (available paired subsets)` -> **80 rows**.
  - SHA-256: `19cd48adbc6a0a320a0423fa1a3f6076b7c42559f45065455d65b4a6068cf59c`
- **`ablation_features.csv`**: `(40 dev x 8 cfgs + 60 test x 9 cfgs) x 2 subbands x 40 trials` -> **68,800 rows**.
  - SHA-256: `752c50f0658277a11f9dc5d6c5c86e5f17b24cd2fda863d85ddb136bb06a70dc`
- **`ablation_wavelet.csv`**: `100 images x 5 wavelets x 2 methods x 2 subbands x 40 trials` -> **80,000 rows**.
  - SHA-256: `cb9a27e814625d2d41159c6da91647e19e9cda779df453e7a84e9689569d9ca7`
- **`ablation_block.csv`**: `100 images x 3 block sizes x 2 methods x 2 subbands x 40 trials` -> **48,000 rows**.
  - SHA-256: `dfdd731ca4103ad641537ad1efaa4e47c0174b6c76213e7cfc6771fda9fe845a`
- **`ablation_payload.csv`**: `100 images x 4 payload sizes x 2 methods x 2 subbands x 40 trials` -> **64,000 rows**.
  - SHA-256: `f210655231170603c9a9f628eaff79153eaa04851f3101b7cb69d11b15dc274d`
- **`ablation_heatmap_data.csv`**: `58 ablation variants aggregated across 13 manuscript attacks` -> **58 rows**.
  - SHA-256: `16dde507d660b338ad9829e9c1dead8af2b599c8bc4a2deeb6659c947352d5cb`

---
*(End of Verification Report)*
