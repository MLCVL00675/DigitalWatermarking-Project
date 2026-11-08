"""
Tests for experiments_v2/stats.py.
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from experiments_v2.stats import (
    compute_bootstrap_mean_median_ci,
    compute_bootstrap_paired_mean_ci,
    compute_cliffs_delta,
    compute_paired_cohens_dz,
    compute_paired_wilcoxon,
    compute_bootstrap_spearman_ci,
    compute_tost_paired,
    compute_power_mde,
    run_stats_pipeline,
)


def test_bootstrap_mean_median_ci():
    rng = np.random.default_rng(123)
    data = rng.normal(0.05, 0.005, size=50)
    res = compute_bootstrap_mean_median_ci(data, num_resamples=1000, rng=rng)
    assert "mean" in res and "median" in res
    assert res["mean_ci_lower"] <= res["mean"] <= res["mean_ci_upper"]
    assert res["median_ci_lower"] <= res["median"] <= res["median_ci_upper"]


def test_cliffs_delta_and_cohens_dz():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.5, 0.55, 0.75, 0.9])
    # x is strictly less than y
    cd = compute_cliffs_delta(x, y)
    assert cd == -1.0

    dz = compute_paired_cohens_dz(x, y)
    assert dz < 0.0

    # Equal arrays
    assert compute_cliffs_delta(x, x) == 0.0
    assert compute_paired_cohens_dz(x, x) == 0.0


def test_paired_wilcoxon_edge_cases():
    # Identical arrays
    x = np.array([0.1, 0.2, 0.3])
    stat, p = compute_paired_wilcoxon(x, x)
    assert p == 1.0

    # Distinct arrays
    y = np.array([0.15, 0.25, 0.35])
    stat, p = compute_paired_wilcoxon(x, y)
    assert p <= 1.0


def test_tost_paired():
    # Exactly equivalent (tiny differences well within margin 0.02)
    rng = np.random.default_rng(42)
    x = rng.normal(0.05, 0.002, size=60)
    y = x + rng.normal(0.0001, 0.001, size=60)

    res_eq = compute_tost_paired(x, y, margin=0.02, alpha=0.05)
    assert res_eq["is_equivalent"] is True
    assert res_eq["p_tost"] < 0.05
    assert -0.02 < res_eq["ci_90_lower"] < res_eq["ci_90_upper"] < 0.02

    # Clearly non-equivalent (difference = 0.05 > 0.02)
    y_shift = x + 0.05
    res_noneq = compute_tost_paired(x, y_shift, margin=0.02, alpha=0.05)
    assert res_noneq["is_equivalent"] is False
    assert res_noneq["p_tost"] > 0.05


def test_spearman_bootstrap_ci():
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 40)
    y = 2.0 * x + rng.normal(0, 0.5, 40)
    rho, p, ci_low, ci_high = compute_bootstrap_spearman_ci(x, y, num_resamples=1000, rng=rng)
    assert rho > 0.90
    assert p < 0.001
    assert ci_low <= rho <= ci_high


def test_stats_pipeline_integration():
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create minimal synthetic severity_raw.csv
        records = []
        for img_id in [f"img_{i}" for i in range(10)]:
            for meth in ["proposed_fixed", "proposed_adaptive"]:
                for sub in ["HL", "LL"]:
                    for att in ["jpeg", "gaussian_blur"]:
                        for sev in ["Q=70", "Q=50"] if att == "jpeg" else ["s=1.0", "s=2.0"]:
                            records.append({
                                "run_id": "r1",
                                "experiment": "test",
                                "image_id": img_id,
                                "payload_id": 0,
                                "method": meth,
                                "subband": sub,
                                "wavelet": "haar",
                                "block_size": 8,
                                "alpha_nominal": 20.0,
                                "alpha_eff_mean": 20.0,
                                "attack": att,
                                "severity": sev,
                                "seed": 0,
                                "psnr_clean": 46.0,
                                "ssim_clean": 0.98,
                                "psnr_attacked": 35.0,
                                "ber": 0.05 if meth == "proposed_fixed" else 0.048,
                                "bcr": 0.95,
                                "nc": 0.99,
                                "extra_json": '{"target_psnr": 46.0}',
                            })
        pd.DataFrame(records).to_csv(os.path.join(tmp_dir, "severity_raw.csv"), index=False)

        # Create minimal dataset_characterization.csv
        char_recs = []
        for i in range(10):
            char_recs.append({
                "image_id": f"img_{i}",
                "texture_score_T": 0.2 + 0.05 * i,
                "texture_stratum": "low" if i < 3 else ("mid" if i < 7 else "high"),
            })
        pd.DataFrame(char_recs).to_csv(os.path.join(tmp_dir, "dataset_characterization.csv"), index=False)

        # Run pipeline with small bootstrap count and 2 permutations for fast test
        run_stats_pipeline(
            results_dir=tmp_dir, num_bootstrap=200, margin=0.02, num_permutations=2, seed=42
        )

        # Verify all output CSVs exist
        expected_files = [
            "stats_summary.csv",
            "stats_paired.csv",
            "stats_mixed.csv",
            "stats_interactions.csv",
            "stats_texture.csv",
            "stats_tost.csv",
        ]
        for ef in expected_files:
            fp = os.path.join(tmp_dir, ef)
            assert os.path.exists(fp), f"Expected file {ef} missing!"
            df = pd.read_csv(fp)
            assert len(df) > 0, f"File {ef} is empty!"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
