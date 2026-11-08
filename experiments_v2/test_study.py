"""
Pytest suite for experiments_v2/study.py.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from experiments_v2.study import (
    get_study_attack_specs,
    get_payload_set,
    _worker_execute_embedding_condition,
    compute_pareto_points,
    compute_paired_differences,
)


def test_study_attack_specs_count():
    specs = get_study_attack_specs()
    assert len(specs) == 13

    # Check 10 deterministic attacks have 1 seed
    # and 3 stochastic attacks have 20 seeds
    total_evals = sum(len(seeds) for _, seeds in specs)
    assert total_evals == 10 * 1 + 3 * 20  # 70 trials per condition!


def test_get_payload_set():
    p2 = get_payload_set(count_per_image=2)
    assert len(p2) == 2
    assert p2[0][0] == "whitened_logo"
    assert p2[1][0] == "balanced_0"

    p6 = get_payload_set(count_per_image=6)
    assert len(p6) == 6
    assert p6[0][0] == "whitened_logo"
    assert p6[5][0] == "balanced_4"


def test_worker_execute_single_condition():
    # Small 512x512 mock image
    cover = np.full((512, 512), 128.0, dtype=np.float64)
    cover[100:200, 100:200] = 200.0
    payload = np.zeros((32, 32), dtype=np.uint8)
    payload[:16, :] = 1

    task_args = (
        "test_run",
        "test_exp",
        "mock_img",
        cover,
        "mock_payload",
        payload,
        "proposed_adaptive",
        "HL",
        30.0,
        "mock_git",
    )

    rows = _worker_execute_embedding_condition(task_args)
    assert len(rows) == 70  # Exactly 70 attack trials
    assert rows[0].psnr_clean > 40.0
    assert rows[0].alpha_eff_mean > 0.0


def test_compute_pareto_and_paired_diffs():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock core raw CSV
        core_csv = os.path.join(tmpdir, "core_raw.csv")
        rows = []
        for m in ["proposed_fixed", "proposed_adaptive"]:
            for a in [20.0, 40.0]:
                for img in ["img1", "img2", "img3"]:
                    rows.append({
                        "run_id": "r1",
                        "experiment": "core_study",
                        "image_id": img,
                        "payload_id": "pl1",
                        "method": m,
                        "subband": "HL",
                        "wavelet": "haar",
                        "block_size": 8,
                        "alpha_nominal": a,
                        "alpha_eff_mean": a,
                        "attack": "no_attack",
                        "severity": "none",
                        "seed": 42,
                        "psnr_clean": 50.0 - a * 0.2,
                        "ssim_clean": 0.99,
                        "psnr_attacked": 50.0 - a * 0.2,
                        "ber": 0.0,
                        "bcr": 100.0,
                        "nc": 1.0,
                        "extra_json": "{}",
                    })
        pd.DataFrame(rows).to_csv(core_csv, index=False)

        pareto_csv = os.path.join(tmpdir, "pareto_points.csv")
        df_p = compute_pareto_points(core_csv, pareto_csv)
        assert os.path.exists(pareto_csv)
        assert len(df_p) == 4  # 2 methods * 2 alphas

        # Create mock matched PSNR raw CSV
        matched_csv = os.path.join(tmpdir, "matched_psnr_raw.csv")
        m_rows = []
        for t_psnr in [40.0, 48.0]:
            for m in ["proposed_fixed", "proposed_adaptive"]:
                for sb in ["HL", "LL"]:
                    for img in ["img1", "img2", "img3"]:
                        m_rows.append({
                            "run_id": "r1",
                            "experiment": f"matched_{t_psnr}",
                            "image_id": img,
                            "payload_id": "pl1",
                            "method": m,
                            "subband": sb,
                            "wavelet": "haar",
                            "block_size": 8,
                            "alpha_nominal": t_psnr,
                            "alpha_eff_mean": 25.0,
                            "attack": "jpeg",
                            "severity": "Q=70",
                            "seed": 42,
                            "psnr_clean": t_psnr,
                            "ssim_clean": 0.98,
                            "psnr_attacked": 35.0,
                            "ber": 0.05 if m == "proposed_adaptive" else 0.08,
                            "bcr": 95.0,
                            "nc": 0.95 if m == "proposed_adaptive" else 0.90,
                            "extra_json": f'{{"target_psnr": {t_psnr}}}',
                        })
        pd.DataFrame(m_rows).to_csv(matched_csv, index=False)

        diffs_csv = os.path.join(tmpdir, "paired_diffs.csv")
        df_diff = compute_paired_differences(matched_csv, diffs_csv)
        assert os.path.exists(diffs_csv)
        assert len(df_diff) > 0
        assert "diff_nc_mean" in df_diff.columns
        assert "p_ttest_nc" in df_diff.columns
