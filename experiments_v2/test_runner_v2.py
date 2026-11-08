"""
Pytest suite for experiments_v2/runner_v2.py.
"""

import os
import numpy as np
import pandas as pd
import pytest
from experiments_v2.runner_v2 import (
    select_benchmark_subset,
    ORDERED_ATTACK_SEVERITIES,
    _worker_condition_v2,
    compute_and_print_breakdown_severities,
    compute_and_print_compound_summary,
)
from experiments_v2.payloads import generate_original_logo, generate_whitened_logo
from experiments_v2.schema import SCHEMA_COLUMNS, ResultRow, init_results_csv, append_result_rows


def test_select_benchmark_subset():
    subset_0 = select_benchmark_subset(
        manifest_csv="results_v2/dataset_manifest.csv", num_images=10, seed=0
    )
    assert len(subset_0) == 10

    # Determinism
    subset_0_again = select_benchmark_subset(
        manifest_csv="results_v2/dataset_manifest.csv", num_images=10, seed=0
    )
    assert subset_0 == subset_0_again

    # Different seed differs
    subset_1 = select_benchmark_subset(
        manifest_csv="results_v2/dataset_manifest.csv", num_images=10, seed=1
    )
    assert subset_0 != subset_1


def test_ordered_attack_severities():
    assert len(ORDERED_ATTACK_SEVERITIES) == 13
    assert "jpeg" in ORDERED_ATTACK_SEVERITIES
    assert "gaussian_noise" in ORDERED_ATTACK_SEVERITIES
    assert "salt_and_pepper" in ORDERED_ATTACK_SEVERITIES
    assert "scaling" in ORDERED_ATTACK_SEVERITIES
    assert "rotation" in ORDERED_ATTACK_SEVERITIES


def test_worker_condition_v2(tmp_path):
    # Test on legacy5_lena
    img_path = "data/dataset_v2/legacy5_lena.png"
    if not os.path.exists(img_path):
        pytest.skip("Test image not found")

    logo = generate_original_logo()
    payload, _ = generate_whitened_logo(logo, seed=42)

    args = (
        "test_run",
        "legacy5_lena",
        img_path,
        "whitened_logo",
        payload,
        "proposed_fixed",
        "HL",
        46.0,
        "dummy_hash",
    )

    cond_key, s_rows, c_rows = _worker_condition_v2(args)
    assert cond_key == "legacy5_lena_proposed_fixed_HL"
    assert len(s_rows) == 220
    assert len(c_rows) == 26

    # Verify first row matches schema
    row = s_rows[0]
    assert row.run_id == "test_run"
    assert row.method == "proposed_fixed"
    assert row.subband == "HL"
    assert 45.0 <= row.psnr_clean <= 47.0


def test_breakdown_calculation(tmp_path):
    csv_file = str(tmp_path / "sample_severity.csv")
    init_results_csv(csv_file)

    # Create dummy rows where JPEG Q=60 has BER > 0.10 and Q=40 has BER > 0.25
    rows = []
    for q, ber_val in [
        ("Q=95", 0.0),
        ("Q=90", 0.01),
        ("Q=80", 0.05),
        ("Q=70", 0.08),
        ("Q=60", 0.15),
        ("Q=50", 0.20),
        ("Q=40", 0.30),
        ("Q=30", 0.40),
    ]:
        rows.append(
            ResultRow(
                run_id="run1",
                experiment="severity_sweep",
                image_id="img1",
                payload_id="whitened_logo",
                method="proposed_adaptive",
                subband="HL",
                wavelet="haar",
                block_size=8,
                alpha_nominal=30.0,
                alpha_eff_mean=30.0,
                attack="jpeg",
                severity=q,
                seed=42,
                psnr_clean=46.0,
                ssim_clean=0.98,
                psnr_attacked=35.0,
                ber=ber_val,
                bcr=1.0 - ber_val,
                nc=1.0 - 2 * ber_val,
                extra_json="{}",
            )
        )
    append_result_rows(csv_file, rows)

    df_bd = compute_and_print_breakdown_severities(csv_file)
    jpeg_row = df_bd[df_bd["attack"] == "jpeg"].iloc[0]
    assert "Q=60" in jpeg_row["HL_Adaptive_BER>0.10"]
    assert "Q=40" in jpeg_row["HL_Adaptive_BER>0.25"]
