"""
Pytest suite for experiments_v2/schema.py.
"""

import os
import tempfile
import pytest
from experiments_v2.schema import (
    SCHEMA_COLUMNS,
    ResultRow,
    make_trial_key,
    init_results_csv,
    load_completed_keys,
    append_result_rows,
)


def test_schema_column_count_and_order():
    expected = [
        "run_id",
        "experiment",
        "image_id",
        "payload_id",
        "method",
        "subband",
        "wavelet",
        "block_size",
        "alpha_nominal",
        "alpha_eff_mean",
        "attack",
        "severity",
        "seed",
        "psnr_clean",
        "ssim_clean",
        "psnr_attacked",
        "ber",
        "bcr",
        "nc",
        "extra_json",
    ]
    assert SCHEMA_COLUMNS == expected
    assert len(SCHEMA_COLUMNS) == 20


def test_result_row_roundtrip():
    row = ResultRow(
        run_id="run_123",
        experiment="alpha_sweep",
        image_id="Lena",
        payload_id="logo_32x32",
        method="Adaptive",
        subband="HL",
        wavelet="haar",
        block_size=8,
        alpha_nominal=30.0,
        alpha_eff_mean=32.4,
        attack="jpeg",
        severity="70",
        seed=42,
        psnr_clean=48.5,
        ssim_clean=0.995,
        psnr_attacked=38.2,
        ber=0.02,
        bcr=98.0,
        nc=0.985,
        extra_json='{"git_commit": "abc", "config": {}}',
    )
    d = row.to_dict()
    assert d["image_id"] == "Lena"
    row2 = ResultRow.from_dict(d)
    assert row2 == row


def test_csv_init_append_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "test_results.csv")
        init_results_csv(csv_file)
        assert os.path.exists(csv_file)

        # Check empty completed keys
        completed = load_completed_keys(csv_file)
        assert len(completed) == 0

        # Append row
        row = ResultRow(
            run_id="run_1",
            experiment="alpha_sweep",
            image_id="Lena",
            payload_id="logo_32x32",
            method="Adaptive",
            subband="HL",
            wavelet="haar",
            block_size=8,
            alpha_nominal=30.0,
            alpha_eff_mean=30.0,
            attack="no_attack",
            severity="0",
            seed=1,
            psnr_clean=50.0,
            ssim_clean=0.99,
            psnr_attacked=50.0,
            ber=0.0,
            bcr=100.0,
            nc=1.0,
            extra_json="{}",
        )
        append_result_rows(csv_file, [row])

        # Completed keys should now have this key
        completed = load_completed_keys(csv_file)
        key = make_trial_key(
            experiment="alpha_sweep",
            image_id="Lena",
            payload_id="logo_32x32",
            method="Adaptive",
            subband="HL",
            wavelet="haar",
            block_size=8,
            alpha_nominal=30.0,
            attack="no_attack",
            severity="0",
            seed=1,
        )
        assert key in completed
