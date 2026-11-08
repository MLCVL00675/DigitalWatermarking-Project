"""
Pytest suite for experiments_v2/harness.py.
"""

import os
import tempfile
import numpy as np
import pytest
from experiments_v2.schema import SCHEMA_COLUMNS
from experiments_v2.attacks import AttackSpec
from experiments_v2.payloads import generate_original_logo
from experiments_v2.common import load_standard_test_images
from experiments_v2.harness import (
    quantize_u8,
    set_global_quantization,
    quantization_context,
    ProposedAdaptiveMethod,
    ProposedFixedMethod,
    create_method,
    run_trial,
    evaluate_grid,
    calibrate_alpha_to_psnr,
)


@pytest.fixture
def sample_data():
    images = load_standard_test_images("data/internet_images")
    logo = generate_original_logo("data/internet_images")
    return images["Lena"], logo


def test_quantize_u8():
    arr = np.array([-10.2, 0.4, 0.6, 254.7, 300.1], dtype=np.float64)
    q = quantize_u8(arr)
    expected = np.array([0.0, 0.0, 1.0, 255.0, 255.0], dtype=np.float64)
    np.testing.assert_array_equal(q, expected)
    assert q.dtype == np.float64


def test_quantization_context():
    set_global_quantization(True)
    with quantization_context(False):
        from experiments_v2 import harness
        assert harness.QUANTIZE is False
    from experiments_v2 import harness
    assert harness.QUANTIZE is True


def test_method_plugins(sample_data):
    cover, payload = sample_data

    for m_name, cls in [("proposed_adaptive", ProposedAdaptiveMethod), ("proposed_fixed", ProposedFixedMethod)]:
        method = create_method(m_name, subband="HL", wavelet="haar")
        assert isinstance(method, cls)
        assert method.name == m_name

        # Embed with quantization
        wm_img, side_info = method.embed(cover, payload, alpha=30.0, quantize=True)
        assert wm_img.shape == (512, 512)
        # Check integer values
        np.testing.assert_array_equal(wm_img, np.round(wm_img))

        # Extract
        extracted = method.extract(wm_img, side_info)
        assert extracted.shape == (32, 32)
        ber = np.mean(np.abs(payload - extracted))
        assert ber == 0.0


def test_run_trial_schema_conformance(sample_data):
    cover, payload = sample_data
    method_cfg = {
        "method": "proposed_adaptive",
        "subband": "HL",
        "wavelet": "haar",
        "block_size": 8,
        "alpha": 30.0,
        "image_id": "Lena",
        "payload_id": "logo_32x32",
    }
    attack_cfg = AttackSpec(attack="no_attack", severity="none", is_stochastic=False)

    row = run_trial(cover, payload, method_cfg, attack_cfg, seed=42, quantize=True)
    for col in SCHEMA_COLUMNS:
        assert col in row
    assert row["ber"] == 0.0
    assert row["nc"] == 1.0


def test_calibrate_alpha_to_psnr(sample_data):
    cover, payload = sample_data
    target = 48.0
    cal_alpha = calibrate_alpha_to_psnr(
        method="proposed_adaptive",
        cover=cover,
        payload=payload,
        target_psnr=target,
        tol=0.10,
        subband="HL",
    )
    assert cal_alpha > 0.0

    # Verify achieved PSNR
    method = ProposedAdaptiveMethod(subband="HL")
    wm_img, _ = method.embed(cover, payload, alpha=cal_alpha, quantize=True)
    from watermark.metrics import compute_psnr
    achieved_psnr = compute_psnr(cover, wm_img)
    assert abs(achieved_psnr - target) <= 0.10


def test_evaluate_grid_resumable(sample_data):
    cover, payload = sample_data
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_grid.csv")
        trials = [
            {
                "cover": cover,
                "payload": payload,
                "method_cfg": {
                    "method": "proposed_fixed",
                    "subband": "HL",
                    "alpha": 30.0,
                    "image_id": "Lena",
                    "payload_id": "logo_32x32",
                },
                "attack_cfg": {
                    "attack": "no_attack",
                    "severity": "none",
                    "is_stochastic": False,
                },
                "seed": 42,
                "quantize": True,
                "experiment": "test_grid",
            }
        ]

        # Pass 1: Run
        df1 = evaluate_grid(trials, csv_path, workers=1)
        assert len(df1) == 1

        # Pass 2: Rerun exact same -> 0 new rows
        df2 = evaluate_grid(trials, csv_path, workers=1)
        assert len(df2) == 1
