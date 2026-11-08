"""
Unit and Integration Tests for Re-implemented Baselines Module (experiments_v2/baselines.py).
"""

import os
import tempfile
import numpy as np
import cv2
import pandas as pd
import pytest

from experiments_v2.baselines import (
    LaiTsaiDwtSvdBaseline,
    DctSvdBaseline,
    DwtDctSvdFixedBaseline,
    DwtDctQimBaseline,
    DwtDctAdaptiveQimBaseline,
    create_baseline_method,
    BASELINE_CLASSES,
    BASELINES_SCHEMA_COLUMNS,
    _worker_baseline_condition,
    compute_baselines_summary,
)
from experiments_v2.harness import METHOD_REGISTRY, quantize_u8, calibrate_alpha_to_psnr
from experiments_v2.payloads import generate_balanced_random_payload
from watermark.metrics import compute_psnr, compute_ber
from experiments_v2.attacks import AttackSpec


@pytest.fixture
def synthetic_cover():
    """Generates a 512x512 deterministic synthetic image."""
    rng = np.random.default_rng(12345)
    return rng.uniform(40.0, 220.0, (512, 512)).astype(np.float64)


@pytest.fixture
def balanced_payload():
    """Generates a balanced 32x32 binary payload (512 ones)."""
    return generate_balanced_random_payload(payload_id=0, target_size=(32, 32))


def test_baseline_registry_and_decoder_types():
    """Verifies all 5 baselines are registered and have valid decoder_type."""
    expected_methods = {
        "lai_tsai_dwt_svd": "reference-based",
        "dct_svd_baseline": "reference-based",
        "hybrid_precedent_fixed": "reference-based",
        "dwt_dct_qim_blind": "blind",
        "dwt_dct_adaptive_qim": "blind",
    }

    for m_name, expected_type in expected_methods.items():
        assert m_name in BASELINE_CLASSES, f"{m_name} missing from BASELINE_CLASSES"
        assert m_name in METHOD_REGISTRY, f"{m_name} missing from harness METHOD_REGISTRY"

        obj = create_baseline_method(m_name)
        assert getattr(obj, "decoder_type", None) == expected_type
        assert obj.decoder_type in ("reference-based", "blind")

        # Verify docstring explicitly states re-implementation and cites papers
        doc = obj.__doc__ or obj.__class__.__doc__
        assert "re-implementation" in doc.lower() or "independent" in doc.lower(), f"Missing re-implementation notice in {m_name}"
        assert "10." in doc or "2001" in doc or "2007" in doc or "2010" in doc, f"Missing paper citation in {m_name}"


def test_all_baselines_clean_ber_zero(synthetic_cover, balanced_payload):
    """Verifies that all 5 baselines achieve BER=0.0000 on clean watermarked images with 8-bit quantization."""
    for m_name in BASELINE_CLASSES.keys():
        method = create_baseline_method(m_name)
        alpha = 75.0 if "qim" in m_name else 25.0

        wm_img, side_info = method.embed(synthetic_cover, balanced_payload, alpha=alpha, quantize=True)
        psnr = compute_psnr(synthetic_cover, wm_img)
        assert psnr > 30.0, f"Unreasonable PSNR {psnr} for {m_name}"

        extracted = method.extract(wm_img, side_info)
        ber = compute_ber(balanced_payload, extracted)
        assert ber == 0.0, f"Clean BER is non-zero ({ber}) for {m_name}"


def test_baselines_calibration(synthetic_cover, balanced_payload):
    """Tests bisection calibration for both reference-based and blind baselines to matched PSNR."""
    test_cases = [
        ("lai_tsai_dwt_svd", 46.0, (1.0, 100.0)),
        ("dwt_dct_qim_blind", 44.0, (5.0, 300.0)),
    ]

    for m_name, target, bounds in test_cases:
        method = create_baseline_method(m_name)
        cal_alpha = calibrate_alpha_to_psnr(
            method=method,
            cover=synthetic_cover,
            payload=balanced_payload,
            target_psnr=target,
            alpha_bounds=bounds,
            tol=0.25,
            max_iter=15,
        )
        wm_img, side_info = method.embed(synthetic_cover, balanced_payload, alpha=cal_alpha, quantize=True)
        achieved_psnr = compute_psnr(synthetic_cover, wm_img)
        assert abs(achieved_psnr - target) <= 0.35, f"Calibration failed for {m_name}: target {target}, achieved {achieved_psnr}"

        extracted = method.extract(wm_img, side_info)
        ber = compute_ber(balanced_payload, extracted)
        assert ber == 0.0, f"Calibrated clean BER non-zero for {m_name}"


def test_worker_and_summary_pipeline(tmp_path, synthetic_cover, balanced_payload):
    """Verifies the worker execution and summary calculation pipeline."""
    img_path = str(tmp_path / "test_img.png")
    cv2.imwrite(img_path, synthetic_cover.astype(np.uint8))

    specs = [
        (AttackSpec(attack="no_attack", severity="none", is_stochastic=False), [42]),
        (AttackSpec(attack="jpeg", severity="Q=70", is_stochastic=False), [42]),
    ]

    task_args = (
        "test_run",
        "test_img",
        img_path,
        "balanced_0",
        balanced_payload,
        "dwt_dct_qim_blind",
        46.0,
        specs,
    )

    cond_key, rows = _worker_baseline_condition(task_args)
    assert len(rows) == 2
    for r in rows:
        assert r["decoder_type"] == "blind"
        assert set(BASELINES_SCHEMA_COLUMNS).issubset(set(r.keys()))

    # Write temporary CSV and verify summary
    raw_csv = str(tmp_path / "baselines_raw.csv")
    summary_csv = str(tmp_path / "baselines_summary.csv")

    df_raw = pd.DataFrame(rows)[BASELINES_SCHEMA_COLUMNS]
    df_raw.to_csv(raw_csv, index=False)

    df_sum = compute_baselines_summary(raw_csv=raw_csv, summary_csv=summary_csv, num_boot=50)
    assert os.path.exists(summary_csv)
    assert len(df_sum) == 2
    assert "decoder_type" in df_sum.columns
    assert "median_ber" in df_sum.columns
    assert "ci_lower" in df_sum.columns
    assert "ci_upper" in df_sum.columns
