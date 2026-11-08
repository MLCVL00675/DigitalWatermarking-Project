"""
Pytest suite for experiments_v2/ablations.py and experiments_v2/transform_methods.py.
"""

import os
import numpy as np
import pytest

from experiments_v2.transform_methods import (
    create_ablation_method,
    DwtSvdMethod,
    DctSvdMethod,
    DwtDctMethod,
    SvdOnlyMethod,
)
from experiments_v2.ablations import (
    bootstrap_median_ci,
    get_feature_weight_configs,
    get_manuscript_attack_specs,
    _worker_ablation_condition,
)
from experiments_v2.payloads import generate_original_logo, generate_whitened_logo


@pytest.fixture
def sample_cover():
    x = np.linspace(10.0, 240.0, 512)
    y = np.linspace(10.0, 240.0, 512)
    xx, yy = np.meshgrid(x, y)
    return (0.5 * xx + 0.5 * yy).astype(np.float64)


@pytest.fixture
def sample_payload():
    logo = generate_original_logo()
    whitened, _ = generate_whitened_logo(logo, seed=42)
    return whitened


def test_transform_method_factory():
    m1 = create_ablation_method("dwt_svd_adaptive", subband="HL")
    assert isinstance(m1, DwtSvdMethod)

    m2 = create_ablation_method("dct_svd_fixed", subband="spatial")
    assert isinstance(m2, DctSvdMethod)

    m3 = create_ablation_method("dwt_dct_adaptive", subband="LL")
    assert isinstance(m3, DwtDctMethod)

    m4 = create_ablation_method("svd_only_fixed", subband="spatial")
    assert isinstance(m4, SvdOnlyMethod)


def test_roundtrip_all_transform_variants(sample_cover, sample_payload):
    alpha = 25.0
    variants = [
        create_ablation_method("dwt_svd_fixed", subband="HL"),
        create_ablation_method("dwt_svd_adaptive", subband="HL"),
        create_ablation_method("dct_svd_fixed", subband="spatial"),
        create_ablation_method("dwt_dct_fixed", subband="HL"),
        create_ablation_method("svd_only_fixed", subband="spatial"),
    ]

    for m in variants:
        wm_img, side_info = m.embed(sample_cover, sample_payload, alpha=alpha, quantize=False)
        assert wm_img.shape == (512, 512)
        extracted = m.extract(wm_img, side_info)
        ber = np.mean(extracted != sample_payload)
        assert ber < 0.01, f"Failed roundtrip for {m.name}: clean BER was {ber}"


def test_bootstrap_median_ci():
    arr = np.array([0.05] * 50 + [0.06] * 50)
    med, low, high = bootstrap_median_ci(arr, num_boot=200)
    assert 0.049 <= med <= 0.061
    assert low <= med <= high


def test_feature_weight_configs():
    cfgs = get_feature_weight_configs()
    assert len(cfgs) == 11
    names = [c["name"] for c in cfgs]
    assert "gradient_only" in names
    assert "variance_only" in names
    assert "entropy_only" in names
    assert "default_40_40_20" in names


def test_manuscript_attacks_count():
    specs = get_manuscript_attack_specs()
    assert len(specs) == 14
    names = [s.attack for s, seeds in specs]
    assert "no_attack" in names
    assert "jpeg" in names
    assert "gaussian_noise" in names
    assert "rotation" in names
