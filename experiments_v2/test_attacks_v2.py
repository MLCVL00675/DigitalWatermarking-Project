"""
Pytest suite for experiments_v2/attacks_v2.py.
Verifies all single attacks and compound degradation chains.
"""

import numpy as np
import pytest
from experiments_v2.attacks_v2 import (
    AttackSpecV2,
    get_single_attack_specs,
    get_compound_chain_specs,
    attack_jpeg,
    attack_gaussian_noise,
    attack_salt_and_pepper,
    attack_gaussian_blur,
    attack_median_filter,
    attack_centered_crop,
    attack_random_crop,
    attack_scaling,
    attack_translation,
    attack_shear,
    attack_rotation,
    attack_contrast,
    attack_gamma,
)


@pytest.fixture
def synthetic_512():
    """Create a 512x512 gradient/texture image in float64."""
    x = np.linspace(10.0, 240.0, 512)
    y = np.linspace(10.0, 240.0, 512)
    xx, yy = np.meshgrid(x, y)
    img = 0.5 * xx + 0.5 * yy
    return img.astype(np.float64)


def test_specs_counts():
    single_specs = get_single_attack_specs()
    compound_specs = get_compound_chain_specs()

    # Verify counts
    assert len(single_specs) > 50  # 8 + 6 + 5 + 6 + 3 + 6 + 6 + 6 + 5 + 3 + 5 + 4 + 4 = 67
    assert len(compound_specs) == 8


def test_single_attacks_signature_and_shape(synthetic_512):
    single_specs = get_single_attack_specs()
    for spec, seeds in single_specs:
        rng = np.random.default_rng(seeds[0]) if spec.is_stochastic else None
        out = spec.apply(synthetic_512, rng=rng)
        assert isinstance(out, np.ndarray), f"Failed for {spec.attack} {spec.severity}"
        assert out.dtype == np.float64, f"Dtype was {out.dtype} for {spec.attack}"
        assert out.shape == synthetic_512.shape, f"Shape changed for {spec.attack}"
        assert not np.isnan(out).any()


def test_compound_chains_signature_and_shape(synthetic_512):
    compound_specs = get_compound_chain_specs()
    for spec, seeds in compound_specs:
        rng = np.random.default_rng(seeds[0]) if spec.is_stochastic else None
        out = spec.apply(synthetic_512, rng=rng)
        assert isinstance(out, np.ndarray), f"Failed for {spec.attack} {spec.severity}"
        assert out.dtype == np.float64, f"Dtype was {out.dtype} for {spec.attack}"
        assert out.shape == synthetic_512.shape, f"Shape changed for {spec.attack}"
        assert not np.isnan(out).any()


def test_stochastic_reproducibility(synthetic_512):
    for att_fn, sev in [
        (attack_gaussian_noise, "0.01"),
        (attack_salt_and_pepper, "1%"),
        (attack_random_crop, "10%"),
    ]:
        rng1 = np.random.default_rng(12345)
        out1 = att_fn(synthetic_512, sev, rng=rng1)

        rng2 = np.random.default_rng(12345)
        out2 = att_fn(synthetic_512, sev, rng=rng2)

        # Bit-for-bit identical with identical seed
        np.testing.assert_array_equal(out1, out2)

        # Different seed must differ
        rng3 = np.random.default_rng(54321)
        out3 = att_fn(synthetic_512, sev, rng=rng3)
        assert not np.array_equal(out1, out3)


def test_real_jpeg_encoding(synthetic_512):
    out_q90 = attack_jpeg(synthetic_512, 90)
    out_q30 = attack_jpeg(synthetic_512, 30)
    assert out_q90.shape == (512, 512)
    assert out_q30.shape == (512, 512)
    # Lower quality should have higher distortion (lower MSE vs original)
    mse_90 = np.mean((out_q90 - synthetic_512) ** 2)
    mse_30 = np.mean((out_q30 - synthetic_512) ** 2)
    assert mse_30 > mse_90


def test_crop_area_equivalence(synthetic_512):
    # Centered crop vs random crop should zero out the exact same number of pixels
    # Area ratio for 10% is 2*(0.1) - (0.1)^2 = 0.19
    c_out = attack_centered_crop(synthetic_512, 10.0)
    rng = np.random.default_rng(42)
    r_out = attack_random_crop(synthetic_512, 10.0, rng=rng)

    c_zero_count = np.sum(c_out == 0.0)
    r_zero_count = np.sum(r_out == 0.0)

    # Difference in zero count should be negligible (< 1% due to rounding of int dimensions)
    assert abs(c_zero_count - r_zero_count) / synthetic_512.size < 0.01
