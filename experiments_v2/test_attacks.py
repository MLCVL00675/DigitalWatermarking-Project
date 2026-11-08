"""
Pytest suite for experiments_v2/attacks.py.
"""

import numpy as np
import pytest
from experiments_v2.attacks import (
    AttackSpec,
    get_standard_attack_specs,
    apply_attack,
)


@pytest.fixture
def test_image():
    # 64x64 synthetic gradient test image
    x = np.linspace(0, 255, 64)
    y = np.linspace(0, 255, 64)
    xx, yy = np.meshgrid(x, y)
    return (0.5 * xx + 0.5 * yy).astype(np.float64)


def test_standard_attack_count():
    specs = get_standard_attack_specs()
    assert len(specs) == 14
    names = [s.attack for s in specs]
    assert "no_attack" in names
    assert "jpeg" in names
    assert "gaussian_noise" in names
    assert "rotation" in names


def test_stochastic_reproducibility(test_image):
    spec = AttackSpec(attack="gaussian_noise", severity="var=0.01", is_stochastic=True)
    
    rng1 = np.random.default_rng(12345)
    out1 = apply_attack(test_image, spec, rng=rng1)
    
    rng2 = np.random.default_rng(12345)
    out2 = apply_attack(test_image, spec, rng=rng2)
    
    # Must be bit-for-bit identical with identical seed
    np.testing.assert_array_equal(out1, out2)
    
    # Different seed must differ
    rng3 = np.random.default_rng(54321)
    out3 = apply_attack(test_image, spec, rng=rng3)
    assert not np.array_equal(out1, out3)


def test_speckle_and_salt_pepper_reproducibility(test_image):
    for att, sev in [("speckle_noise", "var=0.01"), ("salt_and_pepper", "1%")]:
        spec = AttackSpec(attack=att, severity=sev, is_stochastic=True)
        rng1 = np.random.default_rng(42)
        out1 = apply_attack(test_image, spec, rng=rng1)
        rng2 = np.random.default_rng(42)
        out2 = apply_attack(test_image, spec, rng=rng2)
        np.testing.assert_array_equal(out1, out2)


def test_all_attacks_preserve_shape(test_image):
    specs = get_standard_attack_specs()
    for spec in specs:
        rng = np.random.default_rng(999) if spec.is_stochastic else None
        out = apply_attack(test_image, spec, rng=rng)
        assert out.shape == test_image.shape
        assert not np.isnan(out).any()
