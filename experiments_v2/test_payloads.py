"""
Pytest suite for experiments_v2/payloads.py.
"""

import os
import tempfile
import numpy as np
import pytest
from experiments_v2.payloads import (
    generate_original_logo,
    generate_whitened_logo,
    unwhiten_payload,
    generate_balanced_random_payload,
    compute_payload_stats,
    generate_all_payload_stats,
)


def test_original_logo():
    logo = generate_original_logo("data/internet_images")
    assert logo.shape == (32, 32)
    assert set(np.unique(logo)).issubset({0, 1})
    assert logo.dtype == np.uint8


def test_whitened_logo_properties():
    logo = generate_original_logo("data/internet_images")
    whitened, keystream = generate_whitened_logo(logo, seed=42)
    
    assert whitened.shape == (32, 32)
    assert set(np.unique(whitened)).issubset({0, 1})
    
    # Invertible
    recovered = unwhiten_payload(whitened, keystream)
    np.testing.assert_array_equal(recovered, logo)

    # Whitened logo fraction of ones should be close to 0.5 (within 10%)
    frac_ones = np.mean(whitened)
    assert 0.40 <= frac_ones <= 0.60

    # Determinism
    w2, k2 = generate_whitened_logo(logo, seed=42)
    np.testing.assert_array_equal(whitened, w2)
    np.testing.assert_array_equal(keystream, k2)


def test_balanced_random_payload():
    for pid in range(5):
        payload = generate_balanced_random_payload(payload_id=pid)
        assert payload.shape == (32, 32)
        assert set(np.unique(payload)).issubset({0, 1})
        # Exactly 512 ones for 1,024 bits
        assert np.sum(payload) == 512
        assert np.mean(payload) == 0.5

    # Deterministic
    p1 = generate_balanced_random_payload(payload_id=99, seed=123)
    p2 = generate_balanced_random_payload(payload_id=99, seed=123)
    np.testing.assert_array_equal(p1, p2)


def test_payload_statistics_integrity():
    # Balanced payload
    payload = generate_balanced_random_payload(payload_id=0, seed=42)
    stats = compute_payload_stats(payload, "Test", "balanced", "test_0")
    assert stats["total_bits"] == 1024
    assert stats["num_ones"] == 512
    assert stats["fraction_ones"] == 0.5
    assert stats["ber_all_zeros"] == 0.5
    assert stats["ber_all_ones"] == 0.5
    assert np.isclose(stats["ber_random_guess"], 0.5, atol=0.05)


def test_generate_all_payload_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "payload_stats.csv")
        df = generate_all_payload_stats(
            images_dir="data/internet_images",
            output_csv=csv_path,
            n_balanced_samples=3,
        )
        assert os.path.exists(csv_path)
        assert len(df) == 1 + 3 + 3  # 1 orig + 3 whitened + 3 balanced
        assert "fraction_ones" in df.columns
        assert "ber_all_zeros" in df.columns
        assert "ber_random_guess" in df.columns
