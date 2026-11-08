"""
Unit Tests for Runtime, Memory Profiling, and Vectorization Module (experiments_v2/runtime.py).
"""

import os
import json
import numpy as np
import pytest

from experiments_v2.runtime import (
    get_cpu_model_name,
    get_blas_threading_info,
    capture_environment_metadata,
    compute_local_entropy_vectorized,
    block_dct_svd_embed_vectorized,
    verify_vectorization_fidelity,
    measure_memory_and_runtime,
)
from watermark.masking import DeterministicTextureMasker
from watermark.core import CascadedDwtDctSvdWatermarker


def test_environment_capture(tmp_path):
    """Verifies that environment discovery captures all required system specs."""
    env_file = str(tmp_path / "env.json")
    env = capture_environment_metadata(env_file)

    assert os.path.exists(env_file)
    with open(env_file, "r") as f:
        data = json.load(f)

    assert "hardware" in data
    assert "cpu_model" in data["hardware"]
    assert data["hardware"]["cpu_logical_cores"] > 0
    assert data["hardware"]["total_ram_gb"] > 0

    assert "software_versions" in data
    assert "numpy" in data["software_versions"]
    assert "scipy" in data["software_versions"]
    assert "pywavelets" in data["software_versions"]
    assert "opencv" in data["software_versions"]

    assert "blas_configuration" in data


def test_vectorized_entropy_fidelity():
    """Verifies vectorized entropy reproduces loop implementation numerically."""
    rng = np.random.default_rng(123)
    img = rng.uniform(40.0, 220.0, (256, 256)).astype(np.float64)

    masker = DeterministicTextureMasker(window_size=7)
    ent_loop = masker.compute_local_entropy(img, num_bins=16)
    ent_vec = compute_local_entropy_vectorized(img, window_size=7, num_bins=16)

    max_diff = float(np.max(np.abs(ent_loop - ent_vec)))
    assert max_diff < 1e-6, f"Entropy fidelity failed with max_diff={max_diff:.2e}"


def test_vectorized_block_dct_svd_fidelity():
    """Verifies vectorized stacked (N, 8, 8) SVD reproduces loop SVD to < 1e-9."""
    rng = np.random.default_rng(456)
    img = rng.uniform(50.0, 200.0, (256, 256)).astype(np.float64)
    verify_vectorization_fidelity(img)


def test_memory_and_runtime_measurement():
    """Tests runtime and memory measurement utilities."""
    def dummy_task():
        arr = np.ones((500, 500), dtype=np.float64)
        return np.sum(arr * 2.0)

    stats = measure_memory_and_runtime(dummy_task, n_warmup=2, n_repeat=5)

    assert "median_ms" in stats
    assert stats["median_ms"] > 0.0
    assert "iqr_ms" in stats
    assert "tracemalloc_peak_kb" in stats
    assert stats["tracemalloc_peak_kb"] > 0.0
    assert "psutil_rss_delta_kb" in stats
