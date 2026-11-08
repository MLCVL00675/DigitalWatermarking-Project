"""
Pytest suite for experiments_v2/common.py.
"""

import json
import numpy as np
import pytest
from experiments_v2.schema import SCHEMA_COLUMNS, ResultRow
from experiments_v2.attacks import AttackSpec
from experiments_v2.common import (
    get_git_commit_hash,
    ConstantAlphaMasker,
    load_standard_test_images,
    load_standard_watermark,
    evaluate_single_trial,
)


def test_git_commit_hash_retrieval():
    commit = get_git_commit_hash()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_constant_alpha_masker():
    masker = ConstantAlphaMasker(base_alpha=35.0)
    fake_img = np.zeros((512, 512), dtype=np.float64)
    mat = masker.generate_block_alpha_matrix(fake_img, target_subband_shape=(256, 256), block_size=8)
    assert mat.shape == (32, 32)
    assert np.all(mat == 35.0)


def test_dataset_loaders():
    images = load_standard_test_images("data/internet_images")
    assert len(images) == 5
    for name in ["Lena", "Baboon", "Fruits", "Building", "Butterfly"]:
        assert name in images
        assert images[name].shape == (512, 512)
        assert images[name].dtype == np.float64

    pid, wm = load_standard_watermark("data/internet_images")
    assert pid == "logo_32x32"
    assert wm.shape == (32, 32)
    assert set(np.unique(wm)).issubset({0, 1})


def test_evaluate_single_trial_deterministic():
    images = load_standard_test_images("data/internet_images")
    cover = images["Lena"]
    _, wm = load_standard_watermark("data/internet_images")

    spec = AttackSpec(attack="gaussian_noise", severity="var=0.01", is_stochastic=True)

    row1 = evaluate_single_trial(
        run_id="test_run",
        experiment="test_exp",
        image_id="Lena",
        cover_image=cover,
        payload_id="logo_32x32",
        watermark_payload=wm,
        method="Adaptive",
        subband="HL",
        wavelet="haar",
        block_size=8,
        alpha_nominal=30.0,
        attack_spec=spec,
        seed=100,
    )

    row2 = evaluate_single_trial(
        run_id="test_run",
        experiment="test_exp",
        image_id="Lena",
        cover_image=cover,
        payload_id="logo_32x32",
        watermark_payload=wm,
        method="Adaptive",
        subband="HL",
        wavelet="haar",
        block_size=8,
        alpha_nominal=30.0,
        attack_spec=spec,
        seed=100,
    )

    # Identical seed must give identical output
    assert row1.psnr_clean == row2.psnr_clean
    assert row1.psnr_attacked == row2.psnr_attacked
    assert row1.ber == row2.ber
    assert row1.nc == row2.nc

    # Validate schema
    row_dict = row1.to_dict()
    for col in SCHEMA_COLUMNS:
        assert col in row_dict

    extra = json.loads(row1.extra_json)
    assert "git_commit" in extra
    assert "config" in extra


def test_evaluate_single_trial_seed_variation():
    images = load_standard_test_images("data/internet_images")
    cover = images["Lena"]
    _, wm = load_standard_watermark("data/internet_images")

    spec = AttackSpec(attack="gaussian_noise", severity="var=0.01", is_stochastic=True)

    row1 = evaluate_single_trial(
        run_id="test_run",
        experiment="test_exp",
        image_id="Lena",
        cover_image=cover,
        payload_id="logo_32x32",
        watermark_payload=wm,
        method="Adaptive",
        subband="HL",
        wavelet="haar",
        block_size=8,
        alpha_nominal=30.0,
        attack_spec=spec,
        seed=101,
    )

    row2 = evaluate_single_trial(
        run_id="test_run",
        experiment="test_exp",
        image_id="Lena",
        cover_image=cover,
        payload_id="logo_32x32",
        watermark_payload=wm,
        method="Adaptive",
        subband="HL",
        wavelet="haar",
        block_size=8,
        alpha_nominal=30.0,
        attack_spec=spec,
        seed=202,
    )

    # Different seeds should produce slightly different attacked PSNR
    assert row1.psnr_attacked != row2.psnr_attacked
