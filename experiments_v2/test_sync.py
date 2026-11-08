"""
Pytest suite for experiments_v2/sync.py.
"""

import os
import numpy as np
import cv2
import pytest

from experiments_v2.sync import (
    decompose_affine_matrix,
    estimate_geometry_ecc,
    estimate_geometry_features,
    warp_back_and_extract,
    _make_sync_attack_specs,
    _worker_sync_condition,
)
from experiments_v2.payloads import generate_original_logo, generate_whitened_logo


def test_decompose_affine_matrix():
    cx, cy = 256.0, 256.0

    # 1. Rotation 5 degrees
    M_rot = cv2.getRotationMatrix2D((cx, cy), 5.0, 1.0)
    ang, sc, dx, dy = decompose_affine_matrix(M_rot, (cx, cy))
    assert abs(ang - 5.0) < 1e-4
    assert abs(sc - 1.0) < 1e-4
    assert abs(dx) < 1e-4
    assert abs(dy) < 1e-4

    # 2. Scaling 0.9x
    M_sc = np.array([[0.9, 0.0, cx * (1.0 - 0.9)], [0.0, 0.9, cy * (1.0 - 0.9)]], dtype=np.float32)
    ang, sc, dx, dy = decompose_affine_matrix(M_sc, (cx, cy))
    assert abs(ang) < 1e-4
    assert abs(sc - 0.9) < 1e-4
    assert abs(dx) < 1e-4
    assert abs(dy) < 1e-4

    # 3. Translation 8 px
    M_tr = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 8.0]], dtype=np.float32)
    ang, sc, dx, dy = decompose_affine_matrix(M_tr, (cx, cy))
    assert abs(ang) < 1e-4
    assert abs(sc - 1.0) < 1e-4
    assert abs(dx - 8.0) < 1e-4
    assert abs(dy - 8.0) < 1e-4


def test_sync_attack_specs_count():
    specs = _make_sync_attack_specs()
    assert len(specs) == 10  # 5 rot + 2 scale + 2 trans + 1 compound
    names = [s.name for s in specs]
    assert "rotation" in names
    assert "scaling" in names
    assert "translation" in names
    assert "compound" in names


def test_estimator_ecc():
    img_path = "data/dataset_v2/legacy5_lena.png"
    if not os.path.exists(img_path):
        pytest.skip("Test image not found")

    cover = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    h, w = cover.shape
    center = (w / 2.0, h / 2.0)
    M_rot = cv2.getRotationMatrix2D(center, 2.0, 1.0)
    att = cv2.warpAffine(cover.astype(np.float32), M_rot, (w, h), flags=cv2.INTER_LINEAR)

    M_est, succ, meta = estimate_geometry_ecc(att, cover, levels=2, max_iter=30)
    assert succ is True
    assert abs(meta["est_angle_deg"] - 2.0) < 0.2


def test_estimator_features():
    img_path = "data/dataset_v2/legacy5_lena.png"
    if not os.path.exists(img_path):
        pytest.skip("Test image not found")

    cover = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    h, w = cover.shape
    center = (w / 2.0, h / 2.0)
    M_rot = cv2.getRotationMatrix2D(center, 2.0, 1.0)
    att = cv2.warpAffine(cover.astype(np.float32), M_rot, (w, h), flags=cv2.INTER_LINEAR)

    M_est, succ, meta = estimate_geometry_features(att, cover)
    assert succ is True
    assert abs(meta["est_angle_deg"] - 2.0) < 0.3
