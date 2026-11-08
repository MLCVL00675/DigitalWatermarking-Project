"""
Pytest suite for experiments_v2/dataset.py.
"""

import os
import tempfile
import numpy as np
import cv2
import pandas as pd
import pytest

from experiments_v2.dataset import (
    preprocess_image,
    compute_sha256,
    characterize_image,
    plot_dataset_overview,
)


def test_preprocess_image_color_rectangular():
    # 300x400 BGR color test image
    raw = np.full((300, 400, 3), 128, dtype=np.uint8)
    # Add a recognizable region in the center
    raw[100:200, 150:250] = 255
    
    proc = preprocess_image(raw, target_size=512)
    assert proc.shape == (512, 512)
    assert proc.dtype == np.uint8
    assert proc.ndim == 2


def test_preprocess_image_grayscale_square():
    raw = np.full((256, 256), 64, dtype=np.uint8)
    proc = preprocess_image(raw, target_size=512)
    assert proc.shape == (512, 512)
    assert proc.dtype == np.uint8


def test_compute_sha256():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"watermarking_benchmark_test")
        tmp_path = f.name
    try:
        sha = compute_sha256(tmp_path)
        assert len(sha) == 64
        assert isinstance(sha, str)
    finally:
        os.remove(tmp_path)


def test_characterize_image():
    # 512x512 synthetic image with gradients
    x = np.linspace(0, 255, 512)
    y = np.linspace(0, 255, 512)
    xx, yy = np.meshgrid(x, y)
    test_img = (0.5 * xx + 0.5 * yy).astype(np.uint8)

    metrics = characterize_image(test_img)
    
    assert "mean_sobel_gradient" in metrics
    assert "mean_local_std" in metrics
    assert "mean_local_entropy" in metrics
    assert "edge_density" in metrics
    assert "mean_intensity" in metrics
    assert "texture_score_T" in metrics

    assert 0.0 <= metrics["texture_score_T"] <= 1.0
    assert 0.0 <= metrics["edge_density"] <= 1.0
    assert metrics["mean_intensity"] > 0.0


def test_plot_dataset_overview():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 16 mock images
        char_rows = []
        for i in range(16):
            img = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
            img_id = f"test_img_{i:02d}"
            cv2.imwrite(os.path.join(tmpdir, f"{img_id}.png"), img)
            
            stratum = "low" if i < 5 else ("mid" if i < 11 else "high")
            char_rows.append({
                "image_id": img_id,
                "source_dataset": "mock",
                "texture_score_T": 0.1 + (i * 0.05),
                "texture_stratum": stratum,
            })
        df_char = pd.DataFrame(char_rows)

        pdf_path = os.path.join(tmpdir, "dataset_overview.pdf")
        plot_dataset_overview(df_char, tmpdir, pdf_path, q1=0.35, q2=0.65)
        
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 1000
        png_path = pdf_path.replace(".pdf", ".png")
        assert os.path.exists(png_path)
