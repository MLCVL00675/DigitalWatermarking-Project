"""
Unit and Integration Tests for Security Evaluation Module (experiments_v2/security.py).
"""

import os
import tempfile
import numpy as np
import cv2
import pandas as pd
import pytest

from experiments_v2.security import (
    compute_decision_statistic,
    _worker_security_image,
    compute_security_roc_and_summary,
    PLAIN_WORDS_FINDINGS,
)
from experiments_v2.payloads import generate_balanced_random_payload


def test_compute_decision_statistic():
    """Tests bit correct rate calculation."""
    p1 = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    p2 = np.array([1, 0, 1, 0, 0], dtype=np.uint8)
    # 4 out of 5 match -> BCR = 0.80
    assert compute_decision_statistic(p1, p2) == 0.80

    p_same = np.array([0, 1, 0, 1], dtype=np.uint8)
    assert compute_decision_statistic(p_same, p_same) == 1.00

    p_opp = np.array([1, 0, 1, 0], dtype=np.uint8)
    assert compute_decision_statistic(p_same, p_opp) == 0.00


def test_security_worker_pipeline(tmp_path):
    """Tests the worker function on two synthetic images."""
    rng = np.random.default_rng(42)
    img_a = rng.uniform(50.0, 200.0, (512, 512)).astype(np.uint8)
    img_b = rng.uniform(50.0, 200.0, (512, 512)).astype(np.uint8)

    path_a = str(tmp_path / "img_a.png")
    path_b = str(tmp_path / "img_b.png")
    cv2.imwrite(path_a, img_a)
    cv2.imwrite(path_b, img_b)

    task_args = ("test_run", "img_a", path_a, "img_b", path_b, 46.0)
    records = _worker_security_image(task_args)

    assert len(records) > 0
    df = pd.DataFrame(records)

    # Check case names present
    expected_cases = {
        "pos_clean", "pos_jpeg70", "pos_blur", "pos_noise",
        "N1_unwatermarked_cover", "N2_unrelated_image", "N3_cross_side_info",
        "N4_wrong_payload", "N5_different_key_payload", "N6_svd_forgery"
    }
    present_cases = set(df["case_name"].unique())
    assert expected_cases.issubset(present_cases), f"Missing cases: {expected_cases - present_cases}"

    # Verify clean positive BCR == 1.0
    clean_bcr = df[df["case_name"] == "pos_clean"]["bcr"].values
    assert np.all(clean_bcr == 1.0), "Clean positives should have BCR=1.0"

    # Verify N1 has BCR ~ 0.50 on balanced payloads
    n1_balanced = df[(df["case_name"] == "N1_unwatermarked_cover") & (df["payload_type"] == "balanced")]["bcr"].values
    assert np.all(np.isclose(n1_balanced, 0.50, atol=0.01)), f"N1 on balanced should be ~0.50, got {n1_balanced}"

    # Verify N6 Craver SVD ambiguity attack achieves BCR == 1.0000
    n6_bcr = df[df["case_name"] == "N6_svd_forgery"]["bcr"].values
    assert np.all(n6_bcr == 1.0), f"N6 SVD ambiguity attack must achieve BCR=1.0, got {n6_bcr}"


def test_security_roc_summary_pipeline(tmp_path):
    """Tests ROC, AUC, EER and summary generation."""
    # Create synthetic scores dataframe
    rows = []
    # 20 clean positive samples
    for i in range(20):
        rows.append({"method": "proposed_adaptive", "subband": "HL", "payload_type": "balanced",
                     "case_name": "pos_clean", "bcr": 1.00})
    # 20 N1 samples around 0.50
    for i in range(20):
        rows.append({"method": "proposed_adaptive", "subband": "HL", "payload_type": "balanced",
                     "case_name": "N1_unwatermarked_cover", "bcr": 0.50})
    # 20 N6 samples at 1.00
    for i in range(20):
        rows.append({"method": "proposed_adaptive", "subband": "HL", "payload_type": "balanced",
                     "case_name": "N6_svd_forgery", "bcr": 1.00})

    scores_csv = str(tmp_path / "test_scores.csv")
    roc_csv = str(tmp_path / "test_roc.csv")
    summary_csv = str(tmp_path / "test_summary.csv")

    pd.DataFrame(rows).to_csv(scores_csv, index=False)

    df_roc, df_summary = compute_security_roc_and_summary(
        scores_csv=scores_csv,
        roc_csv=roc_csv,
        summary_csv=summary_csv,
        tau_step=0.05,
    )

    assert os.path.exists(roc_csv)
    assert os.path.exists(summary_csv)
    assert len(df_roc) > 0
    assert len(df_summary) > 0

    # For pos_clean vs N1: AUC should be 1.0 and EER should be 0.0
    sub_n1 = df_summary[(df_summary["positive_case"] == "pos_clean") & (df_summary["negative_case"] == "N1_unwatermarked_cover")]
    assert len(sub_n1) == 1
    assert sub_n1.iloc[0]["auc"] == 1.0
    assert sub_n1.iloc[0]["eer"] == 0.0

    # For pos_clean vs N6: N6 has BCR=1.0, so n6_success_rate should be 1.0
    sub_n6 = df_summary[(df_summary["positive_case"] == "pos_clean") & (df_summary["negative_case"] == "N6_svd_forgery")]
    assert len(sub_n6) == 1
    assert sub_n6.iloc[0]["n6_success_rate"] == 1.0
