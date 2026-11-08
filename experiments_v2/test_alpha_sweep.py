"""
Pytest suite for experiments_v2/alpha_sweep.py.
Tests workflow execution, resumability, and output artifacts.
"""

import os
import tempfile
import pandas as pd
import pytest
from experiments_v2.attacks import AttackSpec
from experiments_v2.alpha_sweep import run_alpha_sweep


def test_alpha_sweep_resumable_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_attacks = [
            AttackSpec(attack="no_attack", severity="none", is_stochastic=False),
        ]
        
        # Pass 1: Run 1 condition
        df_summary = run_alpha_sweep(
            images_dir="data/internet_images",
            output_dir=tmpdir,
            fixed_alphas=[30.0],
            adaptive_alphas=[],
            subbands=["HL"],
            seeds=[1],
            workers=1,
            attacks=test_attacks,
        )

        raw_csv = os.path.join(tmpdir, "alpha_sweep_raw.csv")
        assert os.path.exists(raw_csv)
        df_raw1 = pd.read_csv(raw_csv)
        # 5 images * 1 fixed alpha * 1 subband * 1 attack * 1 seed = 5 rows
        assert len(df_raw1) == 5

        by_img_csv = os.path.join(tmpdir, "alpha_sweep_by_image.csv")
        summary_csv = os.path.join(tmpdir, "alpha_sweep_summary.csv")
        assert os.path.exists(by_img_csv)
        assert os.path.exists(summary_csv)

        # Pass 2: Rerun exact same setup -> should detect 5 existing rows and do 0 new trials
        df_summary2 = run_alpha_sweep(
            images_dir="data/internet_images",
            output_dir=tmpdir,
            fixed_alphas=[30.0],
            adaptive_alphas=[],
            subbands=["HL"],
            seeds=[1],
            workers=1,
            attacks=test_attacks,
        )
        df_raw2 = pd.read_csv(raw_csv)
        # Row count should remain 5 (no duplicates)
        assert len(df_raw2) == 5
