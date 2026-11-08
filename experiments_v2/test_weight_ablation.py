"""
Pytest suite for experiments_v2/weight_ablation.py.
Tests weight ablation execution, paired statistics, and resumability.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from experiments_v2.attacks import AttackSpec
from experiments_v2.weight_ablation import (
    get_weight_configurations,
    run_weight_ablation,
)


def test_weight_configurations():
    cfgs = get_weight_configurations()
    assert len(cfgs) == 6
    ids = [c["id"] for c in cfgs]
    assert ids == ["A", "B", "C", "D", "E", "F"]
    for c in cfgs:
        assert np.isclose(c["w_grad"] + c["w_var"] + c["w_ent"], 1.0, atol=1e-3)


def test_weight_ablation_resumable_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_attacks = [
            AttackSpec(attack="no_attack", severity="none", is_stochastic=False),
        ]
        test_cfgs = get_weight_configurations()[:2]  # Config A and B

        df_summary = run_weight_ablation(
            images_dir="data/internet_images",
            output_dir=tmpdir,
            base_alpha=30.0,
            subband="HL",
            seeds=[1],
            workers=1,
            attacks=test_attacks,
            configs=test_cfgs,
        )

        raw_csv = os.path.join(tmpdir, "weight_ablation_raw.csv")
        assert os.path.exists(raw_csv)
        df_raw = pd.read_csv(raw_csv)
        # 2 configs * 5 images * 1 attack * 1 seed = 10 rows
        assert len(df_raw) == 10

        paired_csv = os.path.join(tmpdir, "weight_ablation_paired_tests.csv")
        assert os.path.exists(paired_csv)
        df_paired = pd.read_csv(paired_csv)
        assert len(df_paired) == 1
        assert df_paired.iloc[0]["n_images"] == 5

        # Rerun to check resumability
        run_weight_ablation(
            images_dir="data/internet_images",
            output_dir=tmpdir,
            base_alpha=30.0,
            subband="HL",
            seeds=[1],
            workers=1,
            attacks=test_attacks,
            configs=test_cfgs,
        )
        df_raw2 = pd.read_csv(raw_csv)
        assert len(df_raw2) == 10
