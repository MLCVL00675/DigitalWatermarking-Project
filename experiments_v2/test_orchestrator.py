"""
Pytest suite for experiments_v2 master runner.
"""

import os
import tempfile
import pytest
from experiments_v2.alpha_sweep import run_alpha_sweep
from experiments_v2.weight_ablation import run_weight_ablation
from experiments_v2.attacks import AttackSpec


def test_quick_orchestration():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_attacks = [AttackSpec(attack="no_attack", severity="none", is_stochastic=False)]
        
        # Alpha sweep quick
        df_alpha = run_alpha_sweep(
            images_dir="data/internet_images",
            output_dir=os.path.join(tmpdir, "alpha_sweep"),
            fixed_alphas=[30.0],
            adaptive_alphas=[30.0],
            subbands=["HL"],
            seeds=[1],
            workers=1,
            attacks=test_attacks,
        )
        assert df_alpha is not None

        # Weight ablation quick
        df_weight = run_weight_ablation(
            images_dir="data/internet_images",
            output_dir=os.path.join(tmpdir, "weight_ablation"),
            base_alpha=30.0,
            subband="HL",
            seeds=[1],
            workers=1,
            attacks=test_attacks,
            configs=[{"id": "A", "name": "Proposed (40/40/20)", "w_grad": 0.40, "w_var": 0.40, "w_ent": 0.20}],
        )
        assert df_weight is not None
