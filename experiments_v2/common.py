"""
Common Utilities, Watermarking Testbed Harness, and Multiprocessing Infrastructure
for experiments_v2.

Strictly adheres to:
- Deterministic seeding with np.random.default_rng(seed)
- 20-column long-format ResultRow schema
- Resumability (skips completed trials)
- Multiprocessing with --workers flag
"""

import os
import sys
import json
import subprocess
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np
import cv2

from watermark.core import CascadedDwtDctSvdWatermarker
from watermark.masking import DeterministicTextureMasker
from watermark.metrics import (
    compute_psnr,
    compute_ssim,
    compute_nc,
    compute_ber,
    compute_bcr,
    compute_mse,
)
from experiments_v2.schema import (
    ResultRow,
    make_trial_key,
    load_completed_keys,
    append_result_rows,
)
from experiments_v2.attacks import AttackSpec, apply_attack


@functools.lru_cache(maxsize=1)
def get_git_commit_hash() -> str:
    """Returns the current Git commit hash, cached for deterministic recording."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class ConstantAlphaMasker(DeterministicTextureMasker):
    """
    Fixed-alpha baseline control masker.
    Provides a strictly uniform alpha matrix across all subband blocks.
    """
    def __init__(self, base_alpha: float = 30.0):
        super().__init__(base_alpha=base_alpha)
        self.fixed_alpha = float(base_alpha)

    def generate_block_alpha_matrix(
        self,
        image: np.ndarray,
        target_subband_shape: Tuple[int, int],
        block_size: int = 8,
        base_alpha: Optional[float] = None,
    ) -> np.ndarray:
        sub_h, sub_w = target_subband_shape
        num_blocks_h = sub_h // block_size
        num_blocks_w = sub_w // block_size
        val = float(base_alpha if base_alpha is not None else self.fixed_alpha)
        return np.full((num_blocks_h, num_blocks_w), fill_value=val, dtype=np.float64)


def load_standard_test_images(images_dir: str = "data/internet_images") -> Dict[str, np.ndarray]:
    """
    Loads and standardizes the benchmark test images (512x512 grayscale float64).
    """
    image_configs = [
        ("Lena", "lena.png"),
        ("Baboon", "baboon.jpg"),
        ("Fruits", "fruits.jpg"),
        ("Building", "building.jpg"),
        ("Butterfly", "butterfly.jpg"),
    ]
    images: Dict[str, np.ndarray] = {}
    for name, filename in image_configs:
        path = os.path.join(images_dir, filename)
        if not os.path.exists(path):
            # Try png/jpg fallback
            alt_ext = ".jpg" if filename.endswith(".png") else ".png"
            alt_path = os.path.join(images_dir, filename.rsplit(".", 1)[0] + alt_ext)
            path = alt_path if os.path.exists(alt_path) else path

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Test image '{name}' could not be loaded from '{path}'")
        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        images[name] = img.astype(np.float64)
    return images


def load_standard_watermark(
    images_dir: str = "data/internet_images",
    target_size: Tuple[int, int] = (32, 32),
    filename: str = "watermark_logo.png",
    payload_id: str = "logo_32x32",
) -> Tuple[str, np.ndarray]:
    """
    Loads and binarizes the standard 32x32 binary watermark payload.
    """
    path = os.path.join(images_dir, filename)
    if not os.path.exists(path):
        # Fallback to binary_watermark_32x32.png if present
        alt_path = os.path.join(images_dir, "binary_watermark_32x32.png")
        path = alt_path if os.path.exists(alt_path) else path

    wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Watermark image not found at '{path}'")
    wm = cv2.resize(wm, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    thresh = (float(np.max(wm)) + float(np.min(wm))) / 2.0
    binary_wm = (wm >= thresh).astype(np.uint8)
    return payload_id, binary_wm


def evaluate_single_trial(
    run_id: str,
    experiment: str,
    image_id: str,
    cover_image: np.ndarray,
    payload_id: str,
    watermark_payload: np.ndarray,
    method: str,
    subband: str,
    wavelet: str,
    block_size: int,
    alpha_nominal: float,
    attack_spec: AttackSpec,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
    git_commit: Optional[str] = None,
) -> ResultRow:
    """
    Executes a single, fully deterministic watermarking trial under specified attack.
    Returns a validated ResultRow adhering to the 20-column long-format schema.
    """
    cfg = config or {}
    g_hash = git_commit or get_git_commit_hash()

    # Step 1: Configure masker
    if method == "Fixed":
        masker = ConstantAlphaMasker(base_alpha=alpha_nominal)
    else:
        w_grad = float(cfg.get("weight_gradient", 0.40))
        w_var = float(cfg.get("weight_variance", 0.40))
        w_ent = float(cfg.get("weight_entropy", 0.20))
        win_size = int(cfg.get("window_size", 7))
        alpha_min = float(cfg.get("alpha_min", 0.05))
        alpha_max = float(cfg.get("alpha_max", 0.25))

        masker = DeterministicTextureMasker(
            weight_gradient=w_grad,
            weight_variance=w_var,
            weight_entropy=w_ent,
            window_size=win_size,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            base_alpha=alpha_nominal,
        )

    # Step 2: Initialize engine
    watermarker = CascadedDwtDctSvdWatermarker(
        wavelet=wavelet,
        target_subband=subband,
        block_size=block_size,
        base_alpha=alpha_nominal,
        texture_masker=masker,
    )

    # Step 3: Embed
    wm_result = watermarker.embed(
        cover_image=cover_image,
        watermark_payload=watermark_payload,
        base_alpha=alpha_nominal,
    )
    watermarked_image = wm_result.watermarked_image
    alpha_eff_mean = float(np.mean(wm_result.alpha_matrix))

    # Imperceptibility metrics
    psnr_clean = float(compute_psnr(cover_image, watermarked_image))
    ssim_clean = float(compute_ssim(cover_image, watermarked_image))

    # Step 4: Deterministic Attack
    # Using np.random.default_rng(seed) ensures strict reproducibility
    rng = np.random.default_rng(seed) if attack_spec.is_stochastic else None
    attacked_image = attack_spec.apply(watermarked_image, rng=rng)

    # Step 5: Extract
    deskew = attack_spec.deskew_angle
    ext_result = watermarker.extract(
        attacked_image=attacked_image,
        metadata=wm_result.metadata,
        rotation_sync_angle=deskew,
    )
    extracted_binary = ext_result.extracted_binary

    # Robustness metrics
    psnr_attacked = float(compute_psnr(cover_image, attacked_image))
    ber = float(compute_ber(watermark_payload, extracted_binary))
    bcr = float(compute_bcr(watermark_payload, extracted_binary))
    nc = float(compute_nc(watermark_payload, extracted_binary))

    # Step 6: Construct ResultRow
    extra_data = {
        "git_commit": g_hash,
        "config": cfg,
        "deskew_angle": deskew,
        "is_stochastic": attack_spec.is_stochastic,
    }

    return ResultRow(
        run_id=run_id,
        experiment=experiment,
        image_id=image_id,
        payload_id=payload_id,
        method=method,
        subband=subband,
        wavelet=wavelet,
        block_size=block_size,
        alpha_nominal=float(alpha_nominal),
        alpha_eff_mean=round(alpha_eff_mean, 6),
        attack=attack_spec.attack,
        severity=attack_spec.severity,
        seed=seed,
        psnr_clean=round(psnr_clean, 4),
        ssim_clean=round(ssim_clean, 6),
        psnr_attacked=round(psnr_attacked, 4),
        ber=round(ber, 6),
        bcr=round(bcr, 4),
        nc=round(nc, 6),
        extra_json=json.dumps(extra_data, sort_keys=True),
    )


@dataclass
class TrialTask:
    """Serializable task descriptor for multiprocessing execution."""
    run_id: str
    experiment: str
    image_id: str
    payload_id: str
    method: str
    subband: str
    wavelet: str
    block_size: int
    alpha_nominal: float
    attack_spec: AttackSpec
    seed: int
    config: Dict[str, Any]
    git_commit: str

    def key(self) -> Tuple:
        return make_trial_key(
            experiment=self.experiment,
            image_id=self.image_id,
            payload_id=self.payload_id,
            method=self.method,
            subband=self.subband,
            wavelet=self.wavelet,
            block_size=self.block_size,
            alpha_nominal=self.alpha_nominal,
            attack=self.attack_spec.attack,
            severity=self.attack_spec.severity,
            seed=self.seed,
        )


def _worker_execute_task(task_and_data: Tuple[TrialTask, np.ndarray, np.ndarray]) -> ResultRow:
    """Top-level worker function for multiprocessing."""
    task, cover_image, watermark_payload = task_and_data
    return evaluate_single_trial(
        run_id=task.run_id,
        experiment=task.experiment,
        image_id=task.image_id,
        cover_image=cover_image,
        payload_id=task.payload_id,
        watermark_payload=watermark_payload,
        method=task.method,
        subband=task.subband,
        wavelet=task.wavelet,
        block_size=task.block_size,
        alpha_nominal=task.alpha_nominal,
        attack_spec=task.attack_spec,
        seed=task.seed,
        config=task.config,
        git_commit=task.git_commit,
    )
