"""
Payload Generation and Statistical Integrity Module for experiments_v2.

Implements three 32x32 (1,024-bit) binary payload families:
1. Original logo (watermark_logo.png)
2. Whitened logo (logo XOR PRNG keystream, seeded, ones ≈ 50%)
3. Balanced random payloads (exactly 512 ones, payload_id = 0..N-1)

Computes and saves baseline bit error rates under trivial detectors
(all-zero, all-one, random-guess) to results_v2/payload_stats.csv.
"""

import os
import csv
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2
import pandas as pd


def generate_original_logo(
    images_dir: str = "data/internet_images",
    target_size: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Loads and binarizes the original 32x32 watermark logo."""
    path = os.path.join(images_dir, "watermark_logo.png")
    if not os.path.exists(path):
        alt_path = os.path.join(images_dir, "binary_watermark_32x32.png")
        path = alt_path if os.path.exists(alt_path) else path

    wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise FileNotFoundError(f"Watermark file not found at: {path}")

    wm = cv2.resize(wm, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    thresh = (float(np.max(wm)) + float(np.min(wm))) / 2.0
    binary_wm = (wm >= thresh).astype(np.uint8)
    return binary_wm


def generate_whitened_logo(
    original_logo: Optional[np.ndarray] = None,
    seed: int = 42,
    images_dir: str = "data/internet_images",
    target_size: Tuple[int, int] = (32, 32),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a whitened logo via XOR with a deterministic PRNG keystream.
    ones ≈ 50%.

    Returns:
        (whitened_payload, keystream)
    """
    logo = (
        original_logo
        if original_logo is not None
        else generate_original_logo(images_dir=images_dir, target_size=target_size)
    )
    rng = np.random.default_rng(seed)
    keystream = rng.integers(0, 2, size=logo.shape, dtype=np.uint8)
    whitened = np.bitwise_xor(logo, keystream).astype(np.uint8)
    return whitened, keystream


def unwhiten_payload(whitened_payload: np.ndarray, keystream: np.ndarray) -> np.ndarray:
    """Inverts the whitening transformation via XOR."""
    return np.bitwise_xor(whitened_payload, keystream).astype(np.uint8)


def generate_balanced_random_payload(
    payload_id: int,
    seed: Optional[int] = None,
    target_size: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """
    Generates an i.i.d. balanced random binary payload containing exactly
    half ones (512 ones for 1,024 bits).
    """
    total_bits = target_size[0] * target_size[1]
    num_ones = total_bits // 2
    bits = np.zeros(total_bits, dtype=np.uint8)
    bits[:num_ones] = 1

    actual_seed = seed if seed is not None else int(payload_id)
    rng = np.random.default_rng(actual_seed)
    rng.shuffle(bits)

    return bits.reshape(target_size)


def compute_payload_stats(
    payload: np.ndarray,
    payload_name: str,
    payload_type: str,
    payload_id: str,
    seed: int = 42,
    n_random_trials: int = 100,
) -> Dict[str, Any]:
    """
    Calculates bit distribution and baseline detector error rates.
    """
    flat = payload.ravel().astype(np.float64)
    total_bits = len(flat)
    num_ones = int(np.sum(flat >= 0.5))
    num_zeros = total_bits - num_ones
    frac_ones = float(num_ones) / float(total_bits)

    # BER under trivial detectors
    # All-zero output: errors occur on all ones
    ber_all_zeros = float(np.mean(np.abs(flat - 0.0)))
    # All-one output: errors occur on all zeros
    ber_all_ones = float(np.mean(np.abs(flat - 1.0)))

    # Random-guess output (Bernoulli 0.5)
    rng = np.random.default_rng(seed)
    random_bers = []
    for _ in range(n_random_trials):
        guess = rng.integers(0, 2, size=total_bits).astype(np.float64)
        random_bers.append(float(np.mean(np.abs(flat - guess))))
    ber_random_guess = float(np.mean(random_bers))

    return {
        "payload_id": payload_id,
        "payload_name": payload_name,
        "payload_type": payload_type,
        "total_bits": total_bits,
        "num_ones": num_ones,
        "num_zeros": num_zeros,
        "fraction_ones": round(frac_ones, 4),
        "ber_all_zeros": round(ber_all_zeros, 4),
        "ber_all_ones": round(ber_all_ones, 4),
        "ber_random_guess": round(ber_random_guess, 4),
    }


def generate_all_payload_stats(
    images_dir: str = "data/internet_images",
    output_csv: str = "results_v2/payload_stats.csv",
    n_balanced_samples: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates and records payload statistics for all three payload families.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    stats_list: List[Dict[str, Any]] = []

    # 1. Original Logo
    orig_logo = generate_original_logo(images_dir=images_dir)
    stats_list.append(
        compute_payload_stats(
            orig_logo,
            payload_name="Original Logo (watermark_logo.png)",
            payload_type="original_logo",
            payload_id="logo_orig",
            seed=seed,
        )
    )

    # 2. Whitened Logo (tested with 3 seeds)
    for s in [42, 101, 2024]:
        whitened_logo, _ = generate_whitened_logo(orig_logo, seed=s)
        stats_list.append(
            compute_payload_stats(
                whitened_logo,
                payload_name=f"Whitened Logo (Seed {s})",
                payload_type="whitened_logo",
                payload_id=f"logo_whitened_s{s}",
                seed=s,
            )
        )

    # 3. Balanced Random Payloads (payload_id = 0..N-1)
    for pid in range(n_balanced_samples):
        bal_payload = generate_balanced_random_payload(payload_id=pid, seed=pid + 1000)
        stats_list.append(
            compute_payload_stats(
                bal_payload,
                payload_name=f"Balanced Random Payload #{pid}",
                payload_type="balanced_random",
                payload_id=f"balanced_{pid}",
                seed=pid + 1000,
            )
        )

    df = pd.DataFrame(stats_list)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print(" PAYLOAD STATISTICAL INTEGRITY REPORT (results_v2/payload_stats.csv)")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    return df


if __name__ == "__main__":
    generate_all_payload_stats()
