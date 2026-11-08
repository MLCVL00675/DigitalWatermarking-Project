"""
Deterministic Attack Suite Module for experiments_v2.

Enforces strictly seeded, reproducible attack simulations using np.random.default_rng(seed).
Decouples attack family from severity according to Rule 3.
"""

from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any
import numpy as np
import cv2


@dataclass(frozen=True)
class AttackSpec:
    """Specification of an attack simulation."""
    attack: str              # Canonical attack family name (e.g. 'jpeg', 'gaussian_noise')
    severity: str            # Parameter representation (e.g. 'Q=90', 'var=0.01', '2 deg [Sync]')
    is_stochastic: bool      # Whether this attack requires an RNG
    deskew_angle: Optional[float] = None  # Detector synchronization deskew angle in degrees

    def apply(self, image: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return apply_attack(image, self, rng=rng)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _to_float64(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64)


def attack_no_attack(image: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    return image.copy().astype(np.float64)


def attack_jpeg(image: np.ndarray, quality: int = 70, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    img_u8 = _to_uint8(image)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    success, encimg = cv2.imencode(".jpg", img_u8, encode_param)
    if not success:
        raise RuntimeError(f"JPEG encoding failed with quality={quality}")
    decimg = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
    return _to_float64(decimg)


def attack_gaussian_noise(
    image: np.ndarray,
    variance: float = 0.01,
    mean: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    generator = rng if rng is not None else np.random.default_rng(0)
    img_f = image.astype(np.float64) / 255.0
    sigma = np.sqrt(variance)
    noise = generator.normal(mean, sigma, img_f.shape)
    noisy = np.clip(img_f + noise, 0.0, 1.0) * 255.0
    return noisy


def attack_speckle_noise(
    image: np.ndarray,
    variance: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    generator = rng if rng is not None else np.random.default_rng(0)
    img_f = image.astype(np.float64) / 255.0
    sigma = np.sqrt(variance)
    noise = generator.normal(0.0, sigma, img_f.shape)
    noisy = np.clip(img_f + (img_f * noise), 0.0, 1.0) * 255.0
    return noisy


def attack_salt_and_pepper(
    image: np.ndarray,
    amount: float = 0.01,
    s_vs_p: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    generator = rng if rng is not None else np.random.default_rng(0)
    img_f = image.copy().astype(np.float64)
    num_salt = int(np.ceil(amount * img_f.size * s_vs_p))
    num_pepper = int(np.ceil(amount * img_f.size * (1.0 - s_vs_p)))

    coords_salt = tuple(generator.integers(0, dim, num_salt) for dim in img_f.shape)
    img_f[coords_salt] = 255.0

    coords_pepper = tuple(generator.integers(0, dim, num_pepper) for dim in img_f.shape)
    img_f[coords_pepper] = 0.0

    return img_f


def attack_median_filter(
    image: np.ndarray,
    kernel_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    img_u8 = _to_uint8(image)
    filtered = cv2.medianBlur(img_u8, int(kernel_size))
    return _to_float64(filtered)


def attack_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blurred = cv2.GaussianBlur(image.astype(np.float32), (k, k), sigma)
    return _to_float64(blurred)


def attack_cropping(
    image: np.ndarray,
    crop_ratio: float = 0.10,
    fill_mode: str = "black",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    ch = int(round(h * crop_ratio / 2.0))
    cw = int(round(w * crop_ratio / 2.0))
    cropped = image.copy().astype(np.float64)

    mask = np.zeros((h, w), dtype=bool)
    mask[ch: h - ch, cw: w - cw] = True
    if fill_mode == "black":
        cropped[~mask] = 0.0
    return cropped


def attack_rotation(
    image: np.ndarray,
    angle_deg: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image.astype(np.float32),
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return _to_float64(rotated)


def attack_contrast(
    image: np.ndarray,
    alpha: float = 1.15,
    beta: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    adjusted = np.clip(alpha * image.astype(np.float64) + beta, 0.0, 255.0)
    return adjusted


def attack_quantization_8bit(
    image: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulates 8-bit spatial integer quantization rounding."""
    return np.clip(np.round(image), 0.0, 255.0).astype(np.uint8).astype(np.float64)


def apply_attack(
    image: np.ndarray,
    spec: AttackSpec,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Dispatches the attack function corresponding to spec."""
    att = spec.attack.lower()
    sev = spec.severity

    if att in ("no_attack", "none", "identity"):
        return attack_no_attack(image, rng=rng)
    elif att in ("quantization", "8-bit quantization only", "quantization_8bit", "8-bit quantization"):
        return attack_quantization_8bit(image, rng=rng)
    elif att == "jpeg":
        # Extract quality from severity, e.g. 'Q=90' or '90'
        q_str = sev.replace("Q=", "").strip()
        quality = int(q_str) if q_str.isdigit() else 70
        return attack_jpeg(image, quality=quality, rng=rng)
    elif att == "gaussian_noise":
        var_str = sev.replace("var=", "").strip()
        variance = float(var_str) if var_str else 0.01
        return attack_gaussian_noise(image, variance=variance, rng=rng)
    elif att == "speckle_noise":
        var_str = sev.replace("var=", "").strip()
        variance = float(var_str) if var_str else 0.01
        return attack_speckle_noise(image, variance=variance, rng=rng)
    elif att in ("salt_and_pepper", "salt_pepper_noise"):
        amt_str = sev.replace("%", "").strip()
        try:
            amt = float(amt_str) / 100.0 if "%" in sev else float(amt_str)
        except ValueError:
            amt = 0.01
        return attack_salt_and_pepper(image, amount=amt, rng=rng)
    elif att == "median_filter":
        k_str = sev.split("x")[0].strip()
        kernel_size = int(k_str) if k_str.isdigit() else 3
        return attack_median_filter(image, kernel_size=kernel_size, rng=rng)
    elif att == "gaussian_blur":
        k_str = sev.split("x")[0].strip()
        kernel_size = int(k_str) if k_str.isdigit() else 3
        return attack_gaussian_blur(image, kernel_size=kernel_size, rng=rng)
    elif att == "cropping":
        c_str = sev.replace("%", "").strip()
        try:
            ratio = float(c_str) / 100.0 if "%" in sev else float(c_str)
        except ValueError:
            ratio = 0.10
        return attack_cropping(image, crop_ratio=ratio, rng=rng)
    elif att == "rotation":
        deg_str = sev.split("deg")[0].replace("[Sync]", "").replace("[Unsync]", "").strip()
        try:
            angle = float(deg_str)
        except ValueError:
            angle = 2.0
        return attack_rotation(image, angle_deg=angle, rng=rng)
    elif att in ("contrast", "contrast_adjust"):
        alpha_str = sev.replace("x", "").strip()
        try:
            alpha = float(alpha_str)
        except ValueError:
            alpha = 1.15
        return attack_contrast(image, alpha=alpha, rng=rng)
    else:
        raise ValueError(f"Unknown attack family: '{spec.attack}'")


def get_standard_attack_specs() -> List[AttackSpec]:
    """
    Returns the standard 14 attack configurations matching the literature benchmark.
    """
    return [
        AttackSpec(attack="no_attack", severity="none", is_stochastic=False, deskew_angle=None),
        AttackSpec(attack="jpeg", severity="Q=90", is_stochastic=False, deskew_angle=None),
        AttackSpec(attack="jpeg", severity="Q=70", is_stochastic=False, deskew_angle=None),
        AttackSpec(attack="jpeg", severity="Q=50", is_stochastic=False, deskew_angle=None),
        AttackSpec(attack="gaussian_noise", severity="var=0.01", is_stochastic=True, deskew_angle=None),
        AttackSpec(attack="speckle_noise", severity="var=0.01", is_stochastic=True, deskew_angle=None),
        AttackSpec(attack="salt_and_pepper", severity="1%", is_stochastic=True, deskew_angle=None),
        AttackSpec(attack="median_filter", severity="3x3", is_stochastic=False, deskew_angle=None),
        AttackSpec(attack="gaussian_blur", severity="3x3, s=1.0", is_stochastic=False, deskew_angle=None),
        AttackSpec(attack="cropping", severity="10%", is_stochastic=False, deskew_angle=None),
        AttackSpec(attack="rotation", severity="2 deg [Unsync]", is_stochastic=False, deskew_angle=None),
        AttackSpec(attack="rotation", severity="2 deg [Sync]", is_stochastic=False, deskew_angle=2.0),
        AttackSpec(attack="rotation", severity="5 deg [Sync]", is_stochastic=False, deskew_angle=5.0),
        AttackSpec(attack="contrast", severity="x1.15", is_stochastic=False, deskew_angle=None),
    ]
