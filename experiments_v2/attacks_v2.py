"""
Comprehensive Extended Attack Suite and Compound Degradation Chains Module
for experiments_v2.

Strict requirements:
- All attacks take (img, severity, rng) and return float64 images in [0, 255].
- JPEG goes through real encoding via cv2.imencode/cv2.imdecode.
- Single-attack severity sweeps (>= 6 levels each).
- Compound (sequential) chains applied in exact specified order.
"""

from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Any, Union
import numpy as np
import cv2


@dataclass(frozen=True)
class AttackSpecV2:
    """Specification of an extended attack or compound chain."""
    attack: str
    severity: str
    is_stochastic: bool
    fn: Callable[[np.ndarray, Any, Optional[np.random.Generator]], np.ndarray]
    deskew_angle: Optional[float] = None

    def apply(self, img: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return self.fn(img, self.severity, rng=rng)


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _to_f64(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64)


# =============================================================================
# SINGLE ATTACKS (TAKE (img, severity, rng) -> float64)
# =============================================================================

def attack_jpeg(
    img: np.ndarray,
    severity: Union[int, float, str] = 70,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Real lossy JPEG compression via OpenCV."""
    q = int(str(severity).replace("Q=", "").strip())
    img_u8 = _to_u8(img)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    success, encimg = cv2.imencode(".jpg", img_u8, encode_param)
    if not success:
        raise RuntimeError(f"JPEG encoding failed with quality={q}")
    decimg = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
    return _to_f64(decimg)


def attack_gaussian_noise(
    img: np.ndarray,
    severity: Union[float, str] = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Zero-mean additive Gaussian noise."""
    var = float(str(severity).replace("var=", "").strip())
    gen = rng if rng is not None else np.random.default_rng(0)
    img_f = img.astype(np.float64) / 255.0
    sigma = np.sqrt(var)
    noise = gen.normal(0.0, sigma, img_f.shape)
    noisy = np.clip(img_f + noise, 0.0, 1.0) * 255.0
    return _to_f64(noisy)


def attack_salt_and_pepper(
    img: np.ndarray,
    severity: Union[float, str] = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Impulse salt & pepper noise."""
    s_str = str(severity).replace("%", "").strip()
    amt = float(s_str) / 100.0 if "%" in str(severity) else float(s_str)
    gen = rng if rng is not None else np.random.default_rng(0)

    out = img.copy().astype(np.float64)
    num_salt = int(np.ceil(amt * out.size * 0.5))
    num_pepper = int(np.ceil(amt * out.size * 0.5))

    coords_salt = tuple(gen.integers(0, dim, num_salt) for dim in out.shape)
    out[coords_salt] = 255.0
    coords_pepper = tuple(gen.integers(0, dim, num_pepper) for dim in out.shape)
    out[coords_pepper] = 0.0
    return out


def attack_gaussian_blur(
    img: np.ndarray,
    severity: Union[float, str] = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Linear spatial Gaussian low-pass spatial filtering."""
    sigma = float(str(severity).replace("s=", "").replace("sigma=", "").strip())
    k = max(3, int(2 * np.ceil(2 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(img.astype(np.float32), (k, k), sigma)
    return _to_f64(blurred)


def attack_median_filter(
    img: np.ndarray,
    severity: Union[int, str] = 3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Non-linear median filtering."""
    k = int(str(severity).split("x")[0].strip())
    if k % 2 == 0:
        k += 1
    img_u8 = _to_u8(img)
    filtered = cv2.medianBlur(img_u8, k)
    return _to_f64(filtered)


def attack_centered_crop(
    img: np.ndarray,
    severity: Union[float, str] = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Centered outer border cropping with zero padding."""
    s_str = str(severity).replace("%", "").strip()
    pct = float(s_str)
    ratio = pct / 100.0 if pct > 1.0 else pct

    h, w = img.shape[:2]
    ch = int(round(h * ratio / 2.0))
    cw = int(round(w * ratio / 2.0))

    cropped = img.copy().astype(np.float64)
    mask = np.zeros((h, w), dtype=bool)
    mask[ch: h - ch, cw: w - cw] = True
    cropped[~mask] = 0.0
    return cropped


def attack_random_crop(
    img: np.ndarray,
    severity: Union[float, str] = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Random-position rectangular crop removing the exact same area as centered crop.
    Zero-pads the removed rectangle.
    """
    s_str = str(severity).replace("%", "").strip()
    pct = float(s_str)
    # Area fraction removed by centered crop of ratio r is 2r - r^2
    r = pct / 100.0 if pct > 1.0 else pct
    area_ratio = 2.0 * r - (r ** 2)

    gen = rng if rng is not None else np.random.default_rng(0)
    h, w = img.shape[:2]
    total_area = h * w
    target_area = area_ratio * total_area

    # Aspect ratio random between 0.8 and 1.25
    aspect = gen.uniform(0.8, 1.25)
    crop_h = int(np.clip(np.round(np.sqrt(target_area / aspect)), 1, h))
    crop_w = int(np.clip(np.round(np.sqrt(target_area * aspect)), 1, w))

    # Random top-left corner
    max_y = max(0, h - crop_h)
    max_x = max(0, w - crop_w)
    y0 = int(gen.integers(0, max_y + 1)) if max_y > 0 else 0
    x0 = int(gen.integers(0, max_x + 1)) if max_x > 0 else 0

    out = img.copy().astype(np.float64)
    out[y0: y0 + crop_h, x0: x0 + crop_w] = 0.0
    return out


def attack_scaling(
    img: np.ndarray,
    severity: Union[float, str] = 0.75,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Scale by factor, then restore to 512x512 via cv2.INTER_LINEAR."""
    scale = float(str(severity).replace("x", "").strip())
    h, w = img.shape[:2]
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))

    scaled = cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    restored = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
    return _to_f64(restored)


def attack_translation(
    img: np.ndarray,
    severity: Union[int, float, str] = 4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Spatial translation along both axes with reflected border."""
    shift = float(str(severity).replace("px", "").strip())
    h, w = img.shape[:2]
    mat = np.array([[1.0, 0.0, shift], [0.0, 1.0, shift]], dtype=np.float32)
    shifted = cv2.warpAffine(
        img.astype(np.float32),
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return _to_f64(shifted)


def attack_shear(
    img: np.ndarray,
    severity: Union[float, str] = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Affine shear distortion with reflected border."""
    sh = float(str(severity).strip())
    h, w = img.shape[:2]
    mat = np.array([[1.0, sh, 0.0], [sh, 1.0, 0.0]], dtype=np.float32)
    sheared = cv2.warpAffine(
        img.astype(np.float32),
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return _to_f64(sheared)


def attack_rotation(
    img: np.ndarray,
    severity: Union[float, str] = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Geometric rotation without detector registration."""
    deg_str = str(severity).replace("deg", "").replace("°", "").strip()
    angle = float(deg_str)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img.astype(np.float32),
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return _to_f64(rotated)


def attack_contrast(
    img: np.ndarray,
    severity: Union[float, str] = 1.15,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Linear contrast scaling: I_out = alpha * I + 5.0."""
    c_str = str(severity).replace("x", "").strip()
    alpha = float(c_str)
    adjusted = np.clip(alpha * img.astype(np.float64) + 5.0, 0.0, 255.0)
    return adjusted


def attack_gamma(
    img: np.ndarray,
    severity: Union[float, str] = 1.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Non-linear gamma correction: I_out = 255 * (I / 255)^gamma."""
    gamma = float(str(severity).replace("g=", "").strip())
    norm = np.clip(img.astype(np.float64) / 255.0, 0.0, 1.0)
    adjusted = (norm ** gamma) * 255.0
    return np.clip(adjusted, 0.0, 255.0)


# =============================================================================
# COMPOUND (SEQUENTIAL) DEGRADATION CHAINS
# =============================================================================

def chain_crop10_jpeg70(img: np.ndarray, severity: str = "default", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chain 1: crop10% + JPEG70."""
    step1 = attack_centered_crop(img, severity=10.0, rng=rng)
    step2 = attack_jpeg(step1, severity=70, rng=rng)
    return step2


def chain_rotate2_jpeg70(img: np.ndarray, severity: str = "default", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chain 2: rotate2° (no registration) + JPEG70."""
    step1 = attack_rotation(img, severity=2.0, rng=rng)
    step2 = attack_jpeg(step1, severity=70, rng=rng)
    return step2


def chain_downscale_jpeg70(img: np.ndarray, severity: str = "default", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chain 3: downscale 0.75 then resize back to 512 + JPEG70."""
    step1 = attack_scaling(img, severity=0.75, rng=rng)
    step2 = attack_jpeg(step1, severity=70, rng=rng)
    return step2


def chain_blur_jpeg70(img: np.ndarray, severity: str = "default", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chain 4: blur(s=1) + JPEG70."""
    step1 = attack_gaussian_blur(img, severity=1.0, rng=rng)
    step2 = attack_jpeg(step1, severity=70, rng=rng)
    return step2


def chain_noise_jpeg70(img: np.ndarray, severity: str = "default", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chain 5: Gaussian noise(0.005) + JPEG70."""
    step1 = attack_gaussian_noise(img, severity=0.005, rng=rng)
    step2 = attack_jpeg(step1, severity=70, rng=rng)
    return step2


def chain_jpeg70_blur(img: np.ndarray, severity: str = "default", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chain 6: JPEG70 + blur(s=1)."""
    step1 = attack_jpeg(img, severity=70, rng=rng)
    step2 = attack_gaussian_blur(step1, severity=1.0, rng=rng)
    return step2


def chain_crop_resize_jpeg50(img: np.ndarray, severity: str = "default", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chain 7: crop10% + resize back to 512 + JPEG50."""
    h, w = img.shape[:2]
    ch = int(round(h * 0.10 / 2.0))
    cw = int(round(w * 0.10 / 2.0))
    cropped = img[ch: h - ch, cw: w - cw]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    step2 = attack_jpeg(resized, severity=50, rng=rng)
    return step2


def chain_contrast_noise_jpeg70(img: np.ndarray, severity: str = "default", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Chain 8: contrast(1.15) + noise(0.005) + JPEG70."""
    step1 = attack_contrast(img, severity=1.15, rng=rng)
    step2 = attack_gaussian_noise(step1, severity=0.005, rng=rng)
    step3 = attack_jpeg(step2, severity=70, rng=rng)
    return step3


# =============================================================================
# ATTACK SPECIFICATIONS & SWEEPS GENERATOR
# =============================================================================

def get_single_attack_specs() -> List[Tuple[AttackSpecV2, List[int]]]:
    """
    Returns ordered list of (AttackSpecV2, seed_list) for all single-attack severity sweeps.
    Stochastic attacks run with 10 seeds; deterministic run with [42].
    """
    stoch_seeds = list(range(1, 11))  # 10 seeds
    det_seeds = [42]
    specs = []

    # 1. JPEG quality: 95, 90, 80, 70, 60, 50, 40, 30 (8 levels)
    for q in [95, 90, 80, 70, 60, 50, 40, 30]:
        specs.append((
            AttackSpecV2("jpeg", f"Q={q}", False, attack_jpeg),
            det_seeds,
        ))

    # 2. Gaussian noise: 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02 (6 levels)
    for v in [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]:
        specs.append((
            AttackSpecV2("gaussian_noise", f"var={v}", True, attack_gaussian_noise),
            stoch_seeds,
        ))

    # 3. Salt & pepper: 0.1%, 0.5%, 1%, 2%, 5% (5 levels)
    for sp in ["0.1%", "0.5%", "1%", "2%", "5%"]:
        specs.append((
            AttackSpecV2("salt_and_pepper", sp, True, attack_salt_and_pepper),
            stoch_seeds,
        ))

    # 4. Gaussian blur: sigma 0.5, 0.75, 1, 1.5, 2, 3 (6 levels)
    for s in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        specs.append((
            AttackSpecV2("gaussian_blur", f"s={s}", False, attack_gaussian_blur),
            det_seeds,
        ))

    # 5. Median: 3, 5, 7 (3 levels)
    for m in [3, 5, 7]:
        specs.append((
            AttackSpecV2("median_filter", f"{m}x{m}", False, attack_median_filter),
            det_seeds,
        ))

    # 6. Centered border crop: 2.5%, 5%, 10%, 15%, 20%, 30% (6 levels)
    for c in [2.5, 5.0, 10.0, 15.0, 20.0, 30.0]:
        specs.append((
            AttackSpecV2("centered_crop", f"{c}%", False, attack_centered_crop),
            det_seeds,
        ))

    # 7. Random-position rectangular crop: same areas (6 levels)
    for c in [2.5, 5.0, 10.0, 15.0, 20.0, 30.0]:
        specs.append((
            AttackSpecV2("random_crop", f"{c}%", True, attack_random_crop),
            stoch_seeds,
        ))

    # 8. Scaling: 0.5, 0.75, 0.9, 1.1, 1.25, 1.5 (6 levels)
    for sc in [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]:
        specs.append((
            AttackSpecV2("scaling", f"{sc}x", False, attack_scaling),
            det_seeds,
        ))

    # 9. Translation: 1, 2, 4, 8, 16 px (5 levels)
    for tr in [1, 2, 4, 8, 16]:
        specs.append((
            AttackSpecV2("translation", f"{tr}px", False, attack_translation),
            det_seeds,
        ))

    # 10. Shear: 0.01, 0.02, 0.05 (3 levels)
    for sh in [0.01, 0.02, 0.05]:
        specs.append((
            AttackSpecV2("shear", f"{sh}", False, attack_shear),
            det_seeds,
        ))

    # 11. Rotation: 0.5, 1, 2, 5, 10 degrees (no registration) (5 levels)
    for r in [0.5, 1.0, 2.0, 5.0, 10.0]:
        specs.append((
            AttackSpecV2("rotation", f"{r}deg", False, attack_rotation, deskew_angle=None),
            det_seeds,
        ))

    # 12. Contrast & Gamma: 0.8, 0.9, 1.1, 1.2 (4 levels each)
    for ct in [0.8, 0.9, 1.1, 1.2]:
        specs.append((
            AttackSpecV2("contrast", f"x{ct}", False, attack_contrast),
            det_seeds,
        ))
    for gm in [0.8, 0.9, 1.1, 1.2]:
        specs.append((
            AttackSpecV2("gamma", f"g={gm}", False, attack_gamma),
            det_seeds,
        ))

    return specs


def get_compound_chain_specs() -> List[Tuple[AttackSpecV2, List[int]]]:
    """
    Returns the 8 compound degradation chains applied in exact sequential order.
    """
    stoch_seeds = list(range(1, 11))  # 10 seeds for noise chains
    det_seeds = [42]

    return [
        (AttackSpecV2("compound", "crop10%+JPEG70", False, chain_crop10_jpeg70), det_seeds),
        (AttackSpecV2("compound", "rotate2°+JPEG70", False, chain_rotate2_jpeg70, deskew_angle=None), det_seeds),
        (AttackSpecV2("compound", "downscale0.75+resize+JPEG70", False, chain_downscale_jpeg70), det_seeds),
        (AttackSpecV2("compound", "blur(s=1)+JPEG70", False, chain_blur_jpeg70), det_seeds),
        (AttackSpecV2("compound", "noise(0.005)+JPEG70", True, chain_noise_jpeg70), stoch_seeds),
        (AttackSpecV2("compound", "JPEG70+blur(s=1)", False, chain_jpeg70_blur), det_seeds),
        (AttackSpecV2("compound", "crop10%+resize+JPEG50", False, chain_crop_resize_jpeg50), det_seeds),
        (AttackSpecV2("compound", "contrast+noise+JPEG70", True, chain_contrast_noise_jpeg70), stoch_seeds),
    ]
