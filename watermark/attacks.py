"""
Simulated Attack Suite Module
Provides standard signal processing and geometric distortion simulations
for watermarking robustness evaluation.
"""

from typing import Dict, Any, Callable
import numpy as np
import cv2


class AttackSuite:
    """
    Standardized attack simulation testbench.
    """

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """Converts float/int image to uint8 [0, 255]."""
        img = np.clip(image, 0.0, 255.0)
        return img.astype(np.uint8)

    @staticmethod
    def _to_float64(image: np.ndarray) -> np.ndarray:
        """Converts image to float64 [0.0, 255.0]."""
        return image.astype(np.float64)

    @classmethod
    def no_attack(cls, image: np.ndarray) -> np.ndarray:
        """Identity baseline (no attack)."""
        return image.copy()

    @classmethod
    def jpeg_compression(cls, image: np.ndarray, quality: int = 70) -> np.ndarray:
        """
        Simulates standard lossy JPEG compression and decompression.

        Args:
            image: Input image.
            quality: JPEG quality factor (1 - 100).
        """
        img_u8 = cls._to_uint8(image)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encimg = cv2.imencode(".jpg", img_u8, encode_param)
        if not success:
            raise RuntimeError("JPEG encoding failed.")
        decimg = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
        return cls._to_float64(decimg)

    @classmethod
    def gaussian_noise(
        cls, image: np.ndarray, mean: float = 0.0, variance: float = 0.01
    ) -> np.ndarray:
        """
        Adds zero-mean Gaussian noise.

        Args:
            image: Input image in [0, 255].
            mean: Gaussian mean (normalized).
            variance: Gaussian variance (normalized to [0, 1] range).
        """
        img_f = image.astype(np.float64) / 255.0
        sigma = np.sqrt(variance)
        noise = np.random.normal(mean, sigma, img_f.shape)
        noisy = np.clip(img_f + noise, 0.0, 1.0) * 255.0
        return noisy

    @classmethod
    def speckle_noise(cls, image: np.ndarray, variance: float = 0.01) -> np.ndarray:
        """
        Adds multiplicative speckle noise: I_noisy = I + I * N(0, var).
        """
        img_f = image.astype(np.float64) / 255.0
        sigma = np.sqrt(variance)
        noise = np.random.normal(0.0, sigma, img_f.shape)
        noisy = np.clip(img_f + (img_f * noise), 0.0, 1.0) * 255.0
        return noisy

    @classmethod
    def salt_and_pepper_noise(
        cls, image: np.ndarray, amount: float = 0.01, s_vs_p: float = 0.5
    ) -> np.ndarray:
        """
        Adds salt & pepper impulse noise.
        """
        img_f = image.copy()
        num_salt = np.ceil(amount * img_f.size * s_vs_p)
        num_pepper = np.ceil(amount * img_f.size * (1.0 - s_vs_p))

        # Salt
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_f.shape]
        img_f[tuple(coords)] = 255.0

        # Pepper
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_f.shape]
        img_f[tuple(coords)] = 0.0

        return img_f

    @classmethod
    def median_filter(cls, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Applies non-linear median filtering.
        """
        img_u8 = cls._to_uint8(image)
        filtered = cv2.medianBlur(img_u8, kernel_size)
        return cls._to_float64(filtered)

    @classmethod
    def gaussian_blur(
        cls, image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0
    ) -> np.ndarray:
        """
        Applies linear Gaussian low-pass spatial filtering.
        """
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = cv2.GaussianBlur(image.astype(np.float32), (k, k), sigma)
        return cls._to_float64(blurred)

    @classmethod
    def cropping(
        cls,
        image: np.ndarray,
        crop_ratio: float = 0.10,
        fill_mode: str = "black",
    ) -> np.ndarray:
        """
        Simulates cropping attack (removes outer border) and restores shape with padding.

        Args:
            image: Input image.
            crop_ratio: Fraction of border to crop (e.g. 0.10 = 10% from borders).
            fill_mode: 'black' (fill with 0) or 'original' (restore cropped area with original).
        """
        h, w = image.shape[:2]
        ch = int(round(h * crop_ratio / 2.0))
        cw = int(round(w * crop_ratio / 2.0))

        cropped = image.copy()
        # Zero-out / mask the 10% border area
        mask = np.zeros((h, w), dtype=bool)
        mask[ch : h - ch, cw : w - cw] = True

        if fill_mode == "black":
            cropped[~mask] = 0.0

        return cropped

    @classmethod
    def rotation(
        cls,
        image: np.ndarray,
        angle_deg: float = 2.0,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: float = 0.0,
    ) -> np.ndarray:
        """
        Simulates geometric rotation attack.
        Produces a genuine rotated image with realistic boundary padding.

        Args:
            image: Input image.
            angle_deg: Rotation angle in degrees (e.g., 1.0, 2.0, 5.0, 10.0).
            border_mode: Border extrapolation method (default: cv2.BORDER_CONSTANT).
            border_value: Padding value for constant border (default: 0.0 black).
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(
            image.astype(np.float32),
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=border_value,
        )
        return cls._to_float64(rotated)

    @classmethod
    def rotate_orthogonal(cls, image: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Rotates image by 90 * k degrees counter-clockwise (k=1: 90 deg, k=2: 180 deg, k=3: 270 deg).
        """
        rotated = np.rot90(image, k=k)
        return cls._to_float64(rotated)

    @classmethod
    def contrast_adjustment(
        cls, image: np.ndarray, alpha: float = 1.15, beta: float = 5.0
    ) -> np.ndarray:
        """
        Applies linear contrast scaling and brightness offset: I_out = alpha * I + beta.
        """
        adjusted = np.clip(alpha * image + beta, 0.0, 255.0)
        return cls._to_float64(adjusted)

    @classmethod
    def get_standard_attack_suite(cls) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        """
        Returns an ordered dictionary of all standard benchmarking attack functions.
        """
        return {
            "No Attack": cls.no_attack,
            "JPEG Compression (Q=90)": lambda img: cls.jpeg_compression(img, quality=90),
            "JPEG Compression (Q=70)": lambda img: cls.jpeg_compression(img, quality=70),
            "JPEG Compression (Q=50)": lambda img: cls.jpeg_compression(img, quality=50),
            "Gaussian Noise (var=0.01)": lambda img: cls.gaussian_noise(img, variance=0.01),
            "Speckle Noise (var=0.01)": lambda img: cls.speckle_noise(img, variance=0.01),
            "Salt & Pepper Noise (1%)": lambda img: cls.salt_and_pepper_noise(img, amount=0.01),
            "Median Filter (3x3)": lambda img: cls.median_filter(img, kernel_size=3),
            "Gaussian Blur (3x3, s=1.0)": lambda img: cls.gaussian_blur(img, kernel_size=3, sigma=1.0),
            "Border Cropping (10%)": lambda img: cls.cropping(img, crop_ratio=0.10, fill_mode="black"),
            "Rotation (2 deg)": lambda img: cls.rotation(img, angle_deg=2.0),
            "Rotation (5 deg)": lambda img: cls.rotation(img, angle_deg=5.0),
            "Contrast Adjust (x1.15)": lambda img: cls.contrast_adjustment(img, alpha=1.15, beta=5.0),
        }
