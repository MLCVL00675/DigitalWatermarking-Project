"""
Transform Ablation Methods Module for experiments_v2.

Implements the 4 core transform ablation variants:
1. DwtSvdMethod: 1-level DWT -> Block SVD (no DCT).
2. DctSvdMethod: Spatial Block DCT -> Block SVD (no DWT).
3. DwtDctMethod: 1-level DWT -> Block DCT -> Mid-frequency additive rule (no SVD).
4. SvdOnlyMethod: Spatial Block SVD (no DWT, no DCT).

All variants:
- Subclass BaseWatermarkMethod.
- Provide identical embedding rule (additive) and extraction rule (threshold at 0.5).
- Fully vectorized 4D block array transformations for maximum performance.
- Support both fixed-alpha and adaptive texture-masked alpha.
"""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import pywt
import cv2
from scipy.fftpack import dctn, idctn

from watermark.masking import DeterministicTextureMasker
from experiments_v2.harness import (
    BaseWatermarkMethod,
    QUANTIZE,
    quantize_u8,
)
from experiments_v2.common import ConstantAlphaMasker


# =============================================================================
# 1. DWT + SVD (NO DCT)
# =============================================================================

class DwtSvdMethod(BaseWatermarkMethod):
    """
    1-level DWT decomposition followed directly by block SVD on subband blocks.
    Skips block DCT completely.
    """
    def __init__(
        self,
        subband: str = "HL",
        wavelet: str = "haar",
        block_size: int = 8,
        adaptive: bool = True,
        weight_gradient: float = 0.40,
        weight_variance: float = 0.40,
        weight_entropy: float = 0.20,
        window_size: int = 7,
        alpha_min: float = 0.05,
        alpha_max: float = 0.25,
    ):
        self.subband = subband.upper()
        self.wavelet = wavelet
        self.block_size = block_size
        self.adaptive = adaptive
        self.w_grad = weight_gradient
        self.w_var = weight_variance
        self.w_ent = weight_entropy
        self.window_size = window_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.name = f"dwt_svd_{'adaptive' if adaptive else 'fixed'}"

    def _get_alpha_matrix(self, cover: np.ndarray, alpha: float, grid_shape: Tuple[int, int]) -> np.ndarray:
        if self.adaptive:
            masker = DeterministicTextureMasker(
                weight_gradient=self.w_grad,
                weight_variance=self.w_var,
                weight_entropy=self.w_ent,
                window_size=self.window_size,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                base_alpha=alpha,
            )
            sb_shape = (cover.shape[0] // 2, cover.shape[1] // 2)
            return masker.generate_block_alpha_matrix(cover, sb_shape, self.block_size, alpha).astype(np.float64)
        else:
            return np.full(grid_shape, float(alpha), dtype=np.float64)

    def embed(
        self,
        cover: np.ndarray,
        payload: np.ndarray,
        alpha: float,
        quantize: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = cover.shape[:2]
        bs = self.block_size
        coeffs = pywt.dwt2(cover, self.wavelet)
        LL, (LH, HL, HH) = coeffs
        subband_dict = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}
        target_sb = subband_dict[self.subband].copy()
        sb_h, sb_w = target_sb.shape

        nh, nw = sb_h // bs, sb_w // bs
        grid_shape = (nh, nw)

        # Prepare payload
        if payload.shape != grid_shape:
            w_bin = cv2.resize(payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        alpha_mat = self._get_alpha_matrix(cover, alpha, grid_shape)

        # 4D Block extraction & batched SVD directly on subband
        blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        U, S, Vt = np.linalg.svd(blocks, full_matrices=True)
        orig_s0 = S[:, :, 0].copy()

        # Additive embedding on S0
        S_mod = S.copy()
        S_mod[:, :, 0] += alpha_mat * w_bin

        # Reconstruction
        mod_blocks = np.matmul(U * S_mod[..., None, :], Vt)
        new_sb = target_sb.copy()
        new_sb[: nh * bs, : nw * bs] = mod_blocks.transpose(0, 2, 1, 3).reshape(nh * bs, nw * bs)
        subband_dict[self.subband] = new_sb

        recon = pywt.idwt2((subband_dict["LL"], (subband_dict["LH"], subband_dict["HL"], subband_dict["HH"])), self.wavelet)
        recon = np.clip(recon[:h, :w], 0.0, 255.0)

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            recon = quantize_u8(recon)

        side_info = {
            "method": self.name,
            "subband": self.subband,
            "wavelet": self.wavelet,
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "alpha_matrix": alpha_mat,
            "alpha_eff_mean": float(np.mean(alpha_mat)),
            "orig_s0": orig_s0,
            "quantized": should_quantize,
        }
        return recon, side_info

    def extract(
        self,
        image: np.ndarray,
        side_info: Dict[str, Any],
        deskew_angle: Optional[float] = None,
    ) -> np.ndarray:
        img = image.copy().astype(np.float64)
        if deskew_angle is not None and abs(deskew_angle) > 1e-4:
            img = CascadedDwtDctSvdWatermarker.deskew_image(img, angle_deg=deskew_angle)

        h, w = side_info["cover_shape"]
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        coeffs = pywt.dwt2(img, side_info["wavelet"])
        LL, (LH, HL, HH) = coeffs
        subband_dict = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}
        target_sb = subband_dict[side_info["subband"]]

        bs = side_info["block_size"]
        nh, nw = side_info["grid_shape"]
        blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        s_att_0 = np.linalg.svd(blocks, compute_uv=False)[:, :, 0]

        alpha_mat = side_info["alpha_matrix"]
        orig_s0 = side_info["orig_s0"]

        mask = alpha_mat > 1e-7
        w_raw = np.zeros_like(s_att_0)
        w_raw[mask] = (s_att_0[mask] - orig_s0[mask]) / alpha_mat[mask]
        w_bin = (w_raw >= 0.5).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(w_bin.astype(np.float32), (side_info["watermark_shape"][1], side_info["watermark_shape"][0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return w_bin


# =============================================================================
# 2. DCT + SVD (NO DWT, SPATIAL 16x16 BLOCKS)
# =============================================================================

class DctSvdMethod(BaseWatermarkMethod):
    """
    Direct spatial 2D block DCT (16x16 on 512x512 -> 32x32 blocks), then block SVD.
    Skips DWT completely.
    """
    def __init__(
        self,
        subband: str = "spatial",
        block_size: int = 16,
        adaptive: bool = True,
        weight_gradient: float = 0.40,
        weight_variance: float = 0.40,
        weight_entropy: float = 0.20,
        window_size: int = 7,
        alpha_min: float = 0.05,
        alpha_max: float = 0.25,
    ):
        self.subband = subband.lower()
        self.block_size = block_size
        self.adaptive = adaptive
        self.w_grad = weight_gradient
        self.w_var = weight_variance
        self.w_ent = weight_entropy
        self.window_size = window_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.name = f"dct_svd_{'adaptive' if adaptive else 'fixed'}"

    def _get_alpha_matrix(self, cover: np.ndarray, alpha: float, grid_shape: Tuple[int, int]) -> np.ndarray:
        if self.adaptive:
            masker = DeterministicTextureMasker(
                weight_gradient=self.w_grad,
                weight_variance=self.w_var,
                weight_entropy=self.w_ent,
                window_size=self.window_size,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                base_alpha=alpha,
            )
            # Compute on spatial cover
            return masker.generate_block_alpha_matrix(cover, cover.shape[:2], self.block_size, alpha).astype(np.float64)
        else:
            return np.full(grid_shape, float(alpha), dtype=np.float64)

    def embed(
        self,
        cover: np.ndarray,
        payload: np.ndarray,
        alpha: float,
        quantize: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = cover.shape[:2]
        bs = self.block_size
        nh, nw = h // bs, w // bs
        grid_shape = (nh, nw)

        if payload.shape != grid_shape:
            w_bin = cv2.resize(payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        alpha_mat = self._get_alpha_matrix(cover, alpha, grid_shape)

        blocks = cover[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")
        U, S, Vt = np.linalg.svd(dct_blocks, full_matrices=True)
        orig_s0 = S[:, :, 0].copy()

        S_mod = S.copy()
        S_mod[:, :, 0] += alpha_mat * w_bin

        mod_dct = np.matmul(U * S_mod[..., None, :], Vt)
        mod_blocks = idctn(mod_dct, axes=(-2, -1), type=2, norm="ortho")

        wm_img = cover.copy()
        wm_img[: nh * bs, : nw * bs] = mod_blocks.transpose(0, 2, 1, 3).reshape(nh * bs, nw * bs)
        wm_img = np.clip(wm_img, 0.0, 255.0)

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            wm_img = quantize_u8(wm_img)

        side_info = {
            "method": self.name,
            "subband": self.subband,
            "wavelet": "none",
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "alpha_matrix": alpha_mat,
            "alpha_eff_mean": float(np.mean(alpha_mat)),
            "orig_s0": orig_s0,
            "quantized": should_quantize,
        }
        return wm_img, side_info

    def extract(
        self,
        image: np.ndarray,
        side_info: Dict[str, Any],
        deskew_angle: Optional[float] = None,
    ) -> np.ndarray:
        img = image.copy().astype(np.float64)
        if deskew_angle is not None and abs(deskew_angle) > 1e-4:
            img = CascadedDwtDctSvdWatermarker.deskew_image(img, angle_deg=deskew_angle)

        h, w = side_info["cover_shape"]
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        bs = side_info["block_size"]
        nh, nw = side_info["grid_shape"]
        blocks = img[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")
        s_att_0 = np.linalg.svd(dct_blocks, compute_uv=False)[:, :, 0]

        alpha_mat = side_info["alpha_matrix"]
        orig_s0 = side_info["orig_s0"]

        mask = alpha_mat > 1e-7
        w_raw = np.zeros_like(s_att_0)
        w_raw[mask] = (s_att_0[mask] - orig_s0[mask]) / alpha_mat[mask]
        w_bin = (w_raw >= 0.5).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(w_bin.astype(np.float32), (side_info["watermark_shape"][1], side_info["watermark_shape"][0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return w_bin


# =============================================================================
# 3. DWT + DCT (NO SVD, MID-FREQUENCY COEFFICIENT ADDITIVE EMBEDDING)
# =============================================================================

class DwtDctMethod(BaseWatermarkMethod):
    """
    1-level DWT followed by 8x8 block DCT.
    Modifies one mid-frequency DCT coefficient (3, 3) per block via additive rule.
    Skips SVD completely.
    """
    def __init__(
        self,
        subband: str = "HL",
        wavelet: str = "haar",
        block_size: int = 8,
        adaptive: bool = True,
        weight_gradient: float = 0.40,
        weight_variance: float = 0.40,
        weight_entropy: float = 0.20,
        window_size: int = 7,
        alpha_min: float = 0.05,
        alpha_max: float = 0.25,
    ):
        self.subband = subband.upper()
        self.wavelet = wavelet
        self.block_size = block_size
        self.adaptive = adaptive
        self.w_grad = weight_gradient
        self.w_var = weight_variance
        self.w_ent = weight_entropy
        self.window_size = window_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.name = f"dwt_dct_{'adaptive' if adaptive else 'fixed'}"

    def _get_alpha_matrix(self, cover: np.ndarray, alpha: float, grid_shape: Tuple[int, int]) -> np.ndarray:
        if self.adaptive:
            masker = DeterministicTextureMasker(
                weight_gradient=self.w_grad,
                weight_variance=self.w_var,
                weight_entropy=self.w_ent,
                window_size=self.window_size,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                base_alpha=alpha,
            )
            sb_shape = (cover.shape[0] // 2, cover.shape[1] // 2)
            return masker.generate_block_alpha_matrix(cover, sb_shape, self.block_size, alpha).astype(np.float64)
        else:
            return np.full(grid_shape, float(alpha), dtype=np.float64)

    def embed(
        self,
        cover: np.ndarray,
        payload: np.ndarray,
        alpha: float,
        quantize: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = cover.shape[:2]
        bs = self.block_size
        coeffs = pywt.dwt2(cover, self.wavelet)
        LL, (LH, HL, HH) = coeffs
        subband_dict = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}
        target_sb = subband_dict[self.subband].copy()
        sb_h, sb_w = target_sb.shape

        nh, nw = sb_h // bs, sb_w // bs
        grid_shape = (nh, nw)

        if payload.shape != grid_shape:
            w_bin = cv2.resize(payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        alpha_mat = self._get_alpha_matrix(cover, alpha, grid_shape)

        blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")

        # Select mid-frequency coefficient (3, 3)
        orig_c = dct_blocks[:, :, 3, 3].copy()
        dct_mod = dct_blocks.copy()
        dct_mod[:, :, 3, 3] += alpha_mat * w_bin

        mod_blocks = idctn(dct_mod, axes=(-2, -1), type=2, norm="ortho")
        new_sb = target_sb.copy()
        new_sb[: nh * bs, : nw * bs] = mod_blocks.transpose(0, 2, 1, 3).reshape(nh * bs, nw * bs)
        subband_dict[self.subband] = new_sb

        recon = pywt.idwt2((subband_dict["LL"], (subband_dict["LH"], subband_dict["HL"], subband_dict["HH"])), self.wavelet)
        recon = np.clip(recon[:h, :w], 0.0, 255.0)

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            recon = quantize_u8(recon)

        side_info = {
            "method": self.name,
            "subband": self.subband,
            "wavelet": self.wavelet,
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "alpha_matrix": alpha_mat,
            "alpha_eff_mean": float(np.mean(alpha_mat)),
            "orig_c": orig_c,
            "quantized": should_quantize,
        }
        return recon, side_info

    def extract(
        self,
        image: np.ndarray,
        side_info: Dict[str, Any],
        deskew_angle: Optional[float] = None,
    ) -> np.ndarray:
        img = image.copy().astype(np.float64)
        if deskew_angle is not None and abs(deskew_angle) > 1e-4:
            img = CascadedDwtDctSvdWatermarker.deskew_image(img, angle_deg=deskew_angle)

        h, w = side_info["cover_shape"]
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        coeffs = pywt.dwt2(img, side_info["wavelet"])
        LL, (LH, HL, HH) = coeffs
        subband_dict = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}
        target_sb = subband_dict[side_info["subband"]]

        bs = side_info["block_size"]
        nh, nw = side_info["grid_shape"]
        blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")
        c_att = dct_blocks[:, :, 3, 3]

        alpha_mat = side_info["alpha_matrix"]
        orig_c = side_info["orig_c"]

        mask = alpha_mat > 1e-7
        w_raw = np.zeros_like(c_att)
        w_raw[mask] = (c_att[mask] - orig_c[mask]) / alpha_mat[mask]
        w_bin = (w_raw >= 0.5).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(w_bin.astype(np.float32), (side_info["watermark_shape"][1], side_info["watermark_shape"][0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return w_bin


# =============================================================================
# 4. SVD ONLY (SPATIAL 16x16 BLOCKS, NO DWT, NO DCT)
# =============================================================================

class SvdOnlyMethod(BaseWatermarkMethod):
    """
    Direct spatial block SVD (16x16 on 512x512 -> 32x32 blocks).
    Skips both DWT and DCT completely.
    """
    def __init__(
        self,
        subband: str = "spatial",
        block_size: int = 16,
        adaptive: bool = True,
        weight_gradient: float = 0.40,
        weight_variance: float = 0.40,
        weight_entropy: float = 0.20,
        window_size: int = 7,
        alpha_min: float = 0.05,
        alpha_max: float = 0.25,
    ):
        self.subband = subband.lower()
        self.block_size = block_size
        self.adaptive = adaptive
        self.w_grad = weight_gradient
        self.w_var = weight_variance
        self.w_ent = weight_entropy
        self.window_size = window_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.name = f"svd_only_{'adaptive' if adaptive else 'fixed'}"

    def _get_alpha_matrix(self, cover: np.ndarray, alpha: float, grid_shape: Tuple[int, int]) -> np.ndarray:
        if self.adaptive:
            masker = DeterministicTextureMasker(
                weight_gradient=self.w_grad,
                weight_variance=self.w_var,
                weight_entropy=self.w_ent,
                window_size=self.window_size,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                base_alpha=alpha,
            )
            return masker.generate_block_alpha_matrix(cover, cover.shape[:2], self.block_size, alpha).astype(np.float64)
        else:
            return np.full(grid_shape, float(alpha), dtype=np.float64)

    def embed(
        self,
        cover: np.ndarray,
        payload: np.ndarray,
        alpha: float,
        quantize: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = cover.shape[:2]
        bs = self.block_size
        nh, nw = h // bs, w // bs
        grid_shape = (nh, nw)

        if payload.shape != grid_shape:
            w_bin = cv2.resize(payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        alpha_mat = self._get_alpha_matrix(cover, alpha, grid_shape)

        blocks = cover[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        U, S, Vt = np.linalg.svd(blocks, full_matrices=True)
        orig_s0 = S[:, :, 0].copy()

        S_mod = S.copy()
        S_mod[:, :, 0] += alpha_mat * w_bin

        mod_blocks = np.matmul(U * S_mod[..., None, :], Vt)

        wm_img = cover.copy()
        wm_img[: nh * bs, : nw * bs] = mod_blocks.transpose(0, 2, 1, 3).reshape(nh * bs, nw * bs)
        wm_img = np.clip(wm_img, 0.0, 255.0)

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            wm_img = quantize_u8(wm_img)

        side_info = {
            "method": self.name,
            "subband": self.subband,
            "wavelet": "none",
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "alpha_matrix": alpha_mat,
            "alpha_eff_mean": float(np.mean(alpha_mat)),
            "orig_s0": orig_s0,
            "quantized": should_quantize,
        }
        return wm_img, side_info

    def extract(
        self,
        image: np.ndarray,
        side_info: Dict[str, Any],
        deskew_angle: Optional[float] = None,
    ) -> np.ndarray:
        img = image.copy().astype(np.float64)
        if deskew_angle is not None and abs(deskew_angle) > 1e-4:
            img = CascadedDwtDctSvdWatermarker.deskew_image(img, angle_deg=deskew_angle)

        h, w = side_info["cover_shape"]
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        bs = side_info["block_size"]
        nh, nw = side_info["grid_shape"]
        blocks = img[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        s_att_0 = np.linalg.svd(blocks, compute_uv=False)[:, :, 0]

        alpha_mat = side_info["alpha_matrix"]
        orig_s0 = side_info["orig_s0"]

        mask = alpha_mat > 1e-7
        w_raw = np.zeros_like(s_att_0)
        w_raw[mask] = (s_att_0[mask] - orig_s0[mask]) / alpha_mat[mask]
        w_bin = (w_raw >= 0.5).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(w_bin.astype(np.float32), (side_info["watermark_shape"][1], side_info["watermark_shape"][0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return w_bin


# =============================================================================
# 5. METHOD FACTORY FOR TRANSFORM ABLATIONS
# =============================================================================

def create_ablation_method(method_name: str, subband: str = "HL", **kwargs) -> BaseWatermarkMethod:
    """Factory creating transform ablation methods."""
    m_lower = method_name.lower().strip()
    adaptive = "adaptive" in m_lower

    if "dwt_svd" in m_lower:
        return DwtSvdMethod(subband=subband, adaptive=adaptive, **kwargs)
    elif "dct_svd" in m_lower:
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "wavelet"}
        return DctSvdMethod(subband=subband, adaptive=adaptive, **kwargs_clean)
    elif "dwt_dct" in m_lower:
        return DwtDctMethod(subband=subband, adaptive=adaptive, **kwargs)
    elif "svd_only" in m_lower:
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "wavelet"}
        return SvdOnlyMethod(subband=subband, adaptive=adaptive, **kwargs_clean)
    else:
        from experiments_v2.harness import create_method
        return create_method(method_name, subband=subband, **kwargs)
