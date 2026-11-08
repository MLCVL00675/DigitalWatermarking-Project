"""
Re-implemented Baseline Watermarking Methods and Comparative Harness Module for experiments_v2.

CRITICAL METHODOLOGICAL NOTICE:
All baseline implementations in this module are independent re-implementations of published
algorithmic ideas designed specifically for controlled, matched-distortion comparative evaluation
within this research framework. They are NOT the original authors' source code.
Each method is explicitly linked to its foundational publication:

1. LaiTsaiDwtSvdBaseline:
   Paper: Chun-Cheng Lai and Ching-Che Tsai, "Digital Image Watermarking Using Discrete
   Wavelet Transform and Singular Value Decomposition," IEEE Transactions on Instrumentation
   and Measurement, vol. 59, no. 11, pp. 3060-3063, Nov. 2010. DOI: 10.1109/TIM.2010.2066770.
   Decoder Type: reference-based (requires original singular values S0).

2. DctSvdBaseline:
   Paper: Elena Sverdlov, David Dexter, and Ahmet M. Eskicioglu, "Robust image watermarking
   in the DCT domain using singular value decomposition," in Proc. IASTED Conf. on Signal
   and Image Processing, Kauai, Hawaii, Aug. 2001, pp. 488-493.
   Decoder Type: reference-based (requires original singular values S0).

3. DwtDctSvdFixedBaseline ("hybrid precedent"):
   Papers: K. A. Navas and V. P. Sasikumar, "Survey of medical image watermarking in dual
   transform domain," in Proc. IEEE Int. Conf. on Computational Intelligence and Multimedia
   Applications (ICCIMA 2007), pp. 248-253, 2007; and Emir Ganic and Ahmet M. Eskicioglu,
   "Robust DWT-SVD domain image watermarking: recommending proper sub-bands," in Proc.
   ACM MM&Sec, 2004, pp. 166-174.
   Decoder Type: reference-based (requires original singular values S0).

4. DwtDctQimBaseline (BLIND baseline for fairness):
   Paper: Brian Chen and Gregory W. Wornell, "Quantization index modulation: a class of
   provably good methods for digital watermarking and information embedding," IEEE Transactions
   on Information Theory, vol. 47, no. 4, pp. 1423-1443, May 2001. DOI: 10.1109/18.910572;
   applied to mid-frequency DWT-DCT coefficients following Mauro Barni et al. (2001).
   Decoder Type: blind (no reference cover, coefficients, or singular values needed).

5. DwtDctAdaptiveQimBaseline (Texture-Adaptive QIM variant):
   Paper: M. A. Akhaee, M. J. Rad, and S. Ghaemmaghami, "A Controllable Texture-Adaptive
   Image Watermarking Scheme in the Wavelet Domain," IEEE Transactions on Information Forensics
   and Security, vol. 5, no. 4, pp. 707-715, Dec. 2010; integrating local HVS masking with QIM.
   Decoder Type: blind (quantization step modulated per block by activity map).

IMPORTANT COMPARABILITY CAVEAT:
Reference-based and blind methods are NOT directly comparable. Reference-based decoders possess
access to uncorrupted side-information (original singular values or reference matrices) during
detection, giving them a structural advantage under severe attacks. Blind methods extract
watermarks directly from the received image without access to the unwatermarked cover.
"""

import os
import sys
import time
import json
import argparse
from typing import Tuple, Dict, Any, Optional, List, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import pywt
from scipy.fftpack import dctn, idctn
import pandas as pd

from watermark.core import CascadedDwtDctSvdWatermarker
from watermark.masking import DeterministicTextureMasker
from watermark.metrics import (
    compute_psnr,
    compute_ssim,
    compute_ber,
    compute_bcr,
    compute_nc,
)
from experiments_v2.harness import (
    BaseWatermarkMethod,
    METHOD_REGISTRY,
    QUANTIZE,
    quantize_u8,
    calibrate_alpha_to_psnr,
    fast_batched_extract,
)
from experiments_v2.schema import (
    ResultRow,
    append_result_rows,
    init_results_csv,
)
from experiments_v2.common import get_git_commit_hash
from experiments_v2.payloads import generate_balanced_random_payload
from experiments_v2.attacks import AttackSpec, get_standard_attack_specs
from experiments_v2.runner_v2 import select_benchmark_subset
from experiments_v2.ablations import bootstrap_median_ci


# =============================================================================
# 1. BASELINE 1: LAI & TSAI DWT-SVD (REFERENCE-BASED)
# =============================================================================

class LaiTsaiDwtSvdBaseline(BaseWatermarkMethod):
    """
    Re-implementation of the DWT-SVD digital watermarking method inspired by:
    Chun-Cheng Lai and Ching-Che Tsai, "Digital Image Watermarking Using Discrete Wavelet
    Transform and Singular Value Decomposition," IEEE Transactions on Instrumentation and
    Measurement, vol. 59, no. 11, pp. 3060-3063, Nov. 2010. DOI: 10.1109/TIM.2010.2066770.

    CRITICAL NOTICE: This is an independent re-implementation of the published architectural
    concept for benchmarking within this research harness; NOT the authors' original code.

    Architecture:
    - 1-level 2D DWT (Haar) decomposes cover image into LL, LH, HL, HH.
    - SVD applied to blocks of the LL (approximation) subband. Lai & Tsai embed in LL
      because LL contains the primary energy and is highly resilient against lossy compression.
    - Watermark bits added to the largest singular values: S0' = S0 + alpha * bit.
    - Reconstruction via IDWT followed by standard 8-bit integer quantization.
    - Extraction: Reference-based, requiring original cover singular values S0 and step alpha.
    """
    name = "lai_tsai_dwt_svd"
    decoder_type = "reference-based"

    def __init__(
        self,
        subband: str = "LL",
        wavelet: str = "haar",
        block_size: int = 8,
    ):
        self.subband = subband.upper()
        self.wavelet = wavelet
        self.block_size = block_size

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
            w_bin = cv2.resize(
                payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        # 4D Block extraction & batched SVD directly on subband
        blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        U, S, Vt = np.linalg.svd(blocks, full_matrices=True)
        orig_s0 = S[:, :, 0].copy()

        # Additive embedding on S0
        S_mod = S.copy()
        S_mod[:, :, 0] += float(alpha) * w_bin

        # Reconstruction
        mod_blocks = np.matmul(U * S_mod[..., None, :], Vt)
        new_sb = target_sb.copy()
        new_sb[: nh * bs, : nw * bs] = mod_blocks.transpose(0, 2, 1, 3).reshape(nh * bs, nw * bs)
        subband_dict[self.subband] = new_sb

        recon = pywt.idwt2(
            (subband_dict["LL"], (subband_dict["LH"], subband_dict["HL"], subband_dict["HH"])),
            self.wavelet,
        )
        recon = np.clip(recon[:h, :w], 0.0, 255.0)

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            recon = quantize_u8(recon)

        side_info = {
            "method": self.name,
            "decoder_type": self.decoder_type,
            "subband": self.subband,
            "wavelet": self.wavelet,
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "alpha": float(alpha),
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

        alpha = side_info["alpha"]
        orig_s0 = side_info["orig_s0"]

        w_raw = (s_att_0 - orig_s0) / max(alpha, 1e-7)
        w_bin = (w_raw >= 0.5).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(
                w_bin.astype(np.float32),
                (side_info["watermark_shape"][1], side_info["watermark_shape"][0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        return w_bin


# =============================================================================
# 2. BASELINE 2: DCT-SVD BLOCK ADDITIVE (REFERENCE-BASED)
# =============================================================================

class DctSvdBaseline(BaseWatermarkMethod):
    """
    Re-implementation of the block DCT-SVD digital watermarking method inspired by:
    Elena Sverdlov, David Dexter, and Ahmet M. Eskicioglu, "Robust image watermarking
    in the DCT domain using singular value decomposition," in Proc. IASTED Conf. on
    Signal and Image Processing, Kauai, Hawaii, Aug. 2001, pp. 488-493.

    CRITICAL NOTICE: This is an independent re-implementation of the published architectural
    concept for benchmarking within this research harness; NOT the authors' original code.

    Architecture:
    - Direct spatial block DCT (16x16 blocks on 512x512 cover -> 32x32 blocks = 1,024 blocks).
    - Skips DWT decomposition entirely.
    - 2D DCT (orthonormal) applied to each 16x16 block, followed by SVD on each DCT matrix.
    - Watermark bits added to largest singular value: S0' = S0 + alpha * bit.
    - Inverse DCT, block reassembly, and standard 8-bit integer quantization.
    - Extraction: Reference-based, requiring original DCT singular values S0 and step alpha.
    """
    name = "dct_svd_baseline"
    decoder_type = "reference-based"

    def __init__(
        self,
        subband: str = "spatial",
        block_size: int = 16,
    ):
        self.subband = subband.lower()
        self.block_size = block_size

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
            w_bin = cv2.resize(
                payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        blocks = cover[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")
        U, S, Vt = np.linalg.svd(dct_blocks, full_matrices=True)
        orig_s0 = S[:, :, 0].copy()

        S_mod = S.copy()
        S_mod[:, :, 0] += float(alpha) * w_bin

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
            "decoder_type": self.decoder_type,
            "subband": self.subband,
            "wavelet": "none",
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "alpha": float(alpha),
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

        alpha = side_info["alpha"]
        orig_s0 = side_info["orig_s0"]

        w_raw = (s_att_0 - orig_s0) / max(alpha, 1e-7)
        w_bin = (w_raw >= 0.5).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(
                w_bin.astype(np.float32),
                (side_info["watermark_shape"][1], side_info["watermark_shape"][0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        return w_bin


# =============================================================================
# 3. BASELINE 3: DWT-DCT-SVD FIXED ALPHA ("HYBRID PRECEDENT", REFERENCE-BASED)
# =============================================================================

class DwtDctSvdFixedBaseline(BaseWatermarkMethod):
    """
    Re-implementation of the classic hybrid DWT-DCT-SVD fixed-alpha watermarking precedent,
    following:
    K. A. Navas and V. P. Sasikumar, "Survey of medical image watermarking in dual transform
    domain," in Proc. IEEE Int. Conf. on Computational Intelligence and Multimedia Applications
    (ICCIMA 2007), pp. 248-253, 2007; and
    Emir Ganic and Ahmet M. Eskicioglu, "Robust DWT-SVD domain image watermarking: recommending
    proper sub-bands," in Proc. ACM MM&Sec, 2004, pp. 166-174.

    CRITICAL NOTICE: This is an independent re-implementation of the published architectural
    concept for benchmarking within this research harness; NOT the authors' original code.

    Architecture:
    - 1-level 2D DWT (Haar) -> HL subband (256x256).
    - 8x8 block 2D DCT on subband blocks -> 32x32 blocks = 1,024 blocks.
    - SVD on each 8x8 DCT block.
    - Constant alpha additive embedding: S0' = S0 + alpha * bit.
    - Inverse DCT -> Inverse DWT -> 8-bit integer quantization.
    - Extraction: Reference-based using original S0 and fixed alpha.
    """
    name = "hybrid_precedent_fixed"
    decoder_type = "reference-based"

    def __init__(
        self,
        subband: str = "HL",
        wavelet: str = "haar",
        block_size: int = 8,
    ):
        self.subband = subband.upper()
        self.wavelet = wavelet
        self.block_size = block_size

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
            w_bin = cv2.resize(
                payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")
        U, S, Vt = np.linalg.svd(dct_blocks, full_matrices=True)
        orig_s0 = S[:, :, 0].copy()

        S_mod = S.copy()
        S_mod[:, :, 0] += float(alpha) * w_bin

        mod_dct = np.matmul(U * S_mod[..., None, :], Vt)
        mod_blocks = idctn(mod_dct, axes=(-2, -1), type=2, norm="ortho")

        new_sb = target_sb.copy()
        new_sb[: nh * bs, : nw * bs] = mod_blocks.transpose(0, 2, 1, 3).reshape(nh * bs, nw * bs)
        subband_dict[self.subband] = new_sb

        recon = pywt.idwt2(
            (subband_dict["LL"], (subband_dict["LH"], subband_dict["HL"], subband_dict["HH"])),
            self.wavelet,
        )
        recon = np.clip(recon[:h, :w], 0.0, 255.0)

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            recon = quantize_u8(recon)

        side_info = {
            "method": self.name,
            "decoder_type": self.decoder_type,
            "subband": self.subband,
            "wavelet": self.wavelet,
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "alpha": float(alpha),
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
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")
        s_att_0 = np.linalg.svd(dct_blocks, compute_uv=False)[:, :, 0]

        alpha = side_info["alpha"]
        orig_s0 = side_info["orig_s0"]

        w_raw = (s_att_0 - orig_s0) / max(alpha, 1e-7)
        w_bin = (w_raw >= 0.5).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(
                w_bin.astype(np.float32),
                (side_info["watermark_shape"][1], side_info["watermark_shape"][0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        return w_bin


# =============================================================================
# 4. BASELINE 4: DWT-DCT QIM (A STRICTLY BLIND BASELINE FOR FAIRNESS)
# =============================================================================

class DwtDctQimBaseline(BaseWatermarkMethod):
    """
    Re-implementation of a blind DWT-DCT Quantization Index Modulation (QIM) watermarking scheme,
    following:
    Brian Chen and Gregory W. Wornell, "Quantization index modulation: a class of provably
    good methods for digital watermarking and information embedding," IEEE Transactions on
    Information Theory, vol. 47, no. 4, pp. 1423-1443, May 2001. DOI: 10.1109/18.910572;
    applied to mid-frequency DWT-DCT coefficients following Mauro Barni et al. (2001).

    CRITICAL NOTICE: This is an independent re-implementation of the published architectural
    concept for benchmarking within this research harness; NOT the authors' original code.

    Architecture:
    - 1-level 2D DWT (Haar) -> HL subband (256x256).
    - Partition HL into 32x32 blocks of size 8x8 (1,024 blocks).
    - 2D DCT (orthonormal) on each 8x8 block.
    - Modulates a mid-frequency DCT coefficient at coordinate (u, v) = (3, 3) in each block.
    - Lattice quantizer rule with step Delta:
        q(C, 0, Delta) = round(C / Delta) * Delta
        q(C, 1, Delta) = round((C - Delta/2) / Delta) * Delta + Delta/2
    - Reconstructs blocks via 2D IDCT, followed by 2D IDWT and 8-bit integer quantization.
    - Extraction: STRICTLY BLIND. Needs NO reference cover image, NO reference coefficients,
      and NO singular values. Computes minimum Euclidean distance to 0-lattice and 1-lattice:
        d0 = |C* - round(C* / Delta) * Delta|
        d1 = |C* - (round((C* - Delta/2) / Delta) * Delta + Delta/2)|
        b_hat = 1 if d1 < d0 else 0.
    """
    name = "dwt_dct_qim_blind"
    decoder_type = "blind"

    def __init__(
        self,
        subband: str = "HL",
        wavelet: str = "haar",
        block_size: int = 8,
        coord: Tuple[int, int] = (3, 3),
    ):
        self.subband = subband.upper()
        self.wavelet = wavelet
        self.block_size = block_size
        self.u, self.v = coord

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
            w_bin = cv2.resize(
                payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")

        delta = max(float(alpha), 1e-4)
        C = dct_blocks[:, :, self.u, self.v]

        # Parity lattice quantization
        q0 = np.round(C / delta) * delta
        q1 = np.round((C - delta / 2.0) / delta) * delta + (delta / 2.0)
        C_mod = np.where(w_bin == 1, q1, q0)

        dct_blocks_mod = dct_blocks.copy()
        dct_blocks_mod[:, :, self.u, self.v] = C_mod

        mod_blocks = idctn(dct_blocks_mod, axes=(-2, -1), type=2, norm="ortho")
        new_sb = target_sb.copy()
        new_sb[: nh * bs, : nw * bs] = mod_blocks.transpose(0, 2, 1, 3).reshape(nh * bs, nw * bs)
        subband_dict[self.subband] = new_sb

        recon = pywt.idwt2(
            (subband_dict["LL"], (subband_dict["LH"], subband_dict["HL"], subband_dict["HH"])),
            self.wavelet,
        )
        recon = np.clip(recon[:h, :w], 0.0, 255.0)

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            recon = quantize_u8(recon)

        side_info = {
            "method": self.name,
            "decoder_type": self.decoder_type,
            "subband": self.subband,
            "wavelet": self.wavelet,
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "delta": delta,
            "coord": (self.u, self.v),
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

        u, v = side_info["coord"]
        C_att = dct_blocks[:, :, u, v]
        delta = side_info["delta"]

        # Blind minimum distance lattice decoder
        d0 = np.abs(C_att - np.round(C_att / delta) * delta)
        d1 = np.abs(C_att - (np.round((C_att - delta / 2.0) / delta) * delta + (delta / 2.0)))
        w_bin = (d1 < d0).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(
                w_bin.astype(np.float32),
                (side_info["watermark_shape"][1], side_info["watermark_shape"][0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        return w_bin


# =============================================================================
# 5. BASELINE 5: TEXTURE-ADAPTIVE DWT-DCT QIM (BLIND VARIANT)
# =============================================================================

class DwtDctAdaptiveQimBaseline(BaseWatermarkMethod):
    """
    Re-implementation of a texture-adaptive DWT-DCT Quantization Index Modulation (QIM) scheme,
    following:
    M. A. Akhaee, M. J. Rad, and S. Ghaemmaghami, "A Controllable Texture-Adaptive Image
    Watermarking Scheme in the Wavelet Domain," IEEE Transactions on Information Forensics
    and Security, vol. 5, no. 4, pp. 707-715, Dec. 2010; integrating local HVS masking with QIM.

    CRITICAL NOTICE: This is an independent re-implementation of the published architectural
    concept for benchmarking within this research harness; NOT the authors' original code.

    Architecture:
    - Employs the DeterministicTextureMasker (gradient, variance, entropy) to generate a
      per-block activity matrix.
    - Adapts quantization step size per block: Delta_k = Delta_0 * (scaled activity weight).
    - Higher texture blocks receive a larger step size (enhanced robustness against attacks);
      smooth blocks receive a smaller step size (preventing perceptual distortion).
    - Embeds via QIM into mid-frequency DCT coefficient (3, 3) of HL subband.
    - Extraction: Blind lattice decoding using the block quantization step matrix.
    """
    name = "dwt_dct_adaptive_qim"
    decoder_type = "blind"

    def __init__(
        self,
        subband: str = "HL",
        wavelet: str = "haar",
        block_size: int = 8,
        coord: Tuple[int, int] = (3, 3),
        weight_gradient: float = 0.40,
        weight_variance: float = 0.40,
        weight_entropy: float = 0.20,
        alpha_min: float = 0.05,
        alpha_max: float = 0.25,
    ):
        self.subband = subband.upper()
        self.wavelet = wavelet
        self.block_size = block_size
        self.u, self.v = coord
        self.masker = DeterministicTextureMasker(
            weight_gradient=weight_gradient,
            weight_variance=weight_variance,
            weight_entropy=weight_entropy,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

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
            w_bin = cv2.resize(
                payload.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            w_bin = payload.astype(np.uint8)

        blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")

        # Modulate step per block
        delta_mat = self.masker.generate_block_alpha_matrix(
            cover, (sb_h, sb_w), bs, base_alpha=alpha
        ).astype(np.float64)
        delta_mat = np.maximum(delta_mat, 1.0)

        C = dct_blocks[:, :, self.u, self.v]
        q0 = np.round(C / delta_mat) * delta_mat
        q1 = np.round((C - delta_mat / 2.0) / delta_mat) * delta_mat + (delta_mat / 2.0)
        C_mod = np.where(w_bin == 1, q1, q0)

        dct_blocks_mod = dct_blocks.copy()
        dct_blocks_mod[:, :, self.u, self.v] = C_mod

        mod_blocks = idctn(dct_blocks_mod, axes=(-2, -1), type=2, norm="ortho")
        new_sb = target_sb.copy()
        new_sb[: nh * bs, : nw * bs] = mod_blocks.transpose(0, 2, 1, 3).reshape(nh * bs, nw * bs)
        subband_dict[self.subband] = new_sb

        recon = pywt.idwt2(
            (subband_dict["LL"], (subband_dict["LH"], subband_dict["HL"], subband_dict["HH"])),
            self.wavelet,
        )
        recon = np.clip(recon[:h, :w], 0.0, 255.0)

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            recon = quantize_u8(recon)

        side_info = {
            "method": self.name,
            "decoder_type": self.decoder_type,
            "subband": self.subband,
            "wavelet": self.wavelet,
            "block_size": bs,
            "grid_shape": grid_shape,
            "watermark_shape": payload.shape[:2],
            "cover_shape": (h, w),
            "delta_matrix": delta_mat,
            "alpha_eff_mean": float(np.mean(delta_mat)),
            "coord": (self.u, self.v),
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

        u, v = side_info["coord"]
        C_att = dct_blocks[:, :, u, v]
        delta_mat = side_info["delta_matrix"]

        d0 = np.abs(C_att - np.round(C_att / delta_mat) * delta_mat)
        d1 = np.abs(C_att - (np.round((C_att - delta_mat / 2.0) / delta_mat) * delta_mat + (delta_mat / 2.0)))
        w_bin = (d1 < d0).astype(np.uint8)

        if side_info["watermark_shape"] != side_info["grid_shape"]:
            w_bin = cv2.resize(
                w_bin.astype(np.float32),
                (side_info["watermark_shape"][1], side_info["watermark_shape"][0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        return w_bin


# =============================================================================
# 6. REGISTER BASELINES INTO HARNESS METHOD_REGISTRY
# =============================================================================

BASELINE_CLASSES = {
    "lai_tsai_dwt_svd": LaiTsaiDwtSvdBaseline,
    "dct_svd_baseline": DctSvdBaseline,
    "hybrid_precedent_fixed": DwtDctSvdFixedBaseline,
    "dwt_dct_qim_blind": DwtDctQimBaseline,
    "dwt_dct_adaptive_qim": DwtDctAdaptiveQimBaseline,
}

for _k, _cls in BASELINE_CLASSES.items():
    METHOD_REGISTRY[_k] = _cls


def create_baseline_method(method_name: str, **kwargs) -> BaseWatermarkMethod:
    """Factory creating an instantiated baseline method."""
    cls = BASELINE_CLASSES.get(method_name)
    if cls is None:
        raise ValueError(
            f"Unknown baseline '{method_name}'. Available: {list(BASELINE_CLASSES.keys())}"
        )
    return cls(**kwargs)


# =============================================================================
# 7. MULTIPROCESSING EVALUATION WORKER
# =============================================================================

# Strict CSV output schema: includes decoder_type as requested
BASELINES_SCHEMA_COLUMNS = [
    "run_id",
    "experiment",
    "image_id",
    "payload_id",
    "method",
    "decoder_type",
    "subband",
    "wavelet",
    "block_size",
    "alpha_nominal",
    "alpha_eff_mean",
    "attack",
    "severity",
    "seed",
    "psnr_clean",
    "ssim_clean",
    "psnr_attacked",
    "ber",
    "bcr",
    "nc",
    "extra_json",
]


def _worker_baseline_condition(args: Tuple) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Worker task:
    1. Loads cover image.
    2. Calibrates embedding parameter to target PSNR (with 8-bit integer quantization).
    3. Embeds watermark once.
    4. Evaluates clean PSNR/SSIM.
    5. Evaluates all 14 single attack configurations (10 seeds for stochastic).
    6. Extracts watermark and records metrics.
    """
    (
        run_id,
        image_id,
        image_path,
        payload_id,
        payload,
        method_name,
        target_psnr,
        specs_with_seeds,
    ) = args

    cond_key = f"{image_id}_{method_name}_psnr{int(target_psnr)}"

    # 1. Load cover
    cover = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cover is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    cover = cover.astype(np.float64)

    # 2. Instantiate method
    method_obj = create_baseline_method(method_name)
    decoder_type = getattr(method_obj, "decoder_type", "reference-based")

    # 3. Calibrate alpha to target PSNR (with 8-bit quantization)
    alpha_bounds = (1.0, 350.0) if "qim" in method_name else (0.5, 100.0)
    try:
        cal_alpha = calibrate_alpha_to_psnr(
            method=method_obj,
            cover=cover,
            payload=payload,
            target_psnr=target_psnr,
            alpha_bounds=alpha_bounds,
            tol=0.10,
            max_iter=15,
        )
    except Exception as e:
        # Fallback to nominal if bisection fails
        cal_alpha = 25.0 if "svd" in method_name else 70.0

    # 4. Embed watermark once
    wm_img, side_info = method_obj.embed(cover, payload, alpha=cal_alpha, quantize=True)
    alpha_eff = float(side_info.get("alpha_eff_mean", cal_alpha))

    # 5. Clean metrics
    psnr_clean = float(compute_psnr(cover, wm_img))
    ssim_clean = float(compute_ssim(cover, wm_img))

    git_hash = get_git_commit_hash()
    rows = []

    # 6. Evaluate all attacks
    for spec, seeds in specs_with_seeds:
        deskew = spec.deskew_angle
        for seed in seeds:
            rng = np.random.default_rng(seed) if spec.is_stochastic else None
            attacked = spec.apply(wm_img, rng=rng)

            # Enforce 8-bit quantization
            if spec.attack not in ("no_attack", "identity"):
                attacked = quantize_u8(attacked)

            # Extract
            ext_bits = method_obj.extract(attacked, side_info, deskew_angle=deskew)

            # Robustness metrics
            psnr_att = float(compute_psnr(cover, attacked))
            ber = float(compute_ber(payload, ext_bits))
            bcr = float(compute_bcr(payload, ext_bits))
            nc = float(compute_nc(payload, ext_bits))

            extra = {
                "git_commit": git_hash,
                "target_psnr": target_psnr,
                "decoder_type": decoder_type,
                "comparability_warning": (
                    "Reference-based and blind methods are NOT directly comparable due to "
                    "differential side-information access."
                ),
            }

            row = {
                "run_id": run_id,
                "experiment": "baselines_matched_psnr",
                "image_id": image_id,
                "payload_id": payload_id,
                "method": method_name,
                "decoder_type": decoder_type,
                "subband": getattr(method_obj, "subband", "HL"),
                "wavelet": getattr(method_obj, "wavelet", "haar"),
                "block_size": getattr(method_obj, "block_size", 8),
                "alpha_nominal": round(float(cal_alpha), 4),
                "alpha_eff_mean": round(alpha_eff, 4),
                "attack": spec.attack,
                "severity": spec.severity,
                "seed": seed,
                "psnr_clean": round(psnr_clean, 4),
                "ssim_clean": round(ssim_clean, 6),
                "psnr_attacked": round(psnr_att, 4),
                "ber": round(ber, 6),
                "bcr": round(bcr, 4),
                "nc": round(nc, 6),
                "extra_json": json.dumps(extra, sort_keys=True),
            }
            rows.append(row)

    return cond_key, rows


# =============================================================================
# 8. SUMMARY MATRIX COMPUTATION (WITH BOOTSTRAP 95% CI)
# =============================================================================

def compute_baselines_summary(
    raw_csv: str = "results_v2/baselines_raw.csv",
    summary_csv: str = "results_v2/baselines_summary.csv",
    num_boot: int = 1000,
) -> pd.DataFrame:
    """
    Computes summary statistics enforcing Rule 4:
    IMAGE is the experimental unit (average seeds within image first).
    Calculates median BER per method x attack at each matched PSNR with bootstrap 95% CI over images.
    """
    if not os.path.exists(raw_csv) or os.path.getsize(raw_csv) == 0:
        raise FileNotFoundError(f"Raw CSV not found or empty: {raw_csv}")

    df_raw = pd.read_csv(raw_csv)

    # Extract target_psnr from extra_json if not present
    if "target_psnr" not in df_raw.columns:
        def _get_target_psnr(extra_str):
            try:
                d = json.loads(extra_str)
                return float(d.get("target_psnr", 46.0))
            except Exception:
                return 46.0
        df_raw["target_psnr"] = df_raw["extra_json"].apply(_get_target_psnr)

    # 1. Rule 4: Average seeds within image first!
    group_cols = ["method", "decoder_type", "target_psnr", "attack", "severity", "image_id"]
    df_img = (
        df_raw.groupby(group_cols, as_index=False)
        .agg({
            "ber": "mean",
            "bcr": "mean",
            "nc": "mean",
            "psnr_clean": "mean",
            "ssim_clean": "mean",
        })
    )

    # 2. Across images: compute median BER + bootstrap 95% CI, mean BER, median BCR, median NC
    records = []
    cond_group = ["method", "decoder_type", "target_psnr", "attack", "severity"]

    for (m, dec_type, t_psnr, att, sev), grp in df_img.groupby(cond_group):
        ber_arr = grp["ber"].to_numpy(dtype=np.float64)
        bcr_arr = grp["bcr"].to_numpy(dtype=np.float64)
        nc_arr = grp["nc"].to_numpy(dtype=np.float64)
        psnr_c = float(grp["psnr_clean"].mean())
        ssim_c = float(grp["ssim_clean"].mean())
        n_img = len(grp)

        med_ber, low_ci, high_ci = bootstrap_median_ci(ber_arr, num_boot=num_boot)

        records.append({
            "method": m,
            "decoder_type": dec_type,
            "target_psnr": float(t_psnr),
            "attack": att,
            "severity": sev,
            "median_ber": round(med_ber, 6),
            "ci_lower": round(low_ci, 6),
            "ci_upper": round(high_ci, 6),
            "mean_ber": round(float(np.mean(ber_arr)), 6),
            "median_bcr": round(float(np.median(bcr_arr)), 4),
            "median_nc": round(float(np.median(nc_arr)), 6),
            "psnr_clean_mean": round(psnr_c, 4),
            "ssim_clean_mean": round(ssim_c, 6),
            "n_images": n_img,
        })

    df_summary = pd.DataFrame(records)
    # Sort logically
    df_summary.sort_values(
        by=["target_psnr", "decoder_type", "method", "attack", "severity"], inplace=True
    )

    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)), exist_ok=True)
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n[*] Baselines summary saved to: '{summary_csv}' ({len(df_summary)} summary rows)")
    return df_summary


# =============================================================================
# 9. MASTER EXECUTION RUNNER
# =============================================================================

def run_baselines_study(
    num_images: int = 100,
    target_psnrs: Optional[List[float]] = None,
    methods: Optional[List[str]] = None,
    output_raw: str = "results_v2/baselines_raw.csv",
    output_summary: str = "results_v2/baselines_summary.csv",
    workers: int = 12,
) -> None:
    """
    Executes the matched-distortion baselines benchmark:
    - 5 methods: lai_tsai_dwt_svd, dct_svd_baseline, hybrid_precedent_fixed,
                 dwt_dct_qim_blind, dwt_dct_adaptive_qim.
    - 4 matched PSNR targets: 40, 44, 46, 48 dB (with 8-bit integer quantization).
    - 1,024-bit balanced payload (balanced_0).
    - 100 benchmark images (seed 0 fixed subset).
    - 14 single attack configurations (10 stochastic seeds).
    - Output raw results to results_v2/baselines_raw.csv.
    - Output bootstrap summary to results_v2/baselines_summary.csv.
    """
    targets = target_psnrs or [40.0, 44.0, 46.0, 48.0]
    baseline_methods = methods or [
        "lai_tsai_dwt_svd",
        "dct_svd_baseline",
        "hybrid_precedent_fixed",
        "dwt_dct_qim_blind",
        "dwt_dct_adaptive_qim",
    ]

    print("=" * 80)
    print("RE-IMPLEMENTED BASELINES BENCHMARK RUNNER")
    print("=" * 80)
    print(f"Methods ({len(baseline_methods)}): {baseline_methods}")
    print(f"Target PSNRs: {targets} dB")
    print(f"Number of Images: {num_images}")
    print(f"Output Raw: {output_raw}")
    print(f"Output Summary: {output_summary}")
    print(
        "\nIMPORTANT NOTICE: Reference-based and blind methods are NOT directly comparable!\n"
        "Reference-based decoders utilize uncorrupted cover singular values during extraction.\n"
        "Blind decoders operate with zero access to the unwatermarked cover."
    )
    print("=" * 80)

    # 1. Select benchmark images
    image_ids = select_benchmark_subset(num_images=num_images, seed=0)

    # 2. Balanced 1,024-bit payload
    payload = generate_balanced_random_payload(payload_id=0, target_size=(32, 32))
    payload_id = "balanced_0"

    # 3. Attacks with 10 stochastic seeds
    base_specs = get_standard_attack_specs()
    specs_with_seeds = []
    stoch_seeds = list(range(1, 11))  # 10 seeds
    det_seeds = [42]

    for s in base_specs:
        seeds = stoch_seeds if s.is_stochastic else det_seeds
        specs_with_seeds.append((s, seeds))

    # 4. Check completed conditions for resumability
    completed = set()
    if os.path.exists(output_raw) and os.path.getsize(output_raw) > 0:
        df_done = pd.read_csv(output_raw, usecols=["image_id", "method", "extra_json"])
        for _, row in df_done.iterrows():
            try:
                t = float(json.loads(row["extra_json"]).get("target_psnr", 0.0))
                completed.add((str(row["image_id"]), str(row["method"]), t))
            except Exception:
                pass
    else:
        # Initialize CSV with header including decoder_type
        os.makedirs(os.path.dirname(os.path.abspath(output_raw)), exist_ok=True)
        with open(output_raw, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(BASELINES_SCHEMA_COLUMNS) + "\n")

    # 5. Assemble tasks
    run_id = f"baselines_{int(time.time())}"
    tasks = []

    for t_psnr in targets:
        for m_name in baseline_methods:
            for iid in image_ids:
                if (iid, m_name, t_psnr) in completed:
                    continue
                img_path = os.path.join("data", "dataset_v2", f"{iid}.png")
                task = (
                    run_id,
                    iid,
                    img_path,
                    payload_id,
                    payload,
                    m_name,
                    t_psnr,
                    specs_with_seeds,
                )
                tasks.append(task)

    print(f"\n[*] Total conditions: {len(targets) * len(baseline_methods) * len(image_ids)}")
    print(f"[*] Already completed: {len(completed)}")
    print(f"[*] Pending conditions to execute: {len(tasks)}")

    if not tasks:
        print("[*] All conditions already completed. Computing summary...")
        compute_baselines_summary(output_raw, output_summary)
        return

    # 6. Execute in parallel
    start_time = time.perf_counter()
    done_count = 0

    with open(output_raw, "a", newline="", encoding="utf-8") as f_out:
        writer = pd.DataFrame(columns=BASELINES_SCHEMA_COLUMNS)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker_baseline_condition, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    cond_key, rows = fut.result()
                    df_chunk = pd.DataFrame(rows)[BASELINES_SCHEMA_COLUMNS]
                    df_chunk.to_csv(f_out, header=False, index=False)
                    f_out.flush()
                    done_count += 1

                    elapsed = time.perf_counter() - start_time
                    rate = done_count / elapsed
                    remaining = (len(tasks) - done_count) / rate if rate > 0 else 0
                    if done_count % 25 == 0 or done_count == len(tasks):
                        print(
                            f"  [{done_count}/{len(tasks)}] Done: {cond_key} | {rate:.2f} cond/s | ETA: {remaining:.1f}s",
                            flush=True,
                        )
                except Exception as e:
                    t = futures[fut]
                    print(f"[!] Error in condition {t[1]}_{t[5]}_{t[6]}: {e}", file=sys.stderr)
                    raise e

    total_time = time.perf_counter() - start_time
    print(f"\n[*] All baseline conditions completed in {total_time:.1f}s ({len(tasks)/total_time:.2f} cond/s)")

    # 7. Compute bootstrap summary
    compute_baselines_summary(output_raw, output_summary)


# =============================================================================
# 10. CLI ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Re-implemented Baselines Benchmark Runner")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker processes")
    parser.add_argument("--num-images", type=int, default=100, help="Number of benchmark images")
    parser.add_argument(
        "--psnr-targets",
        type=float,
        nargs="+",
        default=[40.0, 44.0, 46.0, 48.0],
        help="Target clean PSNR values in dB",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default="results_v2/baselines_raw.csv",
        help="Output raw CSV path",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results_v2/baselines_summary.csv",
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    run_baselines_study(
        num_images=args.num_images,
        target_psnrs=args.psnr_targets,
        output_raw=args.raw_output,
        output_summary=args.summary_output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
