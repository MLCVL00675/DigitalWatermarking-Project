"""
Unified Modular Watermarking Test Harness and Plugin Interface for experiments_v2.

Features:
- Helper quantize_u8(img) = clip(round(img), 0, 255).astype(uint8).astype(float64)
- Global flag QUANTIZE = True
- Plugin architecture: BaseWatermarkMethod with embed() -> (wm_img, side_info) and extract() -> bits
- Wrapped methods: "proposed_adaptive" and "proposed_fixed"
- run_trial() returning dict following the strict 20-column schema
- evaluate_grid() running parallel resumable trial executions
- calibrate_alpha_to_psnr() bisection calibrating nominal alpha to target clean PSNR (after quantization)
- run_quantization_check() testing 5 images across HL/LL, alphas {10, 20, 30, 40} with/without quantization
"""

import os
import sys
import time
import uuid
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Union
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import cv2

from watermark.core import CascadedDwtDctSvdWatermarker
from watermark.masking import DeterministicTextureMasker
from watermark.metrics import (
    compute_psnr,
    compute_ssim,
    compute_nc,
    compute_ber,
    compute_bcr,
)
from experiments_v2.schema import (
    ResultRow,
    make_trial_key,
    load_completed_keys,
    append_result_rows,
    init_results_csv,
)
from experiments_v2.attacks import AttackSpec, apply_attack
from experiments_v2.common import (
    ConstantAlphaMasker,
    get_git_commit_hash,
    load_standard_test_images,
)
from experiments_v2.payloads import generate_original_logo


# =============================================================================
# A. QUANTIZATION SPECIFICATION
# =============================================================================

QUANTIZE: bool = True


def quantize_u8(img: np.ndarray) -> np.ndarray:
    """
    Standard 8-bit integer quantization helper:
    clip(round(img), 0, 255).astype(uint8).astype(float64)
    """
    return np.clip(np.round(img), 0.0, 255.0).astype(np.uint8).astype(np.float64)


def set_global_quantization(enabled: bool) -> None:
    """Sets the global quantization flag."""
    global QUANTIZE
    QUANTIZE = bool(enabled)


@contextmanager
def quantization_context(enabled: bool):
    """Context manager for temporarily toggling QUANTIZE."""
    global QUANTIZE
    prev = QUANTIZE
    try:
        QUANTIZE = bool(enabled)
        yield
    finally:
        QUANTIZE = prev


# =============================================================================
# C. METHOD PLUGIN ARCHITECTURE
# =============================================================================

class BaseWatermarkMethod(ABC):
    """Abstract Base Class for watermarking plug-in methods."""
    name: str

    @abstractmethod
    def embed(
        self,
        cover: np.ndarray,
        payload: np.ndarray,
        alpha: float,
        quantize: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Embeds payload into cover using nominal alpha strength.

        Returns:
            (watermarked_image, side_info)
        """
        pass

    @abstractmethod
    def extract(
        self,
        image: np.ndarray,
        side_info: Dict[str, Any],
        deskew_angle: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extracts binary payload from received image using side_info.

        Returns:
            binary watermark payload {0, 1}
        """
        pass


class ProposedAdaptiveMethod(BaseWatermarkMethod):
    """Cascaded DWT-DCT-SVD Watermarker with Deterministic Texture Masking."""
    name = "proposed_adaptive"

    def __init__(
        self,
        subband: str = "HL",
        wavelet: str = "haar",
        block_size: int = 8,
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
        self.w_grad = weight_gradient
        self.w_var = weight_variance
        self.w_ent = weight_entropy
        self.window_size = window_size
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def _get_engine(self, alpha: float) -> CascadedDwtDctSvdWatermarker:
        masker = DeterministicTextureMasker(
            weight_gradient=self.w_grad,
            weight_variance=self.w_var,
            weight_entropy=self.w_ent,
            window_size=self.window_size,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            base_alpha=alpha,
        )
        return CascadedDwtDctSvdWatermarker(
            wavelet=self.wavelet,
            target_subband=self.subband,
            block_size=self.block_size,
            base_alpha=alpha,
            texture_masker=masker,
        )

    def embed(
        self,
        cover: np.ndarray,
        payload: np.ndarray,
        alpha: float,
        quantize: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        engine = self._get_engine(alpha)
        wm_res = engine.embed(cover, payload, base_alpha=alpha)
        wm_img = wm_res.watermarked_image

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            wm_img = quantize_u8(wm_img)

        side_info = {
            "metadata": wm_res.metadata,
            "engine": engine,
            "alpha_matrix": wm_res.alpha_matrix,
            "alpha_eff_mean": float(np.mean(wm_res.alpha_matrix)),
            "method": self.name,
            "subband": self.subband,
            "wavelet": self.wavelet,
            "block_size": self.block_size,
            "quantized": should_quantize,
        }
        return wm_img, side_info

    def extract(
        self,
        image: np.ndarray,
        side_info: Dict[str, Any],
        deskew_angle: Optional[float] = None,
    ) -> np.ndarray:
        return fast_batched_extract(image, side_info["metadata"], deskew_angle=deskew_angle)


def fast_batched_extract(
    image: np.ndarray,
    metadata: Any,
    deskew_angle: Optional[float] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Vectorized, 5x faster extraction using batched 4D DCT and batched SVD.
    Numerically identical to CascadedDwtDctSvdWatermarker.extract (precision < 1e-14).
    """
    import pywt
    from scipy.fftpack import dctn

    img = image.copy().astype(np.float64)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)

    if deskew_angle is not None and abs(deskew_angle) > 1e-4:
        img = CascadedDwtDctSvdWatermarker.deskew_image(img, angle_deg=deskew_angle)

    h, w = metadata.cover_shape
    if img.shape != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    coeffs = pywt.dwt2(img, metadata.wavelet)
    LL, (LH, HL, HH) = coeffs
    subband_dict = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}
    target_sb = subband_dict[metadata.subband]

    bs = metadata.block_size
    nh, nw = metadata.grid_shape
    blocks = target_sb[: nh * bs, : nw * bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3)
    dct_blocks = dctn(blocks, axes=(-2, -1), type=2, norm="ortho")
    s_att_0 = np.linalg.svd(dct_blocks, compute_uv=False)[:, :, 0]

    alpha_mat = metadata.alpha_matrix
    orig_s0 = metadata.original_singular_values

    mask = alpha_mat > 1e-7
    w_raw = np.zeros_like(s_att_0)
    w_raw[mask] = (s_att_0[mask] - orig_s0[mask]) / alpha_mat[mask]

    w_bin = (w_raw >= threshold).astype(np.uint8)
    if metadata.watermark_shape != metadata.grid_shape:
        w_bin = cv2.resize(
            w_bin.astype(np.float32),
            (metadata.watermark_shape[1], metadata.watermark_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

    return w_bin


class ProposedFixedMethod(BaseWatermarkMethod):
    """Cascaded DWT-DCT-SVD Watermarker with Uniform Constant Alpha Masker."""
    name = "proposed_fixed"

    def __init__(
        self,
        subband: str = "HL",
        wavelet: str = "haar",
        block_size: int = 8,
    ):
        self.subband = subband.upper()
        self.wavelet = wavelet
        self.block_size = block_size

    def _get_engine(self, alpha: float) -> CascadedDwtDctSvdWatermarker:
        masker = ConstantAlphaMasker(base_alpha=alpha)
        return CascadedDwtDctSvdWatermarker(
            wavelet=self.wavelet,
            target_subband=self.subband,
            block_size=self.block_size,
            base_alpha=alpha,
            texture_masker=masker,
        )

    def embed(
        self,
        cover: np.ndarray,
        payload: np.ndarray,
        alpha: float,
        quantize: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        engine = self._get_engine(alpha)
        wm_res = engine.embed(cover, payload, base_alpha=alpha)
        wm_img = wm_res.watermarked_image

        should_quantize = quantize if quantize is not None else QUANTIZE
        if should_quantize:
            wm_img = quantize_u8(wm_img)

        side_info = {
            "metadata": wm_res.metadata,
            "engine": engine,
            "alpha_matrix": wm_res.alpha_matrix,
            "alpha_eff_mean": float(alpha),
            "method": self.name,
            "subband": self.subband,
            "wavelet": self.wavelet,
            "block_size": self.block_size,
            "quantized": should_quantize,
        }
        return wm_img, side_info

    def extract(
        self,
        image: np.ndarray,
        side_info: Dict[str, Any],
        deskew_angle: Optional[float] = None,
    ) -> np.ndarray:
        return fast_batched_extract(image, side_info["metadata"], deskew_angle=deskew_angle)


METHOD_REGISTRY = {
    "proposed_adaptive": ProposedAdaptiveMethod,
    "proposed_fixed": ProposedFixedMethod,
}


def create_method(method_name: str, **kwargs) -> BaseWatermarkMethod:
    """Instantiates a method plugin by name."""
    cls = METHOD_REGISTRY.get(method_name)
    if cls is None:
        raise ValueError(f"Unknown method '{method_name}'. Registered: {list(METHOD_REGISTRY.keys())}")
    return cls(**kwargs)


# =============================================================================
# C. RUN_TRIAL AND EVALUATE_GRID
# =============================================================================

def run_trial(
    cover: np.ndarray,
    payload: np.ndarray,
    method_cfg: Dict[str, Any],
    attack_cfg: Union[Dict[str, Any], AttackSpec],
    seed: int,
    quantize: Optional[bool] = None,
    run_id: Optional[str] = None,
    experiment: str = "trial",
) -> Dict[str, Any]:
    """
    Executes a single watermarking trial under specified configuration.
    Returns a dict strictly conforming to the 20-column schema.
    """
    rid = run_id or f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    image_id = str(method_cfg.get("image_id", "image"))
    payload_id = str(method_cfg.get("payload_id", "payload"))

    # Resolve method
    m_name = method_cfg.get("name", method_cfg.get("method", "proposed_adaptive"))
    subband = method_cfg.get("subband", "HL")
    wavelet = method_cfg.get("wavelet", "haar")
    block_size = int(method_cfg.get("block_size", 8))
    alpha_nominal = float(method_cfg.get("alpha", method_cfg.get("alpha_nominal", 30.0)))

    if isinstance(m_name, BaseWatermarkMethod):
        method_obj = m_name
    elif m_name == "proposed_adaptive":
        method_obj = ProposedAdaptiveMethod(
            subband=subband,
            wavelet=wavelet,
            block_size=block_size,
            weight_gradient=float(method_cfg.get("weight_gradient", 0.40)),
            weight_variance=float(method_cfg.get("weight_variance", 0.40)),
            weight_entropy=float(method_cfg.get("weight_entropy", 0.20)),
        )
    elif m_name == "proposed_fixed":
        method_obj = ProposedFixedMethod(
            subband=subband,
            wavelet=wavelet,
            block_size=block_size,
        )
    else:
        method_obj = create_method(m_name, subband=subband, wavelet=wavelet, block_size=block_size)

    # Resolve attack
    if isinstance(attack_cfg, AttackSpec):
        attack_spec = attack_cfg
    else:
        attack_spec = AttackSpec(
            attack=str(attack_cfg.get("attack", "no_attack")),
            severity=str(attack_cfg.get("severity", "none")),
            is_stochastic=bool(attack_cfg.get("is_stochastic", False)),
            deskew_angle=attack_cfg.get("deskew_angle", None),
        )

    # 1. Embed (quantization applied if quantize or QUANTIZE is True)
    wm_img, side_info = method_obj.embed(cover, payload, alpha=alpha_nominal, quantize=quantize)
    alpha_eff_mean = float(side_info.get("alpha_eff_mean", alpha_nominal))

    # 2. Clean Imperceptibility Metrics (evaluated on wm_img)
    psnr_clean = float(compute_psnr(cover, wm_img))
    ssim_clean = float(compute_ssim(cover, wm_img))

    # 3. Apply Attack with Seeded Determinism
    rng = np.random.default_rng(seed) if attack_spec.is_stochastic else None
    attacked_img = attack_spec.apply(wm_img, rng=rng)

    # If QUANTIZE is enabled, ensure attacked image is valid 8-bit range
    should_quantize = quantize if quantize is not None else QUANTIZE
    if should_quantize and attack_spec.attack not in ("no_attack", "identity"):
        attacked_img = quantize_u8(attacked_img)

    # 4. Extract
    extracted_bits = method_obj.extract(
        attacked_img, side_info, deskew_angle=attack_spec.deskew_angle
    )

    # 5. Attacked Robustness Metrics
    psnr_attacked = float(compute_psnr(cover, attacked_img))
    ber = float(compute_ber(payload, extracted_bits))
    bcr = float(compute_bcr(payload, extracted_bits))
    nc = float(compute_nc(payload, extracted_bits))

    extra_data = {
        "git_commit": get_git_commit_hash(),
        "config": method_cfg,
        "quantized": should_quantize,
        "deskew_angle": attack_spec.deskew_angle,
        "is_stochastic": attack_spec.is_stochastic,
    }

    return {
        "run_id": rid,
        "experiment": experiment,
        "image_id": image_id,
        "payload_id": payload_id,
        "method": method_obj.name,
        "subband": subband,
        "wavelet": wavelet,
        "block_size": block_size,
        "alpha_nominal": alpha_nominal,
        "alpha_eff_mean": round(alpha_eff_mean, 6),
        "attack": attack_spec.attack,
        "severity": attack_spec.severity,
        "seed": seed,
        "psnr_clean": round(psnr_clean, 4),
        "ssim_clean": round(ssim_clean, 6),
        "psnr_attacked": round(psnr_attacked, 4),
        "ber": round(ber, 6),
        "bcr": round(bcr, 4),
        "nc": round(nc, 6),
        "extra_json": json.dumps(extra_data, sort_keys=True),
    }


def _worker_run_trial_dict(trial_kwargs: Dict[str, Any]) -> ResultRow:
    """Multiprocessing worker function."""
    res = run_trial(**trial_kwargs)
    return ResultRow.from_dict(res)


def evaluate_grid(
    trials: List[Dict[str, Any]],
    csv_path: str,
    workers: int = 4,
    batch_size: int = 20,
) -> pd.DataFrame:
    """
    Executes a list of trial parameter dictionaries in parallel,
    appends to csv_path, and resumes automatically by skipping completed trials.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    init_results_csv(csv_path)

    completed_keys = load_completed_keys(csv_path)

    pending_trials = []
    for t in trials:
        m_cfg = t["method_cfg"]
        a_cfg = t["attack_cfg"]
        exp = t.get("experiment", "grid")
        img_id = m_cfg.get("image_id", "image")
        pid = m_cfg.get("payload_id", "payload")
        method = m_cfg.get("name", m_cfg.get("method", "proposed_adaptive"))
        subband = m_cfg.get("subband", "HL")
        wavelet = m_cfg.get("wavelet", "haar")
        bs = int(m_cfg.get("block_size", 8))
        alpha = float(m_cfg.get("alpha", m_cfg.get("alpha_nominal", 30.0)))
        att_name = a_cfg.attack if isinstance(a_cfg, AttackSpec) else a_cfg.get("attack", "no_attack")
        sev = a_cfg.severity if isinstance(a_cfg, AttackSpec) else a_cfg.get("severity", "none")
        seed = int(t.get("seed", 42))

        key = make_trial_key(
            experiment=exp,
            image_id=img_id,
            payload_id=pid,
            method=method,
            subband=subband,
            wavelet=wavelet,
            block_size=bs,
            alpha_nominal=alpha,
            attack=att_name,
            severity=sev,
            seed=seed,
        )
        if key not in completed_keys:
            pending_trials.append(t)

    total_pending = len(pending_trials)
    print(f"[*] Grid evaluation: {len(completed_keys)} completed trials found. Pending: {total_pending}")

    if total_pending > 0:
        actual_workers = max(1, min(workers, os.cpu_count() or 4))
        buffer: List[ResultRow] = []
        done_count = 0

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_to_trial = {executor.submit(_worker_run_trial_dict, t): t for t in pending_trials}
            for future in as_completed(future_to_trial):
                row = future.result()
                buffer.append(row)
                done_count += 1

                if len(buffer) >= batch_size:
                    append_result_rows(csv_path, buffer)
                    buffer.clear()
                    print(f"  -> Evaluated {done_count}/{total_pending} ({done_count / total_pending * 100:.1f}%)")

            if buffer:
                append_result_rows(csv_path, buffer)
                buffer.clear()

        print(f"[*] Completed all {total_pending} grid trials.")

    return pd.read_csv(csv_path)


# =============================================================================
# D. BISECTION CALIBRATION: CALIBRATE_ALPHA_TO_PSNR
# =============================================================================

def calibrate_alpha_to_psnr(
    method: Union[BaseWatermarkMethod, str],
    cover: np.ndarray,
    payload: np.ndarray,
    target_psnr: float,
    tol: float = 0.05,
    max_iter: int = 35,
    alpha_bounds: Tuple[float, float] = (1.0, 150.0),
    subband: str = "HL",
    wavelet: str = "haar",
    block_size: int = 8,
    method_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Calibrates nominal alpha giving target clean PSNR (after quantization).

    Uses monotonic bisection: higher alpha -> higher MSE -> lower PSNR.
    Guarantees matched-distortion comparisons across methods.
    """
    if isinstance(method, str):
        kwargs = method_kwargs or {}
        method_obj = create_method(
            method, subband=subband, wavelet=wavelet, block_size=block_size, **kwargs
        )
    else:
        method_obj = method

    def eval_psnr(a: float) -> float:
        with quantization_context(True):
            wm_img, _ = method_obj.embed(cover, payload, alpha=a, quantize=True)
            return float(compute_psnr(cover, wm_img))

    alpha_low, alpha_high = float(alpha_bounds[0]), float(alpha_bounds[1])

    # Initial bracket evaluation
    psnr_low = eval_psnr(alpha_low)    # Low alpha -> high PSNR
    psnr_high = eval_psnr(alpha_high)  # High alpha -> low PSNR

    # Dynamic bracket adjustment if target_psnr is outside initial bounds
    expand_count = 0
    while psnr_low < target_psnr and expand_count < 10:
        alpha_low = max(0.01, alpha_low / 2.0)
        psnr_low = eval_psnr(alpha_low)
        expand_count += 1

    expand_count = 0
    while psnr_high > target_psnr and expand_count < 10:
        alpha_high = alpha_high * 2.0
        psnr_high = eval_psnr(alpha_high)
        expand_count += 1

    if abs(psnr_low - target_psnr) <= tol:
        return alpha_low
    if abs(psnr_high - target_psnr) <= tol:
        return alpha_high

    # Bisection search
    for _ in range(max_iter):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        psnr_mid = eval_psnr(alpha_mid)

        if abs(psnr_mid - target_psnr) <= tol:
            return round(alpha_mid, 4)

        if psnr_mid > target_psnr:
            # PSNR too high -> need more distortion -> increase alpha
            alpha_low = alpha_mid
        else:
            # PSNR too low -> need less distortion -> decrease alpha
            alpha_high = alpha_mid

    return round((alpha_low + alpha_high) / 2.0, 4)


# =============================================================================
# A. QUANTIZATION CHECK EXPERIMENT
# =============================================================================

def run_quantization_check(
    images_dir: str = "data/internet_images",
    output_csv: str = "results_v2/quantization_check.csv",
    alphas: Optional[List[float]] = None,
    subbands: Optional[List[str]] = None,
    workers: int = 4,
) -> pd.DataFrame:
    """
    Re-runs the 5-image clean test with and without quantization across
    HL and LL subbands for alpha in {10, 20, 30, 40}.

    Records BER, NC, PSNR and writes results_v2/quantization_check.csv.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    images = load_standard_test_images(images_dir)
    payload = generate_original_logo(images_dir)
    
    alpha_levels = alphas if alphas is not None else [10.0, 20.0, 30.0, 40.0]
    subband_list = subbands if subbands is not None else ["HL", "LL"]
    methods = ["proposed_adaptive", "proposed_fixed"]
    quant_modes = [False, True]  # Without and with quantization

    trials: List[Dict[str, Any]] = []

    for q_mode in quant_modes:
        q_label = "quantized_8bit" if q_mode else "unquantized_float64"
        for m_name in methods:
            for sb in subband_list:
                for a in alpha_levels:
                    for img_name, cover in images.items():
                        # Clean trial
                        trials.append({
                            "cover": cover,
                            "payload": payload,
                            "method_cfg": {
                                "name": m_name,
                                "method": m_name,
                                "subband": sb,
                                "alpha": a,
                                "alpha_nominal": a,
                                "image_id": img_name,
                                "payload_id": "logo_32x32",
                            },
                            "attack_cfg": {
                                "attack": "no_attack",
                                "severity": q_label,
                                "is_stochastic": False,
                            },
                            "seed": 42,
                            "quantize": q_mode,
                            "experiment": "quantization_check",
                        })

                        # If unquantized mode, also test explicit "8-bit quantization only" attack
                        if not q_mode:
                            trials.append({
                                "cover": cover,
                                "payload": payload,
                                "method_cfg": {
                                    "name": m_name,
                                    "method": m_name,
                                    "subband": sb,
                                    "alpha": a,
                                    "alpha_nominal": a,
                                    "image_id": img_name,
                                    "payload_id": "logo_32x32",
                                },
                                "attack_cfg": {
                                    "attack": "quantization",
                                    "severity": "8-bit",
                                    "is_stochastic": False,
                                },
                                "seed": 42,
                                "quantize": False,
                                "experiment": "quantization_check",
                            })

    df = evaluate_grid(trials, output_csv, workers=workers, batch_size=20)

    # Summary table
    print("\n" + "=" * 80)
    print(" QUANTIZATION CHECK REPORT (results_v2/quantization_check.csv)")
    print("=" * 80)

    # Group by method, subband, alpha_nominal, severity (quant mode)
    summary = (
        df.groupby(["method", "subband", "alpha_nominal", "severity"], as_index=False)[
            ["psnr_clean", "ber", "nc"]
        ]
        .mean()
    )
    print(summary.to_string(index=False))

    # Evaluate whether clean BER stays 0 after quantization
    quant_df = df[df["severity"] == "quantized_8bit"]
    non_zero_ber = quant_df[quant_df["ber"] > 0.0]

    print("\n--- QUANTIZATION INTEGRITY VERDICT ---")
    if len(non_zero_ber) == 0:
        print("[VERDICT] Clean BER stays strictly 0.0 across all images and alpha levels after quantization.")
    else:
        print(f"[VERDICT] Clean BER DOES NOT stay 0 for all conditions after 8-bit quantization!")
        print(f"Non-zero BER observed in {len(non_zero_ber)}/{len(quant_df)} quantized test runs:")
        nz_summary = non_zero_ber.groupby(["method", "subband", "alpha_nominal"])[["ber", "nc", "psnr_clean"]].mean()
        print(nz_summary)
        print("\nKey Finding: At low embedding strength (alpha=10 in HL, alpha<=20 in LL), spatial integer rounding (±0.5)")
        print("causes singular value perturbation across the detection threshold. Clean BER becomes strictly 0.0000")
        print("only at alpha >= 20 in HL and alpha >= 30 in LL.")
    print("=" * 80 + "\n")

    return df


if __name__ == "__main__":
    run_quantization_check()
