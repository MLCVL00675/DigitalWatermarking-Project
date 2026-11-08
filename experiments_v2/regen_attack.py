#!/usr/bin/env python3
"""
Regeneration and Latent Diffusion Threat Model Attack Benchmark (regen_attack.py)
---------------------------------------------------------------------------------
Evaluates watermark robustness under generative and latent-space degradation channels:
  (a) VAE Round-Trip: Encode and decode via public latent-diffusion VAE
      (stabilityai/sd-vae-ft-mse).
  (b) Diffusion Regeneration: Add noise to VAE latents at strength in {0.1, 0.2, 0.3}
      and denoise using a public Stable Diffusion img2img pipeline (empty prompt).

Protocol:
  - 8-bit integer quantized stego images at matched PSNR 46.0 dB.
  - Methods: proposed_fixed and proposed_adaptive.
  - Subbands: HL (mid-frequency) and LL (low-frequency).
  - Canonical 1,024-bit balanced watermark payload.
  - Images resized to model resolution and back to 512x512 via cv2.INTER_AREA.
  - Adaptive resolution: if CUDA is available, runs on 50 images at 512x512;
    if CPU only, runs on 10 images at 256x256 with explicit reporting.
  - Outputs results_v2/regen_raw.csv and summary statistics.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
import pandas as pd
from PIL import Image

import torch
from diffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

from watermark.metrics import compute_psnr, compute_ssim, compute_ber, compute_bcr, compute_nc
from experiments_v2.harness import create_method, calibrate_alpha_to_psnr, quantize_u8
from experiments_v2.payloads import generate_balanced_random_payload

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("regen_attack")


def get_regen_device_and_config(force_num_images: Optional[int] = None) -> Tuple[torch.device, int, int, str]:
    """Determines compute device, image count, resolution, and hardware note."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_images = force_num_images or 50
        res = 512
        note = f"GPU execution ({torch.cuda.get_device_name(0)}): evaluated on {num_images} images at 512x512"
    else:
        device = torch.device("cpu")
        num_images = force_num_images or 10
        res = 256
        note = f"CPU execution (no GPU detected): evaluated on {num_images} benchmark images at 256x256 (resized back to 512x512 with cv2.INTER_AREA)"
    return device, num_images, res, note


def load_regen_models(
    device: torch.device,
    vae_id: str = "stabilityai/sd-vae-ft-mse",
    diffusion_id: str = "segmind/tiny-sd",
) -> Tuple[AutoencoderKL, StableDiffusionImg2ImgPipeline]:
    """Loads the VAE and Stable Diffusion img2img pipelines."""
    logger.info("Loading VAE model: '%s' on %s...", vae_id, device)
    vae = AutoencoderKL.from_pretrained(vae_id)
    vae.to(device=device, dtype=torch.float32).eval()

    logger.info("Loading Diffusion img2img pipeline: '%s' on %s...", diffusion_id, device)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        diffusion_id,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device=device, dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)

    return vae, pipe


def apply_vae_roundtrip(
    img_512: np.ndarray,
    vae: AutoencoderKL,
    device: torch.device,
    model_res: int = 256,
) -> np.ndarray:
    """
    Applies VAE round-trip:
    1. Resize 512x512 -> model_res via cv2.INTER_AREA.
    2. Normalize to [-1, 1] RGB tensor.
    3. vae.encode -> latent_dist.mode() -> vae.decode.
    4. Denormalize, convert to grayscale, resize back to 512x512 via cv2.INTER_AREA.
    """
    if model_res != 512:
        img_res = cv2.resize(img_512, (model_res, model_res), interpolation=cv2.INTER_AREA)
    else:
        img_res = img_512

    # Prepare RGB float32 tensor in [-1, 1]
    rgb = np.stack([img_res, img_res, img_res], axis=0)  # (3, H, W)
    t = torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1.0

    with torch.no_grad():
        posterior = vae.encode(t)
        latents = posterior.latent_dist.mode()
        recon = vae.decode(latents).sample

    # Convert back to float64 [0, 255] grayscale
    recon_clamped = (recon.clamp(-1.0, 1.0) + 1.0) * 127.5
    recon_gray = recon_clamped.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float64)

    if model_res != 512:
        recon_512 = cv2.resize(recon_gray, (512, 512), interpolation=cv2.INTER_AREA)
    else:
        recon_512 = recon_gray

    return np.clip(recon_512, 0.0, 255.0)


def apply_diffusion_regeneration(
    img_512: np.ndarray,
    pipe: StableDiffusionImg2ImgPipeline,
    strength: float,
    device: torch.device,
    model_res: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """
    Applies Stable Diffusion img2img regeneration:
    1. Resize 512x512 -> model_res via cv2.INTER_AREA.
    2. Convert to PIL RGB.
    3. pipe(prompt=' ', strength=strength, num_inference_steps=20).
    4. Convert to float64 grayscale and resize back to 512x512 via cv2.INTER_AREA.
    """
    if model_res != 512:
        img_res = cv2.resize(img_512, (model_res, model_res), interpolation=cv2.INTER_AREA)
    else:
        img_res = img_512

    pil_img = Image.fromarray(np.clip(img_res, 0.0, 255.0).astype(np.uint8)).convert("RGB")
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        out = pipe(
            prompt=" ",
            image=pil_img,
            strength=strength,
            num_inference_steps=20,
            generator=generator,
        ).images[0]

    out_np = np.array(out.convert("L"), dtype=np.float64)

    if model_res != 512:
        out_512 = cv2.resize(out_np, (512, 512), interpolation=cv2.INTER_AREA)
    else:
        out_512 = out_np

    return np.clip(out_512, 0.0, 255.0)


def run_regeneration_benchmark(
    dataset_dir: str = "data/dataset_v2",
    manifest_csv: str = "results_v2/dataset_manifest.csv",
    output_csv: str = "results_v2/regen_raw.csv",
    force_num_images: Optional[int] = None,
    target_psnr: float = 46.0,
) -> pd.DataFrame:
    """Executes the regeneration threat model benchmark."""
    device, num_images, model_res, hw_note = get_regen_device_and_config(force_num_images)

    logger.info("=" * 80)
    logger.info("STARTING REGENERATION & DIFFUSION ATTACK BENCHMARK")
    logger.info("Hardware Config: %s", hw_note)
    logger.info("Device: %s | Target Images: %d | Resolution: %dx%d", device, num_images, model_res, model_res)
    logger.info("Matched Target Clean PSNR: %.1f dB", target_psnr)
    logger.info("=" * 80)

    # 1. Determine image list
    if os.path.exists(manifest_csv):
        df_man = pd.read_csv(manifest_csv)
        image_ids = df_man["image_id"].tolist()[:num_images]
    else:
        import glob
        all_imgs = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))
        image_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_imgs[:num_images]]

    logger.info("Selected %d benchmark images: %s ...", len(image_ids), image_ids[:3])

    # 2. Load models
    vae_id = "stabilityai/sd-vae-ft-mse"
    diffusion_id = "segmind/tiny-sd"
    vae, pipe = load_regen_models(device, vae_id=vae_id, diffusion_id=diffusion_id)

    # 3. Setup payload and conditions
    payload = generate_balanced_random_payload(payload_id=0, target_size=(32, 32))
    methods = ["proposed_fixed", "proposed_adaptive"]
    subbands = ["HL", "LL"]
    diffusion_strengths = [0.1, 0.2, 0.3]

    records: List[Dict[str, Any]] = []
    total_evals = len(image_ids) * len(methods) * len(subbands) * (1 + len(diffusion_strengths))
    eval_idx = 0
    t_start = time.time()

    for img_idx, img_id in enumerate(image_ids, 1):
        img_path = os.path.join(dataset_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            logger.warning("Image path '%s' not found. Skipping.", img_path)
            continue
        cover = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        for meth_name in methods:
            for subband in subbands:
                method = create_method(meth_name, subband=subband)

                # Calibrate alpha to matched PSNR 46 dB
                alpha_opt = calibrate_alpha_to_psnr(
                    method,
                    cover,
                    payload,
                    target_psnr=target_psnr,
                )
                stego_raw, meta = method.embed(cover, payload, alpha=alpha_opt)
                stego_q = quantize_u8(stego_raw)

                clean_psnr = float(compute_psnr(cover, stego_q))
                clean_ssim = float(compute_ssim(cover, stego_q))

                # --- (a) VAE Round-Trip Attack ---
                eval_idx += 1
                recon_vae = apply_vae_roundtrip(stego_q, vae, device, model_res=model_res)
                psnr_vae = float(compute_psnr(stego_q, recon_vae))
                ssim_vae = float(compute_ssim(stego_q, recon_vae))
                ext_vae = method.extract(recon_vae, meta)
                ber_vae = float(compute_ber(ext_vae, payload))
                bcr_vae = float(compute_bcr(ext_vae, payload))
                nc_vae = float(compute_nc(ext_vae, payload))

                records.append({
                    "image_id": img_id,
                    "method": meth_name,
                    "subband": subband,
                    "matched_psnr": target_psnr,
                    "alpha_calibrated": round(alpha_opt, 4),
                    "attack_family": "generative_regen",
                    "attack": "vae_roundtrip",
                    "strength": 0.0,
                    "model_name": vae_id,
                    "model_resolution": model_res,
                    "device": str(device),
                    "clean_psnr": round(clean_psnr, 2),
                    "clean_ssim": round(clean_ssim, 4),
                    "psnr_regen_stego": round(psnr_vae, 2),
                    "ssim_regen_stego": round(ssim_vae, 4),
                    "ber": round(ber_vae, 4),
                    "bcr": round(bcr_vae, 4),
                    "nc": round(nc_vae, 4),
                })

                # --- (b) Diffusion Regeneration Attacks (Strengths 0.1, 0.2, 0.3) ---
                for s in diffusion_strengths:
                    eval_idx += 1
                    recon_diff = apply_diffusion_regeneration(
                        stego_q, pipe, strength=s, device=device, model_res=model_res, seed=42
                    )
                    psnr_diff = float(compute_psnr(stego_q, recon_diff))
                    ssim_diff = float(compute_ssim(stego_q, recon_diff))
                    ext_diff = method.extract(recon_diff, meta)
                    ber_diff = float(compute_ber(ext_diff, payload))
                    bcr_diff = float(compute_bcr(ext_diff, payload))
                    nc_diff = float(compute_nc(ext_diff, payload))

                    records.append({
                        "image_id": img_id,
                        "method": meth_name,
                        "subband": subband,
                        "matched_psnr": target_psnr,
                        "alpha_calibrated": round(alpha_opt, 4),
                        "attack_family": "generative_regen",
                        "attack": f"diffusion_regen_{s:.1f}",
                        "strength": s,
                        "model_name": diffusion_id,
                        "model_resolution": model_res,
                        "device": str(device),
                        "clean_psnr": round(clean_psnr, 2),
                        "clean_ssim": round(clean_ssim, 4),
                        "psnr_regen_stego": round(psnr_diff, 2),
                        "ssim_regen_stego": round(ssim_diff, 4),
                        "ber": round(ber_diff, 4),
                        "bcr": round(bcr_diff, 4),
                        "nc": round(nc_diff, 4),
                    })

        elapsed = time.time() - t_start
        rate = eval_idx / elapsed
        rem = (total_evals - eval_idx) / rate if rate > 0 else 0
        logger.info(
            "[%d/%d images] Processed '%s' | Total evals: %d/%d | %.2f eval/s | ETA: %.1fs",
            img_idx,
            len(image_ids),
            img_id,
            eval_idx,
            total_evals,
            rate,
            rem,
        )

    # 4. Save results CSV
    df_out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    logger.info("Saved Regeneration Attack Results to: '%s' (%d rows)", output_csv, len(df_out))

    # 5. Print summary table
    print("\n" + "=" * 90)
    print("REGENERATION ATTACK BENCHMARK SUMMARY (Matched Clean PSNR = 46.0 dB)")
    print("=" * 90)
    summary_grp = df_out.groupby(["method", "subband", "attack"])[["psnr_regen_stego", "ssim_regen_stego", "ber", "nc"]].median().reset_index()
    print(summary_grp.to_string(index=False))
    print("=" * 90 + "\n")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Regeneration and Latent Diffusion Attack Benchmark")
    parser.add_argument("--dataset-dir", type=str, default="data/dataset_v2")
    parser.add_argument("--manifest-csv", type=str, default="results_v2/dataset_manifest.csv")
    parser.add_argument("--output-csv", type=str, default="results_v2/regen_raw.csv")
    parser.add_argument("--num-images", type=int, default=None, help="Override number of images (default: 50 on GPU, 10 on CPU)")
    parser.add_argument("--target-psnr", type=float, default=46.0, help="Target clean PSNR in dB")
    args = parser.parse_args()

    run_regeneration_benchmark(
        dataset_dir=args.dataset_dir,
        manifest_csv=args.manifest_csv,
        output_csv=args.output_csv,
        force_num_images=args.num_images,
        target_psnr=args.target_psnr,
    )


if __name__ == "__main__":
    main()
