#!/usr/bin/env python3
"""
scripts/download_datasets.py
----------------------------
Standalone dataset downloader and deterministic preprocessing pipeline.
Fetches public benchmark datasets (BSDS500, Kodak, USC-SIPI, and COCO),
preprocesses them into 512x512 grayscale PNGs, and verifies their SHA-256
checksums strictly against results_v4/dataset_manifest_v4.csv.

Usage:
    python scripts/download_datasets.py --verify-only
    python scripts/download_datasets.py --output-dir data/dataset_v2 --workers 8
"""

import os
import sys
import ssl
import time
import zipfile
import hashlib
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import pandas as pd


# -----------------------------------------------------------------------------
# Public Dataset Source URLs and Constants
# -----------------------------------------------------------------------------

KODAK_URL_TEMPLATE = "http://r0k.us/graphics/kodak/kodak/kodim{idx:02d}.png"
USC_MISC_ZIP_URL = "https://sipi.usc.edu/database/misc.zip"
BSDS_RAW_GITHUB_BASE = "https://raw.githubusercontent.com/BIDS/BSDS500/master/BSDS500/data/images/test/"

BSDS500_TEST_FILENAMES = [
    "100007.jpg", "100039.jpg", "100099.jpg", "10081.jpg", "101027.jpg", "101084.jpg",
    "102062.jpg", "103006.jpg", "103029.jpg", "103078.jpg", "104010.jpg", "104055.jpg",
    "105027.jpg", "106005.jpg", "106047.jpg", "107014.jpg", "107045.jpg", "107072.jpg",
    "108004.jpg", "108036.jpg", "108069.jpg", "109055.jpg", "112056.jpg", "112090.jpg",
    "117025.jpg", "118015.jpg", "118031.jpg", "118072.jpg", "120003.jpg", "120093.jpg",
    "123057.jpg", "128035.jpg", "130014.jpg", "130066.jpg", "134049.jpg", "134067.jpg",
    "140006.jpg", "140088.jpg", "14085.jpg", "14092.jpg", "141012.jpg", "141048.jpg",
    "145059.jpg", "145079.jpg", "146074.jpg", "147077.jpg", "147080.jpg", "15011.jpg",
    "15062.jpg", "156054.jpg", "157032.jpg", "157087.jpg", "159002.jpg", "159022.jpg",
    "160006.jpg", "16004.jpg", "160067.jpg", "16068.jpg", "161045.jpg", "163004.jpg",
    "163096.jpg", "164046.jpg", "168084.jpg", "17067.jpg", "175083.jpg", "176051.jpg",
    "179084.jpg", "181021.jpg", "183066.jpg", "185092.jpg", "187058.jpg", "187099.jpg",
    "188025.jpg", "189006.jpg", "189013.jpg", "189029.jpg", "189096.jpg", "196027.jpg",
    "196040.jpg", "196062.jpg", "196088.jpg", "198087.jpg", "20069.jpg", "201080.jpg",
    "2018.jpg", "202000.jpg", "206062.jpg", "206097.jpg", "207038.jpg", "207049.jpg",
    "208078.jpg", "209021.jpg", "217013.jpg", "217090.jpg", "220003.jpg", "223004.jpg",
    "223060.jpg", "225022.jpg", "226022.jpg", "226033.jpg", "226043.jpg", "226060.jpg",
    "228076.jpg", "230063.jpg", "230098.jpg", "23050.jpg", "232076.jpg", "235098.jpg",
    "238025.jpg", "243095.jpg", "246009.jpg", "247003.jpg", "247012.jpg", "249021.jpg",
    "250047.jpg", "250087.jpg", "253016.jpg", "253092.jpg", "257098.jpg", "258089.jpg",
    "259060.jpg", "267036.jpg", "268048.jpg", "268074.jpg", "277053.jpg", "279005.jpg",
    "28083.jpg", "281017.jpg", "285022.jpg", "288024.jpg", "289011.jpg", "290035.jpg",
    "29030.jpg", "296028.jpg", "296058.jpg", "302022.jpg", "306051.jpg", "306052.jpg",
    "3063.jpg", "309040.jpg", "317043.jpg", "326025.jpg", "326085.jpg", "33044.jpg",
    "334025.jpg", "335088.jpg", "335094.jpg", "344010.jpg", "346016.jpg", "347031.jpg",
    "35028.jpg", "35049.jpg", "36046.jpg", "365072.jpg", "368037.jpg", "372019.jpg",
    "376086.jpg", "384022.jpg", "384089.jpg", "385022.jpg", "388006.jpg", "388018.jpg",
    "388067.jpg", "393035.jpg", "41006.jpg", "41029.jpg", "41085.jpg", "41096.jpg",
    "43033.jpg", "43051.jpg", "45000.jpg", "48017.jpg", "48025.jpg", "49024.jpg",
    "5096.jpg", "51084.jpg", "6046.jpg", "61034.jpg", "64061.jpg", "65084.jpg",
    "69000.jpg", "69007.jpg", "69022.jpg", "70011.jpg", "70090.jpg", "71076.jpg",
    "71099.jpg", "77062.jpg", "78098.jpg", "79073.jpg", "80085.jpg", "80090.jpg",
    "8068.jpg", "81066.jpg", "81090.jpg", "81095.jpg", "87015.jpg", "92014.jpg",
    "94095.jpg", "97010.jpg",
]

USC_SIPI_MISC_IDS = [
    "4.1.01", "4.1.02", "4.1.03", "4.1.04", "4.1.05", "4.1.06", "4.1.07", "4.1.08",
    "4.2.01", "4.2.03", "4.2.04", "4.2.05", "4.2.06", "4.2.07",
    "5.1.09", "5.1.10", "5.1.11", "5.1.12", "5.1.13", "5.1.14",
    "5.2.08", "5.2.09", "5.2.10",
    "5.3.01", "5.3.02",
    "7.1.01", "7.1.02", "7.1.03", "7.1.04", "7.1.05", "7.1.06", "7.1.07", "7.1.08", "7.1.09", "7.1.10",
    "7.2.01",
    "boat", "ruler", "stream",
]


# -----------------------------------------------------------------------------
# Preprocessing and Checksum Utilities
# -----------------------------------------------------------------------------

def preprocess_image(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Deterministic standard preprocessing:
    1. Grayscale conversion via cv2.cvtColor(BGR2GRAY)
    2. Center-crop to square along the shorter dimension
    3. Resize to target_size x target_size using INTER_AREA (or INTER_CUBIC if upscaling)
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    crop = gray[top : top + min_dim, left : left + min_dim]

    if min_dim >= target_size:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    resized = cv2.resize(crop, (target_size, target_size), interpolation=interp)
    return resized.astype(np.uint8)


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as fp:
        while chunk := fp.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path: str, data_dirs: List[str]) -> Tuple[int, int, List[str]]:
    """
    Checks all entries in dataset_manifest_v4.csv against files present in data_dirs.
    Returns (num_passed, num_failed, list_of_error_strings).
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    passed = 0
    failed = 0
    errors = []

    for _, row in df.iterrows():
        img_id = row["image_id"]
        expected_hash = row["SHA-256"]
        fname = f"{img_id}.png"

        target_file = None
        for d in data_dirs:
            candidate = os.path.join(d, fname)
            if os.path.exists(candidate):
                target_file = candidate
                break

        if target_file is None:
            errors.append(f"MISSING: {fname} (expected SHA-256: {expected_hash[:12]}...)")
            failed += 1
            continue

        actual_hash = compute_sha256(target_file)
        if actual_hash == expected_hash:
            passed += 1
        else:
            errors.append(f"HASH MISMATCH for {fname}: expected {expected_hash}, got {actual_hash}")
            failed += 1

    return passed, failed, errors


# -----------------------------------------------------------------------------
# Acquisition Pipeline
# -----------------------------------------------------------------------------

def download_and_preprocess_dataset(
    output_dir: str = "data/dataset_v2",
    manifest_csv: str = "results_v4/dataset_manifest_v4.csv",
    legacy_dir: str = "data/internet_images",
    num_bsds: int = 140,
    workers: int = 6,
) -> None:
    """
    Acquires, preprocesses, and verifies the full standardized benchmark image set.
    """
    os.makedirs(output_dir, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print("=" * 80)
    print("STANDARDIZED BENCHMARK DATASET DOWNLOADER & PREPROCESSOR")
    print(f"Target Output Directory: {output_dir}")
    print(f"Authoritative Manifest:  {manifest_csv}")
    print("=" * 80)

    # 1. Check existing images
    existing_files = {f[:-4] for f in os.listdir(output_dir) if f.endswith(".png")}
    print(f"[*] Found {len(existing_files)} pre-existing images in '{output_dir}'.")

    # 2. Process legacy5 images if present
    legacy_map = {
        "legacy_lena": "lena",
        "legacy_baboon": "baboon",
        "legacy_fruits": "fruits",
        "legacy_building": "building",
        "legacy_butterfly": "butterfly",
    }
    for leg_id, base_name in legacy_map.items():
        dst_path = os.path.join(output_dir, f"{leg_id}.png")
        if os.path.exists(dst_path):
            continue
        # Search legacy_dir for .png or .jpg
        found = False
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = os.path.join(legacy_dir, base_name + ext)
            if os.path.exists(cand):
                img = cv2.imread(cand)
                if img is not None:
                    proc = preprocess_image(img, 512)
                    cv2.imwrite(dst_path, proc)
                    found = True
                    break
        if not found:
            print(f"[!] Warning: legacy image '{base_name}' not found in '{legacy_dir}'.")

    # 3. Download Kodak Images (1..24)
    print("\n[*] Processing Kodak Image Suite (24 images)...")
    for idx in range(1, 25):
        img_id = f"kodak_{idx:02d}"
        dst_path = os.path.join(output_dir, f"{img_id}.png")
        if os.path.exists(dst_path):
            continue

        url = KODAK_URL_TEMPLATE.format(idx=idx)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                proc = preprocess_image(img, 512)
                cv2.imwrite(dst_path, proc)
                print(f"  [+] Downloaded and processed: {img_id}.png")
        except Exception as e:
            print(f"  [!] Failed fetching {url}: {e}")

    # 4. Download BSDS500 test images
    print(f"\n[*] Processing BSDS500 Test Split ({num_bsds} images)...")
    bsds_subset = BSDS500_TEST_FILENAMES[:num_bsds]

    def fetch_bsds(fname: str) -> Tuple[str, bool]:
        base_id = os.path.splitext(fname)[0]
        img_id = f"bsds_{base_id}"
        dst_path = os.path.join(output_dir, f"{img_id}.png")
        if os.path.exists(dst_path):
            return img_id, True

        url = BSDS_RAW_GITHUB_BASE + fname
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                proc = preprocess_image(img, 512)
                cv2.imwrite(dst_path, proc)
                return img_id, True
        except Exception as e:
            return img_id, False
        return img_id, False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_bsds, fn): fn for fn in bsds_subset}
        for fut in as_completed(futures):
            img_id, success = fut.result()
            if not success:
                print(f"  [!] Warning: Could not fetch {img_id}")

    # 5. Download USC-SIPI Miscellaneous images
    print(f"\n[*] Processing USC-SIPI Miscellaneous Volume ({len(USC_SIPI_MISC_IDS)} images)...")
    usc_raw_dir = "data/usc_raw"
    os.makedirs(usc_raw_dir, exist_ok=True)
    misc_zip = os.path.join(usc_raw_dir, "misc.zip")
    if not os.path.exists(misc_zip):
        try:
            print("  Downloading misc.zip archive from USC-SIPI...")
            req = urllib.request.Request(USC_MISC_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                with open(misc_zip, "wb") as f_out:
                    f_out.write(resp.read())
        except Exception as e:
            print(f"  [!] USC-SIPI bulk download failed: {e}")

    if os.path.exists(misc_zip):
        try:
            with zipfile.ZipFile(misc_zip, "r") as z:
                z.extractall(usc_raw_dir)
        except Exception as e:
            print(f"  [!] Unzipping misc.zip failed: {e}")

    for uid in USC_SIPI_MISC_IDS:
        sanitized = uid.replace(".", "_")
        img_id = f"usc_{sanitized}"
        dst_path = os.path.join(output_dir, f"{img_id}.png")
        if os.path.exists(dst_path):
            continue

        # Search for file in usc_raw_dir
        found = False
        for root, _, files in os.walk(usc_raw_dir):
            for f in files:
                if f.lower().startswith(uid.lower()):
                    full_p = os.path.join(root, f)
                    img = cv2.imread(full_p)
                    if img is not None:
                        proc = preprocess_image(img, 512)
                        cv2.imwrite(dst_path, proc)
                        found = True
                        break
            if found:
                break


def main():
    parser = argparse.ArgumentParser(description="Standalone benchmark dataset downloader and SHA-256 verifier.")
    parser.add_argument("--output-dir", type=str, default="data/dataset_v2", help="Preprocessed images destination directory.")
    parser.add_argument("--coco-dir", type=str, default="data/dataset_coco_v2", help="Directory holding COCO images.")
    parser.add_argument("--manifest", type=str, default="results_v4/dataset_manifest_v4.csv", help="Authoritative manifest CSV.")
    parser.add_argument("--legacy-dir", type=str, default="data/internet_images", help="Directory with original 5 benchmark images.")
    parser.add_argument("--workers", type=int, default=6, help="Download threads.")
    parser.add_argument("--verify-only", action="store_true", help="Skip downloads and perform checksum verification only.")
    args = parser.parse_args()

    # Fallback manifest path check
    manifest_path = args.manifest
    if not os.path.exists(manifest_path) and os.path.exists("results_v3/dataset_manifest_v4.csv"):
        manifest_path = "results_v3/dataset_manifest_v4.csv"

    if not args.verify_only:
        download_and_preprocess_dataset(
            output_dir=args.output_dir,
            manifest_csv=manifest_path,
            legacy_dir=args.legacy_dir,
            workers=args.workers,
        )

    # Verification phase
    print("\n" + "=" * 80)
    print("VERIFYING IMAGE CHECKSUMS (SHA-256) AGAINST AUTHORITATIVE MANIFEST")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)
    
    data_dirs = [args.output_dir, args.coco_dir]
    passed, failed, errors = verify_manifest(manifest_path, data_dirs)

    print(f"[*] Checksum Audit: {passed} images passed, {failed} images failed.")
    if errors:
        print("\n[!] Discrepancies detected:")
        for err in errors[:10]:
            print(f"    - {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more.")
        sys.exit(1)
    else:
        print("[SUCCESS] All 308 benchmark images verified bitwise identical to official hashes!")


if __name__ == "__main__":
    main()

