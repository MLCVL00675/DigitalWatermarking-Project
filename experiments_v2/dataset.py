"""
Dataset Ingestion, Preprocessing, Characterization, and Stratification Module
for experiments_v2.

Features:
1. Public Dataset Downloaders:
   - BSDS500 (test split, 150-200 images, CC-BY / Berkeley Academic Terms)
   - Kodak Image Suite (24 images, Public Domain / Open Benchmark)
   - USC-SIPI Miscellaneous Volume (39 images, Open Research Benchmark)
   - Legacy 5 Benchmark Images (Lena, Baboon, Fruits, Building, Butterfly, source="legacy5")
2. Deterministic Preprocessing:
   - Grayscale conversion with cv2 (BGR2GRAY)
   - Center-crop to a square on the shorter side
   - Resize to 512x512 with cv2.INTER_AREA
   - Saved as PNG in data/dataset_v2/
3. Manifest Generation (results_v2/dataset_manifest.csv):
   - image_id, source dataset, original filename, original size, license/terms URL, SHA-256
4. Texture Characterization & Stratification (results_v2/dataset_characterization.csv):
   - mean Sobel gradient, mean local std (7x7), mean local entropy,
   - edge density (Canny), mean intensity, mean texture score T from DeterministicTextureMasker
   - texture_stratum: terciles of T (low, mid, high)
5. Overview Plotting:
   - results_v2/figs/dataset_overview.pdf: Histogram of T with tercile lines + 4x4 thumbnail grid
6. Paper Summary Table (dataset, N, license).
"""

import os
import io
import ssl
import json
import zipfile
import hashlib
import urllib.request
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

from watermark.masking import DeterministicTextureMasker


# =============================================================================
# DATASET CONSTANTS AND METADATA
# =============================================================================

LICENSES = {
    "BSDS500": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html",
    "Kodak": "http://r0k.us/graphics/kodak/",
    "USC-SIPI": "https://sipi.usc.edu/database/",
    "legacy5": "https://en.wikipedia.org/wiki/Standard_test_image",
}

# Standard BSDS500 test split image IDs (exact 200 files from BIDS/BSDS500 repository)
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


# =============================================================================
# 2. DETERMINISTIC PREPROCESSING
# =============================================================================

def preprocess_image(img: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Deterministic standard preprocessing:
    1. Convert to grayscale with cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if color.
    2. Center-crop to a square on the shorter side.
    3. Resize to target_size x target_size with cv2.INTER_AREA.

    Returns:
        uint8 2D numpy array with shape (target_size, target_size).
    """
    if img.ndim == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = img[:, :, 0]
    elif img.ndim == 2:
        gray = img.copy()
    else:
        raise ValueError(f"Unsupported image dimensions: {img.shape}")

    h, w = gray.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = gray[top: top + side, left: left + side]

    if cropped.shape != (target_size, target_size):
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    else:
        resized = cropped

    return np.clip(resized, 0, 255).astype(np.uint8)


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hexadecimal checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# 4. TEXTURE CHARACTERIZATION
# =============================================================================

def characterize_image(img: np.ndarray, masker: Optional[DeterministicTextureMasker] = None) -> Dict[str, float]:
    """
    Computes statistical and structural texture features:
    - mean Sobel gradient
    - mean local std (7x7 window)
    - mean local entropy
    - edge density (Canny detector)
    - mean intensity
    - mean texture score T from DeterministicTextureMasker
    """
    tm = masker or DeterministicTextureMasker()
    img_f = img.astype(np.float64)
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)

    # 1. Mean Sobel Gradient Magnitude
    sobel_map = tm.compute_gradient_magnitude(img_f)
    mean_sobel = float(np.mean(sobel_map))

    # 2. Mean Local Standard Deviation (7x7 neighborhood)
    mean_filt = uniform_filter(img_f, size=7, mode="reflect")
    sq_filt = uniform_filter(img_f**2, size=7, mode="reflect")
    local_var = np.maximum(sq_filt - mean_filt**2, 0.0)
    local_std = np.sqrt(local_var)
    mean_std = float(np.mean(local_std))

    # 3. Mean Local Shannon Entropy
    entropy_map = tm.compute_local_entropy(img_f)
    mean_entropy = float(np.mean(entropy_map))

    # 4. Edge Density via Canny
    edges = cv2.Canny(img_u8, threshold1=100, threshold2=200)
    edge_density = float(np.mean(edges > 0))

    # 5. Mean Intensity
    mean_intensity = float(np.mean(img_f))

    # 6. Mean Texture Activity Score T (0.4*grad + 0.4*var + 0.2*ent)
    spatial_mask = tm.generate_spatial_mask(img_f)
    texture_t = float(np.mean(spatial_mask))

    return {
        "mean_sobel_gradient": round(mean_sobel, 6),
        "mean_local_std": round(mean_std, 6),
        "mean_local_entropy": round(mean_entropy, 6),
        "edge_density": round(edge_density, 6),
        "mean_intensity": round(mean_intensity, 4),
        "texture_score_T": round(texture_t, 6),
    }


# =============================================================================
# 1. DATASET DOWNLOADERS
# =============================================================================

def _create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def download_kodak(output_dir: str, workers: int = 4) -> List[Tuple[str, np.ndarray, Tuple[int, int]]]:
    """Downloads all 24 Kodak lossless benchmark images."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    ctx = _create_ssl_context()

    def fetch_kodim(i: int):
        filename = f"kodim{i:02d}.png"
        url = f"http://r0k.us/graphics/kodak/kodak/{filename}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to decode {filename}")
        orig_shape = img.shape[:2]
        return filename, img, orig_shape

    print("[*] Downloading Kodak dataset (24 images)...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_kodim, i): i for i in range(1, 25)}
        for fut in as_completed(futures):
            try:
                fn, img, orig_shape = fut.result()
                results.append((fn, img, orig_shape))
            except Exception as e:
                print(f"  [!] Warning: Failed to fetch Kodak #{futures[fut]}: {e}")

    results.sort(key=lambda x: x[0])
    print(f"[*] Kodak dataset ready: {len(results)}/24 images downloaded.")
    return results


def download_usc_sipi(output_dir: str) -> List[Tuple[str, np.ndarray, Tuple[int, int]]]:
    """Downloads USC-SIPI Miscellaneous volume (misc.zip) and extracts TIFFs."""
    os.makedirs(output_dir, exist_ok=True)
    url = "https://sipi.usc.edu/database/misc.zip"
    ctx = _create_ssl_context()
    results = []

    print("[*] Downloading USC-SIPI Miscellaneous volume (misc.zip, ~10.5 MB)...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=40) as resp:
        zip_bytes = resp.read()

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    for name in sorted(zf.namelist()):
        if name.lower().endswith((".tiff", ".tif")):
            data = zf.read(name)
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                filename = os.path.basename(name)
                results.append((filename, img, img.shape[:2]))

    print(f"[*] USC-SIPI volume ready: {len(results)} benchmark images extracted.")
    return results


def download_bsds500(
    output_dir: str,
    target_count: int = 140,
    workers: int = 8,
) -> List[Tuple[str, np.ndarray, Tuple[int, int]]]:
    """Downloads BSDS500 test split images from the BIDS benchmark mirror."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    ctx = _create_ssl_context()
    selected_files = BSDS500_TEST_FILENAMES[:target_count]

    def fetch_bsds(filename: str):
        url = f"https://raw.githubusercontent.com/BIDS/BSDS500/master/BSDS500/data/images/test/{filename}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to decode {filename}")
        return filename, img, img.shape[:2]

    print(f"[*] Downloading BSDS500 test split ({len(selected_files)} images)...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_bsds, fn): fn for fn in selected_files}
        for fut in as_completed(futures):
            try:
                fn, img, orig_shape = fut.result()
                results.append((fn, img, orig_shape))
            except Exception as e:
                print(f"  [!] Warning: Failed to fetch BSDS500 {futures[fut]}: {e}")

    results.sort(key=lambda x: x[0])
    print(f"[*] BSDS500 dataset ready: {len(results)} images downloaded.")
    return results


def collect_legacy5(images_dir: str = "data/internet_images") -> List[Tuple[str, np.ndarray, Tuple[int, int]]]:
    """Loads the 5 historical baseline benchmark images."""
    legacy_files = [
        ("lena.png", "lena.jpg"),
        ("baboon.jpg", "baboon.png"),
        ("fruits.jpg", "fruits.png"),
        ("building.jpg", "building.png"),
        ("butterfly.jpg", "butterfly.png"),
    ]
    results = []
    for primary, fallback in legacy_files:
        path = os.path.join(images_dir, primary)
        if not os.path.exists(path):
            path = os.path.join(images_dir, fallback)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Legacy image not found: {path}")
        results.append((os.path.basename(path), img, img.shape[:2]))

    print(f"[*] Legacy5 benchmark images ready: {len(results)} images loaded.")
    return results


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

def prepare_dataset_pipeline(
    output_dir: str = "data/dataset_v2",
    manifest_csv: str = "results_v2/dataset_manifest.csv",
    characterization_csv: str = "results_v2/dataset_characterization.csv",
    plot_pdf: str = "results_v2/figs/dataset_overview.pdf",
    legacy_dir: str = "data/internet_images",
    num_bsds: int = 140,
    workers: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the complete dataset preparation, manifest generation,
    characterization, stratification, and visualization pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(manifest_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(characterization_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(plot_pdf)), exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    char_rows: List[Dict[str, Any]] = []
    masker = DeterministicTextureMasker()

    # 1. Gather all image sources
    sources_to_process = []

    # BSDS500
    try:
        bsds_imgs = download_bsds500(output_dir, target_count=num_bsds, workers=workers)
        for fn, img, o_shape in bsds_imgs:
            iid = f"bsds_{os.path.splitext(fn)[0]}"
            sources_to_process.append((iid, "BSDS500", fn, o_shape, LICENSES["BSDS500"], img))
    except Exception as e:
        print(f"[!] Warning: BSDS500 download issue: {e}")

    # Kodak
    try:
        kodak_imgs = download_kodak(output_dir, workers=workers)
        for fn, img, o_shape in kodak_imgs:
            iid = f"kodak_{os.path.splitext(fn)[0]}"
            sources_to_process.append((iid, "Kodak", fn, o_shape, LICENSES["Kodak"], img))
    except Exception as e:
        print(f"[!] Warning: Kodak download issue: {e}")

    # USC-SIPI
    try:
        usc_imgs = download_usc_sipi(output_dir)
        for fn, img, o_shape in usc_imgs:
            iid = f"usc_{os.path.splitext(fn)[0].replace('.', '_')}"
            sources_to_process.append((iid, "USC-SIPI", fn, o_shape, LICENSES["USC-SIPI"], img))
    except Exception as e:
        print(f"[!] Warning: USC-SIPI download issue: {e}")

    # Legacy 5
    try:
        legacy_imgs = collect_legacy5(legacy_dir)
        for fn, img, o_shape in legacy_imgs:
            base_name = os.path.splitext(fn)[0].lower()
            iid = f"legacy5_{base_name}"
            sources_to_process.append((iid, "legacy5", fn, o_shape, LICENSES["legacy5"], img))
    except Exception as e:
        print(f"[!] Warning: Legacy5 loading issue: {e}")

    total_gathered = len(sources_to_process)
    print(f"\n[*] Preprocessing and characterizing {total_gathered} images into '{output_dir}'...")

    # 2. Preprocess, save PNG, compute characterization
    for iid, src, fn, orig_shape, lic_url, raw_img in sources_to_process:
        proc_img = preprocess_image(raw_img, target_size=512)
        save_path = os.path.join(output_dir, f"{iid}.png")
        cv2.imwrite(save_path, proc_img)

        # SHA-256
        sha = compute_sha256(save_path)

        # Manifest entry
        manifest_rows.append({
            "image_id": iid,
            "source dataset": src,
            "original filename": fn,
            "original size": f"{orig_shape[1]}x{orig_shape[0]}",
            "license/terms URL": lic_url,
            "SHA-256": sha,
        })

        # Characterization
        metrics = characterize_image(proc_img, masker=masker)
        metrics["image_id"] = iid
        metrics["source_dataset"] = src
        char_rows.append(metrics)

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_csv, index=False)

    df_char = pd.DataFrame(char_rows)

    # 3. Add texture strata (terciles of T)
    q1 = float(df_char["texture_score_T"].quantile(1.0 / 3.0))
    q2 = float(df_char["texture_score_T"].quantile(2.0 / 3.0))

    def assign_stratum(score: float) -> str:
        if score <= q1:
            return "low"
        elif score <= q2:
            return "mid"
        else:
            return "high"

    df_char["texture_stratum"] = df_char["texture_score_T"].apply(assign_stratum)
    df_char.to_csv(characterization_csv, index=False)

    print(f"[*] Manifest written to: '{manifest_csv}'")
    print(f"[*] Characterization written to: '{characterization_csv}'")
    print(f"    Texture Tercile Cutoffs: Low <= {q1:.4f} < Mid <= {q2:.4f} < High")

    # 4. Generate Overview Plot
    plot_dataset_overview(df_char, output_dir, plot_pdf, q1, q2)

    # 5. Print Paper Summary Table
    print_dataset_summary_table(df_manifest, df_char)

    return df_manifest, df_char


# =============================================================================
# 5. VISUALIZATION AND PLOTTING
# =============================================================================

def plot_dataset_overview(
    df_char: pd.DataFrame,
    images_dir: str,
    output_pdf: str,
    q1: Optional[float] = None,
    q2: Optional[float] = None,
):
    """
    Generates dataset_overview.pdf:
    - Histogram of texture score T with tercile cutoff lines
    - 4x4 thumbnail grid representatively sampled across strata
    """
    t_vals = df_char["texture_score_T"].to_numpy()
    cut1 = q1 if q1 is not None else float(np.quantile(t_vals, 1.0 / 3.0))
    cut2 = q2 if q2 is not None else float(np.quantile(t_vals, 2.0 / 3.0))

    fig = plt.figure(figsize=(16, 7), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2], wspace=0.25)

    # Panel A: Histogram of Texture Score T
    ax_hist = fig.add_subplot(gs[0])
    counts, bins, patches = ax_hist.hist(
        t_vals, bins=25, color="#4a90e2", edgecolor="white", alpha=0.85, density=False
    )

    # Tercile vertical lines
    ax_hist.axvline(cut1, color="#d9534f", linestyle="--", linewidth=2.0, label=f"Tercile 1 ($q_1={cut1:.3f}$)")
    ax_hist.axvline(cut2, color="#f0ad4e", linestyle="--", linewidth=2.0, label=f"Tercile 2 ($q_2={cut2:.3f}$)")

    # Color background zones for strata
    ax_hist.axvspan(min(t_vals), cut1, color="#5bc0de", alpha=0.12, label="Low Texture Stratum")
    ax_hist.axvspan(cut1, cut2, color="#5cb85c", alpha=0.12, label="Mid Texture Stratum")
    ax_hist.axvspan(cut2, max(t_vals), color="#d9534f", alpha=0.12, label="High Texture Stratum")

    ax_hist.set_title("Distribution of Spatial Texture Activity ($T$)\nacross Public Image Benchmark ($N={}$)".format(len(df_char)), fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("Mean Texture Activity Score $T$ (DeterministicTextureMasker)", fontsize=11)
    ax_hist.set_ylabel("Image Count", fontsize=11)
    ax_hist.legend(loc="upper right", frameon=True, fontsize=9)
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    # Panel B: 4x4 Representative Thumbnail Grid
    # Select 4 from Low, 4 from Mid, 4 from High, 4 from Legacy5
    sample_images = []
    
    # 4 Low
    lows = df_char[df_char["texture_stratum"] == "low"].head(4)["image_id"].tolist()
    # 4 Mid
    mids = df_char[df_char["texture_stratum"] == "mid"].head(4)["image_id"].tolist()
    # 4 High
    highs = df_char[df_char["texture_stratum"] == "high"].head(4)["image_id"].tolist()
    # 4 Legacy
    legacies = df_char[df_char["source_dataset"] == "legacy5"].head(4)["image_id"].tolist()
    if len(legacies) < 4:
        legacies += df_char.tail(4 - len(legacies))["image_id"].tolist()

    grid_ids = lows + mids + highs + legacies
    grid_ids = grid_ids[:16]

    # Create sub-grid for thumbnails
    gs_sub = gs[1].subgridspec(4, 4, wspace=0.08, hspace=0.18)
    for idx, iid in enumerate(grid_ids):
        ax_sub = fig.add_subplot(gs_sub[idx])
        img_path = os.path.join(images_dir, f"{iid}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            ax_sub.imshow(img, cmap="gray")
        ax_sub.axis("off")
        
        row_info = df_char[df_char["image_id"] == iid].iloc[0]
        stratum = row_info["texture_stratum"].capitalize()
        t_val = row_info["texture_score_T"]
        short_id = iid.split("_", 1)[1] if "_" in iid else iid
        ax_sub.set_title(f"{short_id}\n[{stratum}, {t_val:.2f}]", fontsize=7, pad=2)

    # Save PDF and PNG
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    png_path = output_pdf.replace(".pdf", ".png")
    fig.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Overview figures saved: '{output_pdf}' and '{png_path}'")


# =============================================================================
# 6. PAPER SUMMARY TABLE
# =============================================================================

def print_dataset_summary_table(df_manifest: pd.DataFrame, df_char: pd.DataFrame):
    """Prints a structured summary table for inclusion in the academic paper."""
    merged = df_manifest.merge(df_char, on="image_id", suffixes=("", "_char"))
    
    summary_records = []
    for src, grp in merged.groupby("source dataset"):
        n_count = len(grp)
        t_mean = grp["texture_score_T"].mean()
        t_std = grp["texture_score_T"].std()
        n_low = (grp["texture_stratum"] == "low").sum()
        n_mid = (grp["texture_stratum"] == "mid").sum()
        n_high = (grp["texture_stratum"] == "high").sum()
        lic = grp["license/terms URL"].iloc[0]
        
        summary_records.append({
            "Dataset Source": src,
            "N": n_count,
            "Mean Texture T (±std)": f"{t_mean:.3f} ± {t_std:.3f}",
            "Low Stratum": n_low,
            "Mid Stratum": n_mid,
            "High Stratum": n_high,
            "License / Terms URL": lic,
        })

    total_row = {
        "Dataset Source": "**Total**",
        "N": len(merged),
        "Mean Texture T (±std)": f"{merged['texture_score_T'].mean():.3f} ± {merged['texture_score_T'].std():.3f}",
        "Low Stratum": (merged["texture_stratum"] == "low").sum(),
        "Mid Stratum": (merged["texture_stratum"] == "mid").sum(),
        "High Stratum": (merged["texture_stratum"] == "high").sum(),
        "License / Terms URL": "Multiple (Open Academic/Research)",
    }
    summary_records.append(total_row)
    df_summary = pd.DataFrame(summary_records)

    print("\n" + "=" * 90)
    print(" BENCHMARK DATASET COMPOSITION (FOR PAPER TABLE)")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90 + "\n")
