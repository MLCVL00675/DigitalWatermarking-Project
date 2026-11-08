"""
Long-format results schema and CSV persistence for experiments_v2.

Strict Schema Order:
1. run_id
2. experiment
3. image_id
4. payload_id
5. method
6. subband
7. wavelet
8. block_size
9. alpha_nominal
10. alpha_eff_mean
11. attack
12. severity
13. seed
14. psnr_clean
15. ssim_clean
16. psnr_attacked
17. ber
18. bcr
19. nc
20. extra_json
"""

import os
import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set, Tuple


SCHEMA_COLUMNS = [
    "run_id",
    "experiment",
    "image_id",
    "payload_id",
    "method",
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


@dataclass
class ResultRow:
    run_id: str
    experiment: str
    image_id: str
    payload_id: str
    method: str
    subband: str
    wavelet: str
    block_size: int
    alpha_nominal: float
    alpha_eff_mean: float
    attack: str
    severity: str
    seed: int
    psnr_clean: float
    ssim_clean: float
    psnr_attacked: float
    ber: float
    bcr: float
    nc: float
    extra_json: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultRow":
        return cls(
            run_id=str(data["run_id"]),
            experiment=str(data["experiment"]),
            image_id=str(data["image_id"]),
            payload_id=str(data["payload_id"]),
            method=str(data["method"]),
            subband=str(data["subband"]),
            wavelet=str(data["wavelet"]),
            block_size=int(data["block_size"]),
            alpha_nominal=float(data["alpha_nominal"]),
            alpha_eff_mean=float(data["alpha_eff_mean"]),
            attack=str(data["attack"]),
            severity=str(data["severity"]),
            seed=int(data["seed"]),
            psnr_clean=float(data["psnr_clean"]),
            ssim_clean=float(data["ssim_clean"]),
            psnr_attacked=float(data["psnr_attacked"]),
            ber=float(data["ber"]),
            bcr=float(data["bcr"]),
            nc=float(data["nc"]),
            extra_json=str(data.get("extra_json", "{}")),
        )


def make_trial_key(
    experiment: str,
    image_id: str,
    payload_id: str,
    method: str,
    subband: str,
    wavelet: str,
    block_size: int,
    alpha_nominal: float,
    attack: str,
    severity: str,
    seed: int,
) -> Tuple:
    """Returns canonical tuple identifying a unique trial for resumability."""
    return (
        str(experiment).strip(),
        str(image_id).strip(),
        str(payload_id).strip(),
        str(method).strip(),
        str(subband).strip(),
        str(wavelet).strip(),
        int(block_size),
        round(float(alpha_nominal), 4),
        str(attack).strip(),
        str(severity).strip(),
        int(seed),
    )


def init_results_csv(csv_path: str) -> None:
    """Initializes the CSV with headers if it does not exist."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(SCHEMA_COLUMNS)


def load_completed_keys(csv_path: str) -> Set[Tuple]:
    """Reads existing CSV rows and returns a set of completed trial keys."""
    if not os.path.exists(csv_path):
        return set()

    completed: Set[Tuple] = set()
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = make_trial_key(
                    experiment=row["experiment"],
                    image_id=row["image_id"],
                    payload_id=row["payload_id"],
                    method=row["method"],
                    subband=row["subband"],
                    wavelet=row["wavelet"],
                    block_size=int(row["block_size"]),
                    alpha_nominal=float(row["alpha_nominal"]),
                    attack=row["attack"],
                    severity=row["severity"],
                    seed=int(row["seed"]),
                )
                completed.add(key)
            except (KeyError, ValueError):
                continue
    return completed


def append_result_rows(csv_path: str, rows: List[ResultRow]) -> None:
    """Appends multiple ResultRow entries to the CSV file safely."""
    if not rows:
        return
    init_results_csv(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([
                r.run_id,
                r.experiment,
                r.image_id,
                r.payload_id,
                r.method,
                r.subband,
                r.wavelet,
                r.block_size,
                r.alpha_nominal,
                r.alpha_eff_mean,
                r.attack,
                r.severity,
                r.seed,
                r.psnr_clean,
                r.ssim_clean,
                r.psnr_attacked,
                r.ber,
                r.bcr,
                r.nc,
                r.extra_json,
            ])
        f.flush()
