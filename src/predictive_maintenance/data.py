from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import pandas as pd
import requests


NASA_CMAPSS_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"


@dataclass(frozen=True)
class CmapssPaths:
    root: Path
    raw_dir: Path
    zip_path: Path

    def __init__(self, root: Path):
        raw_dir = root / "data" / "cmapss_raw"
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "raw_dir", raw_dir)
        object.__setattr__(self, "zip_path", raw_dir / "CMAPSSData.zip")


def project_root() -> Path:
    # repo/src/predictive_maintenance/data.py -> parents[2] == repo/
    return Path(__file__).resolve().parents[2]


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def _find_required_files(raw_dir: Path, fd: str) -> dict[str, Path]:
    need = [f"train_{fd}.txt", f"test_{fd}.txt", f"RUL_{fd}.txt"]
    found: dict[str, Path] = {}
    for name in need:
        matches = list(raw_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(
                f"Could not find {name} after extraction. "
                f"Looked under: {raw_dir}. Contents may have changed."
            )
        found[name] = matches[0]
    return found


def ensure_cmapss_downloaded_and_extracted(paths: CmapssPaths, fd: str = "FD001") -> dict[str, Path]:
    # Download zip if missing
    if not paths.zip_path.exists():
        _download_file(NASA_CMAPSS_ZIP_URL, paths.zip_path)

    # Extract if required files not found yet
    try:
        return _find_required_files(paths.raw_dir, fd)
    except FileNotFoundError:
        _extract_zip(paths.zip_path, paths.raw_dir)
        return _find_required_files(paths.raw_dir, fd)


def _read_cmapss_txt(path: Path) -> pd.DataFrame:
    # whitespace-separated, no header, sometimes irregular spaces
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python")


def standardize_cmapss(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    col_names = (
        ["unit_id", "cycle", "op_1", "op_2", "op_3"]
        + [f"s_{i}" for i in range(1, 22)]
    )

    # Some versions may have extra trailing empty cols; keep only first 26.
    train = train_raw.iloc[:, :26].copy()
    test = test_raw.iloc[:, :26].copy()
    train.columns = col_names
    test.columns = col_names

    return train, test


def load_fd001() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = CmapssPaths(project_root())
    files = ensure_cmapss_downloaded_and_extracted(paths, fd="FD001")

    train_raw = _read_cmapss_txt(files["train_FD001.txt"])
    test_raw = _read_cmapss_txt(files["test_FD001.txt"])
    rul_raw = _read_cmapss_txt(files["RUL_FD001.txt"])

    train, test = standardize_cmapss(train_raw, test_raw)

    rul = rul_raw.copy()
    rul.columns = ["rul_end_of_test"]

    return train, test, rul
