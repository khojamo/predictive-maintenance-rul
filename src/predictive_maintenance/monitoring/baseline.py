from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _quantile_edges(values: np.ndarray, bins: int) -> list[float] | None:
    x = values.astype(float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return None
    qs = np.linspace(0.0, 1.0, bins + 1)
    # numpy>=2 uses method=; keep compatibility
    try:
        edges = np.quantile(x, qs, method="linear")
    except TypeError:
        edges = np.quantile(x, qs, interpolation="linear")

    edges = np.unique(edges)
    if edges.size < 3:
        return None
    return [float(v) for v in edges.tolist()]


def build_baseline_bins(
    latest: pd.DataFrame,
    *,
    bins: int,
    exclude_cols: list[str] | None = None,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float]]]:
    exclude_cols = exclude_cols or []
    feature_cols = [c for c in latest.columns if c not in set(exclude_cols)]

    feature_bins: dict[str, list[float]] = {}
    feature_stats: dict[str, dict[str, float]] = {}

    for col in feature_cols:
        arr = latest[col].to_numpy()
        edges = _quantile_edges(arr, bins=bins)
        if edges is None:
            continue
        feature_bins[col] = edges
        feature_stats[col] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    return feature_bins, feature_stats


def save_baseline(
    *,
    out_dir: Path,
    dataset_name: str,
    bins: int,
    window: int,
    min_periods: int,
    feature_bins: dict[str, list[float]],
    feature_stats: dict[str, dict[str, float]],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_bins = {
        "created_at_utc": _utc_now_iso(),
        "dataset": dataset_name,
        "bins": bins,
        "window": window,
        "min_periods": min_periods,
        "feature_bins": feature_bins,
    }

    bins_path = out_dir / "baseline_bins.json"
    stats_path = out_dir / "baseline_feature_stats.json"

    bins_path.write_text(json.dumps(out_bins, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")

    return bins_path, stats_path
