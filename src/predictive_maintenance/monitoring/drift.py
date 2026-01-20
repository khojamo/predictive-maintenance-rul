from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predictive_maintenance.feature_config import load_feature_config
from predictive_maintenance.features import build_rolling_features


def project_root() -> Path:
    """Repo root path."""
    return Path(__file__).resolve().parents[3]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


BASELINE_BINS_PATH = "reports/baseline_bins.json"
BASELINE_STATS_PATH = "reports/baseline_feature_stats.json"


@dataclass(frozen=True)
class BaselineSpec:
    created_at_utc: str
    dataset: str
    bins: int
    latest_only: bool
    id_col: str
    time_col: str
    feature_bins: dict[str, list[float]]


def _baseline_paths() -> tuple[Path, Path]:
    root = project_root()
    return root / BASELINE_BINS_PATH, root / BASELINE_STATS_PATH


def load_baseline_bins() -> BaselineSpec:
    bins_path, _ = _baseline_paths()
    if not bins_path.exists():
        raise FileNotFoundError(
            f"Baseline bins not found: {bins_path}. Run scripts_make_baseline_bins.py first."
        )
    obj = json.loads(bins_path.read_text(encoding="utf-8"))
    return BaselineSpec(
        created_at_utc=str(obj["created_at_utc"]),
        dataset=str(obj.get("dataset", "FD001")),
        bins=int(obj.get("bins", 10)),
        latest_only=bool(obj.get("latest_only", True)),
        id_col=str(obj.get("id_col", "unit_id")),
        time_col=str(obj.get("time_col", "cycle")),
        feature_bins={k: [float(x) for x in v] for k, v in obj["feature_bins"].items()},
    )


def load_baseline_stats(stats_path: Path | None = None) -> dict[str, dict[str, float]]:
    """
    Load baseline feature stats.
    Supports BOTH formats:
      - {"features": {...}}
      - {...}  (direct mapping)
    """
    if stats_path is None:
        _, stats_path = _baseline_paths()

    if not stats_path.exists():
        return {}

    obj = json.loads(stats_path.read_text(encoding="utf-8"))

    # accept both formats
    feats = obj.get("features", obj)

    out: dict[str, dict[str, float]] = {}
    for k, v in feats.items():
        if isinstance(v, dict) and ("mean" in v) and ("std" in v):
            out[k] = {"mean": float(v["mean"]), "std": float(v["std"])}
    return out




def psi_from_edges(values: np.ndarray, edges: list[float], eps: float = 1e-6) -> float:
    """Population Stability Index vs a quantile-binned baseline.

    Assumption: edges were created from baseline quantiles => baseline expected proportions are ~uniform.
    """
    if len(edges) < 3:
        return 0.0

    e = np.unique(np.asarray(edges, dtype=float))
    if e.size < 3:
        return 0.0

    # histogram counts in bins (len-1)
    counts, _ = np.histogram(values.astype(float), bins=e)
    actual = counts / max(counts.sum(), 1)

    n_bins = len(actual)
    expected = np.full(n_bins, 1.0 / n_bins, dtype=float)

    a = np.clip(actual, eps, 1.0)
    e2 = np.clip(expected, eps, 1.0)

    return float(np.sum((a - e2) * np.log(a / e2)))


def compute_drift(
    raw_rows: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
    top_k: int = 20,
    psi_threshold: float = 0.2,
) -> dict[str, Any]:
    baseline = load_baseline_bins()
    stats = load_baseline_stats()
    feature_cfg = load_feature_config(project_root() / "models" / "feature_config.json")
    if feature_cfg is not None:
        id_col = feature_cfg.id_col
        time_col = feature_cfg.time_col
        signal_cols = feature_cfg.signal_cols
        categorical_cols = feature_cfg.categorical_cols
        categorical_levels = feature_cfg.categorical_levels
    else:
        id_col = "unit_id"
        time_col = "cycle"
        signal_cols = None
        categorical_cols = []
        categorical_levels = {}

    # Build features, align to scoring behavior
    feat = build_rolling_features(
        raw_rows,
        window=window,
        min_periods=min_periods,
        signal_cols=signal_cols,
        categorical_cols=categorical_cols,
        categorical_levels=categorical_levels,
        id_col=id_col,
        time_col=time_col,
    )
    if baseline.latest_only:
        latest = (
            feat.sort_values([id_col, time_col])
            .groupby(id_col, as_index=False)
            .tail(1)
        )
    else:
        latest = feat

    drift_rows: list[dict[str, Any]] = []

    for col, edges in baseline.feature_bins.items():
        if col not in latest.columns:
            continue
        v = latest[col].to_numpy()
        psi = psi_from_edges(v, edges)

        b = stats.get(col, {})
        drift_rows.append(
            {
                "feature": col,
                "psi": float(psi),
                "flag": bool(psi >= psi_threshold),
                "baseline_mean": float(b.get("mean", 0.0)),
                "baseline_std": float(b.get("std", 0.0)),
                "recent_mean": float(np.mean(v)) if len(v) else 0.0,
                "recent_std": float(np.std(v)) if len(v) else 0.0,
            }
        )

    drift_rows.sort(key=lambda r: r["psi"], reverse=True)

    flagged = [r for r in drift_rows if r["flag"]]

    return {
        "created_at_utc": _utc_now_iso(),
        "baseline_created_at_utc": baseline.created_at_utc,
        "dataset": baseline.dataset,
        "psi_threshold": psi_threshold,
        "n_features": len(drift_rows),
        "n_flagged": len(flagged),
        "top": drift_rows[: int(top_k)],
    }
