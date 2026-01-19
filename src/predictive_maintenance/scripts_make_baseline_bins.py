from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from predictive_maintenance.data import load_fd001
from predictive_maintenance.features import build_rolling_features
from predictive_maintenance.labels import add_failure_within_h_label, add_rul_label


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def main() -> None:
    root = project_root()
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))

    fcfg = cfg["features"]
    window = int(fcfg["window"])
    minp = int(fcfg["min_periods"])
    bins = int(cfg.get("monitoring", {}).get("psi_bins", 10))

    # Baseline built from TRAIN features (FD001)
    H = int(cfg["risk_horizon"])
    train, _, _ = load_fd001()
    train = add_failure_within_h_label(add_rul_label(train), horizon=H)

    feat = build_rolling_features(train, window=window, min_periods=minp)

    # Use latest row per unit (matches production scoring behavior)
    latest = feat.sort_values(["unit_id", "cycle"]).groupby("unit_id", as_index=False).tail(1)

    feature_cols = [c for c in latest.columns if c not in {"unit_id", "cycle"}]

    feature_bins: dict[str, list[float]] = {}
    feature_stats: dict[str, dict[str, float]] = {}

    for col in feature_cols:
        arr = latest[col].to_numpy()
        edges = _quantile_edges(arr, bins=bins)
        if edges is None:
            continue
        feature_bins[col] = edges
        feature_stats[col] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    out_bins = {
        "created_at_utc": _utc_now_iso(),
        "dataset": "FD001",
        "bins": bins,
        "window": window,
        "min_periods": minp,
        "feature_bins": feature_bins,
    }

    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    bins_path = out_dir / "baseline_bins.json"
    bins_path.write_text(json.dumps(out_bins, indent=2), encoding="utf-8")

    stats_path = out_dir / "baseline_feature_stats.json"
    stats_path.write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")

    print(f"Wrote baseline bins: {bins_path}")
    print(f"Wrote baseline stats: {stats_path}")
    print("Features tracked:", len(feature_bins))


if __name__ == "__main__":
    main()
