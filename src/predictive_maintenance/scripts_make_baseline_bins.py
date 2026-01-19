from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from predictive_maintenance.data import load_fd001
from predictive_maintenance.features import build_rolling_features
from predictive_maintenance.labels import add_failure_within_h_label, add_rul_label
from predictive_maintenance.monitoring.baseline import build_baseline_bins, save_baseline


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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

    feature_bins, feature_stats = build_baseline_bins(
        latest, bins=bins, exclude_cols=["unit_id", "cycle"]
    )

    out_dir = root / "reports"
    bins_path, stats_path = save_baseline(
        out_dir=out_dir,
        dataset_name="FD001",
        bins=bins,
        window=window,
        min_periods=minp,
        feature_bins=feature_bins,
        feature_stats=feature_stats,
    )

    print(f"Wrote baseline bins: {bins_path}")
    print(f"Wrote baseline stats: {stats_path}")
    print("Features tracked:", len(feature_bins))


if __name__ == "__main__":
    main()
