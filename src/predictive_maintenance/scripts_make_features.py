from __future__ import annotations

from pathlib import Path
import yaml

from predictive_maintenance.data import load_fd001
from predictive_maintenance.labels import add_rul_label, add_failure_within_h_label
from predictive_maintenance.features import build_rolling_features


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    cfg = yaml.safe_load((project_root() / "configs" / "default.yaml").read_text(encoding="utf-8"))
    H = int(cfg["risk_horizon"])
    fcfg = cfg["features"]
    window = int(fcfg["window"])
    minp = int(fcfg["min_periods"])

    train, _, _ = load_fd001()
    train = add_failure_within_h_label(add_rul_label(train), horizon=H)

    X = build_rolling_features(train, window=window, min_periods=minp)

    # Merge labels for training table
    y = train[["unit_id", "cycle", "rul", "fail_within_h"]]
    table = X.merge(y, on=["unit_id", "cycle"], how="inner")

    # Sanity checks
    assert len(table) == len(train), "Feature rows should match raw rows after merge"
    assert table.isna().sum().sum() == 0, "No NaNs expected with min_periods>=1 (std might be 0)"

    out_dir = project_root() / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_features_sample.csv"
    table.head(2000).to_csv(out_path, index=False)

    print("Feature table shape:", table.shape)
    print("Wrote sample:", out_path)


if __name__ == "__main__":
    main()
