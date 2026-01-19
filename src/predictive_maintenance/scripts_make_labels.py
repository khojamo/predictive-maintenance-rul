from __future__ import annotations

from pathlib import Path
import yaml

from predictive_maintenance.data import load_fd001
from predictive_maintenance.labels import add_rul_label, add_failure_within_h_label


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    cfg_path = project_root() / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    H = int(cfg["risk_horizon"])

    train, test, rul = load_fd001()

    train_l = add_rul_label(train)
    train_l = add_failure_within_h_label(train_l, horizon=H)

    # Quick sanity prints
    print("H =", H)
    print("Train columns:", list(train_l.columns))
    print("RUL min/max:", int(train_l["rul"].min()), int(train_l["rul"].max()))
    print("Fail_within_h rate:", float(train_l["fail_within_h"].mean()))

    # Save a small sample for inspection (not required, but useful)
    out_dir = project_root() / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "train_labeled_sample.csv"
    train_l.head(2000).to_csv(sample_path, index=False)
    print("Wrote sample:", sample_path)


if __name__ == "__main__":
    main()
