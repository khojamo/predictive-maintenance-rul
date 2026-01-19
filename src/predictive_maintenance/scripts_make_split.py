from __future__ import annotations

from pathlib import Path
import yaml

from predictive_maintenance.data import load_fd001
from predictive_maintenance.labels import add_rul_label, add_failure_within_h_label
from predictive_maintenance.split import split_by_unit_id, filter_by_units


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    cfg = yaml.safe_load((project_root() / "configs" / "default.yaml").read_text(encoding="utf-8"))
    H = int(cfg["risk_horizon"])
    sp = cfg["split"]

    train_df, _, _ = load_fd001()
    train_df = add_failure_within_h_label(add_rul_label(train_df), horizon=H)

    res = split_by_unit_id(
        train_df,
        train_frac=float(sp["train"]),
        valid_frac=float(sp["valid"]),
        test_frac=float(sp["test"]),
        seed=int(sp["seed"]),
    )

    tr = filter_by_units(train_df, res.train_units)
    va = filter_by_units(train_df, res.valid_units)
    te = filter_by_units(train_df, res.test_units)

    print("Units:", len(res.train_units), len(res.valid_units), len(res.test_units))
    print("Rows :", len(tr), len(va), len(te))
    print("Class rate (train/valid/test):",
          float(tr["fail_within_h"].mean()),
          float(va["fail_within_h"].mean()),
          float(te["fail_within_h"].mean()))

    # Save lists for reproducibility
    out_dir = project_root() / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_units.txt").write_text("\n".join(map(str, sorted(res.train_units))), encoding="utf-8")
    (out_dir / "valid_units.txt").write_text("\n".join(map(str, sorted(res.valid_units))), encoding="utf-8")
    (out_dir / "test_units.txt").write_text("\n".join(map(str, sorted(res.test_units))), encoding="utf-8")
    print("Wrote unit lists to:", out_dir)


if __name__ == "__main__":
    main()
