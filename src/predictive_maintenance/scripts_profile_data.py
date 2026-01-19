from __future__ import annotations

from pathlib import Path

import pandas as pd

from predictive_maintenance.data import load_fd001
from predictive_maintenance.validation import validate_cmapss


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def profile(df: pd.DataFrame) -> str:
    units = df["unit_id"].nunique()
    rows = len(df)
    cycles_min = int(df.groupby("unit_id")["cycle"].max().min())
    cycles_max = int(df.groupby("unit_id")["cycle"].max().max())
    missing = int(df.isna().sum().sum())

    return "\n".join(
        [
            f"- rows: {rows}",
            f"- units: {units}",
            f"- min max-cycle per unit: {cycles_min}",
            f"- max max-cycle per unit: {cycles_max}",
            f"- total missing values: {missing}",
        ]
    )


def main() -> None:
    train, test, rul = load_fd001()

    validate_cmapss(train)
    validate_cmapss(test)

    out = []
    out.append("# CMAPSS FD001 Data Profile\n")
    out.append("## Train\n")
    out.append(profile(train))
    out.append("\n## Test\n")
    out.append(profile(test))
    out.append("\n## RUL\n")
    out.append(f"- rows: {len(rul)}")

    report_path = project_root() / "reports" / "data_profile.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(out), encoding="utf-8")

    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
