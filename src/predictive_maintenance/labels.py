from __future__ import annotations

import pandas as pd


def add_rul_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RUL (remaining useful life) label for CMAPSS train-style data:
    rul = max_cycle_per_unit - cycle
    """
    out = df.copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).astype(int)
    return out


def add_failure_within_h_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Binary label: 1 if failure within next `horizon` cycles, else 0.
    Uses RUL as ground truth.
    """
    if "rul" not in df.columns:
        raise ValueError("Column 'rul' missing. Call add_rul_label() first.")

    out = df.copy()
    out["fail_within_h"] = (out["rul"] <= horizon).astype(int)
    return out
