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


def ensure_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    failure_cycle_col: str = "failure_cycle",
) -> pd.DataFrame:
    """
    Ensure both RUL and fail-within-h labels exist.

    Priority:
      1) Use existing 'rul' if present.
      2) Else derive from failure_cycle_col if present.
      3) Else derive from max cycle per unit_id (CMAPSS-style).
    """
    out = df.copy()

    if "rul" not in out.columns:
        if failure_cycle_col in out.columns:
            out["rul"] = (out[failure_cycle_col] - out["cycle"]).clip(lower=0).astype(int)
        else:
            out = add_rul_label(out)

    if "fail_within_h" not in out.columns:
        out = add_failure_within_h_label(out, horizon=horizon)

    return out
