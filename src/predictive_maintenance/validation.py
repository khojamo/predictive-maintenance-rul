from __future__ import annotations

"""Schema validation.

Pandera is used when available. If pandera isn't installed, validation becomes a
no-op (useful for minimal environments / CI smoke tests).
"""

from typing import Any

import pandas as pd

try:
    import pandera.pandas as pa
    from pandera import Check
except ModuleNotFoundError:  # pragma: no cover
    pa = None  # type: ignore
    Check = None  # type: ignore


def cmapss_schema():
    if pa is None:  # pragma: no cover
        return None

    cols: dict[str, Any] = {
        "unit_id": pa.Column(int, Check.ge(1), nullable=False, coerce=True),
        "cycle": pa.Column(int, Check.ge(1), nullable=False, coerce=True),
        "op_1": pa.Column(float, nullable=False, coerce=True),
        "op_2": pa.Column(float, nullable=False, coerce=True),
        "op_3": pa.Column(float, nullable=False, coerce=True),
    }

    for i in range(1, 22):
        cols[f"s_{i}"] = pa.Column(float, nullable=False, coerce=True)

    return pa.DataFrameSchema(
        columns=cols,
        strict=True,
        checks=[
            Check(lambda df: df.groupby("unit_id")["cycle"].max().min() >= 1),
        ],
    )


def validate_cmapss(df):
    """Validate raw CMAPSS-like sensor rows.

    Returns the validated dataframe (pandera may coerce dtypes).
    """

    schema = cmapss_schema()
    if schema is None:  # pandera missing
        return df
    return schema.validate(df, lazy=True)


def validate_timeseries(
    df,
    *,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
):
    """Lightweight validation for custom time-series datasets."""
    missing = [c for c in [id_col, time_col, *feature_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[id_col] = out[id_col].astype(int)
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    if out[time_col].isna().any():
        raise ValueError(f"Time column '{time_col}' contains non-numeric values.")

    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[[id_col, time_col]].isna().any().any():
        raise ValueError("Found missing values in id/time columns after coercion.")

    return out
