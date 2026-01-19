from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train_units: set[int]
    valid_units: set[int]
    test_units: set[int]


def split_by_unit_id(
    df: pd.DataFrame,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    seed: int = 42,
) -> SplitResult:
    if abs((train_frac + valid_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("train_frac + valid_frac + test_frac must equal 1.0")

    units = np.array(sorted(df["unit_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)

    n = len(units)
    n_train = int(round(n * train_frac))
    n_valid = int(round(n * valid_frac))
    # remainder goes to test (ensures total = n)
    n_test = n - n_train - n_valid

    train_units = set(units[:n_train].tolist())
    valid_units = set(units[n_train : n_train + n_valid].tolist())
    test_units = set(units[n_train + n_valid : n_train + n_valid + n_test].tolist())

    # sanity: disjointness
    if (train_units & valid_units) or (train_units & test_units) or (valid_units & test_units):
        raise RuntimeError("Unit splits are not disjoint (this should never happen).")

    return SplitResult(train_units=train_units, valid_units=valid_units, test_units=test_units)


def filter_by_units(df: pd.DataFrame, units: set[int]) -> pd.DataFrame:
    return df[df["unit_id"].isin(units)].copy()
