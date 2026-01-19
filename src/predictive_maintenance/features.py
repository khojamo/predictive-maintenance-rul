from __future__ import annotations

import pandas as pd

SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
OP_COLS = ["op_1", "op_2", "op_3"]


def build_rolling_features(
    df: pd.DataFrame,
    window: int,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Build rolling-window features per unit_id using ONLY past/current cycles.

    Features per column (op_1..op_3 and s_1..s_21):
      - rolling mean/std/min/max over last `window` cycles
      - last value (current cycle)

    NaN handling (leakage-safe):
      - if min_periods > 1, early cycles have NaNs in rolling stats.
      - rolling mean/min/max are filled with current value (*_last)
      - rolling std is filled with 0.0
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    # Ensure correct order (important for rolling windows)
    out = df.sort_values(["unit_id", "cycle"]).copy()

    # Keep only needed columns (traceable)
    keep = ["unit_id", "cycle"] + OP_COLS + SENSOR_COLS
    out = out[keep]

    g = out.groupby("unit_id", group_keys=False)

    feats = []
    # Rolling stats + current value for each signal
    for col in OP_COLS + SENSOR_COLS:
        r = g[col].rolling(window=window, min_periods=min_periods)

        feats.append(r.mean().reset_index(level=0, drop=True).rename(f"{col}_roll{window}_mean"))
        feats.append(r.std(ddof=0).reset_index(level=0, drop=True).rename(f"{col}_roll{window}_std"))
        feats.append(r.min().reset_index(level=0, drop=True).rename(f"{col}_roll{window}_min"))
        feats.append(r.max().reset_index(level=0, drop=True).rename(f"{col}_roll{window}_max"))

        feats.append(out[col].rename(f"{col}_last"))

    feat_df = pd.concat(feats, axis=1)

    # Attach identifiers for merging with labels later
    feat_df.insert(0, "cycle", out["cycle"].to_numpy())
    feat_df.insert(0, "unit_id", out["unit_id"].to_numpy())

    # --- NaN handling (leakage-safe) ---
    # With min_periods > 1, early cycles per unit have NaNs for rolling stats.
    # We impute using only current info:
    # - roll mean/min/max -> current value (*_last)
    # - roll std          -> 0.0
    for col in OP_COLS + SENSOR_COLS:
        last = f"{col}_last"

        mean_c = f"{col}_roll{window}_mean"
        min_c = f"{col}_roll{window}_min"
        max_c = f"{col}_roll{window}_max"
        std_c = f"{col}_roll{window}_std"

        if mean_c in feat_df.columns:
            feat_df[mean_c] = feat_df[mean_c].fillna(feat_df[last])
        if min_c in feat_df.columns:
            feat_df[min_c] = feat_df[min_c].fillna(feat_df[last])
        if max_c in feat_df.columns:
            feat_df[max_c] = feat_df[max_c].fillna(feat_df[last])
        if std_c in feat_df.columns:
            feat_df[std_c] = feat_df[std_c].fillna(0.0)

    # Final hard check: no NaNs
    if feat_df.isna().any().any():
        bad = feat_df.isna().sum().sort_values(ascending=False)
        top = bad[bad > 0].head(30)
        raise ValueError(f"NaNs remain in features. Top offenders:\n{top}")

    return feat_df
