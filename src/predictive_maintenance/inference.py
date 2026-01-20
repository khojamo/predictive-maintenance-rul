from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from predictive_maintenance.feature_config import FeatureConfig, load_feature_config
from predictive_maintenance.features import OP_COLS, SENSOR_COLS, build_rolling_features
from predictive_maintenance.validation import validate_cmapss, validate_timeseries


def project_root() -> Path:
    # .../src/predictive_maintenance/inference.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Artifacts:
    risk_model: Any
    rul_model: Any
    risk_threshold: float
    window: int
    min_periods: int
    id_col: str
    time_col: str
    signal_cols: list[str]
    categorical_cols: list[str]
    categorical_levels: dict[str, list[str]]
    risk_horizon: int


@lru_cache(maxsize=1)
def load_artifacts() -> Artifacts:
    root = project_root()

    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    window = int(cfg["features"]["window"])
    min_periods = int(cfg["features"]["min_periods"])
    risk_horizon = int(cfg.get("risk_horizon", 30))

    feature_cfg_path = root / "models" / "feature_config.json"
    feature_cfg: FeatureConfig | None = load_feature_config(feature_cfg_path)
    if feature_cfg is not None:
        window = feature_cfg.window
        min_periods = feature_cfg.min_periods
        risk_horizon = feature_cfg.risk_horizon
        id_col = feature_cfg.id_col
        time_col = feature_cfg.time_col
        signal_cols = feature_cfg.signal_cols
        categorical_cols = feature_cfg.categorical_cols
        categorical_levels = feature_cfg.categorical_levels
    else:
        id_col = "unit_id"
        time_col = "cycle"
        signal_cols = OP_COLS + SENSOR_COLS
        categorical_cols = []
        categorical_levels = {}

    th_path = root / "configs" / "thresholds.yaml"
    th = yaml.safe_load(th_path.read_text(encoding="utf-8")) if th_path.exists() else {"risk_threshold": 0.5}
    risk_threshold = float(th.get("risk_threshold", 0.5))

    risk_cal_path = root / "models" / "risk_model_calibrated.joblib"
    risk_base_path = root / "models" / "risk_model.joblib"
    if risk_cal_path.exists():
        risk_model = joblib.load(risk_cal_path)
    else:
        risk_model = joblib.load(risk_base_path)
    rul_model = joblib.load(root / "models" / "rul_model.joblib")

    return Artifacts(
        risk_model=risk_model,
        rul_model=rul_model,
        risk_threshold=risk_threshold,
        window=window,
        min_periods=min_periods,
        id_col=id_col,
        time_col=time_col,
        signal_cols=signal_cols,
        categorical_cols=categorical_cols,
        categorical_levels=categorical_levels,
        risk_horizon=risk_horizon,
    )


def score_latest_cycles(
    raw: pd.DataFrame,
    window: int | None = None,
    min_periods: int | None = None,
    all_cycles: bool = False,
    clamp_rul: bool = True,
) -> pd.DataFrame:
    """
    Input: raw CMAPSS-like rows with:
      unit_id, cycle, op_1..op_3, s_1..s_21
    Output: one scored row per unit_id (latest cycle)
    """
    a = load_artifacts()
    window = a.window if window is None else int(window)
    min_periods = a.min_periods if min_periods is None else int(min_periods)

    # strict schema + type coercion
    if a.id_col == "unit_id" and a.time_col == "cycle" and set(a.signal_cols) == set(OP_COLS + SENSOR_COLS):
        raw = validate_cmapss(raw)
    else:
        raw = validate_timeseries(
            raw,
            id_col=a.id_col,
            time_col=a.time_col,
            feature_cols=a.signal_cols,
            categorical_cols=a.categorical_cols,
        )

    feats = build_rolling_features(
        raw,
        window=window,
        min_periods=min_periods,
        signal_cols=a.signal_cols,
        categorical_cols=a.categorical_cols,
        categorical_levels=a.categorical_levels,
        id_col=a.id_col,
        time_col=a.time_col,
    )

    if all_cycles:
        latest = feats.sort_values([a.id_col, a.time_col]).reset_index(drop=True)
    else:
        latest = (
            feats.sort_values([a.id_col, a.time_col])
            .groupby(a.id_col, as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

    X = latest.drop(columns=[a.id_col, a.time_col])

    risk_proba = a.risk_model.predict_proba(X)[:, 1]
    risk_proba = risk_proba.clip(1e-6, 1 - 1e-6)
    risk_label = (risk_proba >= a.risk_threshold).astype(int)
    rul_pred = a.rul_model.predict(X)
    if clamp_rul:
        rul_pred = np.maximum(rul_pred, 0.0)

    out = latest[[a.id_col, a.time_col]].copy()
    out["risk_proba"] = risk_proba.astype(float)
    out["risk_label"] = risk_label.astype(int)
    out["risk_threshold"] = float(a.risk_threshold)
    out["rul_pred"] = rul_pred.astype(float)

    return out


def score_all_cycles_to_csv(
    raw: pd.DataFrame,
    out_path: Path,
    *,
    window: int | None = None,
    min_periods: int | None = None,
    clamp_rul: bool = True,
) -> tuple[Path, dict[str, float]]:
    """
    Stream all-cycle scores to CSV, one unit at a time (memory-friendly).
    """
    a = load_artifacts()
    window = a.window if window is None else int(window)
    min_periods = a.min_periods if min_periods is None else int(min_periods)

    if a.id_col == "unit_id" and a.time_col == "cycle" and set(a.signal_cols) == set(OP_COLS + SENSOR_COLS):
        raw = validate_cmapss(raw)
    else:
        raw = validate_timeseries(
            raw,
            id_col=a.id_col,
            time_col=a.time_col,
            feature_cols=a.signal_cols,
            categorical_cols=a.categorical_cols,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    header_written = False
    total_rows = 0
    total_units = 0
    alert_count = 0
    risk_sum = 0.0
    rul_sum = 0.0
    rul_min = float("inf")

    for _, unit_df in raw.sort_values([a.id_col, a.time_col]).groupby(a.id_col, as_index=False):
        total_units += 1
        feats = build_rolling_features(
            unit_df,
            window=window,
            min_periods=min_periods,
            signal_cols=a.signal_cols,
            categorical_cols=a.categorical_cols,
            categorical_levels=a.categorical_levels,
            id_col=a.id_col,
            time_col=a.time_col,
        )
        X = feats.drop(columns=[a.id_col, a.time_col])
        risk_proba = a.risk_model.predict_proba(X)[:, 1]
        risk_proba = risk_proba.clip(1e-6, 1 - 1e-6)
        risk_proba = risk_proba.clip(1e-6, 1 - 1e-6)
        risk_label = (risk_proba >= a.risk_threshold).astype(int)
        rul_pred = a.rul_model.predict(X)
        if clamp_rul:
            rul_pred = np.maximum(rul_pred, 0.0)

        out = feats[[a.id_col, a.time_col]].copy()
        out["risk_proba"] = risk_proba.astype(float)
        out["risk_label"] = risk_label.astype(int)
        out["risk_threshold"] = float(a.risk_threshold)
        out["rul_pred"] = rul_pred.astype(float)

        out.to_csv(out_path, mode="a", header=not header_written, index=False)
        header_written = True
        total_rows += len(out)
        alert_count += int((out["risk_label"] == 1).sum())
        risk_sum += float(out["risk_proba"].sum())
        rul_sum += float(out["rul_pred"].sum())
        if len(out):
            rul_min = min(rul_min, float(out["rul_pred"].min()))

    summary = {
        "rows": float(total_rows),
        "units": float(total_units),
        "alert_rate": float(alert_count / total_rows) if total_rows else 0.0,
        "risk_mean": float(risk_sum / total_rows) if total_rows else 0.0,
        "rul_mean": float(rul_sum / total_rows) if total_rows else 0.0,
        "rul_min": float(rul_min) if total_rows else 0.0,
    }
    return out_path, summary
