from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from predictive_maintenance.features import build_rolling_features
from predictive_maintenance.validation import validate_cmapss


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


@lru_cache(maxsize=1)
def load_artifacts() -> Artifacts:
    root = project_root()

    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    window = int(cfg["features"]["window"])
    min_periods = int(cfg["features"]["min_periods"])

    th_path = root / "configs" / "thresholds.yaml"
    th = yaml.safe_load(th_path.read_text(encoding="utf-8")) if th_path.exists() else {"risk_threshold": 0.5}
    risk_threshold = float(th.get("risk_threshold", 0.5))

    risk_model = joblib.load(root / "models" / "risk_model_calibrated.joblib")
    rul_model = joblib.load(root / "models" / "rul_model.joblib")

    return Artifacts(
        risk_model=risk_model,
        rul_model=rul_model,
        risk_threshold=risk_threshold,
        window=window,
        min_periods=min_periods,
    )


def score_latest_cycles(
    raw: pd.DataFrame,
    window: int | None = None,
    min_periods: int | None = None,
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
    raw = validate_cmapss(raw)

    feats = build_rolling_features(raw, window=window, min_periods=min_periods)

    latest = (
        feats.sort_values(["unit_id", "cycle"])
        .groupby("unit_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    X = latest.drop(columns=["unit_id", "cycle"])

    risk_proba = a.risk_model.predict_proba(X)[:, 1]
    risk_proba = risk_proba.clip(1e-6, 1 - 1e-6)
    risk_label = (risk_proba >= a.risk_threshold).astype(int)
    rul_pred = a.rul_model.predict(X)

    out = latest[["unit_id", "cycle"]].copy()
    out["risk_proba"] = risk_proba.astype(float)
    out["risk_label"] = risk_label.astype(int)
    out["risk_threshold"] = float(a.risk_threshold)
    out["rul_pred"] = rul_pred.astype(float)

    return out
