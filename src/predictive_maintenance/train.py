from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import yaml

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
import joblib

from predictive_maintenance.data import load_fd001
from predictive_maintenance.labels import add_rul_label, add_failure_within_h_label
from predictive_maintenance.features import build_rolling_features
from predictive_maintenance.split import split_by_unit_id, filter_by_units


@dataclass(frozen=True)
class Artifacts:
    risk_model_path: Path
    rul_model_path: Path
    metrics_path: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def make_training_table(cfg: dict) -> pd.DataFrame:
    H = int(cfg["risk_horizon"])
    window = int(cfg["features"]["window"])
    minp = int(cfg["features"]["min_periods"])
    sp = cfg["split"]

    train_raw, _, _ = load_fd001()
    labeled = add_failure_within_h_label(add_rul_label(train_raw), horizon=H)

    split_res = split_by_unit_id(
        labeled,
        train_frac=float(sp["train"]),
        valid_frac=float(sp["valid"]),
        test_frac=float(sp["test"]),
        seed=int(sp["seed"]),
    )

    X = build_rolling_features(labeled, window=window, min_periods=minp)
    y = labeled[["unit_id", "cycle", "rul", "fail_within_h"]]
    table = X.merge(y, on=["unit_id", "cycle"], how="inner")

    table["split"] = "train"
    table.loc[table["unit_id"].isin(split_res.valid_units), "split"] = "valid"
    table.loc[table["unit_id"].isin(split_res.test_units), "split"] = "test"

    return table


def train_and_eval(cfg: dict) -> tuple[dict, Pipeline, Pipeline]:
    table = make_training_table(cfg)

    feature_cols = [c for c in table.columns if c not in {"unit_id", "cycle", "rul", "fail_within_h", "split"}]

    tr = table[table["split"] == "train"]
    va = table[table["split"] == "valid"]
    te = table[table["split"] == "test"]

    X_tr, X_va, X_te = tr[feature_cols], va[feature_cols], te[feature_cols]
    y_tr_cls, y_va_cls, y_te_cls = tr["fail_within_h"], va["fail_within_h"], te["fail_within_h"]
    y_tr_reg, y_va_reg, y_te_reg = tr["rul"], va["rul"], te["rul"]

    # --- Risk model (classification) baseline ---
    risk_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    risk_model.fit(X_tr, y_tr_cls)

    va_proba = risk_model.predict_proba(X_va)[:, 1]
    te_proba = risk_model.predict_proba(X_te)[:, 1]

    # threshold 0.5 baseline for now
    va_pred = (va_proba >= 0.5).astype(int)
    te_pred = (te_proba >= 0.5).astype(int)

    # --- RUL model (regression) baseline ---
    rul_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    rul_model.fit(X_tr, y_tr_reg)

    va_rul = rul_model.predict(X_va)
    te_rul = rul_model.predict(X_te)

    metrics = {
        "data": {
            "rows_train": int(len(tr)),
            "rows_valid": int(len(va)),
            "rows_test": int(len(te)),
            "features": int(len(feature_cols)),
        },
        "risk": {
            "valid_pr_auc": float(average_precision_score(y_va_cls, va_proba)),
            "valid_roc_auc": float(roc_auc_score(y_va_cls, va_proba)),
            "valid_f1@0.5": float(f1_score(y_va_cls, va_pred)),
            "test_pr_auc": float(average_precision_score(y_te_cls, te_proba)),
            "test_roc_auc": float(roc_auc_score(y_te_cls, te_proba)),
            "test_f1@0.5": float(f1_score(y_te_cls, te_pred)),
        },
        "rul": {
            "valid_mae": float(mean_absolute_error(y_va_reg, va_rul)),
            "valid_rmse": float(mean_squared_error(y_va_reg, va_rul, squared=False)),
            "test_mae": float(mean_absolute_error(y_te_reg, te_rul)),
            "test_rmse": float(mean_squared_error(y_te_reg, te_rul, squared=False)),
        },
    }

    return metrics, risk_model, rul_model


def main() -> None:
    cfg = yaml.safe_load((project_root() / "configs" / "default.yaml").read_text(encoding="utf-8"))

    metrics, risk_model, rul_model = train_and_eval(cfg)

    models_dir = project_root() / "models"
    reports_dir = project_root() / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    risk_path = models_dir / "risk_model.joblib"
    rul_path = models_dir / "rul_model.joblib"
    metrics_path = reports_dir / "metrics.json"

    joblib.dump(risk_model, risk_path)
    joblib.dump(rul_model, rul_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved:", risk_path)
    print("Saved:", rul_path)
    print("Saved:", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
