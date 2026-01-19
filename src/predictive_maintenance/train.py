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
from predictive_maintenance.feature_config import FeatureConfig, save_feature_config
from predictive_maintenance.monitoring.baseline import build_baseline_bins, save_baseline
from predictive_maintenance.labels import add_rul_label, add_failure_within_h_label
from predictive_maintenance.features import build_rolling_features
from predictive_maintenance.split import split_by_unit_id, filter_by_units
from predictive_maintenance.validation import validate_timeseries
from predictive_maintenance.labels import ensure_labels


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


def make_training_table_from_df(
    df: pd.DataFrame,
    cfg: dict,
    *,
    signal_cols: list[str],
) -> pd.DataFrame:
    H = int(cfg["risk_horizon"])
    window = int(cfg["features"]["window"])
    minp = int(cfg["features"]["min_periods"])
    sp = cfg["split"]

    df = ensure_labels(df, horizon=H)
    df = validate_timeseries(df, id_col="unit_id", time_col="cycle", feature_cols=signal_cols)

    X = build_rolling_features(df, window=window, min_periods=minp, signal_cols=signal_cols)
    y = df[["unit_id", "cycle", "rul", "fail_within_h"]]
    table = X.merge(y, on=["unit_id", "cycle"], how="inner")

    split_res = split_by_unit_id(
        table,
        train_frac=float(sp["train"]),
        valid_frac=float(sp["valid"]),
        test_frac=float(sp["test"]),
        seed=int(sp["seed"]),
    )

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


def train_and_eval_from_df(
    df: pd.DataFrame,
    cfg: dict,
    *,
    signal_cols: list[str],
) -> tuple[dict, Pipeline, Pipeline, list[str]]:
    table = make_training_table_from_df(df, cfg, signal_cols=signal_cols)

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

    return metrics, risk_model, rul_model, feature_cols


def train_models_from_table(
    table: pd.DataFrame,
) -> tuple[
    dict,
    Pipeline,
    Pipeline,
    list[str],
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
]:
    feature_cols = [c for c in table.columns if c not in {"unit_id", "cycle", "rul", "fail_within_h", "split"}]

    tr = table[table["split"] == "train"]
    va = table[table["split"] == "valid"]
    te = table[table["split"] == "test"]

    X_tr, X_va, X_te = tr[feature_cols], va[feature_cols], te[feature_cols]
    y_tr_cls, y_va_cls, y_te_cls = tr["fail_within_h"], va["fail_within_h"], te["fail_within_h"]
    y_tr_reg, y_va_reg, y_te_reg = tr["rul"], va["rul"], te["rul"]

    risk_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    risk_model.fit(X_tr, y_tr_cls)

    va_proba = risk_model.predict_proba(X_va)[:, 1]
    te_proba = risk_model.predict_proba(X_te)[:, 1]

    va_pred = (va_proba >= 0.5).astype(int)
    te_pred = (te_proba >= 0.5).astype(int)

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

    return metrics, risk_model, rul_model, feature_cols, (X_tr, y_tr_cls, X_va, y_va_cls)


def save_trained_artifacts(
    cfg: dict,
    *,
    signal_cols: list[str],
    risk_model: Pipeline,
    rul_model: Pipeline,
    metrics: dict,
    calibrated_risk_model: Pipeline | None = None,
    threshold_cfg: dict | None = None,
    calibration_report: dict | None = None,
) -> None:
    root = project_root()
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    risk_path = models_dir / "risk_model.joblib"
    rul_path = models_dir / "rul_model.joblib"
    metrics_path = reports_dir / "metrics.json"

    joblib.dump(risk_model, risk_path)
    joblib.dump(rul_model, rul_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if calibrated_risk_model is not None:
        joblib.dump(calibrated_risk_model, models_dir / "risk_model_calibrated.joblib")

    if threshold_cfg is not None:
        (root / "configs").mkdir(parents=True, exist_ok=True)
        (root / "configs" / "thresholds.yaml").write_text(
            yaml.safe_dump(threshold_cfg), encoding="utf-8"
        )

    if calibration_report is not None:
        (root / "reports" / "calibration.json").write_text(
            json.dumps(calibration_report, indent=2), encoding="utf-8"
        )

    fcfg = FeatureConfig(
        id_col="unit_id",
        time_col="cycle",
        signal_cols=signal_cols,
        window=int(cfg["features"]["window"]),
        min_periods=int(cfg["features"]["min_periods"]),
        risk_horizon=int(cfg.get("risk_horizon", 30)),
    )
    save_feature_config(fcfg, models_dir / "feature_config.json")


def build_and_save_baseline_from_table(
    table: pd.DataFrame,
    *,
    cfg: dict,
    dataset_name: str,
) -> None:
    fcfg = cfg["features"]
    window = int(fcfg["window"])
    minp = int(fcfg["min_periods"])
    bins = int(cfg.get("monitoring", {}).get("psi_bins", 10))

    train = table[table["split"] == "train"].copy()
    latest = train.sort_values(["unit_id", "cycle"]).groupby("unit_id", as_index=False).tail(1)

    feature_bins, feature_stats = build_baseline_bins(
        latest,
        bins=bins,
        exclude_cols=["unit_id", "cycle", "rul", "fail_within_h", "split"],
    )

    save_baseline(
        out_dir=project_root() / "reports",
        dataset_name=dataset_name,
        bins=bins,
        window=window,
        min_periods=minp,
        feature_bins=feature_bins,
        feature_stats=feature_stats,
    )


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
