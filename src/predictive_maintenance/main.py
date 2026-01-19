from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# --- Your existing project modules ---
# These SHOULD exist based on what you already ran successfully.
from predictive_maintenance.validation import validate_cmapss  # type: ignore

# Try to import your feature builder with a few fallback names
try:
    from predictive_maintenance.features import build_feature_table as _build_feature_table  # type: ignore
except Exception:  # pragma: no cover
    try:
        from predictive_maintenance.features import make_feature_table as _build_feature_table  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import feature table builder from predictive_maintenance.features. "
            "Expected build_feature_table(...) or make_feature_table(...)."
        ) from e


APP_ROOT = Path(__file__).resolve().parents[2]  # project root (../.. from src/api/main.py)
MODELS_DIR = APP_ROOT / "models"
CONFIG_DIR = APP_ROOT / "configs"
REPORTS_DIR = APP_ROOT / "reports"


# -------------------- Simple YAML reader (no extra deps) --------------------
def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    out: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        # try cast
        if v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
        else:
            try:
                if "." in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except Exception:
                out[k] = v
    return out


# -------------------- Model loading --------------------
@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    risk_path = MODELS_DIR / "risk_model_calibrated.joblib"
    rul_path = MODELS_DIR / "rul_model.joblib"
    thresholds_path = CONFIG_DIR / "thresholds.yaml"

    if not risk_path.exists():
        raise RuntimeError(f"Missing model file: {risk_path}")
    if not rul_path.exists():
        raise RuntimeError(f"Missing model file: {rul_path}")

    risk_model = joblib.load(risk_path)
    rul_model = joblib.load(rul_path)
    thresholds = _read_simple_yaml(thresholds_path)
    threshold = float(thresholds.get("risk_threshold", 0.5))

    return {
        "risk_model": risk_model,
        "rul_model": rul_model,
        "risk_threshold": threshold,
    }


def _get_feature_names_from_model(model: Any) -> list[str] | None:
    # common places scikit stores column names
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        # pipeline: try last step
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    # calibrated classifiers sometimes wrap estimators
    if hasattr(model, "calibrated_classifiers_"):
        try:
            est = model.calibrated_classifiers_[0].estimator
            return _get_feature_names_from_model(est)
        except Exception:
            return None
    return None


def _top_linear_drivers(risk_model: Any, X_row: pd.DataFrame, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Fast “why” explanation for linear models:
    approximate contributions in log-odds space.
    If we can’t introspect the model, return [] (still OK for API).
    """
    try:
        # unwrap calibrated model if needed
        est = risk_model
        if hasattr(risk_model, "calibrated_classifiers_"):
            est = risk_model.calibrated_classifiers_[0].estimator

        scaler = None
        clf = None

        # pipeline?
        if hasattr(est, "named_steps"):
            for name, step in est.named_steps.items():
                if hasattr(step, "transform") and "scaler" in name.lower():
                    scaler = step
                if hasattr(step, "coef_"):
                    clf = step
        else:
            if hasattr(est, "coef_"):
                clf = est

        if clf is None:
            return []

        x = X_row.to_numpy()
        if scaler is not None:
            x = scaler.transform(x)

        coef = np.asarray(clf.coef_).reshape(-1)
        contrib = x.reshape(-1) * coef  # log-odds contribution per feature

        feats = list(X_row.columns)
        pairs = sorted(zip(feats, contrib), key=lambda t: abs(float(t[1])), reverse=True)[:top_k]
        return [{"feature": f, "contribution": float(c)} for f, c in pairs]
    except Exception:
        return []


# -------------------- API Schemas --------------------
class PredictRequest(BaseModel):
    rows: list[dict[str, float | int]] = Field(
        ..., description="Raw CMAPSS-like rows with unit_id, cycle, op_1..op_3, s_1..s_21"
    )
    window: int = Field(30, ge=2, le=200, description="Rolling window used in feature engineering")
    explain: bool = Field(False, description="If true, return quick linear driver explanation")
    top_k: int = Field(10, ge=3, le=30, description="Top drivers count")


class PredictResponse(BaseModel):
    unit_id: int
    cycle: int
    risk_proba: float
    risk_label: int
    risk_threshold: float
    rul_pred: float
    top_drivers: list[dict[str, float]] = Field(default_factory=list)


# -------------------- FastAPI app --------------------
app = FastAPI(title="Predictive Maintenance (Risk + RUL)", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    artifacts = load_artifacts()
    metrics_path = REPORTS_DIR / "metrics.json"
    calibration_path = REPORTS_DIR / "calibration.json"
    out: dict[str, Any] = {
        "risk_threshold": artifacts["risk_threshold"],
        "models": {
            "risk": "risk_model_calibrated.joblib",
            "rul": "rul_model.joblib",
        },
    }
    if metrics_path.exists():
        out["metrics"] = pd.read_json(metrics_path, typ="series").to_dict()
    if calibration_path.exists():
        out["calibration"] = pd.read_json(calibration_path, typ="series").to_dict()
    return out


@app.post("/predict", response_model=list[PredictResponse])
def predict(req: PredictRequest) -> list[PredictResponse]:
    artifacts = load_artifacts()
    risk_model = artifacts["risk_model"]
    rul_model = artifacts["rul_model"]
    thr = float(artifacts["risk_threshold"])

    df = pd.DataFrame(req.rows)
    if df.empty:
        raise HTTPException(status_code=422, detail="rows is empty")

    # validate raw
    try:
        validate_cmapss(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation failed: {e}") from e

    # build features (same as training)
    feat = _build_feature_table(df, window=req.window)

    # we score the LAST cycle per unit (what you want operationally)
    feat = feat.sort_values(["unit_id", "cycle"]).groupby("unit_id", as_index=False).tail(1)

    # choose X columns in the correct order
    feature_names = _get_feature_names_from_model(risk_model)
    drop_cols = {"unit_id", "cycle", "rul", "fail_within_h"}
    if feature_names is None:
        X = feat[[c for c in feat.columns if c not in drop_cols]].copy()
    else:
        missing = [c for c in feature_names if c not in feat.columns]
        if missing:
            raise HTTPException(status_code=500, detail=f"Missing required features: {missing[:10]}")
        X = feat[feature_names].copy()

    # predictions
    risk_proba = risk_model.predict_proba(X)[:, 1]
    risk_label = (risk_proba >= thr).astype(int)

    # RUL regression
    rul_pred = np.asarray(rul_model.predict(X), dtype=float)

    # optional quick explanation
    drivers_all: list[list[dict[str, Any]]] = []
    if req.explain:
        for i in range(len(X)):
            drivers_all.append(_top_linear_drivers(risk_model, X.iloc[[i]], top_k=req.top_k))
    else:
        drivers_all = [[] for _ in range(len(X))]

    out: list[PredictResponse] = []
    for i, row in enumerate(feat.itertuples(index=False)):
        out.append(
            PredictResponse(
                unit_id=int(row.unit_id),
                cycle=int(row.cycle),
                risk_proba=float(risk_proba[i]),
                risk_label=int(risk_label[i]),
                risk_threshold=float(thr),
                rul_pred=float(rul_pred[i]),
                top_drivers=[{"feature": d["feature"], "contribution": float(d["contribution"])} for d in drivers_all[i]],
            )
        )
    return out


@app.post("/batch-predict", response_model=list[PredictResponse])
async def batch_predict(
    window: int = 30,
    explain: bool = False,
    top_k: int = 10,
    file: UploadFile = File(...),
) -> list[PredictResponse]:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Please upload a .csv file")

    content = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(content))
    req = PredictRequest(rows=df.to_dict(orient="records"), window=window, explain=explain, top_k=top_k)
    return predict(req)
