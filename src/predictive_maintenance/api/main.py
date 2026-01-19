from __future__ import annotations

import os
import traceback
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request

from predictive_maintenance.predict import predict_one, predict_many
from predictive_maintenance.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    DriftRequest,
    DriftResponse,
)
from predictive_maintenance.monitoring.drift import compute_drift
from predictive_maintenance.monitoring.logging import log_prediction_event
from predictive_maintenance.inference import load_artifacts

app = FastAPI(title="Predictive Maintenance RUL API")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg_path = project_root() / "configs" / "default.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


# ---- DEV-only: return useful error text instead of silent 500 ----
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(tb)  # always print to uvicorn terminal

    if os.getenv("PM_DEBUG", "1") == "1":
        return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": tb})

    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=list[PredictResponse])
def predict(req: PredictRequest) -> list[PredictResponse]:
    request_id = str(uuid4())

    outputs = predict_one(req)  # MUST return list[PredictResponse]

    # Best-effort logging: NEVER break predictions
    try:
        raw_rows = list(req.rows)
        log_prediction_event(
            endpoint="/predict",
            request_id=request_id,
            raw_rows=raw_rows,
            outputs=[o.model_dump() for o in outputs],
        )
    except Exception:
        pass


    return outputs


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(req: BatchPredictRequest) -> BatchPredictResponse:
    request_id = str(uuid4())

    resp = predict_many(req)  # returns BatchPredictResponse(outputs=[...])

    try:
        raw_rows = list(req.rows)
        log_prediction_event(
            endpoint="/batch_predict",
            request_id=request_id,
            raw_rows=raw_rows,
            outputs=[o.model_dump() for o in resp.outputs],
        )
    except Exception:
        pass


    return resp


@app.post("/monitor/drift", response_model=DriftResponse)
def drift(req: DriftRequest) -> DriftResponse:
    cfg = load_config()
    mon = cfg.get("monitoring", {})

    a = load_artifacts()
    window = int(req.window) if req.window is not None else int(a.window)
    min_periods = int(req.min_periods) if req.min_periods is not None else int(a.min_periods)
    psi_threshold = float(req.psi_threshold) if req.psi_threshold is not None else float(mon.get("psi_threshold", 0.20))

    rows_df = pd.DataFrame(list(req.rows))

    try:
        result = compute_drift(
            rows_df,
            window=window,
            min_periods=min_periods,
            top_k=int(req.top_k),
            psi_threshold=psi_threshold,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return DriftResponse(**result)
