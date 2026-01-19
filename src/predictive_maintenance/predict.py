from __future__ import annotations

import pandas as pd

from predictive_maintenance.inference import score_latest_cycles
from predictive_maintenance.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)


def _score(req_rows: list[dict], window: int | None, min_periods: int | None) -> list[PredictResponse]:
    df = pd.DataFrame(req_rows)
    scored = score_latest_cycles(df, window=window, min_periods=min_periods)
    return [PredictResponse(**row) for row in scored.to_dict(orient="records")]


def predict_one(req: PredictRequest) -> list[PredictResponse]:
    rows = [r.model_dump() for r in req.rows]
    return _score(rows, req.window, req.min_periods)


def predict_many(req: BatchPredictRequest) -> BatchPredictResponse:
    rows = [r.model_dump() for r in req.rows]
    preds = _score(rows, req.window, req.min_periods)
    return BatchPredictResponse(outputs=preds)
