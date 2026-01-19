from __future__ import annotations
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    rows: list[dict[str, float | int]]
    window: int | None = Field(None, description="Override rolling window (optional)")
    min_periods: int | None = Field(None, description="Override rolling min_periods (optional)")
    all_cycles: bool = Field(False, description="If true, score every cycle (not just latest).")
    clamp_rul: bool = Field(True, description="Clamp negative RUL predictions to 0.")


class PredictResponse(BaseModel):
    unit_id: int
    cycle: int
    risk_proba: float = Field(..., ge=0.0, le=1.0)
    risk_label: int = Field(..., description="1=alert, 0=ok")
    risk_threshold: float
    rul_pred: float


class BatchPredictRequest(PredictRequest):
    pass


class BatchPredictResponse(BaseModel):
    outputs: list[PredictResponse]


# ---- Monitoring (Package 9) ----
class DriftRequest(PredictRequest):
    psi_threshold: float = Field(0.20, description="Flag drift if PSI >= this")
    top_k: int = Field(20, description="Return top K features by PSI")


class DriftFeature(BaseModel):
    feature: str
    psi: float
    flag: bool
    baseline_mean: float | None = None
    baseline_std: float | None = None
    recent_mean: float | None = None
    recent_std: float | None = None


class DriftResponse(BaseModel):
    created_at_utc: str
    baseline_created_at_utc: str
    dataset: str
    psi_threshold: float
    n_features: int
    n_flagged: int
    top: list[DriftFeature]
