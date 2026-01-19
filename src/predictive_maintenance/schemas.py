from __future__ import annotations
from pydantic import BaseModel, Field


class CMAPSSRow(BaseModel):
    unit_id: int
    cycle: int
    op_1: float
    op_2: float
    op_3: float
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float


class PredictRequest(BaseModel):
    rows: list[CMAPSSRow]
    window: int | None = Field(None, description="Override rolling window (optional)")
    min_periods: int | None = Field(None, description="Override rolling min_periods (optional)")


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
