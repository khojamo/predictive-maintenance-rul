from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _to_py(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return [_to_py(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    return x


def log_prediction_event(*, endpoint: str, request_id: str, raw_rows: list[dict], outputs: list[dict]) -> None:
    root = _project_root()
    path = root / "logs" / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    unit_ids = [r.get("unit_id") for r in raw_rows if "unit_id" in r]
    cycles = [r.get("cycle") for r in raw_rows if "cycle" in r]
    risk_probs = [o.get("risk_proba") for o in outputs if "risk_proba" in o]
    rul_preds = [o.get("rul_pred") for o in outputs if "rul_pred" in o]

    rec = {
        "timestamp_utc": _utc_now(),
        "endpoint": endpoint,
        "request_id": request_id,
        "inputs": {
            "n_rows": len(raw_rows),
            "n_units": len(set(unit_ids)) if unit_ids else 0,
            "cycle_min": min(cycles) if cycles else None,
            "cycle_max": max(cycles) if cycles else None,
        },
        "outputs": {
            "n": len(outputs),
            "risk_mean": float(np.mean(risk_probs)) if risk_probs else None,
            "risk_max": float(np.max(risk_probs)) if risk_probs else None,
            "rul_mean": float(np.mean(rul_preds)) if rul_preds else None,
            "rul_min": float(np.min(rul_preds)) if rul_preds else None,
        },
        "outputs_sample": outputs[:20],
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_py(rec), ensure_ascii=False) + "\n")
