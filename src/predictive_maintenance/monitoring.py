from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from predictive_maintenance.features import build_rolling_features
from predictive_maintenance.validation import validate_cmapss


RAW_COLS = ["unit_id", "cycle", "op_1", "op_2", "op_3"] + [f"s_{i}" for i in range(1, 22)]


def project_root() -> Path:
    # .../src/predictive_maintenance/monitoring.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_fingerprint() -> dict[str, Any]:
    """
    Stable-ish model identifier for logs:
      - hashes the model files that drive predictions
      - includes threshold + feature config
    """
    root = project_root()
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    window = int(cfg["features"]["window"])
    min_periods = int(cfg["features"]["min_periods"])

    th_path = root / "configs" / "thresholds.yaml"
    th = yaml.safe_load(th_path.read_text(encoding="utf-8")) if th_path.exists() else {"risk_threshold": 0.5}

    risk_path = root / "models" / "risk_model_calibrated.joblib"
    rul_path = root / "models" / "rul_model.joblib"

    return {
        "window": window,
        "min_periods": min_periods,
        "risk_threshold": float(th.get("risk_threshold", 0.5)),
        "risk_model_sha256": _sha256_file(risk_path) if risk_path.exists() else None,
        "rul_model_sha256": _sha256_file(rul_path) if rul_path.exists() else None,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_inputs(raw_df: pd.DataFrame) -> dict[str, Any]:
    u = raw_df["unit_id"].unique()
    return {
        "n_rows": int(len(raw_df)),
        "n_units": int(len(u)),
        "unit_id_min": int(np.min(u)) if len(u) else None,
        "unit_id_max": int(np.max(u)) if len(u) else None,
        "unit_id_sample": [int(x) for x in u[:10].tolist()],
        "cycle_min": int(raw_df["cycle"].min()) if len(raw_df) else None,
        "cycle_max": int(raw_df["cycle"].max()) if len(raw_df) else None,
    }


def log_prediction_event(
    *,
    endpoint: str,
    request_id: str,
    raw_rows: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> None:
    root = project_root()
    log_path = root / "logs" / "predictions.jsonl"

    raw_df = pd.DataFrame(raw_rows)
    raw_df = raw_df[RAW_COLS].copy()

    # keep logs reasonable: store full outputs only up to 50 rows
    outputs_sample = outputs[:50]
    risk_probs = [o.get("risk_proba") for o in outputs if "risk_proba" in o]
    rul_preds = [o.get("rul_pred") for o in outputs if "rul_pred" in o]

    record = {
        "timestamp_utc": utc_now_iso(),
        "request_id": request_id,
        "endpoint": endpoint,
        "model": model_fingerprint(),
        "inputs": summarize_inputs(raw_df),
        "outputs_summary": {
            "n_outputs": int(len(outputs)),
            "risk_proba_mean": float(np.mean(risk_probs)) if len(risk_probs) else None,
            "risk_proba_max": float(np.max(risk_probs)) if len(risk_probs) else None,
            "rul_mean": float(np.mean(rul_preds)) if len(rul_preds) else None,
            "rul_min": float(np.min(rul_preds)) if len(rul_preds) else None,
        },
        "outputs_sample": outputs_sample,
    }
    append_jsonl(log_path, record)


def _ensure_raw(df: pd.DataFrame) -> pd.DataFrame:
    # allow extra columns in CSVs; keep only what schema expects
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")
    return df[RAW_COLS].copy()


def latest_feature_matrix(raw: pd.DataFrame, *, window: int, min_periods: int) -> pd.DataFrame:
    """
    Builds rolling features and returns X for the latest cycle per unit.
    This matches how inference works (one decision row per unit).
    """
    raw = _ensure_raw(raw)
    raw = validate_cmapss(raw)

    feats = build_rolling_features(raw, window=window, min_periods=min_periods)

    latest = (
        feats.sort_values(["unit_id", "cycle"])
        .groupby("unit_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    X = latest.drop(columns=["unit_id", "cycle"])
    return X


@dataclass(frozen=True)
class Baseline:
    meta: dict[str, Any]
    features: dict[str, dict[str, Any]]  # feat -> {"edges": [...], "p": [...]}


def _quantile_edges(x: np.ndarray, n_bins: int) -> list[float] | None:
    x = x.astype(float)
    x = x[~np.isnan(x)]
    if x.size < 50:
        return None

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, qs).astype(float)

    # make strictly increasing edges
    edges = np.unique(edges)
    if edges.size < 3:
        return None

    return edges.tolist()


def _bin_proportions(x: np.ndarray, edges: list[float]) -> np.ndarray:
    x = x.astype(float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.zeros(len(edges) - 1, dtype=float)

    # right=False: bins are [e0,e1), ... last bin includes right edge by cut behavior via include_lowest
    cats = pd.cut(x, bins=np.array(edges, dtype=float), include_lowest=True, right=False)
    counts = cats.value_counts(sort=False).to_numpy(dtype=float)
    p = counts / max(counts.sum(), 1.0)
    return p


def build_baseline_bins(
    X_baseline: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> Baseline:
    feats: dict[str, dict[str, Any]] = {}
    for col in X_baseline.columns:
        edges = _quantile_edges(X_baseline[col].to_numpy(), n_bins=n_bins)
        if edges is None:
            continue
        p = _bin_proportions(X_baseline[col].to_numpy(), edges).tolist()
        feats[col] = {"edges": edges, "p": p}

    meta = {
        "created_utc": utc_now_iso(),
        "n_bins": int(n_bins),
        "n_rows": int(len(X_baseline)),
        "n_features": int(len(feats)),
    }
    return Baseline(meta=meta, features=feats)


def save_baseline(baseline: Baseline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": baseline.meta, "features": baseline.features}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_baseline(path: Path) -> Baseline:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Baseline(meta=payload["meta"], features=payload["features"])


def psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """
    Population Stability Index:
      sum((p-q) * ln(p/q))
    p = baseline proportions, q = recent proportions
    """
    p = np.clip(p.astype(float), eps, 1.0)
    q = np.clip(q.astype(float), eps, 1.0)
    return float(np.sum((p - q) * np.log(p / q)))


def compute_drift(
    X_recent: pd.DataFrame,
    baseline: Baseline,
    *,
    psi_threshold: float = 0.2,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for feat, spec in baseline.features.items():
        if feat not in X_recent.columns:
            continue
        edges = spec["edges"]
        p0 = np.array(spec["p"], dtype=float)
        p1 = _bin_proportions(X_recent[feat].to_numpy(), edges)
        val = psi(p0, p1)

        status = "OK"
        if val >= psi_threshold:
            status = "DRIFT"

        rows.append({"feature": feat, "psi": float(val), "status": status})

    rows.sort(key=lambda r: r["psi"], reverse=True)
    return rows[: int(top_k)]


def write_monitoring_report_md(
    path: Path,
    *,
    baseline: Baseline,
    recent_meta: dict[str, Any],
    drift_rows: list[dict[str, Any]],
    psi_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Monitoring Report")
    lines.append("")
    lines.append(f"- Generated (UTC): **{utc_now_iso()}**")
    lines.append(f"- PSI threshold: **{psi_threshold}**")
    lines.append("")
    lines.append("## Baseline")
    lines.append("")
    lines.append(f"- Created (UTC): {baseline.meta.get('created_utc')}")
    lines.append(f"- Bins: {baseline.meta.get('n_bins')}")
    lines.append(f"- Rows: {baseline.meta.get('n_rows')}")
    lines.append(f"- Features with bins: {baseline.meta.get('n_features')}")
    lines.append("")
    lines.append("## Recent batch")
    lines.append("")
    for k, v in recent_meta.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Drift (Top features by PSI)")
    lines.append("")
    lines.append("| Feature | PSI | Status |")
    lines.append("|---|---:|---|")
    for r in drift_rows:
        lines.append(f"| {r['feature']} | {r['psi']:.4f} | {r['status']} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
