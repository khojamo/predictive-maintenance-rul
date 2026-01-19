from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from predictive_maintenance.monitoring.drift import compute_drift


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path, max_lines: int = 2000) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-max_lines:]


def main() -> None:
    root = project_root()
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    fcfg = cfg["features"]
    window = int(fcfg["window"])
    minp = int(fcfg["min_periods"])

    # Use a default recent batch if available
    default_batch = root / "data" / "derived" / "train_labeled_sample.csv"
    batch_path = default_batch if default_batch.exists() else None

    drift_section = "(no baseline bins found; run scripts_make_baseline_bins.py)"
    if batch_path is not None:
        raw = pd.read_csv(batch_path)
        # keep only required raw columns
        keep = ["unit_id", "cycle", "op_1", "op_2", "op_3"] + [f"s_{i}" for i in range(1, 22)]
        raw = raw[[c for c in keep if c in raw.columns]].copy()

        try:
            drift = compute_drift(raw, window=window, min_periods=minp, top_k=15)
            top = pd.DataFrame(drift["top"])
            drift_section = top[["feature", "psi", "flag", "baseline_mean", "recent_mean"]].to_markdown(index=False)
        except FileNotFoundError:
            pass

    # Prediction log summary
    log_path = root / "logs" / "predictions.jsonl"
    events = _read_jsonl(log_path)

    log_section = "(no prediction logs yet)"
    if events:
        df = pd.DataFrame(
            {
                "timestamp_utc": [e.get("timestamp_utc") or e.get("ts_utc") for e in events],
                "endpoint": [e.get("endpoint") for e in events],
                "n_rows": [e.get("inputs", {}).get("n_rows") for e in events],
                "n_units": [e.get("inputs", {}).get("n_units") for e in events],
            }
        )
        log_section = df.tail(20).to_markdown(index=False)

    out = root / "reports" / "monitoring_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        "\n".join(
            [
                "# Monitoring Report (Drift + Prediction Logs)",
                "",
                f"Generated: {_utc_now_iso()}",
                "",
                "## Drift (PSI) — baseline vs recent batch",
                "",
                drift_section,
                "",
                "## Recent prediction events (tail)",
                "",
                log_section,
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
