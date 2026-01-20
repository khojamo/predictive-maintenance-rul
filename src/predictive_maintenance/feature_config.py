from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class FeatureConfig:
    id_col: str
    time_col: str
    signal_cols: list[str]
    categorical_cols: list[str]
    categorical_levels: dict[str, list[str]]
    window: int
    min_periods: int
    risk_horizon: int


def load_feature_config(path: Path) -> FeatureConfig | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return FeatureConfig(
        id_col=str(raw["id_col"]),
        time_col=str(raw["time_col"]),
        signal_cols=list(raw["signal_cols"]),
        categorical_cols=list(raw.get("categorical_cols", [])),
        categorical_levels={str(k): [str(vv) for vv in v] for k, v in raw.get("categorical_levels", {}).items()},
        window=int(raw["window"]),
        min_periods=int(raw["min_periods"]),
        risk_horizon=int(raw["risk_horizon"]),
    )


def save_feature_config(cfg: FeatureConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
