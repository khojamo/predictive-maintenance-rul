from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _project_root() -> Path:
    # repo/src/predictive_maintenance/model_registry.py -> parents[2] == repo/
    return Path(__file__).resolve().parents[2]


def _model_path() -> Path:
    return _project_root() / "models" / "model.joblib"


def ensure_model() -> None:
    """
    Template behavior: if no model exists yet, we create a small deterministic dummy model.
    Later, you'll replace this with your real training pipeline.
    """
    path = _model_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    rng = np.random.default_rng(42)
    X = rng.normal(size=(800, 2))
    # simple rule to create labels (deterministic)
    y = (0.9 * X[:, 0] - 0.6 * X[:, 1] + rng.normal(scale=0.25, size=800) > 0).astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    model.fit(X, y)
    joblib.dump(model, path)


def load_model():
    ensure_model()
    return joblib.load(_model_path())
