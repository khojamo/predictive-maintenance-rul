from __future__ import annotations

from pathlib import Path
import json
import yaml

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import f1_score

from predictive_maintenance.train import make_training_table, project_root


def calibrate_model(
    *,
    risk_model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    out_dir: Path,
    cv: int = 3,
) -> tuple[CalibratedClassifierCV, float, float]:
    """
    Calibrate a risk model and return (calibrated_model, best_threshold, best_f1).
    """
    cal = CalibratedClassifierCV(risk_model, method="isotonic", cv=cv)
    cal.fit(X_train, y_train)

    p_va = cal.predict_proba(X_valid)[:, 1]
    prob_true, prob_pred = calibration_curve(y_valid, p_va, n_bins=10, strategy="uniform")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration Curve (Validation)")
    plt.savefig(fig_dir / "risk_calibration_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (p_va >= t).astype(int)
        f1 = f1_score(y_valid, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return cal, best_t, float(best_f1)


def main() -> None:
    root = project_root()
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))

    table = make_training_table(cfg)
    feature_cols = [c for c in table.columns if c not in {"unit_id", "cycle", "rul", "fail_within_h", "split"}]

    tr = table[table["split"] == "train"]
    va = table[table["split"] == "valid"]

    X_tr, y_tr = tr[feature_cols], tr["fail_within_h"].to_numpy()
    X_va, y_va = va[feature_cols], va["fail_within_h"].to_numpy()

    base = joblib.load(root / "models" / "risk_model.joblib")

    cal, best_t, best_f1 = calibrate_model(
        risk_model=base,
        X_train=X_tr,
        y_train=y_tr,
        X_valid=X_va,
        y_valid=y_va,
        out_dir=root / "reports",
    )

    out_path = root / "models" / "risk_model_calibrated.joblib"
    joblib.dump(cal, out_path)

    th_cfg = {
        "risk_threshold": best_t,
        "threshold_selection": "max_f1_on_validation",
        "calibration": {"method": "isotonic", "cv": 3},
    }
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "thresholds.yaml").write_text(yaml.safe_dump(th_cfg), encoding="utf-8")

    rep = {
        "best_threshold": best_t,
        "best_f1_valid": float(best_f1),
    }
    (root / "reports" / "calibration.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("Saved calibrated model:", out_path)
    print("Saved thresholds:", root / "configs" / "thresholds.yaml")
    print("Saved calibration report:", root / "reports" / "calibration.json")
    print("Wrote figure:", (root / "reports" / "figures" / "risk_calibration_curve.png"))


if __name__ == "__main__":
    main()
