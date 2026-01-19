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


def main() -> None:
    root = project_root()
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))

    table = make_training_table(cfg)
    feature_cols = [c for c in table.columns if c not in {"unit_id", "cycle", "rul", "fail_within_h", "split"}]

    tr = table[table["split"] == "train"]
    va = table[table["split"] == "valid"]

    X_tr, y_tr = tr[feature_cols], tr["fail_within_h"].to_numpy()
    X_va, y_va = va[feature_cols], va["fail_within_h"].to_numpy()

    # Load uncalibrated baseline risk model
    base = joblib.load(root / "models" / "risk_model.joblib")

    # Calibrate on validation-like behavior using CV on train split
    # (We calibrate the already-trained pipeline by wrapping it.)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X_tr, y_tr)

    # Evaluate calibration curve on validation set
    p_va = cal.predict_proba(X_va)[:, 1]

    prob_true, prob_pred = calibration_curve(y_va, p_va, n_bins=10, strategy="uniform")
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title("Calibration Curve (Validation)")
    plt.savefig(fig_dir / "risk_calibration_curve.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Threshold sweep (choose threshold maximizing F1 on validation for now)
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (p_va >= t).astype(int)
        f1 = f1_score(y_va, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    # Save calibrated model
    out_path = root / "models" / "risk_model_calibrated.joblib"
    joblib.dump(cal, out_path)

    # Save threshold config
    th_cfg = {
        "risk_threshold": best_t,
        "threshold_selection": "max_f1_on_validation",
        "calibration": {"method": "isotonic", "cv": 3},
    }
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "thresholds.yaml").write_text(yaml.safe_dump(th_cfg), encoding="utf-8")

    # Save small report json
    rep = {
        "best_threshold": best_t,
        "best_f1_valid": float(best_f1),
    }
    (root / "reports" / "calibration.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("Saved calibrated model:", out_path)
    print("Saved thresholds:", root / "configs" / "thresholds.yaml")
    print("Saved calibration report:", root / "reports" / "calibration.json")
    print("Wrote figure:", fig_dir / "risk_calibration_curve.png")


if __name__ == "__main__":
    main()
