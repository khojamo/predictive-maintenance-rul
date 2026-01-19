from __future__ import annotations

from pathlib import Path
import json
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

from predictive_maintenance.train import make_training_table, project_root


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    root = project_root()
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))

    # load trained models
    risk_model = joblib.load(root / "models" / "risk_model.joblib")
    rul_model = joblib.load(root / "models" / "rul_model.joblib")

    table = make_training_table(cfg)
    feature_cols = [c for c in table.columns if c not in {"unit_id", "cycle", "rul", "fail_within_h", "split"}]

    te = table[table["split"] == "test"]
    X_te = te[feature_cols]
    y_te_cls = te["fail_within_h"].to_numpy()
    y_te_rul = te["rul"].to_numpy()

    # predictions
    proba = risk_model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rul_pred = rul_model.predict(X_te)

    fig_dir = root / "reports" / "figures"
    _ensure_dir(fig_dir)

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_te_cls, proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Risk Model ROC (Test)")
    plt.savefig(fig_dir / "risk_roc.png", dpi=160, bbox_inches="tight")
    plt.close()

    # --- Precision-Recall curve ---
    prec, rec, _ = precision_recall_curve(y_te_cls, proba)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Risk Model Precision–Recall (Test)")
    plt.savefig(fig_dir / "risk_pr.png", dpi=160, bbox_inches="tight")
    plt.close()

    # --- Confusion matrix ---
    cm = confusion_matrix(y_te_cls, pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Risk Confusion Matrix (Test, threshold=0.5)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.savefig(fig_dir / "risk_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close()

    # --- RUL: predicted vs true ---
    plt.figure()
    plt.scatter(y_te_rul, rul_pred, s=6)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("RUL Predicted vs True (Test)")
    plt.savefig(fig_dir / "rul_pred_vs_true.png", dpi=160, bbox_inches="tight")
    plt.close()

    # --- RUL residuals ---
    resid = rul_pred - y_te_rul
    plt.figure()
    plt.hist(resid, bins=50)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title("RUL Residuals (Test)")
    plt.savefig(fig_dir / "rul_residuals_hist.png", dpi=160, bbox_inches="tight")
    plt.close()

    # --- Error by RUL bins ---
    bins = [0, 20, 50, 100, 150, 200, 400]
    cats = pd.cut(y_te_rul, bins=bins, include_lowest=True)
    mae_by_bin = (
        pd.DataFrame({"bin": cats, "abs_err": np.abs(resid)})
        .groupby("bin", observed=False)["abs_err"]
        .mean()
    )

    plt.figure()
    plt.bar(range(len(mae_by_bin)), mae_by_bin.values)
    plt.xticks(range(len(mae_by_bin)), [str(b) for b in mae_by_bin.index], rotation=45, ha="right")
    plt.ylabel("Mean Absolute Error")
    plt.title("RUL MAE by True RUL Bin (Test)")
    plt.savefig(fig_dir / "rul_mae_by_bin.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("Wrote figures to:", fig_dir)


if __name__ == "__main__":
    main()
