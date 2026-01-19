from __future__ import annotations

from pathlib import Path
import yaml
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

from predictive_maintenance.train import make_training_table, project_root


def main() -> None:
    root = project_root()
    cfg = yaml.safe_load((root / "configs" / "default.yaml").read_text(encoding="utf-8"))

    # Use calibrated model for explanations
    model = joblib.load(root / "models" / "risk_model_calibrated.joblib")

    table = make_training_table(cfg)
    feature_cols = [c for c in table.columns if c not in {"unit_id", "cycle", "rul", "fail_within_h", "split"}]

    # Use a small background sample for speed (SHAP needs this)
    tr = table[table["split"] == "train"]
    te = table[table["split"] == "test"]

    X_bg = tr[feature_cols].sample(n=min(300, len(tr)), random_state=42)
    X_te = te[feature_cols].sample(n=min(200, len(te)), random_state=7)

    # Model-agnostic explainer (works for any model)
    # For classifiers, explain probability of class 1 via predict_proba.
    def f(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.Explainer(f, X_bg)
    shap_values = explainer(X_te)

    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Global summary plot
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Summary (Risk Model, sample)")
    plt.savefig(fig_dir / "shap_summary_risk.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Single example (local explanation)
    x_one = X_te.iloc[[0]]
    sv_one = explainer(x_one)

    plt.figure()
    shap.plots.waterfall(sv_one[0], max_display=15, show=False)
    plt.title("SHAP Waterfall (One Prediction)")
    plt.savefig(fig_dir / "shap_waterfall_one.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Save top drivers as text for UI later
    vals = sv_one.values[0]
    names = sv_one.feature_names
    order = np.argsort(np.abs(vals))[::-1][:10]
    top = [(names[i], float(vals[i])) for i in order]

    out_path = root / "reports" / "shap_top_drivers_one.yaml"
    out_path.write_text(yaml.safe_dump({"top_drivers": top}), encoding="utf-8")

    print("Wrote:", fig_dir / "shap_summary_risk.png")
    print("Wrote:", fig_dir / "shap_waterfall_one.png")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
