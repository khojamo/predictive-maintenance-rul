from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st
import yaml

from predictive_maintenance.calibrate import calibrate_model
from predictive_maintenance.feature_catalog import (
    CANONICAL_FEATURES,
    ID_COL,
    TIME_COL,
    LABEL_FAIL_WITHIN_H,
    LABEL_FAILURE_CYCLE,
    LABEL_RUL,
)
from predictive_maintenance.inference import load_artifacts, score_latest_cycles
from predictive_maintenance.monitoring.drift import compute_drift
from predictive_maintenance.train import (
    build_and_save_baseline_from_table,
    make_training_table_from_df,
    save_trained_artifacts,
    train_models_from_table,
)


ROOT = Path(__file__).resolve().parents[3]


def _load_cfg() -> dict:
    cfg_path = ROOT / "configs" / "default.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _auto_map_features(columns: list[str]) -> dict[str, str]:
    norm_cols = {_normalize_name(c): c for c in columns}
    mapping: dict[str, str] = {}
    for feat in CANONICAL_FEATURES:
        key = _normalize_name(feat)
        if key in norm_cols:
            mapping[feat] = norm_cols[key]
    return mapping


def _guess_column(columns: list[str], candidates: list[str]) -> str:
    norm_cols = {_normalize_name(c): c for c in columns}
    for cand in candidates:
        key = _normalize_name(cand)
        if key in norm_cols:
            return norm_cols[key]
    return columns[0] if columns else ""


def _read_upload(file, sheet_name: str | None = None) -> pd.DataFrame:
    file.seek(0)
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file, sheet_name=sheet_name)
    raise ValueError("Unsupported file type. Use CSV or Excel.")


def _coerce_unit_id(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        codes, _ = pd.factorize(series.astype(str))
        return pd.Series(codes + 1, index=series.index)
    return s.astype(int)


def _prepare_df(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    feature_map: dict[str, str],
    rul_col: str | None,
    fail_col: str | None,
    failure_cycle_col: str | None,
    time_is_timestamp: bool,
    missing_strategy: str,
) -> tuple[pd.DataFrame, list[str]]:
    rename_map = {src: dst for dst, src in feature_map.items()}
    rename_map[id_col] = ID_COL
    rename_map[time_col] = TIME_COL
    if rul_col:
        rename_map[rul_col] = LABEL_RUL
    if fail_col:
        rename_map[fail_col] = LABEL_FAIL_WITHIN_H
    if failure_cycle_col:
        rename_map[failure_cycle_col] = LABEL_FAILURE_CYCLE

    out = df.rename(columns=rename_map).copy()

    out[ID_COL] = _coerce_unit_id(out[ID_COL])

    if time_is_timestamp:
        out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
        if out[TIME_COL].isna().any():
            raise ValueError("Timestamp column has invalid values.")
        out = out.sort_values([ID_COL, TIME_COL])
        out[TIME_COL] = out.groupby(ID_COL).cumcount().astype(int) + 1
    else:
        out[TIME_COL] = pd.to_numeric(out[TIME_COL], errors="coerce")
        if out[TIME_COL].isna().any():
            raise ValueError("Time/cycle column has non-numeric values.")
        out[TIME_COL] = out[TIME_COL].astype(int)

    feature_cols = list(feature_map.keys())
    keep_cols = [ID_COL, TIME_COL, *feature_cols]
    for lbl in [LABEL_RUL, LABEL_FAIL_WITHIN_H, LABEL_FAILURE_CYCLE]:
        if lbl in out.columns:
            keep_cols.append(lbl)
    out = out[keep_cols]

    if missing_strategy == "drop_rows":
        out = out.dropna()
    elif missing_strategy == "forward_fill":
        out = out.sort_values([ID_COL, TIME_COL])
        out[feature_cols] = out.groupby(ID_COL)[feature_cols].ffill()
        out[feature_cols] = out[feature_cols].fillna(out[feature_cols].median(numeric_only=True))
    elif missing_strategy == "median":
        out[feature_cols] = out[feature_cols].fillna(out[feature_cols].median(numeric_only=True))

    return out, feature_cols


st.set_page_config(page_title="Predictive Maintenance Studio", layout="wide")
st.title("Predictive Maintenance Studio")

cfg = _load_cfg()

tab_train, tab_score, tab_monitor = st.tabs(["1) Upload & Train", "2) Score", "3) Monitoring"])


with tab_train:
    st.subheader("Upload dataset")
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if file is None:
        st.info("Upload a dataset to begin.")
    else:
        if file.name.lower().endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(file)
            sheet = st.selectbox("Sheet", options=xl.sheet_names)
            df = _read_upload(file, sheet_name=sheet)
        else:
            df = _read_upload(file)

        st.write(f"Rows: {len(df):,}  Columns: {len(df.columns)}")
        st.dataframe(df.head(20), use_container_width=True)

        cols = df.columns.tolist()

        st.subheader("Identify key columns")
        id_guess = _guess_column(cols, ["unit_id", "asset_id", "equipment_id", "id"])
        time_guess = _guess_column(cols, ["cycle", "cycle_index", "timestamp", "time"])
        id_col = st.selectbox("Asset ID column", options=cols, index=cols.index(id_guess))
        time_col = st.selectbox("Time/Cycle column", options=cols, index=cols.index(time_guess))
        time_is_timestamp = st.checkbox("Time column is a timestamp", value=False)

        st.subheader("Label columns (optional)")
        rul_col = st.selectbox("RUL column", options=["(none)"] + cols, index=0)
        fail_col = st.selectbox("Fail-within-h column", options=["(none)"] + cols, index=0)
        failure_cycle_col = st.selectbox("Failure cycle column", options=["(none)"] + cols, index=0)

        rul_col = None if rul_col == "(none)" else rul_col
        fail_col = None if fail_col == "(none)" else fail_col
        failure_cycle_col = None if failure_cycle_col == "(none)" else failure_cycle_col

        st.subheader("Map features")
        auto_map = _auto_map_features(cols)
        mapping_df = pd.DataFrame(
            {
                "canonical_feature": CANONICAL_FEATURES,
                "source_column": [auto_map.get(f, "") for f in CANONICAL_FEATURES],
            }
        )
        mapping_df = st.data_editor(
            mapping_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "source_column": st.column_config.SelectboxColumn(
                    "Your column", options=[""] + cols
                )
            },
        )

        feature_map = {
            row["canonical_feature"]: row["source_column"]
            for _, row in mapping_df.iterrows()
            if row["source_column"]
        }

        dup_sources = (
            pd.Series(feature_map.values()).value_counts().loc[lambda x: x > 1].index.tolist()
        )
        if dup_sources:
            st.error(f"Duplicate mappings detected: {dup_sources}")

        st.subheader("Training options")
        dataset_name = st.text_input("Dataset name", value="custom")
        horizon = st.number_input("Risk horizon (cycles)", min_value=1, max_value=500, value=int(cfg["risk_horizon"]))
        window = st.number_input("Rolling window", min_value=1, max_value=500, value=int(cfg["features"]["window"]))
        min_periods = st.number_input(
            "Min periods", min_value=1, max_value=200, value=int(cfg["features"]["min_periods"])
        )
        missing_strategy = st.selectbox(
            "Missing data handling",
            options=[
                "drop_rows",
                "forward_fill",
                "median",
            ],
        )

        can_train = bool(feature_map) and not dup_sources
        if not can_train:
            st.warning("Map at least one feature and fix duplicate mappings to enable training.")

        if st.button("Train model", type="primary", disabled=not can_train):
            with st.spinner("Preparing data and training models..."):
                cfg["risk_horizon"] = int(horizon)
                cfg["features"]["window"] = int(window)
                cfg["features"]["min_periods"] = int(min_periods)

                prepared, signal_cols = _prepare_df(
                    df,
                    id_col=id_col,
                    time_col=time_col,
                    feature_map=feature_map,
                    rul_col=rul_col,
                    fail_col=fail_col,
                    failure_cycle_col=failure_cycle_col,
                    time_is_timestamp=time_is_timestamp,
                    missing_strategy=missing_strategy,
                )

                table = make_training_table_from_df(prepared, cfg, signal_cols=signal_cols)
                metrics, risk_model, rul_model, _, calib_data = train_models_from_table(table)

                X_tr, y_tr, X_va, y_va = calib_data
                cal_model, best_t, best_f1 = calibrate_model(
                    risk_model=risk_model,
                    X_train=X_tr,
                    y_train=y_tr,
                    X_valid=X_va,
                    y_valid=y_va,
                    out_dir=ROOT / "reports",
                )

                threshold_cfg = {
                    "risk_threshold": float(best_t),
                    "threshold_selection": "max_f1_on_validation",
                    "calibration": {"method": "isotonic", "cv": 3},
                }
                calibration_report = {"best_threshold": float(best_t), "best_f1_valid": float(best_f1)}

                save_trained_artifacts(
                    cfg,
                    signal_cols=signal_cols,
                    risk_model=risk_model,
                    rul_model=rul_model,
                    metrics=metrics,
                    calibrated_risk_model=cal_model,
                    threshold_cfg=threshold_cfg,
                    calibration_report=calibration_report,
                )

                build_and_save_baseline_from_table(table, cfg=cfg, dataset_name=dataset_name)

                mapping_path = ROOT / "models" / "feature_mapping.json"
                mapping_payload = {
                    "id_col": id_col,
                    "time_col": time_col,
                    "time_is_timestamp": bool(time_is_timestamp),
                    "feature_map": feature_map,
                    "label_map": {
                        LABEL_RUL: rul_col,
                        LABEL_FAIL_WITHIN_H: fail_col,
                        LABEL_FAILURE_CYCLE: failure_cycle_col,
                    },
                }
                mapping_path.write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")

                load_artifacts.cache_clear()
                st.success("Training complete. Artifacts saved to models/ and reports/.")
                st.json(metrics)

                st.session_state["last_mapping"] = mapping_payload
                st.session_state["signal_cols"] = signal_cols


with tab_score:
    st.subheader("Score new data")
    models_ready = (ROOT / "models" / "risk_model_calibrated.joblib").exists() and (
        ROOT / "models" / "rul_model.joblib"
    ).exists()
    if not models_ready:
        st.warning("No trained models found. Train a model in the first tab before scoring.")
    score_file = st.file_uploader("Upload CSV or Excel for scoring", type=["csv", "xlsx", "xls"], key="score_file")
    mapping = st.session_state.get("last_mapping")
    use_saved_mapping = False
    if mapping:
        use_saved_mapping = st.checkbox("Use saved mapping from last training", value=True)

    if score_file is not None:
        if score_file.name.lower().endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(score_file)
            sheet = st.selectbox("Sheet (scoring)", options=xl.sheet_names, key="score_sheet")
            score_df = _read_upload(score_file, sheet_name=sheet)
        else:
            score_df = _read_upload(score_file)

        st.dataframe(score_df.head(20), use_container_width=True)

        if mapping and use_saved_mapping:
            id_col = mapping["id_col"]
            time_col = mapping["time_col"]
            feature_map = mapping["feature_map"]
            time_is_timestamp = bool(mapping.get("time_is_timestamp", False))
            rul_col = None
            fail_col = None
            failure_cycle_col = None
        else:
            cols = score_df.columns.tolist()
            id_col = st.selectbox("Asset ID column (scoring)", options=cols, index=0, key="score_id")
            time_col = st.selectbox("Time/Cycle column (scoring)", options=cols, index=1, key="score_time")
            time_is_timestamp = st.checkbox("Time column is a timestamp (scoring)", value=False)

            auto_map = _auto_map_features(cols)
            mapping_df = pd.DataFrame(
                {
                    "canonical_feature": CANONICAL_FEATURES,
                    "source_column": [auto_map.get(f, "") for f in CANONICAL_FEATURES],
                }
            )
            mapping_df = st.data_editor(
                mapping_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "source_column": st.column_config.SelectboxColumn(
                        "Your column", options=[""] + cols
                    )
                },
                key="score_map",
            )
            feature_map = {
                row["canonical_feature"]: row["source_column"]
                for _, row in mapping_df.iterrows()
                if row["source_column"]
            }
            rul_col = None
            fail_col = None
            failure_cycle_col = None

        score_all = st.checkbox("Score all cycles (not just latest)", value=False, key="score_all")

        if st.button("Score data", type="primary", disabled=not models_ready):
            prepared, _ = _prepare_df(
                score_df,
                id_col=id_col,
                time_col=time_col,
                feature_map=feature_map,
                rul_col=rul_col,
                fail_col=fail_col,
                failure_cycle_col=failure_cycle_col,
                time_is_timestamp=time_is_timestamp,
                missing_strategy="median",
            )
            scored = score_latest_cycles(prepared, all_cycles=score_all, clamp_rul=True)

            artifacts = load_artifacts()
            id_col_out = artifacts.id_col
            time_col_out = artifacts.time_col

            n_rows = len(scored)
            n_units = scored[id_col_out].nunique() if id_col_out in scored else 0
            alert_rate = float(scored["risk_label"].mean()) if n_rows else 0.0
            median_rul = float(scored["rul_pred"].median()) if n_rows else 0.0
            min_rul = float(scored["rul_pred"].min()) if n_rows else 0.0

            st.success(f"Scored {n_rows} rows across {n_units} units.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Alert rate", f"{alert_rate:.1%}")
            c2.metric("Median RUL", f"{median_rul:.1f}")
            c3.metric("Min RUL", f"{min_rul:.1f}")

            if score_all:
                st.caption("Scoring mode: all cycles (one row per unit per cycle).")

            alerts = scored[scored["risk_label"] == 1]
            if len(alerts) == 0:
                st.success("No alerts: all rows are below the risk threshold.")
            else:
                st.warning(f"{len(alerts)} alerts flagged. Review high-risk units first.")
                by_unit = (
                    alerts.groupby(id_col_out, as_index=False)
                    .agg({"risk_proba": "max", "rul_pred": "min"})
                    .sort_values(["risk_proba", "rul_pred"], ascending=[False, True])
                    .head(10)
                )
                st.caption("Top 10 units by risk (max risk_proba, min RUL):")
                st.dataframe(by_unit, use_container_width=True)

            st.markdown(
                "- `risk_proba` = probability of failure within the horizon.\n"
                "- `risk_label` = alert if above threshold.\n"
                "- `rul_pred` = estimated remaining useful life (clamped to >= 0).\n"
            )

            st.dataframe(scored.head(50), use_container_width=True)

            st.download_button(
                "Download scores CSV",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="scores.csv",
                mime="text/csv",
            )


with tab_monitor:
    st.subheader("Drift monitoring (PSI)")
    baseline_ready = (ROOT / "reports" / "baseline_bins.json").exists()
    if not baseline_ready:
        st.warning("No baseline bins found. Train a model to generate drift baselines.")
    mon_file = st.file_uploader(
        "Upload recent batch (CSV or Excel)", type=["csv", "xlsx", "xls"], key="mon_file"
    )
    psi_threshold = st.slider("PSI threshold", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    top_k = st.number_input("Top K features", min_value=5, max_value=50, value=20, step=5)

    if mon_file is not None:
        if mon_file.name.lower().endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(mon_file)
            sheet = st.selectbox("Sheet (monitoring)", options=xl.sheet_names, key="mon_sheet")
            mon_df = _read_upload(mon_file, sheet_name=sheet)
        else:
            mon_df = _read_upload(mon_file)

        st.dataframe(mon_df.head(20), use_container_width=True)

        if st.button("Run drift check", type="primary", disabled=not baseline_ready):
            mapping = st.session_state.get("last_mapping")
            if mapping:
                prepared, _ = _prepare_df(
                    mon_df,
                    id_col=mapping["id_col"],
                    time_col=mapping["time_col"],
                    feature_map=mapping["feature_map"],
                    rul_col=None,
                    fail_col=None,
                    failure_cycle_col=None,
                    time_is_timestamp=bool(mapping.get("time_is_timestamp", False)),
                    missing_strategy="median",
                )
            else:
                prepared = mon_df

            artifacts = load_artifacts()
            out = compute_drift(
                prepared,
                window=artifacts.window,
                min_periods=artifacts.min_periods,
                top_k=int(top_k),
                psi_threshold=float(psi_threshold),
            )
            st.success(f"Done. Flagged {out['n_flagged']} / {out['n_features']} features.")
            st.dataframe(pd.DataFrame(out["top"]), use_container_width=True)
