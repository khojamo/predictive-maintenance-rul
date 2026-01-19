import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ML Studio • Predictive Maintenance", layout="wide")
st.title("ML Studio • Predictive Maintenance (Risk + RUL)")

api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")
window = st.sidebar.number_input("window (optional)", value=0, min_value=0, step=1)
min_periods = st.sidebar.number_input("min_periods (optional)", value=0, min_value=0, step=1)

RAW_COLS = ["unit_id", "cycle", "op_1", "op_2", "op_3"] + [f"s_{i}" for i in range(1, 22)]

def _payload(df: pd.DataFrame) -> dict:
    df = df[RAW_COLS].copy()
    payload = {"rows": df.to_dict(orient="records")}
    if window > 0:
        payload["window"] = int(window)
    if min_periods > 0:
        payload["min_periods"] = int(min_periods)
    return payload

def _post(path: str, payload: dict):
    r = requests.post(f"{api_url}{path}", json=payload, timeout=60)
    if not r.ok:
        st.error(f"API error {r.status_code}: {r.text}")
    r.raise_for_status()
    return r.json()


tab1, tab2, tab3 = st.tabs(["Demo (score one unit)", "Batch CSV scoring", "Monitoring (drift)"])

with tab1:
    st.subheader("Score one unit (latest cycle)")
    sample_path = "data/derived/train_labeled_sample.csv"

    try:
        df = pd.read_csv(sample_path)
        unit = st.selectbox("unit_id", sorted(df["unit_id"].unique().tolist()))
        n_last = st.slider("Last N cycles to send", min_value=5, max_value=200, value=60, step=5)

        unit_df = df[df["unit_id"] == unit].sort_values("cycle").tail(n_last)
        st.caption("Preview (raw rows to be sent):")
        st.dataframe(unit_df[RAW_COLS].head(20), use_container_width=True)

        if st.button("Score unit", type="primary"):
            out = _post("/predict", _payload(unit_df))
            st.success("Scored ✅")
            st.json(out)

            if isinstance(out, list) and len(out) > 0:
                o = out[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Risk probability", f"{o['risk_proba']:.3f}")
                c2.metric("Risk label", "ALERT" if o["risk_label"] == 1 else "OK")
                c3.metric("RUL (cycles)", f"{o['rul_pred']:.1f}")

    except FileNotFoundError:
        st.warning("Sample not found: data/derived/train_labeled_sample.csv")

with tab2:
    st.subheader("Batch scoring (CSV upload)")
    st.write("CSV must include: unit_id, cycle, op_1..op_3, s_1..s_21")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head(25), use_container_width=True)

        if st.button("Score CSV", type="primary"):
            out = _post("/batch_predict", _payload(df))  # returns {"outputs":[...]}
            outputs = out["outputs"]
            out_df = pd.DataFrame(outputs)

            st.success(f"Scored ✅ ({len(out_df)} units)")
            st.dataframe(out_df.head(50), use_container_width=True)

            st.download_button(
                "Download results CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_units.csv",
                mime="text/csv",
            )


with tab3:
    st.subheader("Monitoring — drift check (PSI)")
    st.write("Upload a *recent* raw batch (unit_id, cycle, op_1..op_3, s_1..s_21). The app computes rolling features and compares them to the training baseline bins.")

    psi_threshold = st.slider("PSI threshold (flag)", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    top_k = st.number_input("Top K features to display", min_value=5, max_value=50, value=20, step=5)

    mfile = st.file_uploader("Upload recent batch CSV", type=["csv"], key="monitor_file")
    if mfile is not None:
        dfm = pd.read_csv(mfile)
        st.dataframe(dfm.head(25), use_container_width=True)

        if st.button("Run drift check", type="primary"):
            payload = _payload(dfm)
            payload["top_k"] = int(top_k)
            payload["psi_threshold"] = float(psi_threshold)
            out = _post("/monitor/drift", payload)

            st.success(f"Done ✅  Flagged: {out['n_flagged']} / {out['n_features']}")
            top = pd.DataFrame(out["top"])
            st.dataframe(top, use_container_width=True)

            st.download_button(
                "Download drift report CSV",
                data=top.to_csv(index=False).encode("utf-8"),
                file_name="drift_top_features.csv",
                mime="text/csv",
            )
