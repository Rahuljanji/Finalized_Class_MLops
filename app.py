"""Streamlit serving application for the Iris MLOps Pipeline.

Features:
- Live model inference with class probability visualization.
- Production payload and prediction logging for data drift tracking.
- Interactive model metadata and lineage inspector.
- Real-time data drift monitoring dashboard.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from src.monitor_drift import (
    INFERENCE_LOG_PATH,
    REFERENCE_DATA_PATH,
    analyze_drift,
    load_inference_logs,
    log_inference,
)

MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"
SELECTION_PATH = Path(__file__).parent / "reports" / "model_selection.json"
METRICS_PATH = Path(__file__).parent / "reports" / "metrics.json"

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

st.set_page_config(
    page_title="Iris MLOps Production Portal",
    page_icon="🌸",
    layout="wide",
)


@st.cache_resource
def load_champion_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reports():
    selection, metrics = None, None
    if SELECTION_PATH.exists():
        with open(SELECTION_PATH, "r", encoding="utf-8") as f:
            selection = json.load(f)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return selection, metrics


def pretty(label: str) -> str:
    """'Iris-setosa' -> 'Setosa'"""
    return str(label).replace("Iris-", "").title()


# -----------------------------------------------------------------------------
# Sidebar: MLOps Model Lineage & System Info
# -----------------------------------------------------------------------------
model = load_champion_model()
selection, metrics = load_reports()

with st.sidebar:
    st.title("🌸 MLOps Platform")
    st.caption("Automated Champion Model Serving")

    if selection and "champion" in selection:
        champ = selection["champion"]
        crit = champ.get("criteria", {})
        st.subheader("🏆 Active Champion")
        st.info(f"**Model:** {champ.get('display_name', champ.get('model_name'))}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Val Acc", f"{crit.get('validation_accuracy_pct', 0.0):.1f}%")
            st.metric("Latency", f"{crit.get('prediction_time_ms', 0.0):.2f} ms")
        with col_b:
            if metrics:
                st.metric("Test Acc", f"{metrics.get('accuracy', 0.0)*100:.1f}%")
            st.metric("Size", f"{crit.get('model_size_mb', 0.0):.3f} MB")

        with st.expander("Selection Rationale"):
            st.write(champ.get("explanation", "Selected via multi-criteria rules."))
    else:
        st.warning("Model metadata not found. Run `dvc repro` to build.")

    st.divider()
    st.subheader("📡 Live Monitoring")
    log_df = load_inference_logs(INFERENCE_LOG_PATH)
    st.metric("Logged Inferences", len(log_df))

    st.caption("Powered by DVC, MLflow, Optuna & Streamlit")

# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_metrics, tab_monitoring = st.tabs([
    "🌸 Live Inference",
    "📊 Model Lineage & Metrics",
    "🛡️ Data Drift & Monitoring",
])

# ------------------ TAB 1: PREDICTION ------------------
with tab_predict:
    st.header("Flower Species Classification")
    st.write("Provide the flower dimensions below to receive real-time predictions from the production champion model:")

    if model is None:
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. "
            "Please run `dvc repro` to train and select the champion model."
        )
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
        petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=3.8, step=0.1)
    with col2:
        sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
        petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2, step=0.1)

    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES,
    )

    if st.button("Predict Species", type="primary"):
        prediction = model.predict(input_data)[0]
        pretty_pred = pretty(prediction)

        proba_dict = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            classes = [pretty(c) for c in model.classes_]
            proba_dict = {classes[i]: float(proba[i]) for i in range(len(classes))}

        # Log inference to monitoring database
        log_inference(input_data, prediction, proba_dict)

        st.success(f"### Predicted Species: **{pretty_pred}**")

        if proba_dict:
            st.write("##### Prediction Probabilities")
            proba_df = pd.DataFrame(list(proba_dict.items()), columns=["Species", "Probability"]).set_index("Species")
            st.bar_chart(proba_df)

# ------------------ TAB 2: MODEL LINEAGE & METRICS ------------------
with tab_metrics:
    st.header("Model Lineage & Multi-Criteria Benchmark")
    st.write("All 6 candidate models were evaluated across 6 criteria to pick the optimal production champion:")

    comp_csv = Path(__file__).parent / "reports" / "model_comparison.csv"
    if comp_csv.exists():
        comp_df = pd.read_csv(comp_csv)
        st.dataframe(comp_df)
    elif selection:
        st.json(selection)
    else:
        st.info("Run `dvc repro` to generate comparison metrics.")

# ------------------ TAB 3: DATA DRIFT MONITORING ------------------
with tab_monitoring:
    st.header("Production Data Drift & Quality Monitoring")
    st.write(
        "Monitoring production inputs against the reference training dataset using the "
        "**Kolmogorov-Smirnov (KS) two-sample test**."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_check = st.button("Run Drift Check", type="primary")

    log_df = load_inference_logs(INFERENCE_LOG_PATH)
    if not log_df.empty:
        st.write(f"**Total production records logged:** `{len(log_df)}`")

        if run_check or len(log_df) >= 5:
            if REFERENCE_DATA_PATH.exists():
                ref_df = pd.read_csv(REFERENCE_DATA_PATH)
                report = analyze_drift(ref_df, log_df)

                status_color = "red" if report["dataset_drift_detected"] else "green"
                status_msg = "⚠️ DRIFT DETECTED" if report["dataset_drift_detected"] else "✅ HEALTHY DISTRIBUTION"

                st.markdown(f"#### Status: :{status_color}[{status_msg}]")
                st.write(
                    f"Drifted features: **{report['drifted_features']} / {report['total_features']}** "
                    f"({report['drift_share_pct']}%)"
                )

                drift_rows = []
                for feat, d in report["features"].items():
                    drift_rows.append({
                        "Feature": feat,
                        "Status": "DRIFT ⚠️" if d["drift_detected"] else "HEALTHY ✅",
                        "p-value": round(d["p_value"], 4),
                        "KS Statistic": round(d["ks_statistic"], 4),
                        "Reference Mean": round(d["reference_mean"], 2),
                        "Production Mean": round(d["current_mean"], 2),
                        "Mean Shift": round(d["mean_shift"], 2),
                    })
                st.dataframe(pd.DataFrame(drift_rows))

        with st.expander("View Recent Production Logs (Last 10 Requests)"):
            st.dataframe(log_df.tail(10))
    else:
        st.info("No production inferences logged yet. Head over to the 'Live Inference' tab to submit predictions.")
