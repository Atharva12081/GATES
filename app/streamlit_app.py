from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "evidence"

st.set_page_config(page_title="GATES", page_icon="⏱️", layout="wide")


@st.cache_data
def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    predictions = pd.read_csv(EVIDENCE / "predictions_by_time.csv")
    decisions = pd.read_csv(EVIDENCE / "decisions.csv")
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    return predictions, decisions, summary


predictions, decisions, summary = load_evidence()
st.title("GATES")
st.subheader("Group-Aware Time-Efficient Stopping")
st.write(
    "A feasibility system that asks whether a longitudinal in-vitro experiment has enough "
    "evidence to stop, should continue, or is too shifted for automated early action."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Held-out units", int(summary["total"]))
c2.metric("Early-stop coverage", f"{summary['coverage']:.0%}")
c3.metric("Error among early stops", f"{summary['eesr']:.1%}")
c4.metric("Mean saved observations", f"{summary['observation_savings'] * 18:.1f} / 18")

unit = st.selectbox("Held-out experimental unit", predictions["unit_id"].drop_duplicates())
trajectory = predictions.loc[predictions["unit_id"] == unit].sort_values("decision_time")
chosen = decisions.loc[decisions["unit_id"] == unit].iloc[0]

left, right = st.columns([2, 1])
with left:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=trajectory["decision_time"],
            y=trajectory["probability_response"],
            mode="lines+markers",
            name="Predicted response probability",
        )
    )
    figure.add_hline(y=0.5, line_dash="dot", annotation_text="decision boundary")
    figure.add_vline(
        x=float(chosen["decision_time"]), line_dash="dash", annotation_text=str(chosen["decision"])
    )
    figure.update_layout(
        xaxis_title="Hours observed",
        yaxis_title="Probability of low-viability response",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(figure, width="stretch")

with right:
    st.metric("Decision", chosen["decision"])
    st.metric("Decision time", f"{int(chosen['decision_time'])} h")
    st.metric("Scheduled observations saved", int(chosen["observations_saved"]))
    st.metric("Confidence", f"{chosen['confidence']:.1%}")
    trust = (
        "Shifted — automation refused" if bool(chosen["is_ood"]) else "Within fitted distance gate"
    )
    st.write(f"**Trust check:** {trust}")
    st.write(f"**Final viability:** {chosen['viability']:.3f}")

st.divider()
st.subheader("Evidence across held-out units")
plot = px.scatter(
    decisions,
    x="decision_time",
    y="confidence",
    color="decision",
    symbol=decisions["prediction"] == decisions["label"],
    hover_data=["unit_id", "viability", "observations_saved", "is_ood"],
    labels={"decision_time": "Decision time (h)", "confidence": "Prediction confidence"},
)
st.plotly_chart(plot, width="stretch")

with st.expander("Audit record"):
    fields = [
        "unit_id",
        "fold",
        "decision_time",
        "probability_response",
        "confidence_threshold",
        "calibration_certified",
        "ood_score",
        "ood_threshold",
        "decision",
    ]
    st.dataframe(chosen[fields].astype(str).to_frame("value"), width="stretch")
    st.caption(
        "This public dataset has 21 endpoint-labelled units. Confidence-bound certification is "
        "not claimed when calibration sample size is insufficient."
    )
