from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RPE_EVIDENCE = ROOT / "artifacts" / "phase2" / "retinal_rpe_final"
FINAL_EVIDENCE = ROOT / "artifacts" / "final"

st.set_page_config(page_title="GATES evidence demo", page_icon="⏱️", layout="wide")


@st.cache_data
def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    predictions = pd.read_csv(RPE_EVIDENCE / "predictions_by_time.csv")
    decisions = pd.read_csv(RPE_EVIDENCE / "unit_decisions.csv")
    decisions = decisions[
        (decisions["method"] == "gates_full") & np.isclose(decisions["parameter"], 0.05)
    ].copy()
    frontier = pd.read_csv(FINAL_EVIDENCE / "risk_savings_frontier.csv")
    cases = pd.read_csv(FINAL_EVIDENCE / "representative_cases.csv")
    summary = json.loads((FINAL_EVIDENCE / "science_summary.json").read_text())
    return predictions, decisions, frontier, cases, summary


predictions, decisions, frontier, cases, summary = load_evidence()
headline = summary["headline"]

st.title("GATES")
st.subheader("When is a held-out retinal-organoid trajectory safe enough to stop observing?")
st.caption(
    "Frozen retrospective evidence • 988 organoids • 11 experiment-held-out rotations • "
    "no retraining in this app"
)

c1, c2, c3 = st.columns(3)
c1.metric("Observed early-stop error", f"{headline['eesr']:.2%}")
c2.metric("Early-decision coverage", f"{headline['coverage']:.2%}")
c3.metric("Observation savings", f"{headline['observation_savings']:.2%}")

case_labels = {
    "Correct early stop": "Successful early stop",
    "Refusal / full observation": "Difficult case — refusal",
    "Erroneous early stop": "Failure case — erroneous stop",
}
selected_label = st.radio(
    "Evidence case", list(case_labels), format_func=case_labels.get, horizontal=True
)
case = cases[cases["selection_rule"] == selected_label].iloc[0]
unit_id = str(case["unit_id"])
trajectory = predictions[predictions["unit_id"] == unit_id].sort_values("decision_time")
decision = decisions[decisions["unit_id"] == unit_id].iloc[0]
available_times = trajectory["decision_time"].astype(float).tolist()
current_time = st.select_slider(
    "Observations available through",
    options=available_times,
    value=available_times[0],
    format_func=lambda value: f"{value:.0f} h",
)
visible = trajectory[trajectory["decision_time"] <= current_time]
current = visible.iloc[-1]
terminal_time = float(decision["terminal_time"])

if current_time < terminal_time:
    displayed_action = "CONTINUE"
elif decision["decision"] == "STOP":
    displayed_action = "STOP"
else:
    displayed_action = "REFUSE / CONTINUE TO 72 h"

left, right = st.columns([2, 1])
with left:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=visible["decision_time"],
            y=visible["probability"],
            mode="lines+markers",
            name="Predicted P(RPE+)",
        )
    )
    figure.add_hline(y=0.5, line_dash="dot", annotation_text="class boundary")
    if current_time >= terminal_time:
        figure.add_vline(x=terminal_time, line_dash="dash", annotation_text=displayed_action)
    figure.update_layout(
        height=390,
        xaxis_title="Observation time (h)",
        yaxis_title="Predicted P(RPE+)",
        yaxis_range=[0, 1],
        margin=dict(l=30, r=20, t=30, b=30),
    )
    st.plotly_chart(figure, width="stretch")

with right:
    st.metric("Held-out experiment", str(decision["group_id"]))
    st.metric("Observations consumed", int(current["observations_available"]))
    st.metric("Current decision", displayed_action)
    confidence = max(current["probability"], 1 - current["probability"])
    st.metric("Current confidence", f"{confidence:.1%}")
    ood_ratio = float(current["ood_score"] / current["ood_threshold"])
    st.metric("OOD score / threshold", f"{ood_ratio:.2f}")
    predicted_endpoint = "RPE+" if float(current["probability"]) >= 0.5 else "RPE−"
    st.metric("Endpoint prediction", predicted_endpoint)
    ground_truth = "RPE+" if int(decision["endpoint"]) else "RPE−"
    st.metric("Ground truth", ground_truth if current_time >= 72 else "Hidden until 72 h")
    if current_time >= terminal_time:
        st.metric("Observations saved", f"{decision['saved_fraction']:.1%}")
    else:
        st.metric("Observations saved", "pending")

st.divider()
st.subheader("Aggregate risk–savings evidence")
shown_methods = ["cbes_style", "gates_without_ood", "gates_full"]
plot_data = frontier[frontier["method"].isin(shown_methods)].copy()
plot_data["method"] = plot_data["method"].map(
    {
        "cbes_style": "CBES-style",
        "gates_without_ood": "GATES without refusal",
        "gates_full": "Full GATES",
    }
)
plot_data["EESR_percent"] = plot_data["EESR"] * 100
plot_data["savings_percent"] = plot_data["observation_savings"] * 100
risk_plot = px.line(
    plot_data.sort_values("savings_percent"),
    x="savings_percent",
    y="EESR_percent",
    color="method",
    markers=True,
    labels={
        "savings_percent": "Observation savings (%)",
        "EESR_percent": "Erroneous early-stop rate (%)",
        "method": "Method",
    },
)
risk_plot.add_hline(y=5, line_dash="dot", annotation_text="5% operating target")
risk_plot.update_layout(height=430, margin=dict(l=30, r=20, t=30, b=30))
st.plotly_chart(risk_plot, width="stretch")

with st.expander("Exact frozen evidence record"):
    fields = [
        "unit_id",
        "group_id",
        "endpoint",
        "decision",
        "prediction",
        "decision_time",
        "terminal_time",
        "saved_fraction",
        "ood_score",
        "ood_threshold",
        "fold",
        "test_group",
    ]
    st.dataframe(decision[fields].astype(str).to_frame("value"), width="stretch")
    st.caption(
        "The nominal 5% target is not a certified guarantee. Aggregate EESR is 5.69% "
        "(exact 95% CI 3.72–8.29%), and performance varies substantially by experiment."
    )
    st.caption(
        "Audit provenance: gates-science-freeze-v2; source record Zenodo 18198347; "
        "the displayed unit belongs only to its held-out experiment fold."
    )
