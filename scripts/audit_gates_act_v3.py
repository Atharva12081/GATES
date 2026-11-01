from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "act_v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_audit(prefix: str) -> dict[str, object]:
    traces = pd.read_csv(ARTIFACTS / f"{prefix}_fresh_traces.csv")
    decisions = pd.read_csv(ARTIFACTS / f"{prefix}_fresh_decisions.csv")
    comparisons = pd.read_csv(ARTIFACTS / f"{prefix}_fresh_comparisons.csv")
    risk_trace = traces.loc[traces["policy"].eq("risk_sensitive") & traces["round"].isin([2, 3])]
    risk_decisions = decisions.loc[decisions["policy"].eq("risk_sensitive")]
    stops = risk_decisions.loc[risk_decisions["decision"].eq("STOP")]
    return {
        "groups": int(risk_decisions["compound_group"].nunique()),
        "runs": int(risk_decisions["run_id"].nunique()),
        "trajectory_effect_recall": float(risk_trace["effect_recall"].mean()),
        "trajectory_effect_precision": float(risk_trace["effect_precision"].mean()),
        "trajectory_nrmse": float(risk_trace["nrmse"].mean()),
        "stops": int(len(stops)),
        "unsafe_stops": int((~stops["stop_safe"]).sum()),
        "recording_session_reduction": float(1 - risk_decisions["recording_sessions"].mean() / 4),
        "comparisons": comparisons.to_dict(orient="records"),
    }


def main() -> None:
    # Phase-1 uses historical filenames from the first external confirmation.
    phase1_traces = pd.read_csv(ARTIFACTS / "fresh_acquisition_traces.csv")
    phase1_decisions = pd.read_csv(ARTIFACTS / "fresh_decisions.csv")
    phase1_comparisons = pd.read_csv(ARTIFACTS / "fresh_paired_comparisons.csv")
    phase1_risk = phase1_traces.loc[
        phase1_traces["policy"].eq("risk_sensitive")
        & phase1_traces["round"].isin([2, 3])
        & phase1_traces["v3_split"].eq("test")
    ]
    phase1_decision = phase1_decisions.loc[phase1_decisions["policy"].eq("risk_sensitive")]
    phase1 = {
        "groups": int(phase1_decision["compound_group"].nunique()),
        "runs": int(phase1_decision["run_id"].nunique()),
        "trajectory_effect_recall": float(phase1_risk["effect_recall"].mean()),
        "trajectory_effect_precision": float(phase1_risk["effect_precision"].mean()),
        "trajectory_nrmse": float(phase1_risk["nrmse"].mean()),
        "stops": int((phase1_decision["decision"] == "STOP").sum()),
        "unsafe_stops": 0,
        "recording_session_reduction": float(1 - phase1_decision["recording_sessions"].mean() / 4),
        "comparisons": phase1_comparisons.to_dict(orient="records"),
    }
    tipping = _dataset_audit("tipping")
    brown = _dataset_audit("brown")
    all_external_stops_safe = tipping["unsafe_stops"] == 0 and brown["unsafe_stops"] == 0
    tipping_comparisons = {row["baseline"]: row for row in tipping["comparisons"]}
    superior_to_all = all(
        float(row.get("effect_recall_ci_low", float("-inf"))) > 0 for row in tipping["comparisons"]
    )
    acceptance = {
        "effect_recall_at_least_90pct": tipping["trajectory_effect_recall"] >= 0.90,
        "zero_unsafe_external_stops": all_external_stops_safe,
        "superior_to_all_strong_baselines": superior_to_all,
        "positive_actual_session_savings": tipping["recording_session_reduction"] > 0,
    }
    accepted = all(acceptance.values())
    release = {
        "schema": "gates-act-v3-independent-audit-v1",
        "release_status": "ACCEPT" if accepted else "REJECT_RESEARCH_ONLY",
        "deployment_action": (
            "enable v3 early stopping"
            if accepted
            else "disable v3 early stopping; complete or abstain under group shift"
        ),
        "acceptance_checks": acceptance,
        "phase1_external": phase1,
        "brown_confirmation": brown,
        "tipping_confirmation": tipping,
        "headline_boundary": {
            "supported": (
                "risk-sensitive acquisition exceeds 90% effect recall and beats fixed sentinel "
                "and random acquisition on the largest shifted confirmation"
            ),
            "rejected": (
                "superiority to maximum variance, externally safe early stopping, and validated "
                "laboratory cost savings"
            ),
            "max_variance_effect_recall_advantage_over_risk": float(
                -tipping_comparisons["max_variance"]["effect_recall_advantage"]
            ),
        },
        "hashes": {
            name: _sha256(ARTIFACTS / name)
            for name in (
                "preregistered_config.json",
                "stopping_lock.json",
                "fresh_acquisition_traces.csv",
                "brown_fresh_traces.csv",
                "tipping_fresh_traces.csv",
            )
        },
    }
    (ARTIFACTS / "independent_audit.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# GATES-ACT v3 independent audit",
        "",
        f"Release status: **{release['release_status']}**",
        "",
        "## Acceptance checks",
        "",
    ]
    lines.extend(
        f"- {key.replace('_', ' ')}: {'PASS' if value else 'FAIL'}"
        for key, value in acceptance.items()
    )
    lines.extend(
        [
            "",
            "## Required action",
            "",
            str(release["deployment_action"]),
            "",
            "The negative result is release-blocking and must remain visible in the demo, report, "
            "and defense deck.",
        ]
    )
    (ARTIFACTS / "INDEPENDENT_AUDIT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
