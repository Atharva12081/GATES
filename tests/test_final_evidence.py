from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_headline_is_derived_from_master_evidence() -> None:
    master = pd.read_csv(ROOT / "artifacts/final/master_evidence.csv")
    row = master[
        (master["dataset"] == "orgAInoid_RPE_Final")
        & (master["experiment"] == "ALL")
        & (master["method"] == "gates_full")
        & np.isclose(master["risk_target"], 0.05)
    ].iloc[0]
    assert row["test_units"] == 988
    assert row["early_stops"] == 439
    assert row["early_stop_errors"] == 25
    assert np.isclose(row["eesr"], 25 / 439)
    assert np.isclose(row["coverage"], 439 / 988)


def test_experiment_gate_is_not_promoted_to_core_method() -> None:
    summary = (ROOT / "artifacts/final/science_summary.json").read_text()
    assert '"experiment_gate_status": "OPTIONAL RESEARCH EXTENSION"' in summary


def test_class_specific_audit_reconciles_with_frozen_decisions() -> None:
    audit = pd.read_csv(ROOT / "artifacts/final/class_specific_audit.csv")
    assert audit["organoids"].sum() == 988
    assert audit["early_stops"].sum() == 439
    assert audit["erroneous_early_stops"].sum() == 25
    assert audit["refusals"].sum() == 38
