from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gates.act.brown_mea import load_brown_mea
from gates.act.model_v3 import RiskSensitiveSurface
from gates.act.neural_mea import load_neural_mea, neural_coordinates
from gates.act.neural_mea_phase1 import load_phase1_mea, phase1_surface_arrays
from gates.act.tipping_mea import load_tipping_mea, tipping_surface_arrays

ROOT = Path(__file__).resolve().parents[1]


def test_risk_sensitive_surface_returns_finite_probabilistic_prediction() -> None:
    rng = np.random.default_rng(7)
    coordinates = neural_coordinates()
    training = rng.normal(size=(40, 32))
    model = RiskSensitiveSurface(coordinates, training, analog_weight=0.5)
    observed = list(range(8))
    prediction = model.fit_predict(observed, training[0, observed], effect_threshold=0.4)
    assert prediction.mean.shape == (32,)
    assert prediction.covariance.shape == (32, 32)
    assert np.isfinite(prediction.mean).all()
    assert np.all((prediction.effect_probability >= 0) & (prediction.effect_probability <= 1))


def test_phase1_split_is_grouped_and_legacy_overlap_free() -> None:
    legacy = load_neural_mea(ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx")
    phase1 = load_phase1_mea(
        ROOT
        / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx",
        set(legacy.surfaces["compound_group"]),
    )
    meta, values = phase1_surface_arrays(phase1)
    assert phase1.audit["eligible_dtxsid_groups"] == 86
    assert phase1.audit["split_groups"]["test"] == 17
    assert meta.groupby("compound_group")["split"].nunique().max() == 1
    assert all(len(surface) == 32 for surface in values.values())


def test_external_confirmations_have_expected_fresh_group_counts() -> None:
    legacy = load_neural_mea(ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx")
    phase1 = load_phase1_mea(
        ROOT
        / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx",
        set(legacy.surfaces["compound_group"]),
    )
    brown = load_brown_mea(ROOT / "data/neural_mea_brown_raw/brown_mea_2016.csv")
    excluded = set(legacy.surfaces["compound_group"])
    excluded.update(phase1.surfaces["compound_group"])
    excluded.update(brown.surfaces["compound_group"])
    tipping = load_tipping_mea(
        [
            ROOT / "data/neural_mea_tipping_raw/DNTRef_all.csv",
            ROOT / "data/neural_mea_tipping_raw/AllData_SA3_NoDNTRef.csv",
        ],
        excluded,
    )
    meta, values = tipping_surface_arrays(tipping)
    assert brown.audit["fresh_compound_groups"] == 6
    assert tipping.audit["fresh_compound_groups"] == 53
    assert meta["compound_group"].nunique() == 53
    assert all(len(surface) == 32 for surface in values.values())


def test_independent_audit_fails_closed_on_external_evidence() -> None:
    audit = json.loads((ROOT / "artifacts/act_v3/independent_audit.json").read_text())
    assert audit["release_status"] == "REJECT_RESEARCH_ONLY"
    assert audit["acceptance_checks"]["effect_recall_at_least_90pct"] is True
    assert audit["acceptance_checks"]["zero_unsafe_external_stops"] is False
    assert audit["acceptance_checks"]["superior_to_all_strong_baselines"] is False
    assert "disable v3 early stopping" in audit["deployment_action"]
