from __future__ import annotations

import numpy as np

from gates.act.final_policy import (
    RiskLimit,
    minimum_zero_failure_stops,
    released_decision,
    select_cost_aware_max_variance_batch,
)
from gates.act.model_v3 import RiskSensitiveSurface


def _surface() -> RiskSensitiveSurface:
    coordinates = np.asarray([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [1.0, 0.0]])
    training = np.asarray(
        [[0.0, 0.2, 0.7, 1.0], [0.0, -0.1, -0.8, -1.1], [0.0, 0.1, 0.4, 0.9]]
    )
    return RiskSensitiveSurface(coordinates, training, analog_weight=0.0, neighbor_count=3)


def test_risk_limit_requires_59_zero_failure_validation_stops() -> None:
    required = minimum_zero_failure_stops()
    assert required == 59
    assert not RiskLimit(required - 1, 0).authorized
    assert RiskLimit(required, 0).authorized


def test_any_observed_external_failure_keeps_current_gate_closed() -> None:
    gate = RiskLimit(validation_stops=8, unsafe_stops=5)
    assert gate.upper_bound > 0.05
    assert released_decision("STOP", gate, in_distribution=True) == "COMPLETE"
    assert released_decision("STOP", gate, in_distribution=False) == "ABSTAIN"


def test_equal_cost_policy_selects_conditional_maximum_variance() -> None:
    surface = _surface()
    prediction = surface.fit_predict([0], np.asarray([0.0]), effect_threshold=0.4)
    selected = select_cost_aware_max_variance_batch(
        surface,
        prediction,
        observed=[0],
        candidates=np.asarray([1, 2, 3]),
        batch_size=2,
    )
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert selected[0] == int(np.argmax(prediction.std[1:]) + 1)


def test_costs_can_change_selection_without_breaking_constraints() -> None:
    surface = _surface()
    prediction = surface.fit_predict([0], np.asarray([0.0]), effect_threshold=0.4)
    expensive = int(np.argmax(prediction.std[1:]) + 1)
    costs = np.ones(4)
    costs[expensive] = 1_000
    selected = select_cost_aware_max_variance_batch(
        surface,
        prediction,
        observed=[0],
        candidates=np.asarray([1, 2, 3]),
        batch_size=1,
        costs=costs,
    )
    assert selected[0] != expensive
