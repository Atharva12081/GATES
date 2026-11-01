from __future__ import annotations

import numpy as np

from gates.act.decision_utility import AnalogDecisionModel, reference_active


def test_reference_requires_repeatable_effect_across_two_divs() -> None:
    response = np.zeros(32)
    response[[9, 10, 17, 18]] = 0.5
    assert reference_active(response)
    response[[17, 18]] = 0.0
    assert not reference_active(response)


def test_vehicle_cells_do_not_create_reference_activity() -> None:
    response = np.zeros(32)
    response[[0, 8, 16, 24]] = 1.0
    assert not reference_active(response)


def test_analog_probability_tracks_matching_class() -> None:
    responses = np.vstack([np.zeros(32), np.ones(32), np.full(32, 0.9)])
    model = AnalogDecisionModel(responses, np.array([0, 1, 1]), neighbors=3)
    assert model.probability([0, 1], np.array([0.95, 0.95])) > 0.5
    assert model.probability([0, 1], np.array([0.0, 0.0])) < 0.5
