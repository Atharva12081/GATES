from __future__ import annotations

import numpy as np

from gates.act.benchmark import _paired_sign_flip_pvalue, _split_groups
from gates.act.model import AdaptiveResponseSurface, select_initial_design


def test_group_split_is_disjoint() -> None:
    replicates = [1, 2, 3, 4, 5, 6]
    for test in replicates:
        training, calibration = _split_groups(replicates, test)
        assert len(training) == 3
        assert len(calibration) == 2
        assert set(training).isdisjoint(calibration)
        assert test not in training + calibration


def test_act_never_reselects_observed_cell() -> None:
    coordinates = np.array([[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])
    training = np.vstack(
        [coordinates[:, 0] + coordinates[:, 1] + offset for offset in (0, 0.1, -0.1)]
    )
    surface = AdaptiveResponseSurface(coordinates, training)
    observed = select_initial_design(coordinates, size=3)
    prediction = surface.fit_predict(observed, training[0, observed])
    chosen = surface.select_next(prediction, observed, "act", np.random.default_rng(1))
    assert chosen not in observed


def test_uncertainty_falls_after_measurement() -> None:
    coordinates = np.array([[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])
    training = np.vstack(
        [coordinates[:, 0] ** 2 + coordinates[:, 1] + offset for offset in (0, 0.1, -0.1)]
    )
    surface = AdaptiveResponseSurface(coordinates, training)
    observed = select_initial_design(coordinates, size=3)
    first = surface.fit_predict(observed, training[0, observed])
    observed.append(surface.select_next(first, observed, "act", np.random.default_rng(1)))
    second = surface.fit_predict(observed, training[0, observed])
    assert second.uncertainty_score < first.uncertainty_score


def test_sign_flip_pvalue_bounds() -> None:
    pvalue = _paired_sign_flip_pvalue(np.array([0.1, 0.2, 0.3]))
    assert 0.0 <= pvalue <= 1.0
