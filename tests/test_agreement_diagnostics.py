import numpy as np
import pandas as pd

from gates.act.agreement_diagnostics import (
    _calibrate_recall_margin,
    _metrics,
    disagreement_state,
    effect_call,
)


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_group": ["a", "a", "b", "b", "c"],
            "truth": [0.8, 0.0, -0.7, 0.0, 0.9],
            "gates": [0.6, 0.5, -0.5, 0.1, 0.7],
            "selected_residual": [0.7, 0.1, -0.6, 0.5, -0.8],
            "interpolation": [0.5, 0.2, -0.5, 0.2, 0.6],
        }
    )


def test_disagreement_taxonomy_is_exhaustive():
    values = disagreement_state(frame())
    assert values.tolist() == [
        "both_same_sign",
        "gates_only",
        "both_same_sign",
        "residual_only",
        "both_opposite_sign",
    ]


def test_agreement_rejects_opposite_sign_effects():
    np.testing.assert_array_equal(
        effect_call(frame(), "agreement"), [True, False, True, False, False]
    )


def test_crossfit_margin_respects_calibration_recall_target():
    values = frame()
    margin = _calibrate_recall_margin(values, "agreement", 0.95)
    metrics = _metrics(values, "agreement", np.full(len(values), margin))
    assert metrics["protected_recall"] >= 0.95
