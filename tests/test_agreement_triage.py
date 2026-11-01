import numpy as np
import pandas as pd

from gates.act.agreement_triage import _calibrate_margin, _effect_call, _evaluate


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_group": ["a", "a", "b", "b"],
            "truth": [0.8, 0.0, -0.7, 0.0],
            "gates": [0.6, 0.01, -0.5, 0.02],
            "selected_residual": [0.7, 0.02, -0.6, 0.01],
            "interpolation": [0.5, 0.03, -0.5, 0.02],
        }
    )


def test_agreement_requires_threshold_and_sign():
    values = frame()
    np.testing.assert_array_equal(_effect_call(values, "agreement"), [True, False, True, False])
    values.loc[2, "selected_residual"] = 0.6
    assert not _effect_call(values, "agreement")[2]


def test_margin_calibration_protects_calibration_effects():
    values = frame()
    margin = _calibrate_margin(values, "agreement")
    result = _evaluate(values, "agreement", margin)
    assert result["protected_recall"] >= 0.95
    assert result["effect_precision"] == 1.0
