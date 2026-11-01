import pandas as pd

from gates.act.direct_discovery import _metrics


def test_metrics_combine_direct_and_inferred_effects():
    frame = pd.DataFrame(
        {
            "fold": [0, 0, 0, 0],
            "run_id": ["r"] * 4,
            "compound_group": ["g"] * 4,
            "truth": [0.8, 0.7, 0.0, 0.0],
            "directly_observed": [True, False, True, False],
            "effect_call": [True, True, False, False],
            "no_effect_call": [False, False, True, False],
        }
    )
    result = _metrics(frame)
    assert result["total_discovery_recall"] == 1
    assert result["confirmed_effect_precision"] == 1
    assert result["direct_effect_capture"] == 0.5
    assert result["decisive_coverage"] == 0.75
    assert result["run_group_sensitivity"] == 1
    assert result["observed_treated_response_cells"] == 2
    assert result["hidden_treated_response_cells"] == 2


def test_metrics_count_hidden_false_discovery_as_risk():
    frame = pd.DataFrame(
        {
            "fold": [0, 0],
            "run_id": ["r", "r"],
            "compound_group": ["g", "g"],
            "truth": [0.0, 0.0],
            "directly_observed": [True, False],
            "effect_call": [False, True],
            "no_effect_call": [True, False],
        }
    )
    result = _metrics(frame)
    assert result["wrong_decisions"] == 1
    assert result["selective_risk"] == 0.5
    assert result["run_group_precision"] == 0
