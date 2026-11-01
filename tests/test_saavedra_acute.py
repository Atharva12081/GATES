import pandas as pd

from gates.act.saavedra_acute import _aggregate_curves, _evaluate_policy, _metrics


def test_direct_policy_recall_counts_only_selected_reference_effects():
    frame = pd.DataFrame(
        {
            "compound": ["active"] * 7 + ["inactive"] * 7,
            "species": ["human"] * 14,
            "concentration_rank": list(range(7)) * 2,
            "concentration_um": [0.03, 0.1, 0.3, 1, 3, 10, 30] * 2,
            "wmfr_percent_baseline": [100, 100, 100, 100, 100, 100, 200] + [100] * 7,
        }
    )
    curves = _aggregate_curves(frame)
    decisions = _evaluate_policy(curves, (4, 5, 6), "top3")
    result = _metrics(decisions)
    assert result["reference_effect_units"] == 1
    assert result["confirmed_effect_units"] == 1
    assert result["discovery_recall"] == 1
    assert result["effect_precision"] == 1
    assert result["false_discoveries"] == 0


def test_unmeasured_low_dose_effect_remains_unresolved():
    frame = pd.DataFrame(
        {
            "compound": ["nonmonotonic"] * 7,
            "species": ["rat"] * 7,
            "concentration_rank": list(range(7)),
            "concentration_um": [0.03, 0.1, 0.3, 1, 3, 10, 30],
            "wmfr_percent_baseline": [200, 100, 100, 100, 100, 100, 100],
        }
    )
    decisions = _evaluate_policy(_aggregate_curves(frame), (4, 5, 6), "top3")
    result = _metrics(decisions)
    assert result["discovery_recall"] == 0
    assert result["missed_effect_units"] == 1
    assert result["unresolved_units"] == 1
