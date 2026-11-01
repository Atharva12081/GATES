import numpy as np
import pandas as pd

from gates.act.chronological_high_dose import chronological_design
from gates.act.high_dose_falsification import _threshold_layer


def _frame(high: tuple[float, float, float]) -> pd.DataFrame:
    rows = []
    for rank in range(1, 8):
        direct = rank in (5, 6, 7)
        value = high[rank - 5] if direct else 0.0
        rows.append(
            {
                "fold": 0,
                "run_id": "r",
                "compound_group": "g",
                "cell": rank,
                "truth": value,
                "directly_observed": direct,
                "effect_call": False,
                "no_effect_call": False,
            }
        )
    return pd.DataFrame(rows)


def test_adjacent_boundary_rule_requires_two_same_sign_effects():
    result = _threshold_layer(_frame((0.5, 0.6, 0.1)), "rule", release_quiet=False)
    assert bool(result.loc[result["cell"].eq(4), "effect_call"].iloc[0])
    opposite = _threshold_layer(_frame((0.5, -0.6, 0.1)), "rule", release_quiet=False)
    assert not bool(opposite.loc[opposite["cell"].eq(4), "effect_call"].iloc[0])


def test_quiet_rule_releases_only_hidden_lower_doses():
    result = _threshold_layer(_frame((0.1, 0.1, 0.1)), "rule", release_quiet=True)
    assert result.loc[result["cell"].isin([1, 2, 3, 4]), "no_effect_call"].all()


def test_chronological_design_uses_only_past_evidence_and_fixed_daily_budget():
    truth = np.zeros(32)
    truth[6] = 0.5
    observed, log = chronological_design(truth)
    assert len(observed) == 16
    assert all(len(entry["selected_cells"]) == 4 for entry in log)
    assert log[1]["selected_treated_ranks"] == [4, 6, 7]
    for entry in log:
        assert (
            not entry["evidence_cells"]
            or max(cell // 8 for cell in entry["evidence_cells"]) < entry["time_rank"]
        )
