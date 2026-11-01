import pandas as pd

from gates.act.source_aware_controller import _activation, _calls


def frame(truth: list[float], gates: list[float], residual: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "truth": truth,
            "gates": gates,
            "selected_residual": residual,
            "compound_group": [f"g{i}" for i in range(len(truth))],
        }
    )


def test_activation_prefers_agreement_when_removed_calls_are_mostly_false():
    values = frame(
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.8],
        [0.5] * 6,
        [0.1] * 6,
    )
    mode, counts = _activation(values)
    assert mode == "agreement"
    assert counts["false_discoveries_removed"] == 4
    assert counts["true_effect_calls_deferred"] == 2


def test_activation_keeps_gates_when_removed_calls_are_mostly_true():
    values = frame(
        [0.0, 0.8, 0.8, 0.8, 0.8, 0.8],
        [0.5] * 6,
        [0.1] * 6,
    )
    mode, _ = _activation(values)
    assert mode == "gates"


def test_agreement_rejects_opposite_sign_effects():
    values = frame([0.8], [0.5], [-0.6])
    assert not _calls(values, "agreement")[0]
