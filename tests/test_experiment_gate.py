import numpy as np
import pandas as pd

from gates.experiment_gate.nested import (
    apply_nested_experiment_gate,
    fit_nested_experiment_gate,
)
from gates.experiment_gate.signals import binary_entropy


def test_binary_entropy_is_largest_at_half() -> None:
    entropy = binary_entropy(np.array([0.01, 0.5, 0.99]))
    assert entropy[1] > entropy[0]
    assert np.isclose(entropy[0], entropy[2])


def test_nested_gate_selects_failure_aligned_scalar() -> None:
    frame = pd.DataFrame(
        {
            "unsafe": [0, 0, 0, 1, 1, 1],
            "noise": [0.3, 0.8, 0.1, 0.4, 0.2, 0.9],
            "risk_signal": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        }
    )
    fit = fit_nested_experiment_gate(frame, ["noise", "risk_signal"])
    assert fit.signal == "risk_signal"
    assert apply_nested_experiment_gate(fit, 0.85)
    assert not apply_nested_experiment_gate(fit, 0.15)


def test_nested_gate_fails_closed_to_accept_when_development_has_one_class() -> None:
    frame = pd.DataFrame({"unsafe": [0, 0, 0], "signal": [1.0, 2.0, 3.0]})
    fit = fit_nested_experiment_gate(frame, ["signal"])
    assert not apply_nested_experiment_gate(fit, 100.0)
