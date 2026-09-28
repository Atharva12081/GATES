import numpy as np

from gates.policy.stopping import calibrate_threshold, decide


def test_shift_forces_abstention() -> None:
    assert decide(0.99, 0.8, is_ood=True) == "ABSTAIN"


def test_low_confidence_continues() -> None:
    assert decide(0.55, 0.8, is_ood=False) == "CONTINUE"


def test_calibration_never_claims_small_sample_certification() -> None:
    result = calibrate_threshold(np.array([0.99, 0.98, 0.97]), np.array([1, 1, 1]), 0.05)
    assert result.empirical_risk == 0.0
    assert result.certified is False
