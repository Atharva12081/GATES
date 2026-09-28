from pathlib import Path

import numpy as np

from gates.policy.cbes import sequential_calibrate_cbes


def test_cbes_sequential_calibration_keeps_unit_axis_fixed() -> None:
    labels = np.array([0, 1, 1])
    decisions = [np.array([0, -1, 1]), np.array([0, 1, 0])]
    result = sequential_calibrate_cbes(labels, decisions, labels, decisions, alpha=0.34)
    assert len(result.can_stop) == 2
    assert 0 <= result.calibration_eesr <= 1


def test_phase2_claims_do_not_overstate_empirical_evidence() -> None:
    roots = [Path("README.md"), Path("docs"), Path("protocols")]
    files = []
    for root in roots:
        files.extend(root.rglob("*.md") if root.is_dir() else [root])
        if root.is_dir():
            files.extend(root.rglob("*.json"))
    forbidden = (
        "guarantees 5% risk",
        "guaranteed 5% risk",
        "clinically validated",
        "safe for clinical use",
    )
    violations = []
    for path in files:
        text = path.read_text().lower()
        for phrase in forbidden:
            if phrase in text:
                violations.append(f"{path}: {phrase}")
    assert not violations, "Unsupported claims: " + "; ".join(violations)
