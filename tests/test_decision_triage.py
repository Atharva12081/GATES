from __future__ import annotations

import numpy as np
import pandas as pd

from gates.act.decision_triage import _binary_threshold, _triage_thresholds


def test_binary_threshold_preserves_recall() -> None:
    probability = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    label = np.array([1, 1, 1, 0, 0])
    threshold = _binary_threshold(probability, label)
    prediction = probability >= threshold
    assert np.sum(prediction & (label == 1)) / np.sum(label == 1) >= 0.95


def test_triage_thresholds_leave_an_unresolved_region() -> None:
    frame = pd.DataFrame(
        {
            "compound_group": [f"g{i // 4}" for i in range(40)],
            "label": [0, 0, 1, 1] * 10,
        }
    )
    probability = np.tile(np.array([0.05, 0.2, 0.8, 0.95]), 10)
    low, high = _triage_thresholds(frame, probability)
    assert low < high
