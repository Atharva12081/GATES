from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


@dataclass(frozen=True)
class NestedGateFit:
    signal: str
    orientation: int
    threshold: float
    development_auc: float
    development_balanced_accuracy: float


def _oriented_auc(labels: np.ndarray, values: np.ndarray) -> tuple[float, int]:
    if len(np.unique(labels)) < 2:
        return float("nan"), 1
    raw = float(roc_auc_score(labels, values))
    return (raw, 1) if raw >= 0.5 else (1 - raw, -1)


def _fit_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    unique = np.unique(scores)
    candidates = np.r_[
        np.inf,
        unique,
        (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else [],
        -np.inf,
    ]
    best: tuple[float, int, float] | None = None
    best_threshold = float("inf")
    for threshold in candidates:
        predicted = scores >= threshold
        balanced = float(balanced_accuracy_score(labels, predicted))
        candidate = (balanced, -int(predicted.sum()), float(threshold))
        if best is None or candidate > best:
            best = candidate
            best_threshold = float(threshold)
    assert best is not None
    return best_threshold, best[0]


def fit_nested_experiment_gate(
    development: pd.DataFrame,
    candidate_signals: list[str],
    label_column: str = "unsafe",
) -> NestedGateFit:
    labels = development[label_column].to_numpy(int)
    if len(np.unique(labels)) < 2:
        return NestedGateFit(candidate_signals[0], 1, float("inf"), float("nan"), 0.5)
    ranked: list[tuple[float, int, str]] = []
    for order, signal in enumerate(candidate_signals):
        values = development[signal].to_numpy(float)
        auc, orientation = _oriented_auc(labels, values)
        ranked.append((auc, -order, signal))
    _, _, selected = max(ranked)
    auc, orientation = _oriented_auc(labels, development[selected].to_numpy(float))
    oriented = orientation * development[selected].to_numpy(float)
    threshold, balanced = _fit_threshold(labels, oriented)
    return NestedGateFit(selected, orientation, threshold, auc, balanced)


def apply_nested_experiment_gate(fit: NestedGateFit, value: float) -> bool:
    return bool(fit.orientation * value >= fit.threshold)
