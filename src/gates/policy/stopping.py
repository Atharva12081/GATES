from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(errors: int, total: int, confidence: float = 0.95) -> float:
    if total == 0:
        return 1.0
    if errors == total:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, total - errors))


@dataclass(frozen=True)
class ThresholdResult:
    confidence_threshold: float
    calibration_stops: int
    calibration_errors: int
    empirical_risk: float
    upper_risk: float
    certified: bool


def calibrate_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_risk: float,
    min_stops: int = 3,
) -> ThresholdResult:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    confidence = np.maximum(probabilities, 1 - probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    candidates = sorted(set(confidence.tolist()), reverse=True)
    fallback = ThresholdResult(1.01, 0, 0, 0.0, 1.0, False)
    viable: list[ThresholdResult] = []
    for threshold in candidates:
        selected = confidence >= threshold
        total = int(selected.sum())
        if total < min_stops:
            continue
        errors = int((predictions[selected] != labels[selected]).sum())
        empirical = errors / total
        upper = clopper_pearson_upper(errors, total)
        viable.append(
            ThresholdResult(
                confidence_threshold=float(threshold),
                calibration_stops=total,
                calibration_errors=errors,
                empirical_risk=empirical,
                upper_risk=upper,
                certified=upper <= target_risk,
            )
        )
    certified = [item for item in viable if item.certified]
    if certified:
        return max(certified, key=lambda item: item.calibration_stops)
    empirical = [item for item in viable if item.empirical_risk <= target_risk]
    if empirical:
        best = max(empirical, key=lambda item: item.calibration_stops)
        return ThresholdResult(**{**best.__dict__, "certified": False})
    return fallback


def decide(probability: float, threshold: float, is_ood: bool) -> str:
    if is_ood:
        return "ABSTAIN"
    confidence = max(probability, 1 - probability)
    return "STOP" if confidence >= threshold else "CONTINUE"
