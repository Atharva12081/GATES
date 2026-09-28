from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gates.policy.stopping import clopper_pearson_upper


@dataclass(frozen=True)
class GroupThreshold:
    confidence_threshold: float
    calibration_stops: int
    calibration_errors: int
    empirical_risk: float
    worst_group_risk: float
    upper_risk: float
    certified: bool


def calibrate_group_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    target_risk: float,
    minimum_stops_per_group: int = 5,
) -> GroupThreshold:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    confidence = np.maximum(probabilities, 1 - probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    candidates = sorted(set(confidence.tolist()))
    fallback = GroupThreshold(1.01, 0, 0, 0.0, 1.0, 1.0, False)
    valid: list[GroupThreshold] = []
    for threshold in candidates:
        selected = confidence >= threshold
        group_risks: list[float] = []
        enough = True
        for group in np.unique(groups):
            group_selected = selected & (groups == group)
            count = int(group_selected.sum())
            if count < minimum_stops_per_group:
                enough = False
                break
            group_risks.append(
                float((predictions[group_selected] != labels[group_selected]).mean())
            )
        if not enough:
            continue
        errors = int((predictions[selected] != labels[selected]).sum())
        total = int(selected.sum())
        empirical = errors / total
        upper = clopper_pearson_upper(errors, total)
        item = GroupThreshold(
            confidence_threshold=float(threshold),
            calibration_stops=total,
            calibration_errors=errors,
            empirical_risk=empirical,
            worst_group_risk=max(group_risks),
            upper_risk=upper,
            certified=upper <= target_risk,
        )
        if item.worst_group_risk <= target_risk:
            valid.append(item)
    if not valid:
        return fallback
    return max(valid, key=lambda item: item.calibration_stops)
