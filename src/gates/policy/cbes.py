from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CBESCalibration:
    can_stop: tuple[bool, ...]
    calibration_eesr: float
    validation_eesr: float
    hoeffding_epsilon: float
    validation_bound: float
    status: str


def confidence_bound_decision(
    mean_probability: np.ndarray,
    std_probability: np.ndarray,
    kappa: float,
) -> np.ndarray:
    positive = mean_probability - kappa * std_probability >= 0.5
    negative = mean_probability + kappa * std_probability < 0.5
    decision = np.full(len(mean_probability), -1, dtype=int)
    decision[negative] = 0
    decision[positive] = 1
    return decision


def _sequential_eesr(
    labels: np.ndarray,
    decisions_by_time: list[np.ndarray],
    can_stop: np.ndarray,
) -> float:
    stopped = 0
    errors = 0
    for index, label in enumerate(labels):
        for time_index, decisions in enumerate(decisions_by_time):
            if can_stop[time_index] and decisions[index] >= 0:
                stopped += 1
                errors += int(decisions[index] != label)
                break
    return errors / stopped if stopped else 0.0


def sequential_calibrate_cbes(
    calibration_labels: np.ndarray,
    calibration_decisions: list[np.ndarray],
    validation_labels: np.ndarray,
    validation_decisions: list[np.ndarray],
    alpha: float,
    delta: float = 0.05,
) -> CBESCalibration:
    n_times = len(calibration_decisions)
    can_stop = np.zeros(n_times, dtype=bool)
    for index, decisions in enumerate(calibration_decisions):
        selected = decisions >= 0
        errors = int((decisions[selected] != calibration_labels[selected]).sum())
        can_stop[index] = bool(selected.any() and errors / int(selected.sum()) <= alpha)
    for _ in range(n_times + 1):
        calibration_eesr = _sequential_eesr(calibration_labels, calibration_decisions, can_stop)
        if calibration_eesr <= alpha:
            break
        eligible = np.flatnonzero(can_stop)
        if not len(eligible):
            break
        can_stop[eligible[0]] = False
    validation_eesr = _sequential_eesr(validation_labels, validation_decisions, can_stop)
    epsilon = float(np.sqrt(np.log(1 / delta) / (2 * max(len(validation_labels), 1))))
    return CBESCalibration(
        can_stop=tuple(bool(value) for value in can_stop),
        calibration_eesr=float(calibration_eesr),
        validation_eesr=float(validation_eesr),
        hoeffding_epsilon=epsilon,
        validation_bound=float(validation_eesr + epsilon),
        status="calibrated" if can_stop.any() else "conservative_no_stop",
    )
