from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

from gates.act.model_v3 import RiskPrediction, RiskSensitiveSurface


@dataclass(frozen=True)
class RiskLimit:
    """Finite-sample release gate for unsafe early-stop risk.

    The bound is one-sided Clopper-Pearson. It is deliberately a release
    condition, not a claim of conditional or distribution-free validity.
    """

    validation_stops: int
    unsafe_stops: int
    maximum_unsafe_rate: float = 0.05
    alpha: float = 0.05

    @property
    def upper_bound(self) -> float:
        if self.validation_stops <= 0:
            return 1.0
        if self.unsafe_stops >= self.validation_stops:
            return 1.0
        return float(
            beta.ppf(
                1 - self.alpha,
                self.unsafe_stops + 1,
                self.validation_stops - self.unsafe_stops,
            )
        )

    @property
    def authorized(self) -> bool:
        return self.upper_bound <= self.maximum_unsafe_rate


def minimum_zero_failure_stops(maximum_unsafe_rate: float = 0.05, alpha: float = 0.05) -> int:
    """Smallest validation sample whose zero-failure upper bound meets the risk limit."""

    if not 0 < maximum_unsafe_rate < 1 or not 0 < alpha < 1:
        raise ValueError("risk and alpha must lie strictly between zero and one")
    return int(np.ceil(np.log(alpha) / np.log(1 - maximum_unsafe_rate)))


def select_cost_aware_max_variance_batch(
    surface: RiskSensitiveSurface,
    prediction: RiskPrediction,
    observed: list[int],
    candidates: np.ndarray,
    batch_size: int,
    costs: np.ndarray | None = None,
    required: list[int] | None = None,
) -> list[int]:
    """Greedy conditional variance reduction per unit measurement cost.

    Conditioning the covariance after every choice prevents redundant batches.
    Equal costs reduce exactly to the strongest observed baseline.
    """

    candidate_array = np.asarray(candidates, dtype=int)
    if costs is None:
        cost_vector = np.ones(len(surface.coordinates), dtype=float)
    else:
        cost_vector = np.asarray(costs, dtype=float)
        if len(cost_vector) != len(surface.coordinates) or np.any(cost_vector <= 0):
            raise ValueError("costs must be positive and match the response grid")
    selected = list(dict.fromkeys(required or []))
    if len(selected) > batch_size or any(item not in candidate_array for item in selected):
        raise ValueError("required measurements must fit inside the candidate batch")
    covariance = prediction.covariance.copy()
    for chosen in selected:
        denominator = max(float(covariance[chosen, chosen] + surface.noise), 1e-12)
        covariance -= np.outer(covariance[:, chosen], covariance[chosen, :]) / denominator
    while len(selected) < batch_size:
        available = np.asarray(
            [item for item in candidate_array if item not in observed and item not in selected],
            dtype=int,
        )
        if not len(available):
            break
        utility = np.diag(covariance)[available] / cost_vector[available]
        chosen = int(available[int(np.argmax(utility))])
        selected.append(chosen)
        denominator = max(float(covariance[chosen, chosen] + surface.noise), 1e-12)
        covariance -= np.outer(covariance[:, chosen], covariance[chosen, :]) / denominator
    return selected


def released_decision(
    candidate: str,
    risk_limit: RiskLimit,
    in_distribution: bool,
) -> str:
    """Fail closed unless both population shift and empirical risk gates pass."""

    if not in_distribution:
        return "ABSTAIN"
    if candidate == "STOP" and risk_limit.authorized:
        return "STOP"
    return "COMPLETE"
