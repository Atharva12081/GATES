from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class RiskPrediction:
    mean: np.ndarray
    std: np.ndarray
    covariance: np.ndarray
    uncertainty_score: float
    shift_score: float
    effect_probability: np.ndarray


class RiskSensitiveSurface:
    """Shrinkage multi-output Gaussian model with dose-time smoothness.

    The empirical covariance learns recurrent biological response shapes. A
    Matern-like structural kernel prevents unstable conditioning when a response
    pattern or assay batch differs from training.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        training_responses: np.ndarray,
        covariance_blend: float = 0.70,
        noise_fraction: float = 0.06,
        analog_weight: float = 0.50,
        neighbor_count: int = 15,
    ):
        self.coordinates = np.asarray(coordinates, dtype=float)
        training = np.asarray(training_responses, dtype=float)
        if training.ndim != 2 or training.shape[1] != len(coordinates):
            raise ValueError("training_responses must be groups x grid cells")
        self.training_responses = training
        self.prior_mean = np.median(training, axis=0)
        self.robust_scale = np.maximum(
            np.quantile(np.abs(training - self.prior_mean), 0.75, axis=0), 0.08
        )
        centered = training - self.prior_mean
        empirical = LedoitWolf().fit(centered).covariance_
        distance = np.abs(self.coordinates[:, None, :] - self.coordinates[None, :, :])
        scaled = np.sqrt((distance[..., 0] / 0.30) ** 2 + (distance[..., 1] / 0.36) ** 2)
        smooth = (1 + np.sqrt(3) * scaled) * np.exp(-np.sqrt(3) * scaled)
        marginal = np.maximum(np.diag(empirical), 1e-4)
        structural = smooth * np.sqrt(np.outer(marginal, marginal))
        blend = float(np.clip(covariance_blend, 0.0, 1.0))
        self.prior_covariance = blend * empirical + (1 - blend) * structural
        self.noise = max(float(np.median(marginal) * noise_fraction), 1e-5)
        self.response_scale = max(float(np.quantile(np.ptp(training, axis=1), 0.75)), 1e-6)
        self.analog_weight = float(np.clip(analog_weight, 0.0, 1.0))
        self.neighbor_count = min(max(int(neighbor_count), 3), len(training))

    def fit_predict(
        self, observed: list[int], values: np.ndarray, effect_threshold: float
    ) -> RiskPrediction:
        observed_array = np.asarray(observed, dtype=int)
        values = np.asarray(values, dtype=float)
        covariance = self.prior_covariance
        cross = covariance[:, observed_array]
        observed_cov = covariance[np.ix_(observed_array, observed_array)]
        inverse = np.linalg.pinv(observed_cov + np.eye(len(observed_array)) * self.noise)
        residual = values - self.prior_mean[observed_array]
        gaussian_mean = self.prior_mean + cross @ inverse @ residual
        gaussian_posterior = covariance - cross @ inverse @ cross.T

        standardized_difference = (
            self.training_responses[:, observed_array] - values[None, :]
        ) / self.robust_scale[observed_array][None, :]
        distances = np.sqrt(np.mean(standardized_difference**2, axis=1))
        neighbor_indices = np.argsort(distances)[: self.neighbor_count]
        local_distances = distances[neighbor_indices]
        bandwidth = max(float(np.median(local_distances)), 0.10)
        weights = np.exp(-0.5 * (local_distances / bandwidth) ** 2)
        weights /= max(float(weights.sum()), 1e-12)
        neighbors = self.training_responses[neighbor_indices]
        analog_mean = np.sum(neighbors * weights[:, None], axis=0)
        analog_centered = neighbors - analog_mean
        analog_covariance = (analog_centered * weights[:, None]).T @ analog_centered
        analog_covariance += np.eye(len(analog_covariance)) * self.noise

        mean = (1 - self.analog_weight) * gaussian_mean + self.analog_weight * analog_mean
        posterior = (
            1 - self.analog_weight
        ) * gaussian_posterior + self.analog_weight * analog_covariance
        posterior = (posterior + posterior.T) / 2
        posterior += np.eye(len(posterior)) * self.noise
        std = np.sqrt(np.maximum(np.diag(posterior), 1e-12))
        standardized = residual @ inverse @ residual / max(len(observed_array), 1)
        positive = 1 - norm.cdf((effect_threshold - mean) / std)
        negative = norm.cdf((-effect_threshold - mean) / std)
        empirical_effect = np.sum(
            (np.abs(neighbors) >= effect_threshold) * weights[:, None], axis=0
        )
        effect_probability = (1 - self.analog_weight) * (
            positive + negative
        ) + self.analog_weight * empirical_effect
        return RiskPrediction(
            mean=mean,
            std=std,
            covariance=posterior,
            uncertainty_score=float(np.sqrt(np.mean(std**2)) / self.response_scale),
            shift_score=float(np.sqrt(max(standardized, 0.0))),
            effect_probability=np.clip(effect_probability, 0.0, 1.0),
        )

    def risk_sensitive_score(
        self,
        prediction: RiskPrediction,
        candidate: int,
        targets: np.ndarray,
        effect_weight: float,
        direct_bonus: float,
    ) -> float:
        covariance = prediction.covariance
        denominator = max(float(covariance[candidate, candidate] + self.noise), 1e-12)
        reduction = covariance[targets, candidate] ** 2 / denominator
        weights = 1 + effect_weight * prediction.effect_probability[targets]
        integrated = float(np.sum(weights * reduction))
        direct = float(
            direct_bonus
            * prediction.effect_probability[candidate]
            * covariance[candidate, candidate]
        )
        return integrated + direct
