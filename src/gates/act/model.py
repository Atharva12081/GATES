from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def select_initial_design(coordinates: np.ndarray, size: int = 5) -> list[int]:
    """Deterministic farthest-point design with a center anchor."""
    coordinates = np.asarray(coordinates, dtype=float)
    center = np.array([0.5, 0.5])
    chosen = [int(np.argmin(np.linalg.norm(coordinates - center, axis=1)))]
    while len(chosen) < min(size, len(coordinates)):
        distances = np.min(
            np.linalg.norm(coordinates[:, None, :] - coordinates[chosen][None, :, :], axis=2),
            axis=1,
        )
        distances[chosen] = -1.0
        chosen.append(int(np.argmax(distances)))
    return chosen


@dataclass
class SurfacePrediction:
    mean: np.ndarray
    std: np.ndarray
    covariance: np.ndarray
    uncertainty_score: float
    misfit_score: float


class AdaptiveResponseSurface:
    """Hierarchical prior surface plus a GP for replicate-specific residuals."""

    def __init__(self, coordinates: np.ndarray, training_responses: np.ndarray, seed: int = 0):
        self.coordinates = np.asarray(coordinates, dtype=float)
        training_responses = np.asarray(training_responses, dtype=float)
        if training_responses.ndim != 2 or training_responses.shape[1] != len(coordinates):
            raise ValueError("training_responses must be groups x grid cells")
        self.prior_mean = training_responses.mean(axis=0)
        self.prior_sd = np.maximum(training_responses.std(axis=0, ddof=1), 1e-3)
        self.response_scale = max(float(np.ptp(training_responses)), 1e-6)
        self.seed = seed

    def fit_predict(self, observed: list[int], values: np.ndarray) -> SurfacePrediction:
        observed_array = np.asarray(observed, dtype=int)
        values = np.asarray(values, dtype=float)
        residual = values - self.prior_mean[observed_array]
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=[0.35, 0.28], length_scale_bounds="fixed", nu=1.5
        ) + WhiteKernel(noise_level=0.04, noise_level_bounds="fixed")
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            optimizer=None,
            random_state=self.seed,
        )
        model.fit(self.coordinates[observed_array], residual)
        residual_mean, residual_cov = model.predict(self.coordinates, return_cov=True)
        covariance = residual_cov + np.diag((0.35 * self.prior_sd) ** 2)
        std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        normalized_residual = np.abs(residual) / self.prior_sd[observed_array]
        return SurfacePrediction(
            mean=self.prior_mean + residual_mean,
            std=std,
            covariance=covariance,
            uncertainty_score=float(np.sqrt(np.mean(std**2)) / self.response_scale),
            misfit_score=float(np.quantile(normalized_residual, 0.9)),
        )

    def select_next(
        self,
        prediction: SurfacePrediction,
        observed: list[int],
        strategy: str,
        rng: np.random.Generator,
    ) -> int:
        candidates = np.array([i for i in range(len(self.coordinates)) if i not in observed])
        if not len(candidates):
            raise ValueError("no unmeasured assay cells remain")
        if strategy == "random":
            return int(rng.choice(candidates))
        if strategy == "max_variance":
            return int(candidates[np.argmax(prediction.std[candidates])])
        if strategy == "maximin":
            distances = np.min(
                np.linalg.norm(
                    self.coordinates[candidates, None, :]
                    - self.coordinates[np.asarray(observed)][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            return int(candidates[np.argmax(distances)])
        if strategy == "act":
            covariance = prediction.covariance
            remaining_covariance = covariance[np.ix_(candidates, candidates)]
            denominator = np.maximum(np.diag(remaining_covariance), 1e-12)
            reduction = np.sum(remaining_covariance**2, axis=0) / denominator
            return int(candidates[np.argmax(reduction)])
        raise ValueError(f"unknown acquisition strategy: {strategy}")
