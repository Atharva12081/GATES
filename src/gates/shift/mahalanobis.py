from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class MahalanobisGate:
    def __init__(self, quantile: float = 0.95) -> None:
        self.quantile = quantile
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.reference_: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, features: np.ndarray) -> MahalanobisGate:
        transformed = self.scaler.fit_transform(self.imputer.fit_transform(features))
        self.reference_ = transformed
        pairwise = np.sqrt(
            np.mean((transformed[:, None, :] - transformed[None, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(pairwise, np.inf)
        leave_one_out = pairwise.min(axis=1)
        self.threshold_ = float(np.quantile(leave_one_out, self.quantile) * 1.5)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("gate must be fitted before prediction")
        transformed = self.scaler.transform(self.imputer.transform(features))
        distances = np.sqrt(
            np.mean((transformed[:, None, :] - self.reference_[None, :, :]) ** 2, axis=2)
        )
        return distances.min(axis=1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("gate must be fitted before prediction")
        return self.score(features) > self.threshold_
