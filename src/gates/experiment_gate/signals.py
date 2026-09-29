from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


def binary_entropy(probability: pd.Series | np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    return -(values * np.log(values) + (1 - values) * np.log(1 - values))


def _scaled_samples(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    reference_values = scaler.fit_transform(imputer.fit_transform(reference[columns]))
    incoming_values = scaler.transform(imputer.transform(incoming[columns]))
    return reference_values, incoming_values


def _balanced_samples(
    reference: np.ndarray,
    incoming: np.ndarray,
    seed: int,
    maximum: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    size = min(len(reference), len(incoming), maximum)
    reference_index = rng.choice(len(reference), size, replace=False)
    incoming_index = rng.choice(len(incoming), size, replace=False)
    return reference[reference_index], incoming[incoming_index]


def _two_sample_auc(reference: np.ndarray, incoming: np.ndarray, seed: int) -> float:
    reference, incoming = _balanced_samples(reference, incoming, seed)
    values = np.vstack([reference, incoming])
    labels = np.r_[np.zeros(len(reference), dtype=int), np.ones(len(incoming), dtype=int)]
    folds = min(5, len(reference), len(incoming))
    if folds < 2:
        return float("nan")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2_000, random_state=seed),
    )
    probability = cross_val_predict(
        model,
        values,
        labels,
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(labels, probability))


def _distribution_distances(
    reference: np.ndarray,
    incoming: np.ndarray,
    seed: int,
) -> tuple[float, float, float]:
    reference, incoming = _balanced_samples(reference, incoming, seed)
    mean_distance = float(np.linalg.norm(reference.mean(axis=0) - incoming.mean(axis=0)))
    cross = cdist(reference, incoming)
    within_reference = pdist(reference)
    within_incoming = pdist(incoming)
    pooled_distances = np.r_[within_reference, within_incoming, cross.ravel()]
    positive = pooled_distances[pooled_distances > 0]
    bandwidth_squared = float(np.median(positive) ** 2) if len(positive) else 1.0
    gamma = 1 / max(2 * bandwidth_squared, 1e-12)
    kernel_cross = np.exp(-gamma * cross**2).mean()
    kernel_reference = np.exp(-gamma * cdist(reference, reference) ** 2).mean()
    kernel_incoming = np.exp(-gamma * cdist(incoming, incoming) ** 2).mean()
    mmd = float(max(kernel_reference + kernel_incoming - 2 * kernel_cross, 0.0))
    energy = float(
        2 * cross.mean() - cdist(reference, reference).mean() - cdist(incoming, incoming).mean()
    )
    return mean_distance, mmd, energy


def build_experiment_signals(
    prefix_features: pd.DataFrame,
    predictions_at_window: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    prediction_frame = predictions_at_window.copy()
    prediction_frame["entropy"] = binary_entropy(prediction_frame["probability"])
    prediction_frame["ood_ratio"] = (
        prediction_frame["ood_score"] / prediction_frame["ood_threshold"]
    )
    rows: list[dict[str, object]] = []
    groups = sorted(prefix_features["group_id"].astype(str).unique())
    slope_columns = [column for column in feature_columns if column.endswith("__slope")]
    for index, group in enumerate(groups):
        incoming = prefix_features[prefix_features["group_id"].astype(str) == group]
        reference = prefix_features[prefix_features["group_id"].astype(str) != group]
        reference_values, incoming_values = _scaled_samples(reference, incoming, feature_columns)
        mean_distance, mmd, energy = _distribution_distances(
            reference_values, incoming_values, seed + index
        )
        incoming_prediction = prediction_frame[prediction_frame["test_group"].astype(str) == group]
        if slope_columns:
            slope_indices = [feature_columns.index(column) for column in slope_columns]
            slope_norm = np.linalg.norm(incoming_values[:, slope_indices], axis=1)
        else:
            slope_norm = np.zeros(len(incoming_values))
        observations = incoming["observations_available"].to_numpy(float)
        rows.append(
            {
                "experiment": group,
                "two_sample_auc": _two_sample_auc(
                    reference_values, incoming_values, seed + 100 + index
                ),
                "standardized_mean_distance": mean_distance,
                "rbf_mmd": mmd,
                "energy_distance": energy,
                "unit_ood_mean": float(incoming_prediction["ood_ratio"].mean()),
                "unit_ood_p95": float(incoming_prediction["ood_ratio"].quantile(0.95)),
                "unit_ood_fraction": float(incoming_prediction["is_ood"].mean()),
                "prediction_entropy_mean": float(incoming_prediction["entropy"].mean()),
                "prediction_entropy_p90": float(incoming_prediction["entropy"].quantile(0.9)),
                "ensemble_disagreement_mean": float(incoming_prediction["cbes_std"].mean()),
                "ensemble_disagreement_p90": float(incoming_prediction["cbes_std"].quantile(0.9)),
                "ambiguous_prediction_fraction": float(
                    (incoming_prediction["cbes_prediction"] < 0).mean()
                ),
                "predicted_positive_fraction": float(
                    (incoming_prediction["probability"] >= 0.5).mean()
                ),
                "observation_count_mean": float(observations.mean()),
                "observation_count_cv": float(observations.std() / observations.mean()),
                "trajectory_slope_norm_mean": float(slope_norm.mean()),
                "trajectory_slope_norm_p90": float(np.quantile(slope_norm, 0.9)),
            }
        )
    return pd.DataFrame(rows)
