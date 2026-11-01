from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta

SPLIT_SALT = "gates-act-vision-v1"


def brain_organoid_split(metadata: pd.DataFrame) -> pd.Series:
    """Return a clone-balanced, specimen-disjoint deterministic split."""

    required = {"org_id", "Clone"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"metadata must contain {sorted(required)}")
    unique = metadata[["org_id", "Clone"]].drop_duplicates()
    if unique.groupby("org_id")["Clone"].nunique().max() != 1:
        raise ValueError("each organoid must map to exactly one clone")
    mapping: dict[int, str] = {}
    for clone, group in unique.groupby("Clone", sort=True):
        ids = sorted(
            group["org_id"].astype(int),
            key=lambda org_id: hashlib.sha256(
                f"{SPLIT_SALT}:{clone}:{org_id}".encode()
            ).hexdigest(),
        )
        if len(ids) < 4:
            raise ValueError("each clone needs at least four organoids")
        n_test = max(1, round(len(ids) * 0.25))
        n_calibration = max(1, round(len(ids) * 0.1875))
        n_train = len(ids) - n_calibration - n_test
        for org_id in ids[:n_train]:
            mapping[org_id] = "train"
        for org_id in ids[n_train : n_train + n_calibration]:
            mapping[org_id] = "calibration"
        for org_id in ids[n_train + n_calibration :]:
            mapping[org_id] = "test"
    return metadata["org_id"].astype(int).map(mapping)


def ooc_chronological_split(metadata: pd.DataFrame) -> pd.Series:
    """Use acquisition-date groups for a realistic future-batch test."""

    if "imageID" not in metadata:
        raise ValueError("metadata must contain imageID")
    dates = metadata["imageID"].astype(str).str.split("_").str[0]
    unique_dates = sorted(dates.unique())
    if len(unique_dates) != 59:
        raise ValueError(f"expected 59 acquisition dates, found {len(unique_dates)}")
    mapping = {
        date: "train" if index < 39 else "calibration" if index < 49 else "test"
        for index, date in enumerate(unique_dates)
    }
    return dates.map(mapping)


def dice_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_bool = np.asarray(truth, dtype=bool)
    prediction_bool = np.asarray(prediction, dtype=bool)
    denominator = int(truth_bool.sum() + prediction_bool.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(truth_bool, prediction_bool).sum() / denominator)


def one_sided_binomial_upper(
    failures: int,
    trials: int,
    alpha: float = 0.05,
) -> float:
    if trials <= 0:
        return 1.0
    if not 0 <= failures <= trials:
        raise ValueError("failures must be between zero and trials")
    if failures == trials:
        return 1.0
    return float(beta.ppf(1 - alpha, failures + 1, trials - failures))


@dataclass(frozen=True)
class SelectiveThreshold:
    threshold: float
    coverage: float
    failures: int
    accepted: int
    upper_risk: float
    authorized: bool


def calibrate_selective_threshold(
    uncertainty: Iterable[float],
    failure: Iterable[bool],
    maximum_risk: float = 0.10,
    alpha: float = 0.05,
) -> SelectiveThreshold:
    """Maximize coverage subject to a finite-sample upper risk bound.

    Lower uncertainty is better. If no threshold meets the bound, return a
    fail-closed rule that accepts nothing.
    """

    scores = np.asarray(list(uncertainty), dtype=float)
    failures = np.asarray(list(failure), dtype=bool)
    if len(scores) != len(failures) or not len(scores):
        raise ValueError("uncertainty and failure must be non-empty and aligned")
    best: SelectiveThreshold | None = None
    for threshold in np.unique(scores):
        accepted_mask = scores <= threshold
        accepted = int(accepted_mask.sum())
        count = int(failures[accepted_mask].sum())
        upper = one_sided_binomial_upper(count, accepted, alpha)
        candidate = SelectiveThreshold(
            threshold=float(threshold),
            coverage=accepted / len(scores),
            failures=count,
            accepted=accepted,
            upper_risk=upper,
            authorized=upper <= maximum_risk,
        )
        if candidate.authorized and (best is None or candidate.accepted > best.accepted):
            best = candidate
    if best is not None:
        return best
    return SelectiveThreshold(
        threshold=float("-inf"),
        coverage=0.0,
        failures=0,
        accepted=0,
        upper_risk=1.0,
        authorized=False,
    )


def grouped_bootstrap_interval(
    values: Iterable[float],
    groups: Iterable[object],
    *,
    seed: int = 20261024,
    draws: int = 5000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    frame = pd.DataFrame({"value": list(values), "group": list(groups)})
    if frame.empty or frame["group"].nunique() < 2:
        raise ValueError("at least two groups are required")
    grouped = [part["value"].to_numpy(float) for _, part in frame.groupby("group")]
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for index in range(draws):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        estimates[index] = np.concatenate([grouped[item] for item in sampled]).mean()
    tail = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(estimates, [tail, 1 - tail]))
