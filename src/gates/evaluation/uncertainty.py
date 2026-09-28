from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    method: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
        }


def exact_binomial_interval(successes: int, total: int, confidence: float = 0.95) -> Interval:
    if total <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), "undefined")
    result = binomtest(successes, total)
    interval = result.proportion_ci(confidence_level=confidence, method="exact")
    return Interval(successes / total, float(interval.low), float(interval.high), "Clopper-Pearson")


def group_bootstrap_interval(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> Interval:
    groups = frame[group_column].drop_duplicates().to_numpy()
    if len(groups) < 2:
        return Interval(
            float(frame[value_column].mean()),
            float("nan"),
            float("nan"),
            "insufficient groups",
        )
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    grouped = {group: frame.loc[frame[group_column] == group] for group in groups}
    for _ in range(samples):
        selected = rng.choice(groups, size=len(groups), replace=True)
        bootstrap = pd.concat([grouped[group] for group in selected], ignore_index=True)
        estimates.append(float(bootstrap[value_column].mean()))
    alpha = 1 - confidence
    return Interval(
        float(frame[value_column].mean()),
        float(np.quantile(estimates, alpha / 2)),
        float(np.quantile(estimates, 1 - alpha / 2)),
        f"percentile group bootstrap ({len(groups)} groups; exploratory)",
    )
