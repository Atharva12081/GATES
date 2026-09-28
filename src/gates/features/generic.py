from __future__ import annotations

import numpy as np
import pandas as pd

from gates.data.base import LongitudinalDataset


def _slope(values: np.ndarray, times: np.ndarray) -> float:
    keep = np.isfinite(values) & np.isfinite(times)
    if keep.sum() < 2 or np.ptp(times[keep]) == 0:
        return 0.0
    centered_time = times[keep] - times[keep].mean()
    centered_values = values[keep] - values[keep].mean()
    denominator = float(np.sum(centered_time**2))
    return float(np.sum(centered_time * centered_values) / denominator)


def generic_prefix_features(dataset: LongitudinalDataset, decision_time: float) -> pd.DataFrame:
    prefix = dataset.frame.loc[dataset.frame["time"] <= decision_time]
    rows: list[dict[str, object]] = []
    for unit_id, group in prefix.groupby("unit_id", sort=True):
        group = group.sort_values("time")
        times = group["time"].to_numpy(float)
        row: dict[str, object] = {
            "unit_id": unit_id,
            "group_id": str(group["group_id"].iloc[0]),
            "endpoint": int(group["endpoint"].iloc[0]),
            "decision_time": float(decision_time),
            "observations_available": int(len(group)),
            "last_observation_time": float(times[-1]),
        }
        for column in dataset.feature_columns:
            values = group[column].to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"{column}__last"] = float(finite[-1]) if len(finite) else np.nan
            row[f"{column}__mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{column}__delta"] = float(finite[-1] - finite[0]) if len(finite) >= 2 else 0.0
            row[f"{column}__slope"] = _slope(values, times)
        rows.append(row)
    return pd.DataFrame(rows)


def generic_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "unit_id",
        "group_id",
        "endpoint",
        "decision_time",
        "observations_available",
        "last_observation_time",
    }
    return [column for column in frame.columns if column not in excluded]
