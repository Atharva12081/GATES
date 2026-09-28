from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = (
    "pi_fluorescence",
    "organoid_count",
    "organoid_fluorescence",
    "area_mean",
    "area_total",
    "circularity_mean",
    "solidity_mean",
    "eccentricity_mean",
)


def _slope(values: np.ndarray, times: np.ndarray) -> float:
    keep = np.isfinite(values) & np.isfinite(times)
    if keep.sum() < 2 or np.ptp(times[keep]) == 0:
        return 0.0
    return float(np.polyfit(times[keep], values[keep], 1)[0])


def prefix_features(longitudinal: pd.DataFrame, decision_time: int) -> pd.DataFrame:
    prefix = longitudinal.loc[longitudinal["time"] <= decision_time].copy()
    rows: list[dict[str, object]] = []
    for unit_id, group in prefix.groupby("unit_id", sort=True):
        group = group.sort_values("time")
        times = group["time"].to_numpy(float)
        row: dict[str, object] = {
            "unit_id": unit_id,
            "dosage": float(group["dosage"].iloc[0]),
            "replicate": int(group["replicate"].iloc[0]),
            "viability": float(group["viability"].iloc[0]),
            "decision_time": int(decision_time),
            "observations_used": int(len(group)),
        }
        for signal in SIGNALS:
            values = group[signal].to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"{signal}__last"] = float(finite[-1]) if len(finite) else np.nan
            row[f"{signal}__mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{signal}__delta"] = float(finite[-1] - finite[0]) if len(finite) >= 2 else 0.0
            row[f"{signal}__slope"] = _slope(values, times)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"unit_id", "dosage", "replicate", "viability", "decision_time"}
    return [column for column in frame.columns if column not in excluded]
