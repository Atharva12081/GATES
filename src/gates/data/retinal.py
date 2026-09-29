from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gates.data.base import LongitudinalDataset

RETINAL_FEATURES = (
    "area",
    "axis_major_length",
    "axis_minor_length",
    "eccentricity",
    "equivalent_diameter_area",
    "extent",
    "feret_diameter_max",
    "intensity_mean",
    "intensity_std",
    "perimeter",
    "solidity",
    "circularity",
    "blur",
    "roi_contrast",
    "aspect_ratio",
)


def decode_binary_labels(values: pd.Series, *, column: str) -> pd.Series:
    """Decode documented yes/no labels, rejecting missing or unexpected values."""
    normalized = values.astype("string").str.strip().str.lower()
    decoded = normalized.map({"yes": True, "no": False})
    invalid = sorted(normalized[decoded.isna()].drop_duplicates().tolist())
    if invalid:
        raise ValueError(f"{column} contains unsupported labels: {invalid}")
    return decoded.astype(bool)


def load_retinal_morphometrics(path: Path, endpoint: str = "RPE_Final") -> LongitudinalDataset:
    if endpoint not in {"RPE_Final", "Lens_Final"}:
        raise ValueError("endpoint must be RPE_Final or Lens_Final")
    columns = ["experiment", "well", "loop", endpoint, *RETINAL_FEATURES]
    frame = pd.read_csv(path, usecols=columns)
    frame["time"] = frame["loop"].str.extract(r"(\d+)").astype(float) / 2.0
    frame["unit_id"] = frame["experiment"].astype(str) + "|" + frame["well"].astype(str)
    frame["group_id"] = frame["experiment"].astype(str)
    endpoint_values = frame[endpoint]
    if endpoint_values.dtype == bool:
        frame["endpoint"] = endpoint_values.astype(int)
    else:
        frame["endpoint"] = decode_binary_labels(endpoint_values, column=endpoint).astype(int)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    dataset = LongitudinalDataset(
        name=f"orgAInoid_{endpoint}",
        frame=frame[["unit_id", "group_id", "time", "endpoint", *RETINAL_FEATURES]].copy(),
        feature_columns=RETINAL_FEATURES,
        final_time=72.0,
        endpoint_name=endpoint,
    )
    dataset.validate()
    return dataset
