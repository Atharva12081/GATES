from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_pi_assay(raw_dir: Path) -> pd.DataFrame:
    """Load the public OrganoID PI assay as a complete replicate x dose x time cube."""
    frame = pd.read_csv(raw_dir / "PIMeasurements.csv")
    frame.columns = [column.strip().lower() for column in frame.columns]
    frame = frame.rename(columns={"fluorescence": "response"})
    frame = frame.astype({"dosage": float, "replicate": int, "time": float, "response": float})
    if frame.duplicated(["replicate", "dosage", "time"]).any():
        raise ValueError("PI assay contains duplicate replicate-dose-time cells")
    expected = frame.groupby("replicate").size()
    if expected.nunique() != 1:
        raise ValueError("PI assay must expose the same grid for every replicate")
    return frame.sort_values(["replicate", "dosage", "time"]).reset_index(drop=True)


def assay_grid(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[["dosage", "time"]]
        .drop_duplicates()
        .sort_values(["dosage", "time"])
        .reset_index(drop=True)
    )


def coordinate_matrix(grid: pd.DataFrame) -> np.ndarray:
    dose = np.log1p(grid["dosage"].to_numpy(float))
    time = grid["time"].to_numpy(float)
    dose_scale = max(float(dose.max()), 1.0)
    time_scale = max(float(time.max()), 1.0)
    return np.column_stack([dose / dose_scale, time / time_scale])


def response_vector(frame: pd.DataFrame, replicate: int, grid: pd.DataFrame) -> np.ndarray:
    selected = frame.loc[frame["replicate"] == replicate, ["dosage", "time", "response"]]
    merged = grid.merge(selected, on=["dosage", "time"], how="left", validate="one_to_one")
    if merged["response"].isna().any():
        raise ValueError(f"replicate {replicate} does not cover the complete assay grid")
    return np.log1p(merged["response"].to_numpy(float))
