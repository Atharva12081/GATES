from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    return frame


def load_organoid(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    organoids = _normalize_columns(pd.read_csv(raw_dir / "OrganoidMeasurements.csv"))
    pi = _normalize_columns(pd.read_csv(raw_dir / "PIMeasurements.csv"))
    endpoints = _normalize_columns(pd.read_csv(raw_dir / "MTSAssay.csv"))
    for frame in (organoids, pi, endpoints):
        frame["dosage"] = frame["dosage"].astype(float)
        frame["replicate"] = frame["replicate"].astype(int)
    return organoids, pi, endpoints


def audit_dataset(raw_dir: Path) -> dict[str, object]:
    organoids, pi, endpoints = load_organoid(raw_dir)
    endpoint_units = endpoints[["dosage", "replicate"]].drop_duplicates()
    pi_units = pi[["dosage", "replicate"]].drop_duplicates()
    shared = endpoint_units.merge(pi_units, on=["dosage", "replicate"])
    return {
        "endpoint_units": int(len(endpoint_units)),
        "pi_units": int(len(pi_units)),
        "shared_units": int(len(shared)),
        "organoid_rows": int(len(organoids)),
        "pi_rows": int(len(pi)),
        "time_min": int(pi["time"].min()),
        "time_max": int(pi["time"].max()),
        "time_points": sorted(int(v) for v in pi["time"].unique()),
        "endpoint_missing": int(endpoints["viability"].isna().sum()),
        "duplicate_pi_rows": int(pi.duplicated(["dosage", "replicate", "time"]).sum()),
        "replicates_per_dose": endpoints.groupby("dosage")["replicate"].nunique().to_dict(),
    }


def build_longitudinal_table(raw_dir: Path) -> pd.DataFrame:
    organoids, pi, endpoints = load_organoid(raw_dir)

    morphology = organoids.groupby(["dosage", "replicate", "time"], as_index=False).agg(
        organoid_count=("organoid_id", "nunique"),
        organoid_fluorescence=("fluorescence", "sum"),
        area_mean=("area", "mean"),
        area_total=("area", "sum"),
        circularity_mean=("circularity", "mean"),
        solidity_mean=("solidity", "mean"),
        eccentricity_mean=("eccentricity", "mean"),
    )
    table = pi.rename(columns={"fluorescence": "pi_fluorescence"}).merge(
        morphology, how="left", on=["dosage", "replicate", "time"]
    )
    table = table.merge(endpoints, how="inner", on=["dosage", "replicate"])
    table["unit_id"] = table.apply(
        lambda row: f"dose={row['dosage']:g}|rep={int(row['replicate'])}", axis=1
    )
    table = table.replace([np.inf, -np.inf], np.nan).sort_values(["unit_id", "time"])
    return table.reset_index(drop=True)
