from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.neural_mea import DIVS, _canonical_compound

ENDPOINT = "CCTE_Shafer_MEA_dev_firing_rate_mean"
TIME_LABELS = {f"DIV{div}": div for div in DIVS}


def _v3_split(group: str) -> str:
    """Stable preregistered split for the independent Phase-1 release.

    Twenty hash buckets allocate 50%/15%/15%/20% to
    train/development/calibration/test. The identifier is the public DTXSID, so
    repeated plates and aliases cannot cross partitions.
    """
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 20
    if bucket < 10:
        return "train"
    if bucket < 13:
        return "development"
    if bucket < 16:
        return "calibration"
    return "test"


@dataclass(frozen=True)
class Phase1MEAData:
    surfaces: pd.DataFrame
    audit: dict[str, object]


def load_phase1_mea(workbook: Path, legacy_groups: set[str]) -> Phase1MEAData:
    """Load complete vehicle-plus-seven-dose Phase-1 neural MEA surfaces.

    Each treated well is normalized to the matching vehicle population from the
    same plate map and DIV. Public DTXSID values define split groups. Compounds
    overlapping the legacy EPA release are excluded from every v3 claim.
    """
    raw = pd.read_excel(workbook)
    required = {
        "DTXSID",
        "CHEMICAL_NAME",
        "VEHICLE",
        "PLATE_MAP_NAME",
        "PLATE_NAME",
        "WELL_TYPE",
        "DOSE",
        "REPLICATE_ID",
        "TIME_POINT",
        ENDPOINT,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Phase-1 workbook missing columns: {sorted(missing)}")
    raw = raw.loc[raw["TIME_POINT"].isin(TIME_LABELS)].copy()
    raw[ENDPOINT] = pd.to_numeric(raw[ENDPOINT], errors="coerce")
    raw = raw.loc[raw[ENDPOINT].notna()].copy()
    raw["div"] = raw["TIME_POINT"].map(TIME_LABELS).astype(int)

    controls = (
        raw.loc[raw["WELL_TYPE"].eq("VC")]
        .groupby(["PLATE_MAP_NAME", "CHEMICAL_NAME", "div"], as_index=False)[ENDPOINT]
        .median()
        .rename(columns={"CHEMICAL_NAME": "VEHICLE", ENDPOINT: "control"})
    )
    treated = raw.loc[raw["WELL_TYPE"].eq("S")].copy()
    treated["compound_group"] = treated["CHEMICAL_NAME"].map(_canonical_compound)
    treated = treated.loc[~treated["compound_group"].isin(legacy_groups)].copy()
    treated = treated.merge(
        controls,
        on=["PLATE_MAP_NAME", "VEHICLE", "div"],
        how="left",
        validate="many_to_one",
    )
    treated = treated.loc[treated["control"].notna()].copy()
    treated["response"] = np.log1p(treated[ENDPOINT].clip(lower=0)) - np.log1p(
        treated["control"].clip(lower=0)
    )

    aggregate = (
        treated.groupby(
            ["DTXSID", "CHEMICAL_NAME", "compound_group", "PLATE_NAME", "DOSE", "div"],
            as_index=False,
        )["response"]
        .median()
        .sort_values(["DTXSID", "PLATE_NAME", "div", "DOSE"])
    )
    keys = ["DTXSID", "CHEMICAL_NAME", "PLATE_NAME"]
    complete = aggregate.groupby(keys).agg(cells=("response", "size"), doses=("DOSE", "nunique"))
    eligible = set(complete.index[(complete["cells"] == 28) & (complete["doses"] == 7)])
    aggregate = aggregate.loc[
        [tuple(row) in eligible for row in aggregate[keys].itertuples(index=False, name=None)]
    ].copy()
    aggregate["dose_rank"] = aggregate.groupby(keys)["DOSE"].rank(method="dense").astype(int)
    aggregate["time_rank"] = aggregate["div"].map({value: i for i, value in enumerate(DIVS)})
    aggregate["split"] = aggregate["DTXSID"].map(_v3_split)

    vehicle_rows = []
    for record in (
        aggregate[keys + ["compound_group", "split"]].drop_duplicates().itertuples(index=False)
    ):
        for time_rank, div in enumerate(DIVS):
            vehicle_rows.append(
                {
                    "DTXSID": record.DTXSID,
                    "CHEMICAL_NAME": record.CHEMICAL_NAME,
                    "compound_group": record.compound_group,
                    "PLATE_NAME": record.PLATE_NAME,
                    "DOSE": 0.0,
                    "div": div,
                    "response": 0.0,
                    "dose_rank": 0,
                    "time_rank": time_rank,
                    "split": record.split,
                }
            )
    surfaces = pd.concat([aggregate, pd.DataFrame(vehicle_rows)], ignore_index=True)
    surfaces = surfaces.sort_values(keys + ["time_rank", "dose_rank"]).reset_index(drop=True)
    split_groups = surfaces.groupby("split")["DTXSID"].nunique().to_dict()
    split_runs = surfaces.groupby("split")[keys].apply(lambda x: len(x.drop_duplicates())).to_dict()
    audit = {
        "source_rows": int(len(raw)),
        "eligible_complete_runs": int(surfaces[keys].drop_duplicates().shape[0]),
        "eligible_dtxsid_groups": int(surfaces["DTXSID"].nunique()),
        "legacy_overlap_groups_excluded": int(
            raw.loc[raw["WELL_TYPE"].eq("S"), "CHEMICAL_NAME"]
            .map(_canonical_compound)
            .isin(legacy_groups)
            .groupby(raw.loc[raw["WELL_TYPE"].eq("S"), "DTXSID"])
            .any()
            .sum()
        ),
        "split_groups": {str(key): int(value) for key, value in split_groups.items()},
        "split_runs": {str(key): int(value) for key, value in split_runs.items()},
        "endpoint": ENDPOINT,
        "normalization": "median matched-vehicle log1p contrast",
        "grid": "vehicle plus 7 treated dose ranks x DIV 5, 7, 9, 12",
        "split_unit": "DTXSID; repeated plates never cross partitions",
    }
    return Phase1MEAData(surfaces, audit)


def phase1_surface_arrays(data: Phase1MEAData) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    meta_rows: list[dict[str, object]] = []
    values: dict[str, np.ndarray] = {}
    for (dtxsid, compound, plate), group in data.surfaces.groupby(
        ["DTXSID", "CHEMICAL_NAME", "PLATE_NAME"], sort=True
    ):
        ordered = group.sort_values(["time_rank", "dose_rank"])
        if len(ordered) != 32:
            raise ValueError(f"incomplete Phase-1 surface for {dtxsid}/{plate}")
        run_id = f"P1::{dtxsid}::{plate}"
        values[run_id] = ordered["response"].to_numpy(float)
        first = ordered.iloc[0]
        meta_rows.append(
            {
                "run_id": run_id,
                "cohort": "NTP2021",
                "compound": compound,
                "compound_group": dtxsid,
                "split": first["split"],
            }
        )
    return pd.DataFrame(meta_rows), values
