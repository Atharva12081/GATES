from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.neural_mea import DIVS, _canonical_compound, load_neural_mea
from gates.act.neural_mea_phase1 import load_phase1_mea

ENDPOINTS = (
    "firing_rate",
    "burst_rate",
    "active_electrodes",
    "bursting_electrodes",
    "burst_spike_percent",
    "network_spikes",
    "network_spike_percent",
    "correlation",
)

LEGACY_COLUMNS = {
    "firing_rate": "meanfiringrate",
    "burst_rate": "burst.per.min",
    "active_electrodes": "nAE",
    "bursting_electrodes": "nABE",
    "burst_spike_percent": "per.spikes.in.burst",
    "network_spikes": "ns.n",
    "network_spike_percent": "ns.percent.of.spikes.in.ns",
    "correlation": "r",
}

PHASE1_COLUMNS = {
    "firing_rate": "CCTE_Shafer_MEA_dev_firing_rate_mean",
    "burst_rate": "CCTE_Shafer_MEA_dev_burst_rate",
    "active_electrodes": "CCTE_Shafer_MEA_dev_active_electrodes_number",
    "bursting_electrodes": "CCTE_Shafer_MEA_dev_bursting_electrodes_number",
    "burst_spike_percent": "CCTE_Shafer_MEA_dev_per_burst_spike_percent",
    "network_spikes": "CCTE_Shafer_MEA_dev_network_spike_number",
    "network_spike_percent": "CCTE_Shafer_MEA_dev_per_network_spike_spike_percent",
    "correlation": "CCTE_Shafer_MEA_dev_correlation_coefficient_mean",
}


@dataclass(frozen=True)
class MultivariateMEA:
    responses: dict[str, np.ndarray]
    audit: dict[str, object]


def _contrast(values: pd.Series, controls: pd.Series, endpoint: str) -> pd.Series:
    if endpoint == "correlation":
        return values - controls
    return np.log1p(values.clip(lower=0)) - np.log1p(controls.clip(lower=0))


def _legacy_arrays(workbook: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    data = load_neural_mea(workbook)
    grids = data.surfaces[
        ["cohort", "compound", "dose", "div", "dose_rank", "time_rank"]
    ].drop_duplicates()
    frames = []
    for cohort, sheet in (
        ("NTP", "Tab 7-NTP Compound_RawValues"),
        ("TC", "Tab 9-TC Compound_RawValues"),
    ):
        frame = pd.read_excel(workbook, sheet_name=sheet).assign(cohort=cohort)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[raw["DIV"].isin(DIVS)].copy()
    raw["compound_group"] = raw["trt"].map(_canonical_compound)
    eligible = set(data.surfaces["compound_group"])
    raw = raw.loc[raw["compound_group"].isin(eligible)].copy()

    keys = ["cohort", "trt", "Plate.SN", "DIV"]
    for endpoint, column in LEGACY_COLUMNS.items():
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        control_table = (
            raw.loc[raw["dose"].eq(0), keys + [column]]
            .groupby(keys, as_index=False)[column]
            .median()
            .rename(columns={column: f"{column}_control"})
        )
        raw = raw.merge(control_table, on=keys, how="left", validate="many_to_one")
        raw[endpoint] = _contrast(raw[column], raw[f"{column}_control"], endpoint)
        raw = raw.drop(columns=[f"{column}_control"])
    aggregate = raw.groupby(
        ["cohort", "trt", "dose", "DIV"], as_index=False
    )[list(ENDPOINTS)].median()
    aggregate = aggregate.merge(
        grids,
        left_on=["cohort", "trt", "dose", "DIV"],
        right_on=["cohort", "compound", "dose", "div"],
        how="inner",
        validate="one_to_one",
    )
    responses = {}
    for (cohort, compound), group in aggregate.groupby(["cohort", "trt"], sort=True):
        ordered = group.sort_values(["time_rank", "dose_rank"])
        if len(ordered) != 32 or not np.isfinite(ordered[list(ENDPOINTS)]).all().all():
            continue
        responses[f"{cohort}::{compound}"] = ordered[list(ENDPOINTS)].to_numpy(float)
    return responses, {
        "eligible_runs": len(responses),
        "endpoints": list(ENDPOINTS),
        "normalization": "plate/DIV matched-control log1p contrast; correlation raw difference",
    }


def _phase1_arrays(
    workbook: Path, legacy_groups: set[str]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    data = load_phase1_mea(workbook, legacy_groups)
    raw = pd.read_excel(workbook)
    raw = raw.loc[raw["TIME_POINT"].isin([f"DIV{div}" for div in DIVS])].copy()
    raw["div"] = raw["TIME_POINT"].str.removeprefix("DIV").astype(int)
    eligible_runs = set(
        data.surfaces[["DTXSID", "CHEMICAL_NAME", "PLATE_NAME"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    raw = raw.loc[
        [
            (row.DTXSID, row.CHEMICAL_NAME, row.PLATE_NAME) in eligible_runs
            or row.WELL_TYPE == "VC"
            for row in raw.itertuples(index=False)
        ]
    ].copy()
    control_keys = ["PLATE_MAP_NAME", "CHEMICAL_NAME", "div"]
    treated = raw.loc[raw["WELL_TYPE"].eq("S")].copy()
    for endpoint, column in PHASE1_COLUMNS.items():
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
        controls = (
            raw.loc[raw["WELL_TYPE"].eq("VC")]
            .groupby(control_keys, as_index=False)[column]
            .median()
            .rename(columns={"CHEMICAL_NAME": "VEHICLE", column: f"{column}_control"})
        )
        treated[column] = pd.to_numeric(treated[column], errors="coerce")
        treated = treated.merge(
            controls,
            on=["PLATE_MAP_NAME", "VEHICLE", "div"],
            how="left",
            validate="many_to_one",
        )
        treated[endpoint] = _contrast(treated[column], treated[f"{column}_control"], endpoint)
        treated = treated.drop(columns=[f"{column}_control"])
    keys = ["DTXSID", "CHEMICAL_NAME", "PLATE_NAME"]
    aggregate = treated.groupby(keys + ["DOSE", "div"], as_index=False)[
        list(ENDPOINTS)
    ].median()
    aggregate["dose_rank"] = aggregate.groupby(keys)["DOSE"].rank(method="dense").astype(int)
    aggregate["time_rank"] = aggregate["div"].map({value: i for i, value in enumerate(DIVS)})
    vehicle_rows = []
    for record in aggregate[keys].drop_duplicates().itertuples(index=False):
        for time_rank, div in enumerate(DIVS):
            vehicle_rows.append(
                {
                    "DTXSID": record.DTXSID,
                    "CHEMICAL_NAME": record.CHEMICAL_NAME,
                    "PLATE_NAME": record.PLATE_NAME,
                    "DOSE": 0.0,
                    "div": div,
                    "dose_rank": 0,
                    "time_rank": time_rank,
                    **{endpoint: 0.0 for endpoint in ENDPOINTS},
                }
            )
    aggregate = pd.concat([aggregate, pd.DataFrame(vehicle_rows)], ignore_index=True)
    responses = {}
    for (dtxsid, _compound, plate), group in aggregate.groupby(keys, sort=True):
        ordered = group.sort_values(["time_rank", "dose_rank"])
        if len(ordered) != 32 or not np.isfinite(ordered[list(ENDPOINTS)]).all().all():
            continue
        responses[f"P1::{dtxsid}::{plate}"] = ordered[list(ENDPOINTS)].to_numpy(float)
    return responses, {
        "eligible_runs": len(responses),
        "endpoints": list(ENDPOINTS),
        "normalization": "matched vehicle/plate-map/DIV log1p contrast; correlation raw difference",
    }


def load_multivariate_mea(legacy_workbook: Path, phase1_workbook: Path) -> MultivariateMEA:
    legacy_data = load_neural_mea(legacy_workbook)
    legacy, legacy_audit = _legacy_arrays(legacy_workbook)
    phase1, phase1_audit = _phase1_arrays(
        phase1_workbook, set(legacy_data.surfaces["compound_group"])
    )
    overlap = set(legacy) & set(phase1)
    if overlap:
        raise ValueError(f"duplicate run identifiers: {sorted(overlap)[:3]}")
    return MultivariateMEA(
        responses={**legacy, **phase1},
        audit={"legacy": legacy_audit, "phase1": phase1_audit},
    )
