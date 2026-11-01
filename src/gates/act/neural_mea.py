from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_SHEETS = (
    ("NTP", "Tab 7-NTP Compound_RawValues"),
    ("TC", "Tab 9-TC Compound_RawValues"),
)
DIVS = (5, 7, 9, 12)
DOSE_RANKS = tuple(range(8))


def _canonical_compound(name: str) -> str:
    """Conservatively bind known biological-repeat labels to one split group."""
    value = re.sub(r"\s+-\s+(ToxCast G-8|NTP)$", "", str(name), flags=re.IGNORECASE)
    aliases = {
        "aspirin": "acetylsalicylic acid",
        "tetraethylthiuram disulfide": "disulfiram",
        "estradiol": "17beta estradiol",
    }
    return aliases.get(value.casefold(), value.casefold())


def _stable_split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "calibration"
    return "test"


@dataclass(frozen=True)
class NeuralMEAData:
    surfaces: pd.DataFrame
    audit: dict[str, object]


def load_neural_mea(workbook: Path, endpoint: str = "meanfiringrate") -> NeuralMEAData:
    """Load complete 8-dose x 4-DIV compound surfaces from the EPA NFA workbook.

    Responses are median technical-replicate contrasts against the matched zero-dose
    well on the same plate and DIV: log(1 + treated) - log(1 + vehicle).
    """
    frames = []
    for cohort, sheet in RAW_SHEETS:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        required = {"Plate.SN", "DIV", "trt", "dose", endpoint}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{sheet} missing columns: {sorted(missing)}")
        frames.append(frame.assign(cohort=cohort))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[raw["DIV"].isin(DIVS)].copy()
    rows: list[dict[str, object]] = []
    for (cohort, compound, plate, div), group in raw.groupby(
        ["cohort", "trt", "Plate.SN", "DIV"], sort=True
    ):
        controls = group.loc[group["dose"].eq(0), endpoint].dropna()
        if controls.empty:
            continue
        control = float(controls.median())
        for record in group.itertuples(index=False):
            value = getattr(record, endpoint)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "compound": compound,
                    "compound_group": _canonical_compound(compound),
                    "plate": plate,
                    "div": int(div),
                    "dose": float(record.dose),
                    "response": float(np.log1p(float(value)) - np.log1p(control)),
                }
            )
    replicate_values = pd.DataFrame(rows)
    aggregate = (
        replicate_values.groupby(
            ["cohort", "compound", "compound_group", "dose", "div"], as_index=False
        )["response"]
        .median()
        .sort_values(["cohort", "compound", "div", "dose"])
    )
    cell_counts = aggregate.groupby(["cohort", "compound"]).size()
    dose_counts = aggregate.groupby(["cohort", "compound"])["dose"].nunique()
    eligible = cell_counts.index[(cell_counts.eq(32) & dose_counts.eq(8)).to_numpy()]
    eligible_keys = set(eligible.tolist())
    aggregate = aggregate.loc[
        [
            (cohort, compound) in eligible_keys
            for cohort, compound in aggregate[["cohort", "compound"]].itertuples(
                index=False, name=None
            )
        ]
    ].copy()
    aggregate["dose_rank"] = (
        aggregate.groupby(["cohort", "compound"])["dose"].rank(method="dense").astype(int) - 1
    )
    aggregate["time_rank"] = aggregate["div"].map({value: i for i, value in enumerate(DIVS)})
    aggregate["split"] = aggregate["compound_group"].map(_stable_split)
    split_groups = aggregate.groupby("split")["compound_group"].nunique().to_dict()
    split_runs = (
        aggregate.groupby("split")
        .apply(
            lambda frame: frame[["cohort", "compound"]].drop_duplicates().shape[0],
            include_groups=False,
        )
        .to_dict()
    )
    audit = {
        "endpoint": endpoint,
        "raw_rows": int(len(raw)),
        "raw_compound_runs": int(raw[["cohort", "trt"]].drop_duplicates().shape[0]),
        "eligible_complete_runs": int(aggregate[["cohort", "compound"]].drop_duplicates().shape[0]),
        "canonical_groups": int(aggregate["compound_group"].nunique()),
        "technical_replicates_per_cell": replicate_values.groupby(
            ["cohort", "compound", "dose", "div"]
        )
        .size()
        .value_counts()
        .sort_index()
        .to_dict(),
        "split_groups": {str(key): int(value) for key, value in split_groups.items()},
        "split_runs": {str(key): int(value) for key, value in split_runs.items()},
        "normalization": "median plate-matched log1p(treated) minus log1p(zero-dose)",
        "grid": "8 within-compound dose ranks x DIV 5, 7, 9, 12",
    }
    return NeuralMEAData(aggregate.reset_index(drop=True), audit)


def surface_arrays(data: NeuralMEAData) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    meta_rows = []
    values: dict[str, np.ndarray] = {}
    for (cohort, compound), group in data.surfaces.groupby(["cohort", "compound"], sort=True):
        ordered = group.sort_values(["time_rank", "dose_rank"])
        if len(ordered) != 32:
            raise ValueError(f"incomplete surface for {cohort}/{compound}")
        run_id = f"{cohort}::{compound}"
        values[run_id] = ordered["response"].to_numpy(float)
        first = ordered.iloc[0]
        meta_rows.append(
            {
                "run_id": run_id,
                "cohort": cohort,
                "compound": compound,
                "compound_group": first["compound_group"],
                "split": first["split"],
            }
        )
    return pd.DataFrame(meta_rows), values


def neural_coordinates() -> np.ndarray:
    return np.array(
        [(dose_rank / 7, time_rank / 3) for time_rank in range(4) for dose_rank in range(8)],
        dtype=float,
    )
