from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.neural_mea import DIVS, _canonical_compound


@dataclass(frozen=True)
class BrownMEAData:
    surfaces: pd.DataFrame
    audit: dict[str, object]


def load_brown_mea(csv_path: Path) -> BrownMEAData:
    """Load complete seven-dose, four-DIV runs from Brown et al. 2016."""
    raw = pd.read_csv(csv_path)
    required = {"Plate.SN", "DIV", "trt", "dose", "meanfiringrate"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Brown MEA data missing columns: {sorted(missing)}")
    raw = raw.loc[raw["DIV"].isin(DIVS)].copy()
    raw["meanfiringrate"] = pd.to_numeric(raw["meanfiringrate"], errors="coerce")
    raw = raw.loc[raw["meanfiringrate"].notna()].copy()
    aggregate = (
        raw.groupby(["Plate.SN", "trt", "dose", "DIV"], as_index=False)["meanfiringrate"]
        .median()
        .sort_values(["Plate.SN", "trt", "DIV", "dose"])
    )
    keys = ["Plate.SN", "trt"]
    complete = aggregate.groupby(keys).agg(
        cells=("meanfiringrate", "size"), doses=("dose", "nunique"), divs=("DIV", "nunique")
    )
    eligible = set(
        complete.index[complete["cells"].eq(28) & complete["doses"].eq(7) & complete["divs"].eq(4)]
    )
    aggregate = aggregate.loc[
        [tuple(row) in eligible for row in aggregate[keys].itertuples(index=False, name=None)]
    ].copy()
    controls = (
        aggregate.loc[aggregate["dose"].eq(0), keys + ["DIV", "meanfiringrate"]]
        .rename(columns={"meanfiringrate": "control"})
        .drop_duplicates(keys + ["DIV"])
    )
    aggregate = aggregate.merge(controls, on=keys + ["DIV"], validate="many_to_one")
    aggregate["response"] = np.log1p(aggregate["meanfiringrate"].clip(lower=0)) - np.log1p(
        aggregate["control"].clip(lower=0)
    )
    aggregate["dose_rank"] = aggregate.groupby(keys)["dose"].rank(method="dense").astype(int) - 1
    aggregate["time_rank"] = aggregate["DIV"].map({div: i for i, div in enumerate(DIVS)})
    aggregate["compound_group"] = aggregate["trt"].map(_canonical_compound)
    audit = {
        "raw_rows": int(len(raw)),
        "complete_runs": int(len(eligible)),
        "fresh_compound_groups": int(aggregate["compound_group"].nunique()),
        "technical_replicates_per_cell": raw.groupby(["Plate.SN", "trt", "dose", "DIV"])
        .size()
        .value_counts()
        .sort_index()
        .to_dict(),
        "endpoint": "meanfiringrate",
        "normalization": "median within-run log1p dose minus log1p zero-dose",
        "grid": "zero dose plus 6 treated dose ranks x DIV 5, 7, 9, 12",
    }
    return BrownMEAData(aggregate, audit)


def brown_surface_arrays(data: BrownMEAData):
    meta_rows = []
    values: dict[str, np.ndarray] = {}
    for (plate, compound), group in data.surfaces.groupby(["Plate.SN", "trt"], sort=True):
        ordered = group.sort_values(["time_rank", "dose_rank"])
        run_id = f"BROWN::{plate}::{compound}"
        values[run_id] = ordered["response"].to_numpy(float)
        meta_rows.append(
            {
                "run_id": run_id,
                "compound": compound,
                "compound_group": _canonical_compound(compound),
            }
        )
    return pd.DataFrame(meta_rows), values


def seven_dose_surface(values: np.ndarray) -> np.ndarray:
    """Interpolate an 8-rank training surface to vehicle plus six treated ranks."""
    source = np.asarray(values, dtype=float).reshape(4, 8)
    target = np.zeros((4, 7), dtype=float)
    target[:, 0] = source[:, 0]
    source_rank = np.arange(1, 8, dtype=float)
    target_rank = np.linspace(1, 7, 6)
    for time_rank in range(4):
        target[time_rank, 1:] = np.interp(target_rank, source_rank, source[time_rank, 1:])
    return target.ravel()


def seven_dose_coordinates() -> np.ndarray:
    return np.asarray(
        [(dose_rank / 6, time_rank / 3) for time_rank in range(4) for dose_rank in range(7)],
        dtype=float,
    )
