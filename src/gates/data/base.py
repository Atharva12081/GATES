from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"unit_id", "group_id", "time", "endpoint"}


@dataclass(frozen=True)
class LongitudinalDataset:
    name: str
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    final_time: float
    endpoint_name: str

    def validate(self) -> None:
        missing = REQUIRED_COLUMNS - set(self.frame.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        if self.frame.duplicated(["unit_id", "time"]).any():
            raise ValueError("unit-time rows must be unique")
        if (self.frame.groupby("unit_id")["group_id"].nunique() > 1).any():
            raise ValueError("group_id must be constant within an experimental unit")
        if (self.frame.groupby("unit_id")["endpoint"].nunique() > 1).any():
            raise ValueError("endpoint must be constant within an experimental unit")
        if not set(self.feature_columns).issubset(self.frame.columns):
            raise ValueError("one or more declared feature columns are missing")
        if not np.isfinite(self.frame["time"].to_numpy(float)).all():
            raise ValueError("time contains non-finite values")

    @property
    def independent_units(self) -> int:
        return int(self.frame["unit_id"].nunique())

    @property
    def independent_groups(self) -> int:
        return int(self.frame["group_id"].nunique())
