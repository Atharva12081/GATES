from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int
    endpoint_threshold: float
    target_risk: float
    minimum_stop_time: int
    decision_times: tuple[int, ...]
    train_fraction: float
    calibration_fraction: float
    ood_quantile: float
    bootstrap_samples: int

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = json.loads(path.read_text())
        raw["decision_times"] = tuple(raw["decision_times"])
        return cls(**raw)
