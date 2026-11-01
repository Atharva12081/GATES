from __future__ import annotations

import json
from pathlib import Path

from gates.act.high_dose_falsification import run_high_dose_falsification

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = run_high_dose_falsification(
        ROOT / "artifacts/act/research/direct_discovery/direct_discovery_decisions.parquet",
        ROOT / "artifacts/act/research/high_dose_falsification",
    )
    print(json.dumps(result, indent=2))
