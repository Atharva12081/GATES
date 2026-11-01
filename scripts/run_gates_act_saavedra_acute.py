from __future__ import annotations

import json
from pathlib import Path

from gates.act.saavedra_acute import run_saavedra_benchmark

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = run_saavedra_benchmark(
        ROOT / "data/research/saavedra_2021_acute_raw/MEA_data_figures_normalized.xlsx",
        ROOT / "artifacts/act/research/external_evidence",
    )
    print(json.dumps(result, indent=2))
