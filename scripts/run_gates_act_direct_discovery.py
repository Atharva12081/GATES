from __future__ import annotations

import json
from pathlib import Path

from gates.act.direct_discovery import run_direct_discovery_benchmark

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    artifact = ROOT / "artifacts/act/research/competitor_transfer"
    result = run_direct_discovery_benchmark(
        artifact / "h1_oof_predictions_with_selected.parquet",
        artifact / "h6_agreement_triage.json",
        ROOT / "artifacts/act/research/direct_discovery",
    )
    print(json.dumps(result, indent=2))
