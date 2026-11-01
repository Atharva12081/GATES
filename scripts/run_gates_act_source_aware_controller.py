from __future__ import annotations

import json
from pathlib import Path

from gates.act.source_aware_controller import run_source_aware_controller

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    artifact = ROOT / "artifacts/act/research/competitor_transfer"
    result = run_source_aware_controller(
        artifact / "h1_oof_predictions_with_selected.parquet",
        artifact / "h6_agreement_triage.json",
        ROOT / "artifacts/act/research/source_shift",
    )
    print(json.dumps(result, indent=2))
