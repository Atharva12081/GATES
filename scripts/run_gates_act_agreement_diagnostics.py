from __future__ import annotations

import json
from pathlib import Path

from gates.act.agreement_diagnostics import run_agreement_diagnostics

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    artifact = ROOT / "artifacts/act/research/competitor_transfer"
    result = run_agreement_diagnostics(
        artifact / "h1_oof_predictions_with_selected.parquet",
        artifact / "h6_agreement_triage.json",
        artifact,
    )
    print(json.dumps(result, indent=2))
