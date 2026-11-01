from __future__ import annotations

import json
from pathlib import Path

from gates.act.opportunity_audit import run_opportunity_audit

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = run_opportunity_audit(
        ROOT / "artifacts/act/research/direct_discovery/direct_discovery_decisions.parquet",
        ROOT / "artifacts/act/research/scientific_opportunity",
    )
    print(json.dumps(result, indent=2))
