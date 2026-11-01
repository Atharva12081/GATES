from __future__ import annotations

import json
from pathlib import Path

from gates.act.hesi_supplement_audit import run_hesi_supplement_audit

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = run_hesi_supplement_audit(
        ROOT / "data/hesi_2026_supplementary_raw",
        ROOT / "artifacts/act/research/source_shift",
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)
