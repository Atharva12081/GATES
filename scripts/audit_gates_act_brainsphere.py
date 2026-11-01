from __future__ import annotations

import json
from pathlib import Path

from gates.act.brainsphere_audit import audit_brainsphere

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = audit_brainsphere(
        ROOT / "data/research/carstens_2025_brainsphere_supplement",
        ROOT / "artifacts/act/research/external_evidence",
    )
    print(json.dumps(result, indent=2))
