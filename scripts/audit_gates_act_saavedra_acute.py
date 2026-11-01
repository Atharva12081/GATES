from __future__ import annotations

import json
from pathlib import Path

from gates.act.saavedra_acute import extract_wmfr

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/research/saavedra_2021_acute_raw/MEA_data_figures_normalized.xlsx"
ARTIFACT = ROOT / "artifacts/act/research/external_evidence"


def main() -> None:
    frame, recomputed = extract_wmfr(WORKBOOK)
    stored = json.loads((ARTIFACT / "saavedra_audit.json").read_text())
    summary = json.loads((ARTIFACT / "saavedra_summary.json").read_text())
    checks = {
        "audit_reproduces_exactly": recomputed == stored,
        "official_bytes_match": recomputed["checks"]["official_file_hash_matches"],
        "replicate_count_claim_fails": not recomputed["checks"][
            "treatment_rows_have_five_or_six_values"
        ],
        "plate_or_well_identifiers_absent": not recomputed["checks"][
            "plate_or_well_identifiers_present"
        ],
        "cross_concentration_identity_absent": not recomputed["checks"][
            "stable_cross_concentration_replicate_identity"
        ],
        "audit_status_fail": recomputed["status"] == "FAIL",
        "analysis_quarantined": summary["analysis_status"] == "REJECTED_AFTER_AUDIT",
        "confirmation_gate_rejected": summary["prespecified_gate"] == "NO-GO_DATA_INELIGIBLE",
        "no_longitudinal_claim": not recomputed["structure"]["longitudinal_dose_time_eligible"],
        "extracted_rows_nonempty": len(frame) == 823,
    }
    receipt = {
        "schema": "gates-act-saavedra-acute-independent-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "disposition": (
            "Preserve the numerical files as rejected preliminary evidence; exclude them from "
            "independent confirmation, model selection, and canonical claims."
        ),
    }
    (ARTIFACT / "saavedra_independent_audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
