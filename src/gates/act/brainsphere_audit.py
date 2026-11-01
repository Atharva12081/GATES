from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook

EXPECTED_SHA256 = {
    "MOESM1_ESM.xlsx": "097bbf92ea4f3b81fddce98d6e1647f1c22d80724506aa4a0cb3decd35b1b2d2",
    "MOESM2_ESM.pdf": "e52ac1dc3e1d6fbc38d54e2c6126f991125ae2c8919856512e9e3af0e1e01b77",
    "MOESM3_ESM.pdf": "75c39f16a31133e1d5364fcec22e4755d33a8087cbcf0f72cdad7b7b9bb8f0f3",
    "MOESM4_ESM.jpg": "5b272df6b6ef3f5db29d2d82f894113115b3c282bca08be0bf290ff91e20c7e2",
    "MOESM5_ESM.pdf": "60051dc2ce713cb5911947462d9e59f7f0ef4aa4e46feb843e794a8401da22ae",
    "carstens_2025_article.pdf": (
        "a5eb38ea517ffde3986346c5ca8748f71796277d818f8c0a2d49fd65c4e7993c"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_brainsphere(directory: Path, output: Path) -> dict[str, object]:
    hashes = {name: _sha256(directory / name) for name in EXPECTED_SHA256}
    workbook = load_workbook(directory / "MOESM1_ESM.xlsx", read_only=True, data_only=True)
    sheets = []
    raw_identity_columns = {
        "well",
        "well_id",
        "wellplate.id",
        "plate",
        "experiment",
        "spheroid",
        "day",
        "recording_day",
        "concentration",
        "response",
    }
    raw_table_candidate = False
    for sheet in workbook.worksheets:
        header = [str(cell.value).strip().casefold() for cell in sheet[1] if cell.value is not None]
        overlap = sorted(raw_identity_columns & set(header))
        raw_table_candidate |= len(overlap) >= 5
        sheets.append(
            {
                "sheet": sheet.title,
                "rows": sheet.max_row,
                "columns": sheet.max_column,
                "header": [cell.value for cell in sheet[1] if cell.value is not None],
                "raw_identity_fields_found": overlap,
            }
        )
    article_text = (directory / "carstens_2025_article.txt").read_text(errors="replace")
    supplement5_text = (directory / "MOESM5_ESM.txt").read_text(errors="replace")
    checks = {
        "official_files_match_locked_hashes": hashes == EXPECTED_SHA256,
        "article_reports_seven_longitudinal_days": all(
            token in article_text for token in ("days 1, 3, 5, 7, 9, 11 and 13", "seven recording")
        ),
        "article_reports_at_least_three_biological_replicates": (
            "at least three biological" in article_text
            and "one well per plate" in article_text
            and "separate experiments" in article_text
        ),
        "article_reports_one_vehicle_control_per_plate": (
            "one control well per" in article_text and "treated with DMSO" in article_text
        ),
        "article_reports_full_raw_data_only_on_request": (
            "full raw dataset is available from" in article_text
            and "reasonable request" in article_text
        ),
        "xlsx_has_no_row_level_dose_time_identity_table": not raw_table_candidate,
        "supplement5_is_graphical_trajectory_output": (
            "M03869" in supplement5_text
            and "Spheroid Number" in supplement5_text
            and "Day" in supplement5_text
        ),
    }
    eligible = bool(
        checks["official_files_match_locked_hashes"]
        and checks["article_reports_seven_longitudinal_days"]
        and checks["article_reports_at_least_three_biological_replicates"]
        and not checks["xlsx_has_no_row_level_dose_time_identity_table"]
    )
    result = {
        "schema": "gates-act-carstens-brainsphere-eligibility-v1",
        "source": {
            "doi": "10.1007/s00204-025-04043-x",
            "article_url": "https://link.springer.com/article/10.1007/s00204-025-04043-x",
            "license": "CC BY 4.0",
            "public_supplement_files": 5,
            "sha256": hashes,
        },
        "reported_design": {
            "chemicals": 10,
            "exposure_days": 13,
            "recording_days": [1, 3, 5, 7, 9, 11, 13],
            "biological_replicates": "at least three wells on separate plates/experiments",
            "technical_replicates": "multiple BrainSpheres/ROIs within each well",
            "control": "one DMSO well per plate",
            "primary_public_analysis": (
                "per-ROI trajectories integrated to AUC, averaged within well and then used for "
                "concentration-response modeling"
            ),
        },
        "workbook": {"sheets": sheets, "raw_table_candidate": raw_table_candidate},
        "checks": checks,
        "eligibility": "PASS" if eligible else "FAIL",
        "blocker": (
            "The public XLSX contains baseline variability, endpoint mappings and fitted hit/AC50 "
            "summaries, not per-well dose-by-day responses. Supplemental PDFs show plate/spheroid "
            "trajectories graphically but do not expose a machine-readable numerical table. "
            "Digitizing figures or inventing cross-day identities is prohibited."
        ),
        "raw_data_status": (
            "not possessed; article states available from authors on reasonable request"
        ),
        "allowed_use": (
            "independent publication and assay-design evidence only; "
            "no GATES-ACT outcome validation"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "brainsphere_eligibility_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
