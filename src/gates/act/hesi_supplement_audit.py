from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from openpyxl import load_workbook

EXPECTED_HASHES = {
    "mmc1.pptx": "67e9c76c517ad00a6fa780b11bfd4e0c7543b70cdb73fac99609da4b90cfb0fc",
    "mmc2.xlsx": "4100da9b21c8dea936f8a8fe7a40f1962bb746beb32cf221e2e400ecbea2d54a",
    "mmc3.xlsx": "c152818536f2e78a71c376c458b6b926c179cc3bdc55ae8b27354794460d46ba",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slide_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("ppt/slides/slide1.xml"))
    namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    return "\n".join(node.text or "" for node in root.findall(".//a:t", namespace))


def _workbook_profile(path: Path, species: str) -> dict[str, object]:
    workbook = load_workbook(path, read_only=False, data_only=False)
    sheets = []
    endpoint_reference: list[str] | None = None
    site_counts = set()
    all_compounds = []
    for sheet in workbook.worksheets:
        endpoints = [str(sheet.cell(row=row, column=1).value).strip() for row in range(4, 43)]
        if endpoint_reference is None:
            endpoint_reference = endpoints
        if endpoints != endpoint_reference:
            raise ValueError(f"endpoint schema drift in {path.name}/{sheet.title}")
        site_columns = [
            column
            for column in range(2, sheet.max_column + 1)
            if str(sheet.cell(row=1, column=column).value or "").startswith("Site ")
        ]
        site_labels = sorted(
            {str(sheet.cell(row=1, column=column).value) for column in site_columns}
        )
        site_counts.add(len(site_labels))
        concentration_labels = []
        for column in range(2, sheet.max_column + 1):
            label = sheet.cell(row=2, column=column).value
            header = str(sheet.cell(row=1, column=column).value or "")
            if header.startswith("Site ") and isinstance(label, str):
                concentration_labels.append(label.strip())
        unique_concentrations = list(dict.fromkeys(concentration_labels))
        compound = re.sub(r"^(human|rat)_", "", sheet.title, flags=re.IGNORECASE)
        all_compounds.append(compound)
        numeric_site = 0
        missing_site = 0
        formulas = 0
        for row in range(4, 43):
            for column in site_columns:
                value = sheet.cell(row=row, column=column).value
                if isinstance(value, str) and value.startswith("="):
                    formulas += 1
                elif isinstance(value, (int, float)):
                    numeric_site += 1
                elif value is None:
                    missing_site += 1
        sheets.append(
            {
                "sheet": sheet.title,
                "compound_token": compound,
                "rows": sheet.max_row,
                "columns": sheet.max_column,
                "site_labels": site_labels,
                "site_column_count": len(site_columns),
                "duplicate_site_header_instances": len(site_columns)
                - len(unique_concentrations) * len(site_labels),
                "concentration_labels": unique_concentrations,
                "endpoint_count": len(endpoints),
                "numeric_site_cells": numeric_site,
                "missing_site_cells": missing_site,
                "formula_site_cells": formulas,
            }
        )
    return {
        "file": path.name,
        "species": species,
        "sheet_count": len(sheets),
        "compound_tokens": all_compounds,
        "site_counts_observed": sorted(site_counts),
        "site_identity": "anonymous ordinal labels only",
        "endpoints": endpoint_reference,
        "sheets": sheets,
    }


def run_hesi_supplement_audit(raw: Path, output: Path) -> dict[str, object]:
    hashes = {name: _sha256(raw / name) for name in EXPECTED_HASHES}
    slide = _slide_text(raw / "mmc1.pptx")
    profiles = [
        _workbook_profile(raw / "mmc2.xlsx", "human_iPSC"),
        _workbook_profile(raw / "mmc3.xlsx", "primary_rat_cortical"),
    ]
    checks = {
        "publisher_files_match_locked_sha256": hashes == EXPECTED_HASHES,
        "eleven_compound_sheets_per_species": all(
            profile["sheet_count"] == 11 for profile in profiles
        ),
        "human_site_count_is_four_or_five": profiles[0]["site_counts_observed"] == [4, 5],
        "six_rat_anonymous_sites": profiles[1]["site_counts_observed"] == [6],
        "thirty_nine_endpoints": all(len(profile["endpoints"]) == 39 for profile in profiles),
        "slide_names_epa_as_one_contributor": "EPA: 35 wells" in slide and "EPA: 36 wells" in slide,
        "slide_claims_seven_human_laboratories": "7 laboratories using human" in slide,
    }
    result = {
        "schema": "gates-act-hesi-supplement-audit-v1",
        "evidence_class": "independent publisher supplement; metadata audit only",
        "article": {
            "doi": "10.1016/j.neuro.2026.103519",
            "pii": "S0161813X26001403",
            "license": "CC BY 4.0",
            "reported_design": "10 pro-convulsants, 3 negative controls, 7 laboratories",
            "data_availability": "Data available in supplemental materials",
        },
        "official_epa_record": {
            "doi": "10.23719/d-6pnm",
            "dataset_id": "698a1c56e4b0731a6170bc9a",
            "scope": "Shafer/EPA contribution; EPA states it was one of seven laboratories",
            "access_observation_2026_10_26": (
                "dataset page redirects to login; dataset metadata, files, download and metadata "
                "API routes each return HTTP 401"
            ),
        },
        "publisher_files": {
            "urls": {
                "mmc1.pptx": "https://ars.els-cdn.com/content/image/1-s2.0-S0161813X26001403-mmc1.pptx",
                "mmc2.xlsx": "https://ars.els-cdn.com/content/image/1-s2.0-S0161813X26001403-mmc2.xlsx",
                "mmc3.xlsx": "https://ars.els-cdn.com/content/image/1-s2.0-S0161813X26001403-mmc3.xlsx",
            },
            "sha256": hashes,
        },
        "workbooks": profiles,
        "missing_for_model_validation": [
            "named site-to-laboratory mapping",
            "plate and well identifiers",
            "replicate and culture-batch identifiers",
            "assay timing and treatment sequence",
            "baseline and normalization provenance",
            "within-site uncertainty or replicate-level outcomes",
            "two of the thirteen compounds reported by the article",
            "unambiguous fifth human site label for linopirdine (Site 4 is duplicated)",
        ],
        "compatibility": {
            "within_laboratory_compound_generalization": "not identifiable",
            "new_laboratory_transfer": "not provenance-safe",
            "chronological_dose_time_screening": "incompatible",
            "allowed_use": "descriptive supplement integrity and missingness audit only",
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "hesi_supplement_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
