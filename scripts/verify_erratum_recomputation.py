from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PROVENANCE_COLUMNS = {"commit_hash"}
IGNORED_JSON_KEYS = {"commit_hash", "output_dir", "protocol_commit"}
POST_FREEZE_JSON = {"MANIFEST.json", "erratum.json", "erratum_impact_report.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_csv(frozen: Path, recomputed: Path) -> None:
    left = pd.read_csv(frozen)
    right = pd.read_csv(recomputed)
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        raise AssertionError(f"CSV shape/schema changed: {frozen.relative_to(ROOT)}")
    for column in left.columns:
        if column in IGNORED_PROVENANCE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(left[column]):
            if not np.allclose(
                left[column].to_numpy(float),
                right[column].to_numpy(float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                raise AssertionError(f"numeric result changed: {frozen.relative_to(ROOT)}:{column}")
        else:
            left_values = left[column].fillna("<NA>").astype(str)
            right_values = right[column].fillna("<NA>").astype(str)
            if not left_values.equals(right_values):
                raise AssertionError(f"text result changed: {frozen.relative_to(ROOT)}:{column}")


def normalize_json(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: normalize_json(item)
            for key, item in value.items()
            if key not in IGNORED_JSON_KEYS
        }
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    return value


def compare_json(frozen: Path, recomputed: Path) -> None:
    left = normalize_json(json.loads(frozen.read_text()))
    right = normalize_json(json.loads(recomputed.read_text()))
    if left != right:
        raise AssertionError(f"JSON result changed: {frozen.relative_to(ROOT)}")


def compare_tree(frozen_root: Path, recomputed_root: Path) -> tuple[int, int]:
    csv_count = 0
    json_count = 0
    for frozen in sorted(frozen_root.rglob("*.csv")):
        compare_csv(frozen, recomputed_root / frozen.relative_to(frozen_root))
        csv_count += 1
    for frozen in sorted(frozen_root.rglob("*.json")):
        if frozen.name in POST_FREEZE_JSON:
            continue
        compare_json(frozen, recomputed_root / frozen.relative_to(frozen_root))
        json_count += 1
    return csv_count, json_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recomputed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-at", required=True)
    args = parser.parse_args()

    comparisons = [
        (
            "RPE endpoint campaign and OOD ablation",
            ROOT / "artifacts/phase2/retinal_rpe_final",
            args.recomputed / "phase2/retinal_rpe_final",
        ),
        (
            "Lens endpoint campaign",
            ROOT / "artifacts/phase2/retinal_lens_final",
            args.recomputed / "phase2/retinal_lens_final",
        ),
        (
            "RPE robustness campaigns",
            ROOT / "artifacts/phase2/retinal_rpe_stress",
            args.recomputed / "phase2/retinal_rpe_stress",
        ),
        (
            "Nested experiment gate",
            ROOT / "artifacts/phase3/orgainoid_experiment_gate",
            args.recomputed / "phase3/orgainoid_experiment_gate",
        ),
        (
            "Final evidence tables and summaries",
            ROOT / "artifacts/final",
            args.recomputed / "final",
        ),
    ]
    records: list[dict[str, object]] = []
    for name, frozen, recomputed in comparisons:
        csv_count, json_count = compare_tree(frozen, recomputed)
        records.append(
            {
                "artifact_group": name,
                "classification": "RECOMPUTED_IDENTICAL",
                "csv_files_compared": csv_count,
                "json_files_compared": json_count,
                "comparison": (
                    "numeric values within 1e-12 and text exactly equal; commit/path "
                    "provenance excluded"
                ),
            }
        )

    frozen_pngs = sorted((ROOT / "artifacts/final/figures").glob("*.png"))
    for frozen in frozen_pngs:
        recomputed = args.recomputed / "final/figures" / frozen.name
        if sha256(frozen) != sha256(recomputed):
            raise AssertionError(f"rendered PNG changed: {frozen.relative_to(ROOT)}")
    records.append(
        {
            "artifact_group": "Final rendered figures",
            "classification": "RECOMPUTED_IDENTICAL",
            "png_files_compared": len(frozen_pngs),
            "comparison": "pixel render files are byte-identical",
        }
    )
    records.insert(
        0,
        {
            "artifact_group": "Retinal dataset label audit",
            "classification": "CORRECTED_METADATA_ONLY",
            "before_sha256": "074192eb98219c1f5c2b6de819c8903018f6bc2031291c8fb27f10a62bfb3bcb",
            "after_sha256": sha256(ROOT / "evidence/phase2/retinal_dataset_audit.json"),
            "impact": (
                "Corrected positive/negative counts and prevalence; no model input or result "
                "changed."
            ),
        },
    )
    report = {
        "verified_at": args.verified_at,
        "frozen_baseline": {
            "tag": "gates-science-freeze-v1",
            "commit": "d701487df78595f835215b60523a95c154e0b16d",
            "phase2_evidence_commit": "02ecb77976801b38228d4f09717b9f6593b7dc1d",
            "manifest_sha256": "2def9352762ddfbb9735fb8c01e3e71d9cf4dd15cbc8b9bf83da97652cdb95d7",
        },
        "comparison_policy": {
            "numeric_tolerance": 1e-12,
            "ignored_provenance": sorted(IGNORED_PROVENANCE_COLUMNS | IGNORED_JSON_KEYS),
            "reason": "Recomputation provenance changes do not represent scientific results.",
        },
        "artifacts": records,
        "classification_counts": {
            "UNAFFECTED": 0,
            "CORRECTED_METADATA_ONLY": 1,
            "RECOMPUTED_IDENTICAL": len(records) - 1,
            "RESULT_CHANGED": 0,
        },
        "conclusion": (
            "The erratum changes audit metadata only; all recomputed science is identical."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["classification_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
