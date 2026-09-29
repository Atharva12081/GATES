from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "Justfile",
    "data/manifests/retinal_sources.json",
    "docs/final/ERRATUM.md",
    "evidence/phase2/retinal_dataset_audit.json",
    "artifacts/final/erratum.json",
    "artifacts/final/erratum_impact_report.json",
    "scripts/audit_retinal_dataset.py",
    "scripts/fetch_retinal_data.py",
    "scripts/freeze_science_v2.py",
    "scripts/run_phase2_retinal.py",
    "scripts/verify_erratum_recomputation.py",
    "scripts/verify_final_evidence.py",
    "src/gates/data/retinal.py",
    "tests/test_retinal_audit.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-commit", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.evidence_commit], cwd=ROOT, text=True
    ).strip()
    records = []
    for relative in FILES:
        path = ROOT / relative
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "freeze_name": "gates-science-freeze-v2",
        "status": "v1 science preserved; audit metadata and data acquisition corrected",
        "evidence_commit": commit,
        "supersedes": {
            "tag": "gates-science-freeze-v1",
            "scope": "audit metadata and deterministic data acquisition only",
        },
        "result_changed_count": 0,
        "dataset_sha256": "b73890e767f63e9cc464597444e5159429172312d6e4539adbf83f128663ff61",
        "files": records,
    }
    output = ROOT / "artifacts/final/MANIFEST_V2.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
