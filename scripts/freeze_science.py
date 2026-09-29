from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-commit",
        required=True,
        help="Commit containing the finalized evidence before this manifest is added.",
    )
    args = parser.parse_args()
    evidence_commit = git_output("rev-parse", args.evidence_commit)
    final_dir = ROOT / "artifacts" / "final"
    tracked_roots = [
        final_dir,
        ROOT / "docs" / "phase3" / "CLAIM_FREEZE.md",
        ROOT / "docs" / "final",
        ROOT / "app" / "streamlit_app.py",
        ROOT / "scripts" / "run_phase3_experiment_gate.py",
        ROOT / "scripts" / "finalize_phase3_submission.py",
        ROOT / "scripts" / "freeze_science.py",
        ROOT / "scripts" / "verify_final_evidence.py",
        ROOT / "protocols" / "phase2_protocol.json",
        ROOT / "protocols" / "phase3" / "orgainoid_experiment_gate_protocol.json",
        ROOT / "uv.lock",
    ]
    files: list[Path] = []
    for target in tracked_roots:
        if target.is_dir():
            files.extend(path for path in target.rglob("*") if path.is_file())
        else:
            files.append(target)
    files = sorted(path for path in files if path.name != "MANIFEST.json")
    dataset_audit = json.loads(
        (ROOT / "evidence" / "phase2" / "retinal_dataset_audit.json").read_text()
    )
    phase2_manifest = json.loads((ROOT / "artifacts" / "phase2" / "MANIFEST.json").read_text())
    manifest = {
        "freeze_name": "gates-science-freeze-v1",
        "status": "science frozen; model development stopped",
        "evidence_commit": evidence_commit,
        "manifest_parent_commit": git_output("rev-parse", "HEAD"),
        "environment_lock": {
            "path": "uv.lock",
            "sha256": sha256(ROOT / "uv.lock"),
        },
        "dataset": {
            "path": "data/phase2_raw/orgainoid_morphometrics.csv",
            "sha256": dataset_audit["source"]["sha256"],
            "rows": dataset_audit["counts"]["rows"],
            "independent_organoids": dataset_audit["counts"]["independent_organoids"],
            "independent_experiments": dataset_audit["counts"]["independent_experiments"],
        },
        "protocol_hashes": {
            "phase2": sha256(ROOT / "protocols" / "phase2_protocol.json"),
            "phase3_experiment_gate": sha256(
                ROOT / "protocols" / "phase3" / "orgainoid_experiment_gate_protocol.json"
            ),
        },
        "phase2_immutable": {
            "tag": "gates-phase2-evidence-v1",
            "code_commit": phase2_manifest["phase2_code_commit"],
            "evidence_commit": git_output("rev-list", "-n", "1", "gates-phase2-evidence-v1"),
            "manifest_sha256": sha256(ROOT / "artifacts" / "phase2" / "MANIFEST.json"),
        },
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (final_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
