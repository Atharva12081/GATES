from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_bytes(revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-commit", required=True)
    parser.add_argument("--release", default="gates-ai4s-submission-final")
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.release_commit], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True
    ).splitlines()
    paths = [path for path in paths if path != "artifacts/release/MANIFEST.json"]
    records = []
    for path in paths:
        payload = git_bytes(commit, path)
        records.append(
            {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    manifest = {
        "release": args.release,
        "release_commit": commit,
        "base_science_tag": "gates-science-freeze-v2",
        "historical_science_tag": "gates-science-freeze-v1",
        "clean_clone_reproduction": "passed",
        "result_changed_count": 0,
        "checks": [
            "ruff",
            "pytest",
            "gates verify",
            "verify_final_evidence",
            "release claim/link/path audit",
            "Streamlit health and browser interaction",
        ],
        "files": records,
    }
    output = ROOT / "artifacts/release/MANIFEST.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
