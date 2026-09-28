from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "phase2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    retinal = []
    for endpoint in ("rpe_final", "lens_final"):
        directory = ARTIFACTS / f"retinal_{endpoint}"
        for name in ("master_results.csv", "aggregate_results.csv"):
            path = directory / name
            frame = pd.read_csv(path)
            frame["commit_hash"] = commit
            frame.to_csv(path, index=False)
        retinal.append(pd.read_csv(directory / "master_results.csv"))

    feasibility = pd.read_csv(ARTIFACTS / "organoid_feasibility_uncertainty/fold_results.csv")
    feasibility_rows = []
    for row in feasibility.itertuples(index=False):
        feasibility_rows.append(
            {
                "dataset": "OrganoID_gemcitabine_feasibility_v0",
                "held_out_group": str(row.test_replicate),
                "method": "gates_full",
                "operating_point": "frozen_v0",
                "risk_target": 0.20,
                "independent_test_units": row.independent_units,
                "stopped_early": row.stopped,
                "incorrect_early": row.incorrect_early,
                "EESR": row.EESR,
                "FP_EESR": np.nan,
                "FN_EESR": np.nan,
                "coverage": row.coverage,
                "abstention_rate": 0.0,
                "observation_savings": row.observation_savings_all,
                "observation_savings_stopped": row.observation_savings_stopped,
                "mean_decision_time": row.mean_decision_time,
                "median_decision_time": np.nan,
                "calibration_metric": "empirical replicate EESR",
                "OOD_status": "enabled",
                "seed_config": 20260929,
                "commit_hash": "2036b8aeff8e86fb8438bd3358c65329f6ac27de",
            }
        )
    master = pd.concat([pd.DataFrame(feasibility_rows), *retinal], ignore_index=True)
    master.to_csv(ARTIFACTS / "master_results.csv", index=False)

    manifest_files = []
    for path in sorted(ARTIFACTS.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest_files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "phase2_code_commit": commit,
        "feasibility_tag": "gates-organoid-feasibility-v0",
        "protocol": "protocols/phase2_protocol.json",
        "files": manifest_files,
    }
    (ARTIFACTS / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
