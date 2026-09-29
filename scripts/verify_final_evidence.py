from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "final"
REQUIRED_COLUMNS = {
    "dataset",
    "endpoint",
    "experiment",
    "method",
    "risk_target",
    "test_units",
    "early_stops",
    "early_stop_errors",
    "eesr",
    "eesr_ci_low",
    "eesr_ci_high",
    "coverage",
    "abstention_rate",
    "observation_savings",
    "mean_decision_time",
    "median_decision_time",
    "false_positive_early_stops",
    "false_negative_early_stops",
    "ood_enabled",
    "experiment_gate_enabled",
    "seed",
    "protocol_hash",
    "commit_hash",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_git_blob(revision: str, path: str) -> str:
    payload = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
    return hashlib.sha256(payload).hexdigest()


def selected(master: pd.DataFrame, method: str, risk: float) -> pd.Series:
    return master[
        (master["dataset"] == "orgAInoid_RPE_Final")
        & (master["experiment"] == "ALL")
        & (master["method"] == method)
        & np.isclose(master["risk_target"], risk)
    ].iloc[0]


def main() -> None:
    master = pd.read_csv(FINAL / "master_evidence.csv")
    assert REQUIRED_COLUMNS == set(master.columns), "master evidence schema changed"
    assert not master.duplicated(
        ["dataset", "experiment", "method", "risk_target", "seed"]
    ).any(), "duplicate evidence rows"

    full = selected(master, "gates_full", 0.05)
    without = selected(master, "gates_without_ood", 0.05)
    cbes = selected(master, "cbes_style", 0.05)
    gate_005 = selected(master, "gates_full_plus_experiment_gate", 0.05)
    gate_010 = selected(master, "gates_full_plus_experiment_gate", 0.10)
    expected = {
        "full_eesr": (full["eesr"], 25 / 439),
        "full_coverage": (full["coverage"], 439 / 988),
        "full_savings": (full["observation_savings"], 0.12263832658569498),
        "without_refusal_eesr": (without["eesr"], 41 / 477),
        "cbes_eesr": (cbes["eesr"], 69 / 445),
        "gate_005_eesr": (gate_005["eesr"], 8 / 315),
        "gate_010_eesr": (gate_010["eesr"], 76 / 465),
    }
    for name, (observed, target) in expected.items():
        assert np.isclose(observed, target, atol=1e-12), f"headline mismatch: {name}"
    assert int(without["early_stop_errors"] - full["early_stop_errors"]) == 16

    figure_sources = pd.read_csv(FINAL / "figure_sources.csv")
    assert len(figure_sources) == 12
    for stem in figure_sources["figure"]:
        for extension in ("png", "svg"):
            path = FINAL / "figures" / f"{stem}.{extension}"
            assert path.exists() and path.stat().st_size > 1_000, f"missing figure: {path}"

    forensics = json.loads((FINAL / "e007_e012_forensics.json").read_text())
    assert forensics["E007"]["diagnosis"]["classification"] == (
        "possible conditional/concept shift"
    )
    assert forensics["E012"]["diagnosis"]["classification"] == "calibration failure"

    manifest_path = FINAL / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["files"]:
            observed = sha256_git_blob("gates-science-freeze-v1", record["path"])
            assert observed == record["sha256"], f"v1 hash mismatch: {record['path']}"
    manifest_v2_path = FINAL / "MANIFEST_V2.json"
    if manifest_v2_path.exists():
        manifest_v2 = json.loads(manifest_v2_path.read_text())
        for record in manifest_v2["files"]:
            observed = sha256_git_blob("gates-science-freeze-v2", record["path"])
            assert observed == record["sha256"], f"v2 hash mismatch: {record['path']}"
        impact = json.loads((FINAL / "erratum_impact_report.json").read_text())
        assert impact["classification_counts"]["RESULT_CHANGED"] == 0
    print(
        "Final evidence verified: schema, frozen headlines, 12 figure pairs, forensics, "
        "and freeze-aware manifest hashes."
    )


if __name__ == "__main__":
    main()
