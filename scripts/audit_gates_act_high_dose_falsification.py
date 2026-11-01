from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.direct_discovery import _metrics

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/act/research/high_dose_falsification"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    decisions_path = ARTIFACT / "high_dose_falsification_decisions.parquet"
    metrics_path = ARTIFACT / "high_dose_falsification_metrics.csv"
    summary_path = ARTIFACT / "high_dose_falsification_summary.json"
    decisions = pd.read_parquet(decisions_path)
    stored = pd.read_csv(metrics_path).set_index("policy")
    summary = json.loads(summary_path.read_text())
    policies = sorted(decisions["policy"].unique())
    identity_columns = ["fold", "run_id", "compound_group", "cell", "truth", "source"]
    identities = {
        policy: part[identity_columns].sort_values(identity_columns[:-1]).reset_index(drop=True)
        for policy, part in decisions.groupby("policy")
    }
    reference = identities[policies[0]]
    per_run = decisions.groupby(["policy", "fold", "run_id", "compound_group"]).agg(
        rows=("cell", "size"), observed=("directly_observed", "sum")
    )
    checks: dict[str, bool] = {
        "four_mechanistic_layers_present": len(policies) == 4,
        "identical_reference_rows": all(value.equals(reference) for value in identities.values()),
        "twelve_treated_plus_four_vehicle_cells": bool(
            per_run["rows"].eq(28).all() and per_run["observed"].eq(12).all()
        ),
        "effect_and_no_effect_never_overlap": not bool(
            (decisions["effect_call"] & decisions["no_effect_call"]).any()
        ),
        "development_evidence_labeled": "development" in summary["evidence_class"],
        "ai_discovery_gate_rejected": summary["ai_discovery_gate"] == "NO-GO",
        "early_stopping_disabled": summary["early_stopping"] == "disabled",
    }
    for policy, part in decisions.groupby("policy"):
        recomputed = _metrics(part)
        checks[f"metrics_recompute_{policy}"] = all(
            np.isclose(float(stored.loc[policy, key]), float(value), atol=1e-12, rtol=0)
            for key, value in recomputed.items()
        )
    direct = stored.loc["direct_only"]
    full = stored.loc["gates_agreement_release"]
    incremental = summary["incremental_ai_vs_direct_only"]
    checks["incremental_recall_recomputes"] = np.isclose(
        incremental["effect_recall_points"],
        full["total_discovery_recall"] - direct["total_discovery_recall"],
        atol=1e-12,
        rtol=0,
    )
    checks["incremental_active_groups_recomputes"] = (
        incremental["additional_active_run_groups"]
        == full["confirmed_active_run_groups"] - direct["confirmed_active_run_groups"]
    )
    checks = {key: bool(value) for key, value in checks.items()}
    receipt = {
        "schema": "gates-act-high-dose-falsification-independent-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sha256": {
            path.name: _sha256(path) for path in (decisions_path, metrics_path, summary_path)
        },
    }
    (ARTIFACT / "high_dose_falsification_audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
