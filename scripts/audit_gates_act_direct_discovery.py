from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/act/research/direct_discovery"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["truth"].abs().to_numpy() >= 0.40
    effect = frame["effect_call"].to_numpy(dtype=bool)
    no_effect = frame["no_effect_call"].to_numpy(dtype=bool)
    direct = frame["directly_observed"].to_numpy(dtype=bool)
    decisive = effect | no_effect
    correct = (effect & truth) | (no_effect & ~truth)
    group = (
        frame.assign(reference_effect=truth, called_effect=effect)
        .groupby(["fold", "run_id", "compound_group"], as_index=False)[
            ["reference_effect", "called_effect"]
        ]
        .any()
    )
    active = group["reference_effect"].to_numpy(dtype=bool)
    called = group["called_effect"].to_numpy(dtype=bool)
    tp = int((truth & effect).sum())
    fp = int((~truth & effect).sum())
    fn = int((truth & no_effect).sum())
    group_tp = int((active & called).sum())
    group_fp = int((~active & called).sum())
    return {
        "rows": len(frame),
        "run_groups": len(group),
        "observed_treated_response_cells": int(direct.sum()),
        "hidden_treated_response_cells": int((~direct).sum()),
        "reference_effect_cells": int(truth.sum()),
        "direct_effect_cells": int((truth & direct).sum()),
        "inferred_true_effect_cells": int((~direct & truth & effect).sum()),
        "inferred_false_effect_cells": int((~direct & ~truth & effect).sum()),
        "confirmed_effect_precision": tp / max(tp + fp, 1),
        "total_discovery_recall": tp / max(int(truth.sum()), 1),
        "hidden_effect_call_recall": int((~direct & truth & effect).sum())
        / max(int((~direct & truth).sum()), 1),
        "protected_recall": 1 - fn / max(int(truth.sum()), 1),
        "decisive_coverage": float(decisive.mean()),
        "selective_risk": int((decisive & ~correct).sum()) / max(int(decisive.sum()), 1),
        "wrong_decisions": int((decisive & ~correct).sum()),
        "missed_effects": fn,
        "unresolved_cells": int((~decisive).sum()),
        "direct_effect_capture": int((direct & truth).sum()) / max(int(truth.sum()), 1),
        "active_run_groups": int(active.sum()),
        "confirmed_active_run_groups": group_tp,
        "run_group_sensitivity": group_tp / max(int(active.sum()), 1),
        "run_group_precision": group_tp / max(group_tp + group_fp, 1),
    }


def main() -> None:
    decisions_path = ARTIFACT / "direct_discovery_decisions.parquet"
    metrics_path = ARTIFACT / "direct_discovery_metrics.csv"
    summary_path = ARTIFACT / "direct_discovery_summary.json"
    decisions = pd.read_parquet(decisions_path)
    stored_metrics = pd.read_csv(metrics_path).set_index("policy")
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
    direct = decisions.loc[decisions["directly_observed"]]
    truth = direct["truth"].abs() >= 0.40
    checks: dict[str, bool] = {
        "five_policies_present": len(policies) == 5,
        "identical_truth_rows_for_every_policy": all(
            identity.equals(reference) for identity in identities.values()
        ),
        "each_run_has_28_treated_decision_cells": bool(per_run["rows"].eq(28).all()),
        "each_run_observes_12_treated_cells": bool(per_run["observed"].eq(12).all()),
        "common_vehicle_accounting_makes_16_of_32": bool(
            ((per_run["observed"] + 4).eq(16) & (per_run["rows"] + 4).eq(32)).all()
        ),
        "effect_and_no_effect_never_overlap": not bool(
            (decisions["effect_call"] & decisions["no_effect_call"]).any()
        ),
        "direct_effect_calls_equal_observed_reference": bool(
            direct["effect_call"].to_numpy(dtype=bool).tolist() == truth.to_numpy().tolist()
        ),
        "direct_no_effect_calls_equal_observed_reference_complement": bool(
            direct["no_effect_call"].to_numpy(dtype=bool).tolist() == (~truth).to_numpy().tolist()
        ),
        "held_rows_have_one_outer_fold": bool(
            decisions.groupby(["policy", "run_id", "compound_group"])["fold"].nunique().eq(1).all()
        ),
        "early_stopping_disabled": summary["early_stopping"] == "disabled",
        "canonical_integration_rejected": summary["canonical_integration"].startswith("NO-GO"),
        "h1_gate_rejected": summary["research_gate"] == "NO-GO",
        "h2_gate_rejected": summary["matched_risk_h2_gate"] == "NO-GO",
    }
    tolerance = 1e-12
    for policy, part in decisions.groupby("policy"):
        recomputed = _metrics(part)
        stored = stored_metrics.loc[policy]
        checks[f"metrics_recompute_{policy}"] = all(
            np.isclose(float(stored[key]), float(value), atol=tolerance, rtol=0)
            for key, value in recomputed.items()
        )
    bootstrap = summary["bootstrap_highest_three_minus_conditional_max_variance"]
    checks["recall_interval_does_not_clear_zero"] = bootstrap["recall"]["ci95"][0] <= 0
    checks["precision_interval_favors_highest_three"] = bootstrap["precision"]["ci95"][0] > 0
    checks["risk_interval_favors_highest_three"] = bootstrap["risk"]["ci95"][1] < 0
    receipt = {
        "schema": "gates-act-direct-discovery-independent-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cost_interpretation": (
            "12/28 treated response cells are directly observed per run; four shared vehicle "
            "controls make the physical assay budget 16/32. Decision metrics exclude vehicle cells."
        ),
        "evidence_boundary": (
            "Repeated nested development on 99 previously examined compound groups; not "
            "independent confirmation and not a laboratory-session saving."
        ),
        "sha256": {
            decisions_path.name: _sha256(decisions_path),
            metrics_path.name: _sha256(metrics_path),
            summary_path.name: _sha256(summary_path),
        },
    }
    output = ARTIFACT / "direct_discovery_audit.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
