from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.direct_discovery import _metrics

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/act/research/chronological_high_dose"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    decisions_path = ARTIFACT / "chronological_adaptive_decisions.parquet"
    log_path = ARTIFACT / "chronological_selection_log.json"
    metrics_path = ARTIFACT / "chronological_policy_metrics.csv"
    summary_path = ARTIFACT / "chronological_summary.json"
    decisions = pd.read_parquet(decisions_path)
    selection = pd.read_json(log_path)
    stored = pd.read_csv(metrics_path).set_index("policy")
    summary = json.loads(summary_path.read_text())
    per_run = selection.groupby(["fold", "run_id", "compound_group"]).agg(
        batches=("time_rank", "size"),
        selected=("selected_cells", lambda values: sum(len(value) for value in values)),
    )
    evidence_is_past = all(
        not row.evidence_cells
        or max(int(cell) // 8 for cell in row.evidence_cells) < int(row.time_rank)
        for row in selection.itertuples(index=False)
    )
    recomputed = _metrics(decisions)
    checks: dict[str, bool] = {
        "all_adaptive_evidence_is_from_past_batches": evidence_is_past,
        "four_batches_per_run": bool(per_run["batches"].eq(4).all()),
        "sixteen_physical_cells_per_run": bool(per_run["selected"].eq(16).all()),
        "twelve_treated_decision_cells_per_run": bool(
            decisions.groupby(["fold", "run_id", "compound_group"])["directly_observed"]
            .sum()
            .eq(12)
            .all()
        ),
        "metrics_recompute": all(
            np.isclose(
                float(stored.loc["high_anchor_temporal_boundary", key]),
                float(value),
                atol=1e-12,
                rtol=0,
            )
            for key, value in recomputed.items()
        ),
        "chronology_summary_passes": summary["chronology_audit"]["status"] == "PASS",
        "research_gate_rejected": summary["research_gate"] == "NO-GO",
        "early_stopping_disabled": summary["early_stopping"] == "disabled",
    }
    checks = {key: bool(value) for key, value in checks.items()}
    receipt = {
        "schema": "gates-act-chronological-high-dose-independent-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sha256": {
            path.name: _sha256(path)
            for path in (decisions_path, log_path, metrics_path, summary_path)
        },
    }
    (ARTIFACT / "chronological_audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
