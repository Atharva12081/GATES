from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from gates.act.benchmark_v3 import _load_sources

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/act/research/decision_utility"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    predictions = pd.read_csv(OUTPUT / "out_of_fold_predictions.csv")
    summary = json.loads((OUTPUT / "summary.json").read_text())
    legacy = ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx"
    phase1 = (
        ROOT
        / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx"
    )
    _, _, meta, _ = _load_sources(legacy, phase1)
    evaluation_groups = set(predictions["compound_group"].astype(str))
    sealed_groups = set(
        meta.loc[
            meta["v3_split"].isin(["calibration", "test", "legacy_frozen"]),
            "compound_group",
        ].astype(str)
    )
    expected_policies = {
        "decision_voi",
        "max_variance",
        "maximin",
        "fixed_sentinel",
        "random",
        "risk_sensitive",
        "any_observed_effect",
        "two_observed_effects",
        "prior_prevalence",
    }
    counts = predictions.groupby(["compound_group", "policy"])["round"].nunique()
    checks = {
        "no_sealed_group_overlap": not bool(evaluation_groups & sealed_groups),
        "outer_groups_equal_99": len(evaluation_groups) == 99,
        "all_policies_present": set(predictions["policy"]) == expected_policies,
        "three_rounds_per_group_policy": bool(counts.eq(3).all()),
        "chronological_budgets": set(
            predictions[["round", "observations"]].itertuples(index=False, name=None)
        )
        == {(1, 8), (2, 12), (3, 16)},
        "stopping_disabled": summary["stopping"] == "disabled; ALERT is non-terminating",
        "go_gate_failed": summary["go"] is False,
        "canonical_action_preserved": summary["canonical_action"].startswith("NO-GO"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"decision-utility audit failed: {checks}")
    protocol = ROOT / "protocols/gates_act_decision_utility.md"
    report = ROOT / "docs/act/TECHNICAL_REPORT.md"
    deck_source = ROOT / "scripts/build_act_deck.mjs"
    report_pdf = ROOT / "output/pdf/GATES_ACT_Technical_Report.pdf"
    deck = ROOT / "output/presentation/GATES_ACT_Five_Minute_Defense.pptx"
    presentation_untouched = all(
        path.stat().st_mtime < protocol.stat().st_mtime
        for path in (report, deck_source, report_pdf, deck)
    )
    checks["presentation_untouched_during_study"] = presentation_untouched
    if not presentation_untouched:
        raise RuntimeError("a canonical presentation artifact changed after protocol lock")
    evidence = {
        "schema": "gates-act-decision-utility-audit-v1",
        "checks": checks,
        "evidence_boundary": {
            "evaluation_groups": len(evaluation_groups),
            "sealed_group_overlap": sorted(evaluation_groups & sealed_groups),
            "claim_level": "nested development only",
        },
        "hashes": {
            str(path.relative_to(ROOT)): _sha(path)
            for path in (
                protocol,
                ROOT / "src/gates/act/decision_utility.py",
                ROOT / "scripts/run_gates_act_decision_utility.py",
                OUTPUT / "summary.json",
                OUTPUT / "out_of_fold_predictions.csv",
                OUTPUT / "matched_budget_metrics.csv",
                OUTPUT / "metric_intervals.csv",
                OUTPUT / "paired_loss_comparisons.csv",
                legacy,
                phase1,
            )
        },
    }
    (OUTPUT / "audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
