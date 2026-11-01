from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.benchmark_v3 import _load_sources

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/act/research/decision_triage"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binary_metrics(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    label = frame["label"].astype(bool).to_numpy()
    prediction = frame[column].astype(bool).to_numpy()
    tp = int(np.sum(label & prediction))
    fp = int(np.sum(~label & prediction))
    fn = int(np.sum(label & ~prediction))
    tn = int(np.sum(~label & ~prediction))
    return {
        "hidden_cells": len(frame),
        "effect_cells": int(label.sum()),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "specificity": tn / max(tn + fp, 1),
        "false_discoveries": fp,
        "false_negatives": fn,
        "wrong_per_100": 100 * (fp + fn) / len(frame),
        "utility_loss": (5 * fn + fp) / len(frame),
    }


def _triage_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    label = frame["label"].astype(bool).to_numpy()
    confirmed = frame["triage"].eq("CONFIRMED_EFFECT").to_numpy()
    no_effect = frame["triage"].eq("NO_EFFECT_DETECTED").to_numpy()
    unresolved = ~(confirmed | no_effect)
    tp = int(np.sum(label & confirmed))
    fp = int(np.sum(~label & confirmed))
    fn = int(np.sum(label & no_effect))
    correct = int(np.sum((label & confirmed) | (~label & no_effect)))
    return {
        "hidden_cells": len(frame),
        "effect_cells": int(label.sum()),
        "protected_recall": 1 - fn / max(int(label.sum()), 1),
        "confirmed_precision": tp / max(tp + fp, 1),
        "confirmed_effect_recall": tp / max(int(label.sum()), 1),
        "decisive_coverage": float(np.mean(~unresolved)),
        "unresolved_coverage": float(np.mean(unresolved)),
        "false_discoveries": fp,
        "missed_effects": fn,
        "correct_decisions_per_100": 100 * correct / len(frame),
        "wrong_decisions_per_100": 100 * (fp + fn) / len(frame),
        "utility_loss": (5 * fn + fp) / len(frame),
    }


def _same_metrics(actual: dict[str, object], expected: dict[str, object]) -> bool:
    if actual.keys() != expected.keys():
        return False
    for key, value in actual.items():
        reference = expected[key]
        if isinstance(value, (float, np.floating)):
            if not math.isclose(float(value), float(reference), rel_tol=0, abs_tol=1e-12):
                return False
        elif value != reference:
            return False
    return True


def main() -> None:
    decisions = pd.read_csv(OUTPUT / "out_of_fold_cell_decisions.csv")
    summary = json.loads((OUTPUT / "summary.json").read_text())
    legacy = ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx"
    phase1 = (
        ROOT
        / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx"
    )
    _, _, meta, _ = _load_sources(legacy, phase1)
    evaluated = set(decisions["compound_group"].astype(str))
    expected = set(
        meta.loc[meta["v3_split"].eq("train"), "compound_group"].astype(str)
    )
    sealed = set(
        meta.loc[
            meta["v3_split"].isin(["calibration", "test", "legacy_frozen"]),
            "compound_group",
        ].astype(str)
    )
    protocol = ROOT / "protocols/gates_act_decision_triage.md"
    presentation_paths = (
        ROOT / "docs/act/TECHNICAL_REPORT.md",
        ROOT / "scripts/build_act_deck.mjs",
        ROOT / "output/pdf/GATES_ACT_Technical_Report.pdf",
        ROOT / "output/presentation/GATES_ACT_Five_Minute_Defense.pptx",
        ROOT / "app/act_demo.py",
    )

    checks: dict[str, bool] = {
        "evaluation_is_exact_train_partition": evaluated == expected,
        "outer_groups_equal_99": len(evaluated) == 99,
        "no_sealed_group_overlap": not bool(evaluated & sealed),
        "summary_reports_zero_sealed_groups": summary["sealed_groups_used"] == 0,
        "nested_development_claim_only": summary["evidence_level"]
        == "repeated nested development; not fresh external validation",
        "only_locked_budgets": set(
            decisions[["round", "observations"]].itertuples(index=False, name=None)
        )
        == {(2, 12), (3, 16)},
        "unique_hidden_cell_predictions": not decisions.duplicated(
            ["run_id", "round", "cell"]
        ).any(),
        "one_outer_fold_per_group": decisions.groupby("compound_group")[
            "outer_fold"
        ].nunique().eq(1).all(),
        "one_selection_per_fold_round": decisions.groupby(["outer_fold", "round"])[
            [
                "binary_threshold",
                "low_threshold",
                "high_threshold",
                "selected_feature_set",
                "selected_model",
                "selected_calibrator",
            ]
        ].nunique().eq(1).all().all(),
        "candidate_rule_reproduces": decisions["candidate_binary"].astype(bool).equals(
            decisions["probability"].ge(decisions["binary_threshold"])
        ),
        "canonical_rule_reproduces": decisions["canonical_binary"].astype(bool).equals(
            decisions["upper_effect_bound"].ge(0.4)
        ),
        "posterior_mean_rule_reproduces": decisions["mean_binary"].astype(bool).equals(
            decisions["abs_mean"].ge(0.4)
        ),
        "posterior_probability_rule_reproduces": decisions[
            "probability_binary"
        ].astype(bool).equals(decisions["effect_probability"].ge(0.5)),
        "dose_supported_rule_reproduces": decisions[
            "dose_supported_binary"
        ].astype(bool).equals(
            decisions["abs_mean"].ge(0.4)
            & decisions["same_div_neighbor_abs"].ge(0.4)
        ),
        "triage_rule_reproduces": pd.Series(
            np.where(
                decisions["probability"].ge(decisions["high_threshold"]),
                "CONFIRMED_EFFECT",
                np.where(
                    decisions["probability"].le(decisions["low_threshold"]),
                    "NO_EFFECT_DETECTED",
                    "NEEDS_MORE_EVIDENCE",
                ),
            )
        ).equals(decisions["triage"]),
        "probabilities_finite_and_bounded": bool(
            np.isfinite(decisions["probability"]).all()
            and decisions["probability"].between(0, 1).all()
        ),
        "stopping_disabled": summary["stopping"] == "disabled",
        "go_gate_failed": summary["go"] is False,
        "canonical_action_preserved": summary["canonical_action"].startswith("NO-GO"),
        "presentation_untouched_during_study": all(
            path.stat().st_mtime < protocol.stat().st_mtime for path in presentation_paths
        ),
    }

    for reported in summary["round_summaries"]:
        frame = decisions.loc[decisions["round"].eq(reported["round"])]
        independently_computed = {
            "candidate": _binary_metrics(frame, "candidate_binary"),
            "canonical": _binary_metrics(frame, "canonical_binary"),
            "posterior_mean": _binary_metrics(frame, "mean_binary"),
            "posterior_probability": _binary_metrics(frame, "probability_binary"),
            "dose_supported": _binary_metrics(frame, "dose_supported_binary"),
            "triage": _triage_metrics(frame),
        }
        for name, metrics in independently_computed.items():
            checks[f"round_{reported['round']}_{name}_metrics_reproduce"] = (
                _same_metrics(metrics, reported[name])
            )

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"decision-triage audit failed: {failed}")
    checks = {name: bool(passed) for name, passed in checks.items()}

    hashed_paths = (
        protocol,
        ROOT / "src/gates/act/multivariate_mea.py",
        ROOT / "src/gates/act/decision_triage.py",
        ROOT / "scripts/run_gates_act_decision_triage.py",
        ROOT / "tests/test_decision_triage.py",
        OUTPUT / "research_context.json",
        OUTPUT / "summary.json",
        OUTPUT / "out_of_fold_cell_decisions.csv",
        OUTPUT / "ablations.csv",
        OUTPUT / "fold_selections.json",
        OUTPUT / "direct_effects.csv",
        OUTPUT / "direct_effect_summary.csv",
        legacy,
        phase1,
    )
    evidence = {
        "schema": "gates-act-decision-triage-audit-v1",
        "checks": checks,
        "evidence_boundary": {
            "evaluated_groups": len(evaluated),
            "sealed_group_overlap": sorted(evaluated & sealed),
            "claim_level": "repeated nested development only",
            "external_confirmation_reused": False,
        },
        "hashes": {
            str(path.relative_to(ROOT)): _sha(path) for path in hashed_paths
        },
    }
    (OUTPUT / "audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
