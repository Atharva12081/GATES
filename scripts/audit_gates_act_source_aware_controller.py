from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/act/research/source_shift"


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["truth"].abs().to_numpy() >= 0.40
    effect = frame["effect_call"].to_numpy(dtype=bool)
    no_effect = frame["no_effect_call"].to_numpy(dtype=bool)
    decisive = effect | no_effect
    correct = (effect & truth) | (no_effect & ~truth)
    tp = int((effect & truth).sum())
    fp = int((effect & ~truth).sum())
    fn = int((no_effect & truth).sum())
    tn = int((no_effect & ~truth).sum())
    wrong = int((decisive & ~correct).sum())
    return {
        "rows": len(frame),
        "effect_precision": tp / max(tp + fp, 1),
        "effect_call_recall": tp / max(int(truth.sum()), 1),
        "protected_recall": 1 - fn / max(int(truth.sum()), 1),
        "decisive_specificity": tn / max(tn + fp, 1),
        "decisive_coverage": float(decisive.mean()),
        "selective_risk": wrong / max(int(decisive.sum()), 1),
        "confirmed_effects": int(effect.sum()),
        "no_effect_decisions": int(no_effect.sum()),
        "wrong_decisions": wrong,
        "missed_effects": fn,
        "unresolved": int((~decisive).sum()),
    }


def equal_metric(observed: object, expected: object) -> bool:
    if isinstance(observed, (float, np.floating)):
        return bool(np.isclose(observed, expected, rtol=0, atol=1e-12))
    return bool(observed == expected)


def main() -> None:
    decisions = pd.read_parquet(ARTIFACT / "source_aware_controller_decisions.parquet")
    aggregate = pd.read_csv(ARTIFACT / "source_aware_controller_aggregate.csv")
    by_source = pd.read_csv(ARTIFACT / "source_aware_controller_by_source.csv")
    receipt = pd.read_csv(ARTIFACT / "source_aware_controller_calibration.csv")
    summary = json.loads((ARTIFACT / "source_aware_controller_summary.json").read_text())

    identity = ["fold", "run_id", "compound_group", "context", "cell", "truth", "source"]
    reference = (
        decisions.loc[
            decisions["calibration_mode"].eq("recall") & decisions["policy"].eq("gates_global"),
            identity,
        ]
        .sort_values(identity)
        .reset_index(drop=True)
    )
    identity_equal = True
    aggregate_equal = True
    source_equal = True
    no_conflicts = True
    rows_per_policy_equal = True
    for (calibration_mode, policy), part in decisions.groupby(["calibration_mode", "policy"]):
        candidate_identity = part[identity].sort_values(identity).reset_index(drop=True)
        identity_equal &= candidate_identity.equals(reference)
        rows_per_policy_equal &= len(part) == int(summary["rows_per_policy"])
        no_conflicts &= not bool((part["effect_call"] & part["no_effect_call"]).any())
        expected = aggregate.loc[
            aggregate["calibration_mode"].eq(calibration_mode) & aggregate["policy"].eq(policy)
        ].iloc[0]
        for name, value in metrics(part).items():
            aggregate_equal &= equal_metric(value, expected[name])
        for source, source_part in part.groupby("source"):
            expected_source = by_source.loc[
                by_source["calibration_mode"].eq(calibration_mode)
                & by_source["policy"].eq(policy)
                & by_source["source"].eq(source)
            ].iloc[0]
            for name, value in metrics(source_part).items():
                source_equal &= equal_metric(value, expected_source[name])

    original = pd.read_parquet(
        ROOT / "artifacts/act/research/competitor_transfer/h1_oof_predictions_with_selected.parquet"
    )
    key = ["fold", "run_id", "compound_group", "context", "cell", "truth"]
    wide = (
        original.loc[original["method"].isin({"gates", "selected_residual", "interpolation"})]
        .pivot(index=key, columns="method", values="prediction")
        .reset_index()
    )
    saved = json.loads(
        (ROOT / "artifacts/act/research/competitor_transfer/h6_agreement_triage.json").read_text()
    )["conditional_max_variance_policy"]["contexts_by_fold"]
    contexts = list(itertools.combinations(range(1, 8), 3))
    selected_mask = np.logical_or.reduce(
        [
            wide["fold"].eq(int(fold)) & wide["context"].eq(contexts.index(tuple(context)))
            for fold, context in saved.items()
        ]
    )
    selected_input = wide.loc[selected_mask]
    original_identity = ["fold", "run_id", "compound_group", "context", "cell", "truth"]
    input_rows_equal = (
        selected_input[original_identity].sort_values(original_identity).reset_index(drop=True)
    ).equals(reference[original_identity].sort_values(original_identity).reset_index(drop=True))
    folds_disjoint = True
    for row in receipt.itertuples(index=False):
        calibration = reference.loc[~reference["fold"].eq(row.fold)]
        if row.policy in {"agreement_source_calibrated", "source_aware_controller"}:
            calibration = calibration.loc[calibration["source"].eq(row.source)]
        folds_disjoint &= not bool(calibration["fold"].eq(row.fold).any())
        folds_disjoint &= calibration["compound_group"].nunique() == row.calibration_groups

    strongest = aggregate.loc[
        aggregate["calibration_mode"].eq("recall")
        & aggregate["policy"].eq(summary["strongest_observed_policy"])
    ].iloc[0]
    strongest_equal = all(
        equal_metric(strongest[name], value)
        for name, value in summary["strongest_observed_metrics"].items()
        if name != "calibration_mode"
    )
    checks = {
        "identical_held_out_rows_for_every_policy": bool(identity_equal),
        "decision_rows_match_locked_prediction_input": bool(input_rows_equal),
        "1696_rows_per_policy_and_calibration_mode": bool(rows_per_policy_equal),
        "effect_and_no_effect_calls_never_overlap": bool(no_conflicts),
        "aggregate_metrics_recomputed_exactly": bool(aggregate_equal),
        "source_metrics_recomputed_exactly": bool(source_equal),
        "calibration_excludes_held_out_outer_fold": bool(folds_disjoint),
        "strongest_policy_summary_matches_table": bool(strongest_equal),
        "canonical_integration_remains_disabled": summary["canonical_integration"]
        == "not authorized without independent confirmation",
        "early_stopping_remains_disabled": summary["early_stopping"] == "disabled",
    }
    result = {
        "schema": "gates-act-source-aware-controller-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "evidence_class": summary["evidence_class"],
        "groups": int(reference["compound_group"].nunique()),
        "rows_per_policy": len(reference),
    }
    (ARTIFACT / "source_aware_controller_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
