from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.agreement_triage import _calibrate_margin, _effect_call, _score, _wide

CONTEXTS = list(itertools.combinations(range(1, 8), 3))
FIXED_POLICIES = {
    "highest_three_doses": (5, 6, 7),
    "maximin": (1, 4, 7),
    "middle_high_sentinels": (3, 5, 7),
}


def _assemble_decisions(
    wide: pd.DataFrame,
    folds: list[int],
    context: tuple[int, int, int],
    policy: str,
    margin: float,
) -> pd.DataFrame:
    context_index = CONTEXTS.index(tuple(context))
    hidden = wide.loc[wide["fold"].isin(folds) & wide["context"].eq(context_index)].copy()
    hidden["policy"] = policy
    hidden["directly_observed"] = False
    hidden["effect_call"] = _effect_call(hidden, "agreement")
    hidden["no_effect_call"] = _score(hidden, "agreement") < margin
    hidden["margin"] = margin

    full = wide[["fold", "run_id", "compound_group", "cell", "truth"]].drop_duplicates()
    held_full = full.loc[full["fold"].isin(folds)].copy()
    hidden_keys = hidden[["fold", "run_id", "compound_group", "cell"]]
    observed = held_full.merge(
        hidden_keys.assign(hidden=True),
        on=["fold", "run_id", "compound_group", "cell"],
        how="left",
    )
    observed = observed.loc[observed["hidden"].isna()].drop(columns="hidden")
    truth = observed["truth"].abs().to_numpy() >= 0.40
    observed["context"] = context_index
    observed["gates"] = np.nan
    observed["interpolation"] = np.nan
    observed["selected_residual"] = np.nan
    observed["policy"] = policy
    observed["directly_observed"] = True
    observed["effect_call"] = truth
    observed["no_effect_call"] = ~truth
    observed["margin"] = margin
    return pd.concat([observed, hidden], ignore_index=True)


def _policy_decisions(
    wide: pd.DataFrame,
    contexts_by_fold: dict[int, tuple[int, int, int]],
    policy: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pieces = []
    calibration = []
    for fold, context in sorted(contexts_by_fold.items()):
        context_index = CONTEXTS.index(tuple(context))
        calibration_hidden = wide.loc[
            ~wide["fold"].eq(fold) & wide["context"].eq(context_index)
        ].copy()
        margin = _calibrate_margin(calibration_hidden, "agreement")
        pieces.append(_assemble_decisions(wide, [int(fold)], tuple(context), policy, margin))
        calibration.append(
            {
                "policy": policy,
                "fold": int(fold),
                "context": list(context),
                "context_index": context_index,
                "calibration_groups": int(calibration_hidden["compound_group"].nunique()),
                "hidden_no_effect_margin": margin,
            }
        )
    decisions = pd.concat(pieces, ignore_index=True)
    decisions["source"] = np.where(
        decisions["run_id"].str.startswith("P1::"), "phase1_external", "legacy_epa"
    )
    return decisions, calibration


def _matched_risk_decisions(
    wide: pd.DataFrame,
    baseline_contexts: dict[int, tuple[int, int, int]],
    candidate_context: tuple[int, int, int] = (5, 6, 7),
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    folds = sorted(int(value) for value in wide["fold"].unique())
    pieces = []
    receipts = []
    for fold in folds:
        calibration_folds = [value for value in folds if value != fold]
        baseline_context = baseline_contexts[fold]
        baseline_hidden = wide.loc[
            wide["fold"].isin(calibration_folds)
            & wide["context"].eq(CONTEXTS.index(baseline_context))
        ]
        baseline_margin = _calibrate_margin(baseline_hidden, "agreement")
        baseline_calibration = _assemble_decisions(
            wide,
            calibration_folds,
            baseline_context,
            "calibration_baseline",
            baseline_margin,
        )
        baseline_metrics = _metrics(baseline_calibration)

        candidate_base = _assemble_decisions(
            wide,
            calibration_folds,
            candidate_context,
            "calibration_candidate",
            0.0,
        )
        feasible = []
        for margin in np.linspace(0, 0.20, 201):
            trial = candidate_base.copy()
            hidden = ~trial["directly_observed"].to_numpy(dtype=bool)
            no_effect = trial["no_effect_call"].to_numpy(dtype=bool)
            no_effect[hidden] = _score(trial.loc[hidden], "agreement") < margin
            trial["no_effect_call"] = no_effect
            trial_metrics = _metrics(trial)
            if (
                trial_metrics["selective_risk"] <= baseline_metrics["selective_risk"] + 1e-12
                and trial_metrics["protected_recall"]
                >= baseline_metrics["protected_recall"] - 1e-12
            ):
                feasible.append((trial_metrics["decisive_coverage"], float(margin), trial_metrics))
        if feasible:
            _, margin, selected_metrics = max(feasible, key=lambda value: (value[0], value[1]))
        else:
            margin = 0.0
            selected_metrics = _metrics(candidate_base)
        pieces.append(
            _assemble_decisions(
                wide,
                [fold],
                candidate_context,
                "highest_three_matched_risk",
                margin,
            )
        )
        receipts.append(
            {
                "policy": "highest_three_matched_risk",
                "fold": fold,
                "context": list(candidate_context),
                "calibration_groups": int(
                    wide.loc[wide["fold"].isin(calibration_folds), "compound_group"].nunique()
                ),
                "hidden_no_effect_margin": margin,
                "baseline_calibration_risk": baseline_metrics["selective_risk"],
                "baseline_calibration_protected_recall": baseline_metrics["protected_recall"],
                "candidate_calibration_risk": selected_metrics["selective_risk"],
                "candidate_calibration_protected_recall": selected_metrics["protected_recall"],
                "candidate_calibration_coverage": selected_metrics["decisive_coverage"],
            }
        )
    decisions = pd.concat(pieces, ignore_index=True)
    decisions["source"] = np.where(
        decisions["run_id"].str.startswith("P1::"), "phase1_external", "legacy_epa"
    )
    return decisions, receipts


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["truth"].abs().to_numpy() >= 0.40
    effect = frame["effect_call"].to_numpy(dtype=bool)
    no_effect = frame["no_effect_call"].to_numpy(dtype=bool)
    direct = frame["directly_observed"].to_numpy(dtype=bool)
    decisive = effect | no_effect
    correct = (effect & truth) | (no_effect & ~truth)
    tp = int((effect & truth).sum())
    fp = int((effect & ~truth).sum())
    fn = int((no_effect & truth).sum())
    wrong = int((decisive & ~correct).sum())
    hidden_truth = truth & ~direct
    group = (
        frame.assign(reference_effect=truth, called_effect=effect)
        .groupby(["fold", "run_id", "compound_group"], as_index=False)[
            ["reference_effect", "called_effect"]
        ]
        .any()
    )
    reference_active = group["reference_effect"].to_numpy(dtype=bool)
    called_active = group["called_effect"].to_numpy(dtype=bool)
    group_tp = int((reference_active & called_active).sum())
    group_fp = int((~reference_active & called_active).sum())
    return {
        "rows": len(frame),
        "run_groups": len(group),
        "observed_treated_response_cells": int(direct.sum()),
        "hidden_treated_response_cells": int((~direct).sum()),
        "reference_effect_cells": int(truth.sum()),
        "direct_effect_cells": int((direct & truth).sum()),
        "inferred_true_effect_cells": int((~direct & effect & truth).sum()),
        "inferred_false_effect_cells": int((~direct & effect & ~truth).sum()),
        "confirmed_effect_precision": tp / max(tp + fp, 1),
        "total_discovery_recall": tp / max(int(truth.sum()), 1),
        "hidden_effect_call_recall": int((effect & hidden_truth).sum())
        / max(int(hidden_truth.sum()), 1),
        "protected_recall": 1 - fn / max(int(truth.sum()), 1),
        "decisive_coverage": float(decisive.mean()),
        "selective_risk": wrong / max(int(decisive.sum()), 1),
        "wrong_decisions": wrong,
        "missed_effects": fn,
        "unresolved_cells": int((~decisive).sum()),
        "direct_effect_capture": int((direct & truth).sum()) / max(int(truth.sum()), 1),
        "active_run_groups": int(reference_active.sum()),
        "confirmed_active_run_groups": int((reference_active & called_active).sum()),
        "run_group_sensitivity": group_tp / max(int(reference_active.sum()), 1),
        "run_group_precision": group_tp / max(group_tp + group_fp, 1),
    }


def _bootstrap(
    decisions: pd.DataFrame,
    candidate: str,
    baseline: str,
    seed: int = 20261026,
    replicates: int = 10000,
) -> dict[str, object]:
    clusters = sorted(decisions["compound_group"].unique())
    arrays = {}
    for policy in (candidate, baseline):
        part = decisions.loc[decisions["policy"].eq(policy)].copy()
        truth = part["truth"].abs() >= 0.40
        effect = part["effect_call"].astype(bool)
        no_effect = part["no_effect_call"].astype(bool)
        decisive = effect | no_effect
        cell = (
            part.assign(
                truth_count=truth.astype(int),
                tp=(truth & effect).astype(int),
                fp=(~truth & effect).astype(int),
                wrong=(decisive & ~((truth & effect) | (~truth & no_effect))).astype(int),
                decisive_count=decisive.astype(int),
            )
            .groupby("compound_group")[["truth_count", "tp", "fp", "wrong", "decisive_count"]]
            .sum()
        )
        run = (
            part.assign(reference_effect=truth, called_effect=effect)
            .groupby(["compound_group", "run_id"], as_index=False)[
                ["reference_effect", "called_effect"]
            ]
            .any()
        )
        run["active"] = run["reference_effect"].astype(int)
        run["active_tp"] = (run["reference_effect"] & run["called_effect"]).astype(int)
        run_counts = run.groupby("compound_group")[["active", "active_tp"]].sum()
        arrays[policy] = cell.join(run_counts).reindex(clusters).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    values = {}
    for policy in (candidate, baseline):
        totals = arrays[policy][sampled].sum(axis=1)
        values[policy] = {
            "recall": totals[:, 1] / totals[:, 0],
            "precision": totals[:, 1] / np.maximum(totals[:, 1] + totals[:, 2], 1),
            "risk": totals[:, 3] / np.maximum(totals[:, 4], 1),
            "group_sensitivity": totals[:, 6] / totals[:, 5],
        }
    result = {"clusters": len(clusters), "replicates": replicates, "seed": seed}
    for metric in ("recall", "precision", "risk", "group_sensitivity"):
        difference = values[candidate][metric] - values[baseline][metric]
        result[metric] = {
            "difference": float(np.mean(difference)),
            "ci95": [
                float(np.quantile(difference, 0.025)),
                float(np.quantile(difference, 0.975)),
            ],
        }
    return result


def run_direct_discovery_benchmark(
    predictions: Path, triage: Path, output: Path
) -> dict[str, object]:
    wide = _wide(predictions)
    folds = sorted(int(value) for value in wide["fold"].unique())
    stored = json.loads(triage.read_text())["conditional_max_variance_policy"]["contexts_by_fold"]
    policies = {
        "conditional_max_variance": {int(fold): tuple(context) for fold, context in stored.items()},
        **{name: {fold: context for fold in folds} for name, context in FIXED_POLICIES.items()},
    }
    decision_frames = []
    calibration_rows = []
    for policy, contexts in policies.items():
        decisions, calibration = _policy_decisions(wide, contexts, policy)
        decision_frames.append(decisions)
        calibration_rows.extend(calibration)
    matched_decisions, matched_calibration = _matched_risk_decisions(
        wide, policies["conditional_max_variance"]
    )
    decision_frames.append(matched_decisions)
    calibration_rows.extend(matched_calibration)
    all_decisions = pd.concat(decision_frames, ignore_index=True)

    aggregate = []
    by_source = []
    for policy, part in all_decisions.groupby("policy"):
        aggregate.append({"policy": policy, **_metrics(part)})
        for source, source_part in part.groupby("source"):
            by_source.append({"policy": policy, "source": source, **_metrics(source_part)})
    aggregate_frame = pd.DataFrame(aggregate)
    source_frame = pd.DataFrame(by_source)

    random_rows = []
    for index, context in enumerate(CONTEXTS):
        decisions, _ = _policy_decisions(
            wide, {fold: context for fold in folds}, f"random_context_{index:02d}"
        )
        random_rows.append({"context": list(context), **_metrics(decisions)})
    random_frame = pd.DataFrame(random_rows)

    bootstrap = _bootstrap(all_decisions, "highest_three_doses", "conditional_max_variance")
    metrics = aggregate_frame.set_index("policy")
    candidate = metrics.loc["highest_three_doses"]
    matched_candidate = metrics.loc["highest_three_matched_risk"]
    baseline = metrics.loc["conditional_max_variance"]
    source = source_frame.set_index(["policy", "source"])
    candidate_sources = source.loc["highest_three_doses"]
    baseline_sources = source.loc["conditional_max_variance"]
    gate = bool(
        candidate["total_discovery_recall"] - baseline["total_discovery_recall"] >= 0.03
        and bootstrap["recall"]["ci95"][0] > 0
        and candidate["confirmed_effect_precision"] >= baseline["confirmed_effect_precision"]
        and candidate["selective_risk"] <= baseline["selective_risk"]
        and candidate["run_group_sensitivity"] > baseline["run_group_sensitivity"]
        and (
            candidate_sources["total_discovery_recall"]
            >= baseline_sources["total_discovery_recall"]
        ).all()
        and candidate["observed_treated_response_cells"]
        == baseline["observed_treated_response_cells"]
    )
    h2_gate = bool(
        matched_candidate["total_discovery_recall"] >= candidate["total_discovery_recall"]
        and matched_candidate["confirmed_effect_precision"]
        >= candidate["confirmed_effect_precision"]
        and matched_candidate["selective_risk"] <= baseline["selective_risk"]
        and matched_candidate["protected_recall"] >= baseline["protected_recall"]
        and matched_candidate["decisive_coverage"] - candidate["decisive_coverage"] >= 0.10
    )

    output.mkdir(parents=True, exist_ok=True)
    all_decisions.to_parquet(output / "direct_discovery_decisions.parquet", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "direct_discovery_calibration.csv", index=False)
    aggregate_frame.to_csv(output / "direct_discovery_metrics.csv", index=False)
    source_frame.to_csv(output / "direct_discovery_by_source.csv", index=False)
    random_frame.to_json(
        output / "direct_discovery_random_contexts.json", orient="records", indent=2
    )
    result = {
        "schema": "gates-act-direct-discovery-benchmark-v1",
        "evidence_class": "post-hoc hypothesis evaluated as repeated nested development",
        "groups": int(wide["compound_group"].nunique()),
        "run_groups": int(wide.groupby(["run_id", "compound_group"]).ngroups),
        "budget": {
            "per_run": "12 of 28 treated response cells plus 4 shared vehicle controls",
            "total_assay_cells": "16 of 32",
            "decision_metric_denominator": "treated response cells only",
        },
        "effect_reference": "absolute matched-control log1p contrast >= 0.40",
        "aggregate": aggregate_frame.to_dict(orient="records"),
        "by_source": source_frame.to_dict(orient="records"),
        "random_contexts": {
            "count": len(random_frame),
            "mean_total_discovery_recall": float(random_frame["total_discovery_recall"].mean()),
            "min_total_discovery_recall": float(random_frame["total_discovery_recall"].min()),
            "max_total_discovery_recall": float(random_frame["total_discovery_recall"].max()),
        },
        "bootstrap_highest_three_minus_conditional_max_variance": bootstrap,
        "research_gate": "GO" if gate else "NO-GO",
        "matched_risk_h2_gate": "GO" if h2_gate else "NO-GO",
        "canonical_integration": "NO-GO without independent confirmation",
        "early_stopping": "disabled",
        "three_pointer_parity": (
            "not exact: AnchorBoost sparse-concentration forecasting is not chronological "
            "dose-time acquisition"
        ),
    }
    (output / "direct_discovery_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
