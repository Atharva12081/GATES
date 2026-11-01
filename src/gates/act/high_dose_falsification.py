from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.direct_discovery import _bootstrap, _metrics

EFFECT_THRESHOLD = 0.40
QUIET_THRESHOLD = 0.20


def _direct_layer(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    truth = result["truth"].abs() >= EFFECT_THRESHOLD
    direct = result["directly_observed"].astype(bool)
    result["policy"] = name
    result["effect_call"] = direct & truth
    result["no_effect_call"] = direct & ~truth
    result["margin"] = 0.0
    return result


def _threshold_layer(frame: pd.DataFrame, name: str, release_quiet: bool) -> pd.DataFrame:
    result = _direct_layer(frame, name)
    result["dose_rank"] = result["cell"] % 8
    result["time_rank"] = result["cell"] // 8
    observed = result.loc[result["directly_observed"]].copy()
    observed["dose_rank"] = observed["cell"] % 8
    observed["time_rank"] = observed["cell"] // 8
    values = observed.set_index(["fold", "run_id", "compound_group", "time_rank", "dose_rank"])[
        "truth"
    ]
    group_key = ["fold", "run_id", "compound_group", "time_rank"]
    for key, index in result.groupby(group_key).groups.items():
        high = np.asarray([values[(*key, rank)] for rank in (5, 6, 7)], dtype=float)
        hidden_index = [row for row in index if not bool(result.at[row, "directly_observed"])]
        boundary = (
            abs(high[0]) >= EFFECT_THRESHOLD
            and abs(high[1]) >= EFFECT_THRESHOLD
            and np.sign(high[0]) == np.sign(high[1])
        )
        if boundary:
            rank4 = [row for row in hidden_index if int(result.at[row, "dose_rank"]) == 4]
            result.loc[rank4, "effect_call"] = True
        if release_quiet and bool(np.all(np.abs(high) < QUIET_THRESHOLD)):
            lower = [row for row in hidden_index if int(result.at[row, "dose_rank"]) <= 4]
            result.loc[lower, "no_effect_call"] = True
    return result.drop(columns=["dose_rank", "time_rank"])


def run_high_dose_falsification(decisions: Path, output: Path) -> dict[str, object]:
    all_decisions = pd.read_parquet(decisions)
    full = all_decisions.loc[all_decisions["policy"].eq("highest_three_doses")].copy()
    cmv_full = all_decisions.loc[all_decisions["policy"].eq("conditional_max_variance")].copy()
    full["policy"] = "gates_agreement_release"
    layers = pd.concat(
        [
            _direct_layer(full, "direct_only"),
            _threshold_layer(full, "adjacent_boundary_rule", release_quiet=False),
            _threshold_layer(full, "strict_high_dose_rule", release_quiet=True),
            full,
        ],
        ignore_index=True,
    )
    aggregate_rows = []
    source_rows = []
    for policy, part in layers.groupby("policy"):
        aggregate_rows.append({"policy": policy, **_metrics(part)})
        for source, source_part in part.groupby("source"):
            source_rows.append({"policy": policy, "source": source, **_metrics(source_part)})
    aggregate = pd.DataFrame(aggregate_rows)
    by_source = pd.DataFrame(source_rows)
    indexed = aggregate.set_index("policy")
    full_metrics = indexed.loc["gates_agreement_release"]
    direct_metrics = indexed.loc["direct_only"]
    boundary_metrics = indexed.loc["adjacent_boundary_rule"]
    bootstrap_full_direct = _bootstrap(layers, "gates_agreement_release", "direct_only")
    bootstrap_full_boundary = _bootstrap(
        layers, "gates_agreement_release", "adjacent_boundary_rule"
    )
    ai_gate = bool(
        full_metrics["total_discovery_recall"] - direct_metrics["total_discovery_recall"] >= 0.03
        and full_metrics["confirmed_active_run_groups"]
        > direct_metrics["confirmed_active_run_groups"]
        and full_metrics["confirmed_effect_precision"]
        >= direct_metrics["confirmed_effect_precision"]
        and full_metrics["selective_risk"] <= direct_metrics["selective_risk"]
    )
    direct_tp = int(direct_metrics["direct_effect_cells"])
    full_tp = int(full_metrics["direct_effect_cells"] + full_metrics["inferred_true_effect_cells"])
    cmv_direct = _metrics(_direct_layer(cmv_full, "conditional_max_variance_direct_only"))
    cmv_full_metrics = _metrics(cmv_full)
    high_full_recall_gain = (
        full_metrics["total_discovery_recall"] - cmv_full_metrics["total_discovery_recall"]
    )
    direct_placement_gain = (
        direct_metrics["total_discovery_recall"] - cmv_direct["total_discovery_recall"]
    )
    result = {
        "schema": "gates-act-high-dose-falsification-v1",
        "evidence_class": "repeated nested development on previously examined groups",
        "budget": "12/28 treated cells plus four vehicle controls per run; 16/32 physical cells",
        "aggregate": aggregate.to_dict(orient="records"),
        "by_source": by_source.to_dict(orient="records"),
        "incremental_ai_vs_direct_only": {
            "true_effect_cells": full_tp - direct_tp,
            "false_effect_cells": int(full_metrics["inferred_false_effect_cells"]),
            "wrong_decisions": int(
                full_metrics["wrong_decisions"] - direct_metrics["wrong_decisions"]
            ),
            "missed_effects": int(
                full_metrics["missed_effects"] - direct_metrics["missed_effects"]
            ),
            "effect_recall_points": float(
                full_metrics["total_discovery_recall"] - direct_metrics["total_discovery_recall"]
            ),
            "coverage_points": float(
                full_metrics["decisive_coverage"] - direct_metrics["decisive_coverage"]
            ),
            "additional_active_run_groups": int(
                full_metrics["confirmed_active_run_groups"]
                - direct_metrics["confirmed_active_run_groups"]
            ),
            "fraction_of_confirmed_true_effects_due_to_inference": float(
                (full_tp - direct_tp) / max(full_tp, 1)
            ),
        },
        "dose_placement_vs_conditional_max_variance": {
            "highest_three_direct_only": {
                "confirmed_effect_precision": float(direct_metrics["confirmed_effect_precision"]),
                "total_discovery_recall": float(direct_metrics["total_discovery_recall"]),
                "wrong_decisions": int(direct_metrics["wrong_decisions"]),
                "run_group_sensitivity": float(direct_metrics["run_group_sensitivity"]),
            },
            "conditional_max_variance_direct_only": {
                "confirmed_effect_precision": float(cmv_direct["confirmed_effect_precision"]),
                "total_discovery_recall": float(cmv_direct["total_discovery_recall"]),
                "wrong_decisions": int(cmv_direct["wrong_decisions"]),
                "run_group_sensitivity": float(cmv_direct["run_group_sensitivity"]),
            },
            "direct_measurement_recall_gain_points": float(direct_placement_gain),
            "inference_recall_contribution_difference_points": float(
                high_full_recall_gain - direct_placement_gain
            ),
            "net_full_system_recall_gain_points": float(high_full_recall_gain),
            "net_full_system_precision_gain_points": float(
                full_metrics["confirmed_effect_precision"]
                - cmv_full_metrics["confirmed_effect_precision"]
            ),
            "net_full_system_wrong_decision_difference": int(
                full_metrics["wrong_decisions"] - cmv_full_metrics["wrong_decisions"]
            ),
        },
        "incremental_ai_vs_adjacent_boundary_rule": {
            "effect_recall_points": float(
                full_metrics["total_discovery_recall"] - boundary_metrics["total_discovery_recall"]
            ),
            "coverage_points": float(
                full_metrics["decisive_coverage"] - boundary_metrics["decisive_coverage"]
            ),
            "wrong_decisions": int(
                full_metrics["wrong_decisions"] - boundary_metrics["wrong_decisions"]
            ),
        },
        "bootstrap_full_minus_direct": bootstrap_full_direct,
        "bootstrap_full_minus_boundary": bootstrap_full_boundary,
        "ai_discovery_gate": "GO" if ai_gate else "NO-GO",
        "canonical_integration": "NO-GO without independent confirmation",
        "early_stopping": "disabled",
    }
    output.mkdir(parents=True, exist_ok=True)
    layers.to_parquet(output / "high_dose_falsification_decisions.parquet", index=False)
    aggregate.to_csv(output / "high_dose_falsification_metrics.csv", index=False)
    by_source.to_csv(output / "high_dose_falsification_by_source.csv", index=False)
    (output / "high_dose_falsification_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
