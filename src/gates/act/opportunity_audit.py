from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EFFECT_THRESHOLD = 0.40


def _agreement_score(frame: pd.DataFrame) -> pd.Series:
    same_sign = np.sign(frame["gates"]) == np.sign(frame["selected_residual"])
    return pd.Series(
        np.where(
            same_sign,
            np.minimum(frame["gates"].abs(), frame["selected_residual"].abs()),
            0.0,
        ),
        index=frame.index,
    ).fillna(0.0)


def _safe_binary_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    if labels.nunique() < 2:
        return {"auroc": float("nan"), "average_precision": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def run_opportunity_audit(decisions_path: Path, output: Path) -> dict[str, object]:
    all_decisions = pd.read_parquet(decisions_path)
    frame = all_decisions.loc[all_decisions["policy"].eq("highest_three_doses")].copy()
    frame["dose_rank"] = frame["cell"] % 8
    frame["day_index"] = frame["cell"] // 8
    frame["reference_effect"] = frame["truth"].abs() >= EFFECT_THRESHOLD
    frame["agreement_score"] = _agreement_score(frame)
    keys = ["fold", "source", "run_id", "compound_group"]

    group_rows = []
    for key, part in frame.groupby(keys, sort=True):
        effects = part.loc[part["reference_effect"]]
        days = set(effects["day_index"].astype(int))
        high_effect = bool((part["reference_effect"] & part["dose_rank"].ge(5)).any())
        active = bool(part["reference_effect"].any())
        direct_active = bool((part["reference_effect"] & part["directly_observed"]).any())
        called_active = bool(part["effect_call"].any())
        observed_score = float(part.loc[part["directly_observed"], "truth"].abs().max())
        hidden_score = float(part.loc[~part["directly_observed"], "agreement_score"].max())
        group_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "reference_active": active,
                "direct_high_dose_active": direct_active,
                "gates_called_active": called_active,
                "lower_dose_only": active and not high_effect,
                "transient": bool(days.intersection({0, 1, 2})) and 3 not in days,
                "delayed": not bool(days.intersection({0, 1})) and bool(days.intersection({2, 3})),
                "observed_max_absolute_response": observed_score,
                "hidden_agreement_score": hidden_score,
            }
        )
    groups = pd.DataFrame(group_rows)
    missed = groups.loc[groups["reference_active"] & ~groups["direct_high_dose_active"]].copy()
    missed_gates = groups.loc[groups["reference_active"] & ~groups["gates_called_active"]].copy()

    direct_negative = groups.loc[~groups["direct_high_dose_active"]].copy()
    direct_negative["hidden_active"] = direct_negative["reference_active"]
    legal_metrics = {
        "observed_high_dose_score": _safe_binary_metrics(
            direct_negative["hidden_active"],
            direct_negative["observed_max_absolute_response"],
        ),
        "saved_oof_agreement_score": _safe_binary_metrics(
            direct_negative["hidden_active"], direct_negative["hidden_agreement_score"]
        ),
    }
    for score in ("observed_max_absolute_response", "hidden_agreement_score"):
        direct_negative[f"rank_by_{score}"] = direct_negative[score].rank(
            ascending=False, method="min"
        )

    effect_counts = frame.groupby(keys + ["day_index"])["reference_effect"].sum()
    oracle_direct_effects = int(effect_counts.clip(upper=3).sum())
    reference_effects = int(frame["reference_effect"].sum())
    fixed_direct_effects = int((frame["reference_effect"] & frame["directly_observed"]).sum())
    full_gates_effects = int((frame["reference_effect"] & frame["effect_call"]).sum())
    active_groups = int(groups["reference_active"].sum())
    fixed_active = int((groups["reference_active"] & groups["direct_high_dose_active"]).sum())

    missed_effect_rows = frame.merge(missed[keys], on=keys, how="inner")
    missed_effect_rows = missed_effect_rows.loc[
        missed_effect_rows["reference_effect"],
        keys
        + [
            "cell",
            "day_index",
            "dose_rank",
            "truth",
            "directly_observed",
            "gates",
            "selected_residual",
            "agreement_score",
            "effect_call",
            "no_effect_call",
        ],
    ].copy()

    group_opportunity_points = (active_groups - fixed_active) / max(active_groups, 1)
    best_legal = max(
        legal_metrics.values(),
        key=lambda value: (
            np.nan_to_num(value["auroc"], nan=-1.0),
            np.nan_to_num(value["average_precision"], nan=-1.0),
        ),
    )
    gate_conditions = {
        "at_least_five_percent_or_five_missed_active_groups": bool(
            len(missed) >= 5 or group_opportunity_points >= 0.05
        ),
        "oracle_adds_at_least_five_group_sensitivity_points": bool(
            group_opportunity_points >= 0.05
        ),
        "legal_identifiability_clears_prespecified_gate": bool(
            best_legal["auroc"] >= 0.80 and best_legal["average_precision"] >= 0.30
        ),
        "opportunity_is_not_only_cell_mapping": bool(len(missed) > 0),
    }
    pursue_method = all(gate_conditions.values())
    result = {
        "schema": "gates-act-scientific-opportunity-audit-v1",
        "evidence_class": "descriptive development audit on previously examined groups",
        "population": {
            "run_groups": int(len(groups)),
            "compound_groups": int(groups["compound_group"].nunique()),
            "active_run_groups": active_groups,
            "reference_effect_cells": reference_effects,
        },
        "biological_patterns": {
            "lower_dose_only_active_groups": int(groups["lower_dose_only"].sum()),
            "transient_active_groups": int(
                (groups["reference_active"] & groups["transient"]).sum()
            ),
            "delayed_active_groups": int((groups["reference_active"] & groups["delayed"]).sum()),
            "fixed_high_dose_direct_missed_active_groups": int(len(missed)),
            "full_gates_missed_active_groups": int(len(missed_gates)),
        },
        "missed_active_groups": missed.to_dict(orient="records"),
        "legal_identifiability_among_high_dose_negative_groups": {
            "groups": int(len(direct_negative)),
            "hidden_active_groups": int(direct_negative["hidden_active"].sum()),
            "metrics": legal_metrics,
            "warning": (
                "Only two positives are present; scores are descriptive and no threshold was tuned."
            ),
        },
        "oracle_upper_bound": {
            "run_group_sensitivity": 1.0,
            "fixed_high_dose_direct_run_group_sensitivity": fixed_active / max(active_groups, 1),
            "additional_discoverable_active_groups": active_groups - fixed_active,
            "run_group_sensitivity_opportunity_points": group_opportunity_points,
            "direct_cell_capture": oracle_direct_effects / max(reference_effects, 1),
            "direct_effect_cells": oracle_direct_effects,
            "gain_over_fixed_direct_capture_points": (oracle_direct_effects - fixed_direct_effects)
            / max(reference_effects, 1),
            "gain_over_full_fixed_gates_recall_points": (oracle_direct_effects - full_gates_effects)
            / max(reference_effects, 1),
            "diagnostic_only": True,
        },
        "new_method_gate": {
            "conditions": gate_conditions,
            "decision": "PURSUE_ONE_METHOD" if pursue_method else "STOP_ALGORITHMIC_SEARCH",
        },
        "canonical_change": "NONE",
        "early_stopping": "disabled",
    }
    output.mkdir(parents=True, exist_ok=True)
    groups.to_csv(output / "opportunity_group_patterns.csv", index=False)
    direct_negative.to_csv(output / "opportunity_legal_identifiability.csv", index=False)
    missed_effect_rows.to_csv(output / "opportunity_missed_effect_cells.csv", index=False)
    (output / "opportunity_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
