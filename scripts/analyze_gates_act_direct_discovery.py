from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/act/research/direct_discovery"


def main() -> None:
    frame = pd.read_parquet(ARTIFACT / "direct_discovery_decisions.parquet")
    truth = frame.loc[frame["policy"].eq("conditional_max_variance")].copy()
    truth["dose_rank"] = truth["cell"] % 8
    truth["day_index"] = truth["cell"] // 8
    truth["reference_effect"] = truth["truth"].abs() >= 0.40
    dose = (
        truth.groupby("dose_rank", as_index=False)
        .agg(
            treated_cells=("reference_effect", "size"),
            effect_cells=("reference_effect", "sum"),
            effect_rate=("reference_effect", "mean"),
            mean_absolute_effect=("truth", lambda values: values.abs().mean()),
        )
        .sort_values("dose_rank")
    )
    day = (
        truth.groupby("day_index", as_index=False)
        .agg(
            treated_cells=("reference_effect", "size"),
            effect_cells=("reference_effect", "sum"),
            effect_rate=("reference_effect", "mean"),
        )
        .sort_values("day_index")
    )
    source_rows = []
    for (policy, source), part in frame.loc[
        frame["policy"].isin(["conditional_max_variance", "highest_three_doses"])
    ].groupby(["policy", "source"]):
        effect = part["truth"].abs() >= 0.40
        decisive = part["effect_call"] | part["no_effect_call"]
        correct = (effect & part["effect_call"]) | (~effect & part["no_effect_call"])
        source_rows.append(
            {
                "policy": policy,
                "source": source,
                "reference_effect_cells": int(effect.sum()),
                "direct_effect_cells": int((effect & part["directly_observed"]).sum()),
                "direct_effect_capture": float(
                    (effect & part["directly_observed"]).sum() / effect.sum()
                ),
                "total_discovery_recall": float(
                    (effect & part["effect_call"]).sum() / effect.sum()
                ),
                "false_effect_calls": int((~effect & part["effect_call"]).sum()),
                "wrong_decisions": int((decisive & ~correct).sum()),
            }
        )
    source = pd.DataFrame(source_rows)
    candidate = frame.loc[frame["policy"].eq("highest_three_doses")].copy()
    candidate["reference_effect"] = candidate["truth"].abs() >= 0.40
    missed_groups = (
        candidate.groupby(["source", "run_id", "compound_group"], as_index=False)
        .agg(reference_active=("reference_effect", "any"), called_active=("effect_call", "any"))
        .query("reference_active and not called_active")
    )
    dose.to_csv(ARTIFACT / "effect_distribution_by_dose_rank.csv", index=False)
    day.to_csv(ARTIFACT / "effect_distribution_by_day.csv", index=False)
    source.to_csv(ARTIFACT / "direct_discovery_mechanism_by_source.csv", index=False)
    missed_groups.to_csv(ARTIFACT / "highest_three_missed_active_groups.csv", index=False)
    receipt = {
        "schema": "gates-act-direct-discovery-mechanism-v1",
        "analysis_class": "descriptive diagnosis on the same exposed development groups",
        "dose_concentration": {
            "all_reference_effect_cells": int(dose["effect_cells"].sum()),
            "effect_cells_at_ranks_5_to_7": int(
                dose.loc[dose["dose_rank"].isin([5, 6, 7]), "effect_cells"].sum()
            ),
            "fraction_at_ranks_5_to_7": float(
                dose.loc[dose["dose_rank"].isin([5, 6, 7]), "effect_cells"].sum()
                / dose["effect_cells"].sum()
            ),
            "rank_7_effect_cells": int(dose.loc[dose["dose_rank"].eq(7), "effect_cells"].iloc[0]),
            "rank_7_effect_rate": float(dose.loc[dose["dose_rank"].eq(7), "effect_rate"].iloc[0]),
        },
        "mechanism": (
            "Frozen conditional maximum variance selected ranks (3,5,6) in four folds and "
            "(2,5,6) in one, never rank 7. The fixed high-dose panel directly observes ranks "
            "5,6,7, moving concentrated high-dose effects from uncertain inference into direct "
            "calls."
        ),
        "residual_failure": {
            "missed_active_run_groups": len(missed_groups),
            "groups": missed_groups[["source", "run_id", "compound_group"]].to_dict(
                orient="records"
            ),
        },
        "interpretation_limit": (
            "This explains the observed development delta but is also the post-hoc origin of the "
            "candidate policy; it is not independent validation."
        ),
    }
    (ARTIFACT / "direct_discovery_mechanism.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
