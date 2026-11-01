from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE = ROOT / "artifacts/act/research/chronological_high_dose"
FIXED = ROOT / "artifacts/act/research/direct_discovery/direct_discovery_decisions.parquet"
THRESHOLD = 0.40


def main() -> None:
    decisions = pd.read_parquet(ADAPTIVE / "chronological_adaptive_decisions.parquet")
    selection = pd.read_json(ADAPTIVE / "chronological_selection_log.json")
    fixed = pd.read_parquet(FIXED)
    fixed = fixed.loc[fixed["policy"].eq("highest_three_doses")].copy()
    decisions["time_rank"] = decisions["cell"] // 8
    decisions["dose_rank"] = decisions["cell"] % 8
    fixed["time_rank"] = fixed["cell"] // 8
    fixed["dose_rank"] = fixed["cell"] % 8

    switches = selection.loc[
        selection["selected_treated_ranks"].map(lambda ranks: 4 in ranks)
    ].copy()
    lookup = decisions.set_index(["fold", "run_id", "compound_group", "time_rank", "dose_rank"])[
        "truth"
    ]
    switch_rows = []
    for row in switches.itertuples(index=False):
        key = (row.fold, row.run_id, row.compound_group, row.time_rank)
        rank4 = float(lookup[(*key, 4)])
        rank5 = float(lookup[(*key, 5)])
        switch_rows.append(
            {
                "fold": row.fold,
                "run_id": row.run_id,
                "compound_group": row.compound_group,
                "source": row.source,
                "time_rank": row.time_rank,
                "trigger": row.trigger,
                "rank4_truth": rank4,
                "omitted_rank5_truth": rank5,
                "rank4_effect": abs(rank4) >= THRESHOLD,
                "omitted_rank5_effect": abs(rank5) >= THRESHOLD,
            }
        )
    substitutions = pd.DataFrame(switch_rows)

    day_rows = []
    for name, frame in (("adaptive", decisions), ("fixed_high", fixed)):
        for day, part in frame.groupby("time_rank"):
            direct = part["directly_observed"].astype(bool)
            truth = part["truth"].abs() >= THRESHOLD
            day_rows.append(
                {
                    "policy": name,
                    "time_rank": int(day),
                    "direct_effect_cells": int((direct & truth).sum()),
                    "all_reference_effect_cells": int(truth.sum()),
                }
            )
    by_day = pd.DataFrame(day_rows)
    gained = substitutions["rank4_effect"] & ~substitutions["omitted_rank5_effect"]
    lost = ~substitutions["rank4_effect"] & substitutions["omitted_rank5_effect"]
    both = substitutions["rank4_effect"] & substitutions["omitted_rank5_effect"]
    neither = ~substitutions["rank4_effect"] & ~substitutions["omitted_rank5_effect"]
    result = {
        "schema": "gates-act-chronological-mechanism-v1",
        "evidence_class": "post hoc diagnosis on development groups; not a tuning target",
        "eligible_future_batches": int(len(selection) - selection["time_rank"].eq(0).sum()),
        "rank4_substitutions": int(len(substitutions)),
        "trigger_counts": {
            str(key): int(value) for key, value in switches["trigger"].value_counts().items()
        },
        "substitution_effect_accounting": {
            "gained_rank4_only_effects": int(gained.sum()),
            "lost_omitted_rank5_only_effects": int(lost.sum()),
            "both_are_effects": int(both.sum()),
            "neither_is_effect": int(neither.sum()),
            "net_direct_effect_change": int(gained.sum() - lost.sum()),
        },
        "interpretation": (
            "The temporal trigger is useful only if newly observed rank-4 effects exceed effects "
            "lost by omitting rank 5; this accounting is descriptive and was not used to revise "
            "the frozen policy."
        ),
    }
    substitutions.to_csv(ADAPTIVE / "chronological_substitution_diagnostics.csv", index=False)
    by_day.to_csv(ADAPTIVE / "chronological_direct_effects_by_day.csv", index=False)
    (ADAPTIVE / "chronological_mechanism.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
