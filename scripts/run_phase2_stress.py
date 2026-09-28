from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from gates.data.base import LongitudinalDataset
from gates.data.retinal import load_retinal_morphometrics
from gates.evaluation.phase2 import run_phase2_campaign


def copy_dataset(base: LongitudinalDataset, name: str, frame: pd.DataFrame) -> LongitudinalDataset:
    result = LongitudinalDataset(
        name=name,
        frame=frame.sort_values(["unit_id", "time"]).reset_index(drop=True),
        feature_columns=base.feature_columns,
        final_time=base.final_time,
        endpoint_name=base.endpoint_name,
    )
    result.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    base = load_retinal_morphometrics(args.input, "RPE_Final")
    rng = np.random.default_rng(args.seed)
    features = list(base.feature_columns)
    scales = base.frame[features].std().replace(0, 1)
    scenarios: list[tuple[str, LongitudinalDataset, int, str]] = []

    missing = base.frame.copy()
    keep = (missing.time == missing.groupby("unit_id").time.transform("max")) | (
        rng.random(len(missing)) >= 0.20
    )
    scenarios.append(
        (
            "missing_20pct",
            copy_dataset(base, "stress_missing", missing[keep]),
            args.seed,
            "logistic",
        )
    )

    irregular = base.frame.copy()
    keep = (irregular.time == irregular.groupby("unit_id").time.transform("max")) | (
        rng.random(len(irregular)) >= 0.35
    )
    irregular.loc[keep, "time"] += rng.uniform(-0.2, 0.2, keep.sum())
    scenarios.append(
        (
            "irregular_sampling",
            copy_dataset(base, "stress_irregular", irregular[keep]),
            args.seed,
            "logistic",
        )
    )

    noisy = base.frame.copy()
    noisy[features] = (
        noisy[features] + rng.normal(size=noisy[features].shape) * scales.to_numpy() * 0.25
    )
    scenarios.append(
        (
            "measurement_noise_025sd",
            copy_dataset(base, "stress_noise", noisy),
            args.seed,
            "logistic",
        )
    )

    dropped = base.frame.copy()
    dropped_features = features[::2]
    dropped[dropped_features] = np.nan
    scenarios.append(
        (
            "dropped_half_features",
            copy_dataset(base, "stress_dropped", dropped),
            args.seed,
            "logistic",
        )
    )

    corrupted = base.frame.copy()
    chosen_units = set(rng.choice(corrupted.unit_id.unique(), 99, replace=False))
    affected = corrupted.unit_id.isin(chosen_units) & (corrupted.time <= 24)
    direction = rng.choice([-1, 1], (affected.sum(), len(features)))
    corrupted.loc[affected, features] += direction * scales.to_numpy() * 3
    scenarios.append(
        (
            "corrupt_early_10pct",
            copy_dataset(base, "stress_corrupt", corrupted),
            args.seed,
            "logistic",
        )
    )

    reduced = base.frame[
        (np.isclose(base.frame.time % 2, 0))
        | (base.frame.time == base.frame.groupby("unit_id").time.transform("max"))
    ]
    scenarios.append(
        ("reduced_2h_cadence", copy_dataset(base, "stress_cadence", reduced), args.seed, "logistic")
    )

    unit_labels = base.frame.drop_duplicates("unit_id")[["unit_id", "group_id", "endpoint"]]
    retained_units = []
    for _, part in unit_labels.groupby("group_id"):
        retained_units.extend(part[part.endpoint == 0].unit_id.tolist())
        positives = part[part.endpoint == 1].unit_id.to_numpy()
        retained_units.extend(
            rng.choice(positives, max(1, len(positives) // 2), replace=False).tolist()
            if len(positives)
            else []
        )
    imbalance = base.frame[base.frame.unit_id.isin(retained_units)]
    scenarios.append(
        (
            "altered_class_balance",
            copy_dataset(base, "stress_balance", imbalance),
            args.seed,
            "logistic",
        )
    )

    scenarios.append(("alternate_seed", base, args.seed + 1, "logistic"))
    scenarios.append(("decision_tree_predictor", base, args.seed, "decision_tree"))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    rows = []
    for name, dataset, seed, model_kind in scenarios:
        scenario_output = args.output / name
        run_phase2_campaign(
            dataset,
            scenario_output,
            seed,
            commit,
            bootstrap_estimators=4,
            model_kind=model_kind,
        )
        aggregate = pd.read_csv(scenario_output / "aggregate_results.csv")
        selected = aggregate[
            (aggregate.method == "gates_full") & aggregate.risk_target.isin([0.05, 0.10])
        ].copy()
        selected.insert(0, "stress_scenario", name)
        selected["model_kind"] = model_kind
        selected["input_independent_units"] = dataset.independent_units
        rows.append(selected)
    matrix = pd.concat(rows, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output / "stress_matrix.csv", index=False)
    metadata = {
        "seed": args.seed,
        "notes": {
            "missing_20pct": (
                "Randomly removes 20% of observations while retaining each unit's last row."
            ),
            "irregular_sampling": (
                "Randomly removes 35% and jitters retained timestamps by up to 0.2 h."
            ),
            "measurement_noise_025sd": (
                "Adds independent Gaussian feature noise at 0.25 pooled feature SD."
            ),
            "dropped_half_features": (
                f"Masks {len(dropped_features)} of {len(features)} features: {dropped_features}."
            ),
            "corrupt_early_10pct": (
                "Adds signed 3-SD corruption through 24 h to 99 randomly selected units."
            ),
            "reduced_2h_cadence": "Retains measurements on a 2 h grid and each unit's last row.",
            "altered_class_balance": (
                "Retains all negatives and a random half of positives within every experiment."
            ),
            "alternate_seed": "Repeats the unchanged data with seed + 1.",
            "decision_tree_predictor": (
                "Replaces regularized logistic regression with a depth-4 decision tree."
            ),
            "natural_shift_and_ood": (
                "The primary experiment-held-out matrix and OOD ablation supply the "
                "shifted-experiment and OOD-sample tests."
            ),
        },
    }
    (args.output / "stress_protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
