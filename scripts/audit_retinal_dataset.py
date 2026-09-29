from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gates.data.retinal import RETINAL_FEATURES, decode_binary_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    columns = [
        "experiment",
        "well",
        "loop",
        "Condition",
        "file_name",
        "RPE_Final",
        "Lens_Final",
        *RETINAL_FEATURES,
    ]
    frame = pd.read_csv(args.input, usecols=columns)
    frame["unit_id"] = frame["experiment"].astype(str) + "|" + frame["well"].astype(str)
    frame["time"] = frame["loop"].str.extract(r"(\d+)").astype(float) / 2
    frame["RPE_Final_decoded"] = decode_binary_labels(frame["RPE_Final"], column="RPE_Final")
    frame["Lens_Final_decoded"] = decode_binary_labels(
        frame["Lens_Final"], column="Lens_Final"
    )
    units = frame.drop_duplicates("unit_id")
    counts = frame.groupby("unit_id").size()
    experiment_rows = []
    for experiment, part in units.groupby("experiment", sort=True):
        rpe_positive = int(part["RPE_Final_decoded"].sum())
        lens_positive = int(part["Lens_Final_decoded"].sum())
        experiment_rows.append(
            {
                "experiment": str(experiment),
                "independent_organoids": len(part),
                "RPE_positive": rpe_positive,
                "RPE_negative": int(len(part) - rpe_positive),
                "RPE_prevalence": rpe_positive / len(part),
                "Lens_positive": lens_positive,
                "Lens_negative": int(len(part) - lens_positive),
                "Lens_prevalence": lens_positive / len(part),
            }
        )
    rpe_positive = int(units["RPE_Final_decoded"].sum())
    lens_positive = int(units["Lens_Final_decoded"].sum())
    digest = hashlib.sha256()
    with args.input.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    audit = {
        "source": {
            "article": "https://doi.org/10.1371/journal.pbio.3003597",
            "source_data": "https://doi.org/10.5281/zenodo.18198347",
            "file": "Extended_Data_2.csv",
            "sha256": digest.hexdigest(),
            "license": "CC-BY-4.0",
            "redistribution": (
                "Attribution required; raw source remains excluded from git because it is 382 MiB."
            ),
        },
        "hierarchy": {
            "generalization_boundary": "experiment",
            "independent_unit": "organoid well within experiment",
            "observation": "one segmented bright-field image/morphometric row at a loop",
        },
        "counts": {
            "rows": len(frame),
            "independent_experiments": int(frame["experiment"].nunique()),
            "independent_organoids": int(frame["unit_id"].nunique()),
            "possible_time_points": int(frame["loop"].nunique()),
            "cadence_hours": 0.5,
            "maximum_duration_hours": float(frame["time"].max()),
            "full_144_observation_organoids": int((counts == 144).sum()),
            "organoids_below_72_observations": int((counts < 72).sum()),
            "organoids_below_12_observations": int((counts < 12).sum()),
        },
        "endpoint_label_totals": {
            "RPE_Final": {
                "positive": rpe_positive,
                "negative": int(len(units) - rpe_positive),
                "prevalence": rpe_positive / len(units),
            },
            "Lens_Final": {
                "positive": lens_positive,
                "negative": int(len(units) - lens_positive),
                "prevalence": lens_positive / len(units),
            },
        },
        "observation_count_quantiles": {
            "minimum": int(counts.min()),
            "q01": float(counts.quantile(0.01)),
            "q05": float(counts.quantile(0.05)),
            "median": float(counts.median()),
            "maximum": int(counts.max()),
        },
        "integrity": {
            "duplicate_experiment_well_loop_rows": int(
                frame.duplicated(["experiment", "well", "loop"]).sum()
            ),
            "duplicate_file_names": int(frame["file_name"].duplicated().sum()),
            "missing_RPE_final": int(frame["RPE_Final"].isna().sum()),
            "missing_Lens_final": int(frame["Lens_Final"].isna().sum()),
            "feature_missing_cells": {
                feature: int(frame[feature].isna().sum()) for feature in RETINAL_FEATURES
            },
        },
        "conditions_by_independent_unit": {
            str(key): int(value) for key, value in units["Condition"].value_counts().items()
        },
        "experiments": experiment_rows,
        "endpoints": {
            "RPE_Final": "whether the organoid has an RPE-positive final morphology label",
            "Lens_Final": "whether the organoid has a lens-positive final morphology label",
        },
        "authors_split_design": (
            "The authors distinguish within-training validation from a test set acquired in "
            "unrelated experiments. GATES does not reuse their image-classification split; it "
            "makes every complete experiment the held-out test boundary in turn."
        ),
        "leakage_channels_and_controls": [
            "Repeated frames from one organoid: all frames remain within its experiment partition.",
            (
                "Experiment-specific acquisition and morphology: test experiment is excluded "
                "from all preprocessing, fitting, calibration, and OOD thresholds."
            ),
            (
                "Observation availability can encode experiment: availability columns are "
                "excluded from predictor features."
            ),
            (
                "Final labels repeat on every frame: used only as unit-level outcome and never "
                "as a predictor."
            ),
            "Well/file names can identify experiment: excluded from predictor features.",
        ],
        "limitations": [
            (
                "The source table has 114,510 audited rows rather than every image cited in the "
                "paper, consistent with source-table filtering."
            ),
            "Trajectory completeness is highly variable and experiment-dependent.",
            (
                "RPE and lens prevalence differs sharply across experiments, including "
                "all-negative held-out experiments."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
