from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gates.evaluation.uncertainty import exact_binomial_interval
from gates.experiment_gate.signals import binary_entropy

FINAL_COLUMNS = [
    "dataset",
    "endpoint",
    "experiment",
    "method",
    "risk_target",
    "test_units",
    "early_stops",
    "early_stop_errors",
    "eesr",
    "eesr_ci_low",
    "eesr_ci_high",
    "coverage",
    "abstention_rate",
    "observation_savings",
    "mean_decision_time",
    "median_decision_time",
    "false_positive_early_stops",
    "false_negative_early_stops",
    "ood_enabled",
    "experiment_gate_enabled",
    "seed",
    "protocol_hash",
    "commit_hash",
]

METHOD_LABELS = {
    "fixed_time": "Fixed time",
    "naive_confidence": "Naive confidence",
    "calibrated_non_group": "Non-group calibration",
    "cbes_style": "CBES-style",
    "gates_without_ood": "GATES without refusal",
    "gates_full": "Full GATES",
    "gates_full_plus_experiment_gate": "Full GATES + experiment gate",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def endpoint_from_dataset(dataset: str) -> str:
    if "Lens" in dataset:
        return "Lens_Final"
    return "RPE_Final"


def normalize_phase2(
    frame: pd.DataFrame,
    protocol_hash: str,
    *,
    method_suffix: str = "",
) -> pd.DataFrame:
    rows = []
    for record in frame.to_dict("records"):
        early_stops = int(record["stopped_early"])
        errors = int(record["incorrect_early"])
        interval = exact_binomial_interval(errors, early_stops)
        fp_value = record.get("FP_EESR", 0)
        fn_value = record.get("FN_EESR", 0)
        fp_rate = float(fp_value) if pd.notna(fp_value) else 0.0
        fn_rate = float(fn_value) if pd.notna(fn_value) else 0.0
        method = str(record["method"])
        risk_target = record.get("risk_target")
        if pd.isna(risk_target):
            method = f"{method}@{record['operating_point']}"
        rows.append(
            {
                "dataset": record["dataset"],
                "endpoint": endpoint_from_dataset(str(record["dataset"])),
                "experiment": record["held_out_group"],
                "method": f"{method}{method_suffix}",
                "risk_target": risk_target,
                "test_units": int(record["independent_test_units"]),
                "early_stops": early_stops,
                "early_stop_errors": errors,
                "eesr": float(record["EESR"]),
                "eesr_ci_low": float(record.get("EESR_exact_lower", interval.lower)),
                "eesr_ci_high": float(record.get("EESR_exact_upper", interval.upper)),
                "coverage": float(record["coverage"]),
                "abstention_rate": float(record["abstention_rate"]),
                "observation_savings": float(record["observation_savings"]),
                "mean_decision_time": float(record["mean_decision_time"]),
                "median_decision_time": float(record["median_decision_time"]),
                "false_positive_early_stops": int(round(fp_rate * early_stops)),
                "false_negative_early_stops": int(round(fn_rate * early_stops)),
                "ood_enabled": str(record.get("OOD_status", "disabled")) == "enabled",
                "experiment_gate_enabled": False,
                "seed": int(record["seed_config"]),
                "protocol_hash": protocol_hash,
                "commit_hash": record["commit_hash"],
            }
        )
    return pd.DataFrame(rows, columns=FINAL_COLUMNS)


def metrics_from_decisions(frame: pd.DataFrame) -> dict[str, float | int]:
    stopped = frame[frame["decision"] == "STOP"]
    error = stopped["prediction"] != stopped["endpoint"]
    false_positive = error & (stopped["prediction"] == 1) & (stopped["endpoint"] == 0)
    false_negative = error & (stopped["prediction"] == 0) & (stopped["endpoint"] == 1)
    interval = exact_binomial_interval(int(error.sum()), len(stopped))
    return {
        "test_units": len(frame),
        "early_stops": len(stopped),
        "early_stop_errors": int(error.sum()),
        "eesr": float(error.mean()) if len(stopped) else 0.0,
        "eesr_ci_low": interval.lower,
        "eesr_ci_high": interval.upper,
        "coverage": float(len(stopped) / len(frame)),
        "abstention_rate": float((frame["decision"] == "ABSTAIN").mean()),
        "observation_savings": float(frame["saved_fraction"].mean()),
        "mean_decision_time": float(frame["terminal_time"].mean()),
        "median_decision_time": float(frame["terminal_time"].median()),
        "false_positive_early_stops": int(false_positive.sum()),
        "false_negative_early_stops": int(false_negative.sum()),
    }


def experiment_gate_rows(
    decisions: pd.DataFrame,
    protocol_hash: str,
    protocol_commit: str,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for risk, risk_frame in decisions.groupby("risk_target", sort=True):
        groups = [("ALL", risk_frame), *risk_frame.groupby("group_id", sort=True)]
        for experiment, part in groups:
            rows.append(
                {
                    "dataset": "orgAInoid_RPE_Final",
                    "endpoint": "RPE_Final",
                    "experiment": experiment,
                    "method": "gates_full_plus_experiment_gate",
                    "risk_target": float(risk),
                    **metrics_from_decisions(part),
                    "ood_enabled": True,
                    "experiment_gate_enabled": True,
                    "seed": seed,
                    "protocol_hash": protocol_hash,
                    "commit_hash": protocol_commit,
                }
            )
    return pd.DataFrame(rows, columns=FINAL_COLUMNS)


def stop_prediction_table(
    decisions: pd.DataFrame, predictions: pd.DataFrame, experiment: str
) -> pd.DataFrame:
    part = decisions[(decisions["group_id"] == experiment) & (decisions["decision"] == "STOP")]
    result = predictions.merge(
        part[["unit_id", "decision_time", "prediction", "endpoint"]],
        on=["unit_id", "decision_time", "endpoint"],
        how="inner",
        validate="one_to_one",
    )
    result["error"] = result["prediction"] != result["endpoint"]
    result["selective_confidence"] = np.maximum(result["probability"], 1 - result["probability"])
    result["entropy"] = binary_entropy(result["probability"])
    result["ood_ratio"] = result["ood_score"] / result["ood_threshold"]
    return result


def forensic_record(
    experiment: str,
    decisions: pd.DataFrame,
    predictions: pd.DataFrame,
    signals: pd.DataFrame,
    condition_counts: dict[str, dict[str, int]],
) -> dict:
    part = decisions[decisions["group_id"] == experiment]
    stopped = part[part["decision"] == "STOP"]
    errors = stopped[stopped["prediction"] != stopped["endpoint"]]
    stop_predictions = stop_prediction_table(decisions, predictions, experiment)
    at_12h = predictions[
        (predictions["test_group"] == experiment) & (predictions["decision_time"] == 12)
    ]
    signal = signals[signals["experiment"] == experiment].iloc[0]
    ranks = {}
    for column in signals.columns.drop("experiment"):
        ranks[column] = {
            "rank_high_to_low": int(
                signals[column].rank(ascending=False, method="min").loc[signal.name]
            ),
            "of": len(signals),
            "value": float(signal[column]),
        }
    correct = stop_predictions[~stop_predictions["error"]]
    incorrect = stop_predictions[stop_predictions["error"]]
    if experiment == "E007":
        diagnosis = {
            "classification": "possible conditional/concept shift",
            "confidence": "moderate; mechanism remains unidentified",
            "reason": (
                "Errors are concentrated despite non-extreme generic feature-distance ranks, "
                "near-middle endpoint prevalence, and no erroneous stopped unit crossing the "
                "unit OOD threshold. This does not identify a biological mechanism."
            ),
        }
    else:
        diagnosis = {
            "classification": "calibration failure",
            "confidence": "moderate",
            "reason": (
                "All six errors are high-confidence false positives at 48 h. Input-shift scores "
                "are not extreme, and confidence/disagreement do not separate erroneous from "
                "correct stops. A concurrent conditional shift cannot be excluded."
            ),
        }
    return {
        "status": "post-hoc numerical forensic analysis; no model or threshold changed",
        "experiment": experiment,
        "unit_count": len(part),
        "positive_endpoint_fraction": float(part["endpoint"].mean()),
        "condition_counts": condition_counts.get(experiment, {}),
        "phase2_005": metrics_from_decisions(part),
        "error_direction": {
            "false_positive": int(((errors["prediction"] == 1) & (errors["endpoint"] == 0)).sum()),
            "false_negative": int(((errors["prediction"] == 0) & (errors["endpoint"] == 1)).sum()),
        },
        "error_decision_times": {
            str(key): int(value)
            for key, value in errors["decision_time"].value_counts().sort_index().items()
        },
        "early_12h": {
            "mean_probability": float(at_12h["probability"].mean()),
            "predicted_positive_fraction": float((at_12h["probability"] >= 0.5).mean()),
            "mean_entropy": float(binary_entropy(at_12h["probability"]).mean()),
            "mean_ensemble_disagreement": float(at_12h["cbes_std"].mean()),
            "brier_score": float(np.mean((at_12h["probability"] - at_12h["endpoint"]) ** 2)),
        },
        "stopped_unit_comparison": {
            "erroneous_stops": len(incorrect),
            "correct_stops": len(correct),
            "erroneous_mean_confidence": float(incorrect["selective_confidence"].mean()),
            "correct_mean_confidence": float(correct["selective_confidence"].mean()),
            "erroneous_mean_entropy": float(incorrect["entropy"].mean()),
            "correct_mean_entropy": float(correct["entropy"].mean()),
            "erroneous_mean_disagreement": float(incorrect["cbes_std"].mean()),
            "correct_mean_disagreement": float(correct["cbes_std"].mean()),
            "erroneous_mean_ood_ratio": float(incorrect["ood_ratio"].mean()),
            "correct_mean_ood_ratio": float(correct["ood_ratio"].mean()),
            "erroneous_ood_threshold_crossings": int((incorrect["ood_ratio"] >= 1).sum()),
        },
        "signal_values_and_ranks": ranks,
        "missingness_proxy": {
            "mean_observations_available_by_12h": float(signal["observation_count_mean"]),
            "observation_count_cv": float(signal["observation_count_cv"]),
            "note": "Availability is reported descriptively and was excluded from predictors.",
        },
        "diagnosis": diagnosis,
        "insufficient_evidence_for": [
            "a biological mechanism",
            "ground-truth error",
            "a uniquely identified shift mechanism",
        ],
    }


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_frontiers(ci: pd.DataFrame, figure_dir: Path) -> None:
    palette = {
        "fixed_time": "#9ca3af",
        "naive_confidence": "#f59e0b",
        "calibrated_non_group": "#56b4e9",
        "cbes_style": "#8b5cf6",
        "gates_without_ood": "#e69f00",
        "gates_full": "#0b3c5d",
    }
    for x_column, stem, xlabel in (
        ("observation_savings", "01_rpe_risk_savings_frontier", "Observation savings (%)"),
        ("coverage", "02_rpe_risk_coverage_frontier", "Early-decision coverage (%)"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for method, part in ci.groupby("method"):
            if method not in palette:
                continue
            part = part.sort_values(x_column)
            ax.plot(
                part[x_column] * 100,
                part["EESR"] * 100,
                "o-",
                color=palette[method],
                label=METHOD_LABELS.get(method, method),
                linewidth=1.8,
                markersize=4,
            )
            if method == "gates_full" and {
                "EESR_group_bootstrap_lower",
                "EESR_group_bootstrap_upper",
            }.issubset(part.columns):
                ax.fill_between(
                    part[x_column] * 100,
                    part["EESR_group_bootstrap_lower"] * 100,
                    part["EESR_group_bootstrap_upper"] * 100,
                    color=palette[method],
                    alpha=0.14,
                    label="Full GATES 95% experiment bootstrap",
                )
        ax.set(xlabel=xlabel, ylabel="Erroneous early-stop rate (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, figure_dir, stem)


def plot_per_experiment(master: pd.DataFrame, figure_dir: Path) -> None:
    part = master[
        (master["dataset"] == "orgAInoid_RPE_Final")
        & (master["method"] == "gates_full")
        & np.isclose(master["risk_target"], 0.05)
        & (master["experiment"] != "ALL")
    ].sort_values("experiment")
    for column, stem, ylabel in (
        ("eesr", "03_per_experiment_eesr", "EESR (%)"),
        ("observation_savings", "04_per_experiment_observation_savings", "Savings (%)"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        values = part[column] * 100
        colors = ["#d55e00" if value > 5 and column == "eesr" else "#0072b2" for value in values]
        if column == "eesr":
            low = np.maximum(values - part["eesr_ci_low"] * 100, 0)
            high = np.maximum(part["eesr_ci_high"] * 100 - values, 0)
            valid = part["early_stops"] > 0
            errors = np.vstack([low.where(valid, 0), high.where(valid, 0)])
            ax.bar(part["experiment"], values, color=colors, yerr=errors, capsize=3)
        else:
            ax.bar(part["experiment"], values, color=colors)
        if column == "eesr":
            ax.axhline(5, color="black", linestyle="--", linewidth=1, label="5% target")
            ax.legend()
        ax.set(xlabel="Held-out experiment", ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
        save_figure(fig, figure_dir, stem)


def plot_operating_comparisons(operating: pd.DataFrame, figure_dir: Path) -> None:
    order = [
        "fixed_time",
        "naive_confidence",
        "calibrated_non_group",
        "cbes_style",
        "gates_without_ood",
        "gates_full",
    ]
    part = operating[operating["method"].isin(order)].copy()
    part["method"] = pd.Categorical(part["method"], order, ordered=True)
    part = part.sort_values("method")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(part))
    eesr = part["EESR"] * 100
    eesr_errors = np.vstack(
        [eesr - part["EESR_exact_lower"] * 100, part["EESR_exact_upper"] * 100 - eesr]
    )
    ax.bar(
        x - 0.18,
        eesr,
        width=0.36,
        label="EESR (exact 95% CI)",
        color="#d55e00",
        yerr=eesr_errors,
        capsize=3,
    )
    ax.bar(
        x + 0.18,
        part["observation_savings"] * 100,
        width=0.36,
        label="Observation savings",
        color="#0072b2",
    )
    ax.set_xticks(x, [METHOD_LABELS[str(item)] for item in part["method"]], rotation=25, ha="right")
    ax.set_ylabel("Percent")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, figure_dir, "05_baseline_comparison_005")

    ablation = part[part["method"].isin(["gates_without_ood", "gates_full"])]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
    for ax, column, title in zip(
        axes,
        ["EESR", "coverage", "observation_savings"],
        ["EESR", "Coverage", "Savings"],
        strict=True,
    ):
        ax.bar(
            ["No refusal", "Full GATES"],
            ablation[column] * 100,
            color=["#e69f00", "#0b3c5d"],
        )
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=0.2)
    save_figure(fig, figure_dir, "06_ood_refusal_ablation")


def plot_gate_results(comparison: pd.DataFrame, figure_dir: Path) -> None:
    for risk, stem in ((0.05, "07_experiment_gate_005"), (0.10, "08_experiment_gate_010")):
        part = comparison[
            (np.isclose(comparison["risk_target"], risk))
            & comparison["method"].isin(
                ["phase2_full_gates", "phase2_full_gates_plus_experiment_gate"]
            )
        ].copy()
        part["label"] = part["method"].map(
            {
                "phase2_full_gates": "Full GATES",
                "phase2_full_gates_plus_experiment_gate": "+ experiment gate",
            }
        )
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
        for ax, column, title in zip(
            axes,
            ["EESR", "coverage", "observation_savings"],
            ["EESR", "Coverage", "Savings"],
            strict=True,
        ):
            if column == "EESR":
                estimate = part[column] * 100
                errors = np.vstack(
                    [
                        estimate - part["EESR_exact_lower"] * 100,
                        part["EESR_exact_upper"] * 100 - estimate,
                    ]
                )
                ax.bar(
                    part["label"],
                    estimate,
                    color=["#0072b2", "#111827"],
                    yerr=errors,
                    capsize=3,
                )
            else:
                ax.bar(part["label"], part[column] * 100, color=["#0072b2", "#111827"])
            ax.set_title(title)
            ax.set_ylabel("Percent")
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.2)
        save_figure(fig, figure_dir, stem)


def plot_trajectory(
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    unit_id: str,
    figure_dir: Path,
    stem: str,
    title: str,
) -> None:
    trajectory = predictions[predictions["unit_id"] == unit_id].sort_values("decision_time")
    decision = decisions[decisions["unit_id"] == unit_id].iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    axes[0].plot(trajectory["decision_time"], trajectory["probability"], "o-", color="#2563eb")
    axes[0].axhline(0.5, color="grey", linestyle=":")
    axes[0].axvline(decision["terminal_time"], color="#111827", linestyle="--")
    axes[0].set(ylabel="Predicted P(RPE+)", title=f"{title}: {unit_id}")
    ratio = trajectory["ood_score"] / trajectory["ood_threshold"]
    axes[1].plot(trajectory["decision_time"], ratio, "o-", color="#7c3aed")
    axes[1].axhline(1, color="#b91c1c", linestyle="--", label="OOD threshold")
    axes[1].axvline(decision["terminal_time"], color="#111827", linestyle="--")
    axes[1].set(xlabel="Decision time (h)", ylabel="OOD score / threshold")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    save_figure(fig, figure_dir, stem)


def plot_cases(
    predictions: pd.DataFrame, decisions: pd.DataFrame, figure_dir: Path
) -> pd.DataFrame:
    stopped = decisions[decisions["decision"] == "STOP"].copy()
    stopped["error"] = stopped["prediction"] != stopped["endpoint"]
    experiment_errors = stopped.groupby("group_id")["error"].sum()
    safe_experiments = experiment_errors[experiment_errors == 0].index
    success = (
        stopped[(~stopped["error"]) & stopped["group_id"].isin(safe_experiments)]
        .sort_values(["saved_fraction", "unit_id"], ascending=[False, True])
        .iloc[0]
    )
    erroneous = (
        stopped[stopped["error"]]
        .sort_values(["decision_time", "unit_id"], ascending=[True, True])
        .iloc[0]
    )
    abstention = decisions[decisions["decision"] == "ABSTAIN"].sort_values("unit_id").iloc[0]
    selections = [
        (success, "09_representative_successful_stop", "Correct early stop"),
        (abstention, "10_representative_abstention", "Refusal / full observation"),
        (erroneous, "11_representative_erroneous_stop", "Erroneous early stop"),
    ]
    rows = []
    for row, stem, title in selections:
        plot_trajectory(predictions, decisions, row["unit_id"], figure_dir, stem, title)
        rows.append(
            {
                "figure": stem,
                "unit_id": row["unit_id"],
                "experiment": row["group_id"],
                "decision": row["decision"],
                "prediction": int(row["prediction"]),
                "endpoint": int(row["endpoint"]),
                "decision_time": float(row["terminal_time"]),
                "saved_fraction": float(row["saved_fraction"]),
                "selection_rule": title,
            }
        )
    e007 = stop_prediction_table(decisions, predictions, "E007")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for error, part in e007.groupby("error"):
        ax.scatter(
            part["ood_ratio"],
            part["selective_confidence"] * 100,
            label="Erroneous stop" if error else "Correct stop",
            color="#d55e00" if error else "#0072b2",
            alpha=0.8,
        )
    ax.axvline(1, color="black", linestyle="--", label="OOD threshold")
    ax.set(xlabel="OOD score / threshold", ylabel="Stopping confidence (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, figure_dir, "12_e007_failure_case")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", type=Path, default=Path("artifacts/phase2"))
    parser.add_argument(
        "--phase3", type=Path, default=Path("artifacts/phase3/orgainoid_experiment_gate")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/phase2_raw/orgainoid_morphometrics.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/final"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(exist_ok=True)

    phase2_protocol = Path("protocols/phase2_protocol.json")
    phase3_protocol = Path("protocols/phase3/orgainoid_experiment_gate_protocol.json")
    phase2_hash = sha256(phase2_protocol)
    phase3_hash = sha256(phase3_protocol)
    phase3_summary = json.loads((args.phase3 / "run_summary.json").read_text())

    phase2_master = pd.read_csv(args.phase2 / "master_results.csv")
    master_parts = [normalize_phase2(phase2_master, phase2_hash)]
    for endpoint_directory in ("retinal_rpe_final", "retinal_lens_final"):
        aggregate_frame = pd.read_csv(args.phase2 / endpoint_directory / "aggregate_results.csv")
        master_parts.append(normalize_phase2(aggregate_frame, phase2_hash))
    stress = pd.read_csv(args.phase2 / "retinal_rpe_stress/stress_matrix.csv")
    for scenario, part in stress.groupby("stress_scenario", sort=True):
        master_parts.append(normalize_phase2(part, phase2_hash, method_suffix=f"__{scenario}"))
    gated_decisions = pd.read_csv(args.phase3 / "experiment_gated_unit_decisions.csv")
    master_parts.append(
        experiment_gate_rows(
            gated_decisions,
            phase3_hash,
            phase3_summary["protocol_commit"],
            20260929,
        )
    )
    master = pd.concat(master_parts, ignore_index=True)
    master.to_csv(args.output / "master_evidence.csv", index=False)

    rpe_dir = args.phase2 / "retinal_rpe_final"
    frontier = pd.read_csv(rpe_dir / "aggregate_group_bootstrap_ci.csv")
    decisions = pd.read_csv(rpe_dir / "unit_decisions.csv")
    decisions_005 = decisions[
        (decisions["method"] == "gates_full") & np.isclose(decisions["parameter"], 0.05)
    ].copy()
    predictions = pd.read_csv(rpe_dir / "predictions_by_time.csv")
    signals = pd.read_csv(args.phase3 / "experiment_signal_table.csv")
    comparison = pd.read_csv(args.phase3 / "phase3_comparison.csv")
    gate_summary = pd.read_csv(args.phase3 / "experiment_gate_summary.csv")
    operating_005 = pd.read_csv(args.phase3 / "operating_point_005.csv")

    frontier.to_csv(args.output / "risk_savings_frontier.csv", index=False)
    master[
        (master["dataset"] == "orgAInoid_RPE_Final")
        & (master["experiment"] != "ALL")
        & master["method"].isin(["gates_full", "gates_full_plus_experiment_gate"])
    ].to_csv(args.output / "per_experiment_results.csv", index=False)
    gate_summary.merge(
        comparison,
        on="risk_target",
        how="left",
        validate="one_to_many",
    ).to_csv(args.output / "error_prevention_abstention_tradeoff.csv", index=False)
    shutil.copy2(
        args.phase3 / "failure_detection_auc.csv",
        args.output / "candidate_failure_score_comparison.csv",
    )
    stress.to_csv(args.output / "robustness_summary.csv", index=False)

    condition_frame = pd.read_csv(
        args.data, usecols=["experiment", "well", "Condition"]
    ).drop_duplicates(["experiment", "well"])
    condition_counts = {
        str(experiment): {
            str(key): int(value) for key, value in part["Condition"].value_counts().items()
        }
        for experiment, part in condition_frame.groupby("experiment")
    }
    forensic = {
        experiment: forensic_record(
            experiment, decisions_005, predictions, signals, condition_counts
        )
        for experiment in ("E007", "E012")
    }
    (args.output / "e007_e012_forensics.json").write_text(json.dumps(forensic, indent=2) + "\n")

    negative_rows = []
    lens = pd.read_csv(args.phase2 / "retinal_lens_final/aggregate_results.csv")
    lens_row = lens[(lens["method"] == "gates_full") & np.isclose(lens["risk_target"], 0.05)].iloc[
        0
    ]
    negative_rows.append(
        {
            "result": "Lens transfer",
            "eesr": lens_row["EESR"],
            "observation_savings": lens_row["observation_savings"],
            "source": "retinal_lens_final/aggregate_results.csv",
        }
    )
    for scenario in [
        "irregular_sampling",
        "measurement_noise_025sd",
        "dropped_half_features",
        "altered_class_balance",
        "decision_tree_predictor",
    ]:
        row = stress[
            (stress["stress_scenario"] == scenario) & np.isclose(stress["risk_target"], 0.05)
        ].iloc[0]
        negative_rows.append(
            {
                "result": scenario,
                "eesr": row["EESR"],
                "observation_savings": row["observation_savings"],
                "source": "retinal_rpe_stress/stress_matrix.csv",
            }
        )
    gate_010 = comparison[
        np.isclose(comparison["risk_target"], 0.10)
        & (comparison["method"] == "phase2_full_gates_plus_experiment_gate")
    ].iloc[0]
    negative_rows.append(
        {
            "result": "experiment_gate_at_10pct",
            "eesr": gate_010["EESR"],
            "observation_savings": gate_010["observation_savings"],
            "source": "phase3_comparison.csv",
        }
    )
    pd.DataFrame(negative_rows).to_csv(args.output / "negative_results.csv", index=False)

    plot_frontiers(frontier, figure_dir)
    plot_per_experiment(master, figure_dir)
    plot_operating_comparisons(operating_005, figure_dir)
    plot_gate_results(comparison, figure_dir)
    cases = plot_cases(predictions, decisions_005, figure_dir)
    cases.to_csv(args.output / "representative_cases.csv", index=False)

    figure_sources = [
        ("01_rpe_risk_savings_frontier", "risk_savings_frontier.csv"),
        ("02_rpe_risk_coverage_frontier", "risk_savings_frontier.csv"),
        ("03_per_experiment_eesr", "master_evidence.csv"),
        ("04_per_experiment_observation_savings", "master_evidence.csv"),
        ("05_baseline_comparison_005", "operating_point_005.csv"),
        ("06_ood_refusal_ablation", "operating_point_005.csv"),
        ("07_experiment_gate_005", "phase3_comparison.csv"),
        ("08_experiment_gate_010", "phase3_comparison.csv"),
        ("09_representative_successful_stop", "representative_cases.csv; predictions_by_time.csv"),
        ("10_representative_abstention", "representative_cases.csv; predictions_by_time.csv"),
        ("11_representative_erroneous_stop", "representative_cases.csv; predictions_by_time.csv"),
        ("12_e007_failure_case", "e007_e012_forensics.json; predictions_by_time.csv"),
    ]
    pd.DataFrame(figure_sources, columns=["figure", "evidence_source"]).to_csv(
        args.output / "figure_sources.csv", index=False
    )

    headline = master[
        (master["dataset"] == "orgAInoid_RPE_Final")
        & (master["experiment"] == "ALL")
        & (master["method"] == "gates_full")
        & np.isclose(master["risk_target"], 0.05)
    ].iloc[0]
    summary = {
        "status": "science frozen after verification; no further model development",
        "headline_master_evidence_selector": {
            "dataset": headline["dataset"],
            "experiment": "ALL",
            "method": "gates_full",
            "risk_target": 0.05,
        },
        "headline": {
            "test_units": int(headline["test_units"]),
            "eesr": float(headline["eesr"]),
            "eesr_exact_95_ci": [float(headline["eesr_ci_low"]), float(headline["eesr_ci_high"])],
            "coverage": float(headline["coverage"]),
            "observation_savings": float(headline["observation_savings"]),
        },
        "experiment_gate_status": "OPTIONAL RESEARCH EXTENSION",
        "experiment_gate_interpretation": (
            "At the pre-specified 5% target, experiment-level refusal removed a substantial "
            "number of unsafe early decisions. However, this behavior did not transfer to the "
            "10% operating point, indicating that experiment-level failure prediction remains "
            "unstable."
        ),
        "phase2_protocol_hash": phase2_hash,
        "phase3_protocol_hash": phase3_hash,
        "environment_lock_hash": sha256(Path("uv.lock")),
    }
    (args.output / "science_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
