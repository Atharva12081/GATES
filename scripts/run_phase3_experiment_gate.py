from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from gates.data.retinal import load_retinal_morphometrics
from gates.evaluation.uncertainty import exact_binomial_interval
from gates.experiment_gate.nested import (
    apply_nested_experiment_gate,
    fit_nested_experiment_gate,
)
from gates.experiment_gate.signals import binary_entropy, build_experiment_signals
from gates.features.generic import generic_feature_columns, generic_prefix_features


def decision_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    stopped = frame[frame["decision"] == "STOP"]
    error = stopped["prediction"] != stopped["endpoint"]
    false_positive = error & (stopped["prediction"] == 1) & (stopped["endpoint"] == 0)
    false_negative = error & (stopped["prediction"] == 0) & (stopped["endpoint"] == 1)
    interval = exact_binomial_interval(int(error.sum()), len(stopped))
    return {
        "independent_units": len(frame),
        "stopped_early": len(stopped),
        "incorrect_early": int(error.sum()),
        "EESR": float(error.mean()) if len(stopped) else 0.0,
        "EESR_exact_lower": interval.lower,
        "EESR_exact_upper": interval.upper,
        "FP_errors": int(false_positive.sum()),
        "FN_errors": int(false_negative.sum()),
        "FP_EESR": float(false_positive.sum() / len(stopped)) if len(stopped) else 0.0,
        "FN_EESR": float(false_negative.sum() / len(stopped)) if len(stopped) else 0.0,
        "coverage": float(len(stopped) / len(frame)),
        "abstention_rate": float((frame["decision"] == "ABSTAIN").mean()),
        "observation_savings": float(frame["saved_fraction"].mean()),
        "mean_decision_time": float(frame["terminal_time"].mean()),
        "median_decision_time": float(frame["terminal_time"].median()),
    }


def group_bootstrap_metrics(
    frame: pd.DataFrame, seed: int, draws: int = 10_000
) -> dict[str, float]:
    rows = []
    for _, part in frame.groupby("group_id"):
        stopped = part["decision"] == "STOP"
        rows.append(
            (
                len(part),
                int(stopped.sum()),
                int((stopped & (part["prediction"] != part["endpoint"])).sum()),
                float(part["saved_fraction"].sum()),
                float(part["terminal_time"].sum()),
            )
        )
    summary = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(summary), np.full(len(summary), 1 / len(summary)), draws)
    totals = weights @ summary
    eesr = np.full(draws, np.nan)
    np.divide(totals[:, 2], totals[:, 1], out=eesr, where=totals[:, 1] > 0)
    values = {
        "EESR": eesr,
        "coverage": totals[:, 1] / totals[:, 0],
        "observation_savings": totals[:, 3] / totals[:, 0],
        "mean_decision_time": totals[:, 4] / totals[:, 0],
    }
    result = {}
    for name, samples in values.items():
        finite = samples[np.isfinite(samples)]
        result[f"{name}_group_bootstrap_lower"] = float(np.quantile(finite, 0.025))
        result[f"{name}_group_bootstrap_upper"] = float(np.quantile(finite, 0.975))
    return result


def oriented_auc(labels: pd.Series, values: pd.Series) -> tuple[float, float, str]:
    if labels.nunique() < 2 or values.nunique() < 2:
        return float("nan"), float("nan"), "undefined"
    raw = float(roc_auc_score(labels, values))
    if raw >= 0.5:
        return raw, raw, "higher_is_unsafe"
    return raw, 1 - raw, "lower_is_unsafe"


def make_refused(frame: pd.DataFrame) -> pd.DataFrame:
    refused = frame.copy()
    refused["decision"] = "ABSTAIN"
    refused["prediction"] = refused["endpoint"]
    refused["decision_time"] = 72.0
    refused["terminal_time"] = 72.0
    refused["saved_fraction"] = 0.0
    return refused


def select_baseline_row(aggregate: pd.DataFrame, method: str, target: float) -> dict:
    candidates = aggregate[aggregate["method"] == method].copy()
    feasible = candidates[candidates["EESR"] <= target]
    if len(feasible):
        selected = feasible.sort_values(["observation_savings", "coverage"], ascending=False).iloc[
            0
        ]
        status = "target-compatible observed point"
    else:
        selected = candidates.sort_values(
            ["EESR", "observation_savings"], ascending=[True, False]
        ).iloc[0]
        status = "no target-compatible point; safest tested point shown"
    result = selected.to_dict()
    result["risk_target"] = target
    result["selection_status"] = status
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text())
    protocol_hash = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    candidate_signals = protocol["candidate_batch_signals"]
    dataset = load_retinal_morphometrics(args.data, "RPE_Final")
    prefix = generic_prefix_features(dataset, protocol["early_batch_window_hours"])
    columns = generic_feature_columns(prefix)
    predictions = pd.read_csv(args.phase2 / "predictions_by_time.csv")
    decisions = pd.read_csv(args.phase2 / "unit_decisions.csv")
    aggregate = pd.read_csv(args.phase2 / "aggregate_results.csv")
    window_predictions = predictions[
        predictions["decision_time"] == protocol["early_batch_window_hours"]
    ].copy()
    signals = build_experiment_signals(prefix, window_predictions, columns, args.seed)
    signals.to_csv(args.output / "experiment_signal_table.csv", index=False)

    gate_rows = []
    gated_decisions = []
    comparison_rows = []
    failure_auc_rows = []
    nested_states: dict[float, dict[str, bool]] = {}
    for risk in protocol["risk_targets"]:
        full = decisions[
            (decisions["method"] == "gates_full") & np.isclose(decisions["parameter"], risk)
        ].copy()
        without = decisions[
            (decisions["method"] == "gates_without_ood") & np.isclose(decisions["parameter"], risk)
        ].copy()
        outcomes = []
        for experiment, part in full.groupby("group_id"):
            metrics = decision_metrics(part)
            outcomes.append(
                {
                    "experiment": str(experiment),
                    "unsafe": int(metrics["incorrect_early"] > 0),
                    **metrics,
                    "class_balance": float(part["endpoint"].mean()),
                }
            )
        outcome_frame = pd.DataFrame(outcomes)
        analysis = signals.merge(outcome_frame, on="experiment", validate="one_to_one")
        nested_states[risk] = {}
        for outer_fold, experiment in enumerate(sorted(analysis["experiment"])):
            development = analysis[analysis["experiment"] != experiment]
            test = analysis[analysis["experiment"] == experiment].iloc[0]
            fit = fit_nested_experiment_gate(development, candidate_signals)
            refused = apply_nested_experiment_gate(fit, float(test[fit.signal]))
            nested_states[risk][experiment] = refused
            state = "REFUSE_EARLY_STOPPING" if refused else "ACCEPT_AUTOMATION"
            experiment_decisions = full[full["group_id"].astype(str) == experiment]
            before = decision_metrics(experiment_decisions)
            after_frame = make_refused(experiment_decisions) if refused else experiment_decisions
            after = decision_metrics(after_frame)
            unit_ood_summary = {
                "mean": test["unit_ood_mean"],
                "p95": test["unit_ood_p95"],
                "fraction": test["unit_ood_fraction"],
            }
            entropy_summary = {
                "mean": test["prediction_entropy_mean"],
                "p90": test["prediction_entropy_p90"],
            }
            shift_statistics = {
                "two_sample_auc": test["two_sample_auc"],
                "standardized_mean_distance": test["standardized_mean_distance"],
                "rbf_mmd": test["rbf_mmd"],
                "energy_distance": test["energy_distance"],
            }
            gate_rows.append(
                {
                    "risk_target": risk,
                    "experiment": experiment,
                    "batch_gate_signal": fit.signal,
                    "batch_gate_score": fit.orientation * float(test[fit.signal]),
                    "batch_gate_threshold": fit.threshold,
                    "batch_gate_state": state,
                    "development_auc": fit.development_auc,
                    "development_balanced_accuracy": fit.development_balanced_accuracy,
                    "true_phase2_eesr_without_batch_gate": before["EESR"],
                    "true_eesr_with_batch_gate": after["EESR"],
                    "early_stop_errors_prevented": before["incorrect_early"]
                    - after["incorrect_early"],
                    "units_deferred": len(experiment_decisions) if refused else 0,
                    "observation_savings_lost": before["observation_savings"]
                    - after["observation_savings"],
                    "unit_coverage_lost": before["coverage"] - after["coverage"],
                    "experiment_size": len(experiment_decisions),
                    "class_balance": test["class_balance"],
                    "unit_ood_summary": json.dumps(unit_ood_summary, sort_keys=True),
                    "prediction_entropy_summary": json.dumps(entropy_summary, sort_keys=True),
                    "shift_statistics": json.dumps(shift_statistics, sort_keys=True),
                    "outer_fold": outer_fold,
                    "config_hash": protocol_hash,
                    "unsafe_experiment": bool(test["unsafe"]),
                }
            )
            after_frame = after_frame.copy()
            after_frame["method"] = "gates_full_plus_experiment_gate"
            after_frame["risk_target"] = risk
            after_frame["batch_gate_state"] = state
            gated_decisions.append(after_frame)

        gated = pd.concat(
            [frame for frame in gated_decisions if np.isclose(frame["risk_target"].iloc[0], risk)],
            ignore_index=True,
        )
        for method, frame in (
            ("gates_without_unit_ood", without),
            ("phase2_full_gates", full),
            ("phase2_full_gates_plus_experiment_gate", gated),
        ):
            comparison_rows.append(
                {
                    "risk_target": risk,
                    "method": method,
                    **decision_metrics(frame),
                    **group_bootstrap_metrics(frame, args.seed + int(risk * 1000)),
                }
            )
        labels = analysis["unsafe"].astype(int)
        for signal in candidate_signals:
            raw, oriented, direction = oriented_auc(labels, analysis[signal])
            failure_auc_rows.append(
                {
                    "analysis_level": "experiment_posthoc_descriptive",
                    "risk_target": risk,
                    "signal": signal,
                    "n": len(analysis),
                    "positive_events": int(labels.sum()),
                    "auc_raw": raw,
                    "auc_oriented": oriented,
                    "unsafe_direction": direction,
                }
            )

        stopped = full[full["decision"] == "STOP"].copy()
        stop_predictions = predictions.merge(
            stopped[["unit_id", "decision_time", "prediction", "endpoint"]],
            on=["unit_id", "decision_time", "endpoint"],
            how="inner",
            validate="one_to_one",
        )
        stop_predictions["error"] = (
            stop_predictions["prediction"] != stop_predictions["endpoint"]
        ).astype(int)
        stop_predictions["unit_ood_ratio"] = (
            stop_predictions["ood_score"] / stop_predictions["ood_threshold"]
        )
        stop_predictions["prediction_entropy"] = binary_entropy(stop_predictions["probability"])
        stop_predictions["ensemble_disagreement"] = stop_predictions["cbes_std"]
        stop_predictions["negative_selective_confidence"] = 1 - np.maximum(
            stop_predictions["probability"], 1 - stop_predictions["probability"]
        )
        stop_predictions["ambiguous_prediction"] = (stop_predictions["cbes_prediction"] < 0).astype(
            int
        )
        unit_signals = [
            "unit_ood_ratio",
            "prediction_entropy",
            "ensemble_disagreement",
            "negative_selective_confidence",
            "ambiguous_prediction",
        ]
        for signal in unit_signals:
            raw, oriented, direction = oriented_auc(
                stop_predictions["error"], stop_predictions[signal]
            )
            failure_auc_rows.append(
                {
                    "analysis_level": "unit_at_early_stop",
                    "risk_target": risk,
                    "signal": signal,
                    "n": len(stop_predictions),
                    "positive_events": int(stop_predictions["error"].sum()),
                    "auc_raw": raw,
                    "auc_oriented": oriented,
                    "unsafe_direction": direction,
                }
            )

    gate_results = pd.DataFrame(gate_rows)
    gate_results.to_csv(args.output / "experiment_gate_results.csv", index=False)
    gate_summary_rows = []
    for risk, part in gate_results.groupby("risk_target", sort=True):
        refused = part["batch_gate_state"] == "REFUSE_EARLY_STOPPING"
        unsafe = part["unsafe_experiment"].astype(bool)
        true_positive = int((refused & unsafe).sum())
        false_positive = int((refused & ~unsafe).sum())
        false_negative = int((~refused & unsafe).sum())
        true_negative = int((~refused & ~unsafe).sum())
        gate_summary_rows.append(
            {
                "risk_target": risk,
                "unsafe_experiments": int(unsafe.sum()),
                "safe_experiments": int((~unsafe).sum()),
                "unsafe_experiments_correctly_refused": true_positive,
                "safe_experiments_incorrectly_refused": false_positive,
                "unsafe_experiments_missed": false_negative,
                "safe_experiments_correctly_accepted": true_negative,
                "experiment_sensitivity": true_positive / int(unsafe.sum()),
                "experiment_specificity": true_negative / int((~unsafe).sum()),
                "early_stop_errors_prevented": int(part["early_stop_errors_prevented"].sum()),
                "units_deferred": int(part["units_deferred"].sum()),
                "observation_savings_sacrificed": float(
                    part["observation_savings_lost"].sum() / part["experiment_size"].sum()
                ),
                "unit_coverage_sacrificed": float(
                    (part["unit_coverage_lost"] * part["experiment_size"]).sum()
                    / part["experiment_size"].sum()
                ),
                "classification": "nested retrospective, not prospective",
            }
        )
    pd.DataFrame(gate_summary_rows).to_csv(args.output / "experiment_gate_summary.csv", index=False)
    gated_frame = pd.concat(gated_decisions, ignore_index=True)
    gated_frame.to_csv(args.output / "experiment_gated_unit_decisions.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.output / "phase3_comparison.csv", index=False)
    pd.DataFrame(failure_auc_rows).to_csv(args.output / "failure_detection_auc.csv", index=False)

    operating_rows = []
    for risk in protocol["risk_targets"]:
        for method in ("fixed_time", "naive_confidence"):
            operating_rows.append(select_baseline_row(aggregate, method, risk))
        for method in (
            "calibrated_non_group",
            "cbes_style",
            "gates_without_ood",
            "gates_full",
        ):
            row = (
                aggregate[
                    (aggregate["method"] == method) & np.isclose(aggregate["risk_target"], risk)
                ]
                .iloc[0]
                .to_dict()
            )
            row["selection_status"] = "frozen Phase-2 target"
            operating_rows.append(row)
        gate_row = (
            comparison[
                (comparison["risk_target"] == risk)
                & (comparison["method"] == "phase2_full_gates_plus_experiment_gate")
            ]
            .iloc[0]
            .to_dict()
        )
        gate_row["dataset"] = "orgAInoid_RPE_Final"
        gate_row["held_out_group"] = "ALL"
        gate_row["operating_point"] = str(risk)
        gate_row["selection_status"] = "nested retrospective experiment gate"
        operating_rows.append(gate_row)
    operating = pd.DataFrame(operating_rows)
    operating[operating["risk_target"] == 0.05].to_csv(
        args.output / "operating_point_005.csv", index=False
    )
    operating[operating["risk_target"] == 0.10].to_csv(
        args.output / "operating_point_010.csv", index=False
    )

    conditions = pd.read_csv(
        args.data, usecols=["experiment", "well", "Condition", "RPE_Final"]
    ).drop_duplicates(["experiment", "well"])
    full_005 = decisions[
        (decisions["method"] == "gates_full") & np.isclose(decisions["parameter"], 0.05)
    ]
    e007 = full_005[full_005["group_id"] == "E007"]
    e007_stopped = e007[e007["decision"] == "STOP"]
    e007_errors = e007_stopped[e007_stopped["prediction"] != e007_stopped["endpoint"]]
    e007_predictions = predictions[
        (predictions["test_group"] == "E007") & (predictions["decision_time"] == 12)
    ].copy()
    e007_signal = signals[signals["experiment"] == "E007"].iloc[0]
    e007_stop_predictions = predictions.merge(
        e007_stopped[["unit_id", "decision_time", "prediction", "endpoint"]],
        on=["unit_id", "decision_time", "endpoint"],
        how="inner",
        validate="one_to_one",
    )
    e007_stop_predictions["error"] = (
        e007_stop_predictions["prediction"] != e007_stop_predictions["endpoint"]
    )
    e007_stop_predictions["selective_confidence"] = np.maximum(
        e007_stop_predictions["probability"], 1 - e007_stop_predictions["probability"]
    )
    e007_stop_predictions["unit_ood_ratio"] = (
        e007_stop_predictions["ood_score"] / e007_stop_predictions["ood_threshold"]
    )
    e007_stop_predictions["ensemble_disagreement"] = e007_stop_predictions["cbes_std"]
    rank_signals = (
        "two_sample_auc",
        "standardized_mean_distance",
        "rbf_mmd",
        "energy_distance",
        "unit_ood_mean",
        "unit_ood_fraction",
        "prediction_entropy_mean",
        "ensemble_disagreement_mean",
        "trajectory_slope_norm_mean",
    )
    cross_experiment_ranks = {
        key: {
            "rank_high_to_low": int(
                signals[key].rank(ascending=False, method="min")[e007_signal.name]
            ),
            "of": len(signals),
        }
        for key in rank_signals
    }
    correct_stops = e007_stop_predictions[~e007_stop_predictions["error"]]
    erroneous_stops = e007_stop_predictions[e007_stop_predictions["error"]]
    forensic = {
        "status": "post-hoc forensic analysis; no Phase-2 model or threshold changed",
        "experiment": "E007",
        "sample_count": len(e007),
        "positive_endpoint_fraction": float(e007["endpoint"].mean()),
        "condition_counts": {
            str(key): int(value)
            for key, value in conditions[conditions["experiment"] == "E007"]["Condition"]
            .value_counts()
            .items()
        },
        "phase2_005": decision_metrics(e007),
        "error_direction": {
            "false_positive": int(
                ((e007_errors["prediction"] == 1) & (e007_errors["endpoint"] == 0)).sum()
            ),
            "false_negative": int(
                ((e007_errors["prediction"] == 0) & (e007_errors["endpoint"] == 1)).sum()
            ),
        },
        "error_decision_times": {
            str(key): int(value)
            for key, value in e007_errors["decision_time"].value_counts().sort_index().items()
        },
        "input_shift_signals": {
            key: float(e007_signal[key])
            for key in (
                "two_sample_auc",
                "standardized_mean_distance",
                "rbf_mmd",
                "energy_distance",
                "unit_ood_mean",
                "unit_ood_fraction",
            )
        },
        "cross_experiment_ranks": cross_experiment_ranks,
        "prediction_at_12h": {
            "mean_probability": float(e007_predictions["probability"].mean()),
            "predicted_positive_fraction": float((e007_predictions["probability"] >= 0.5).mean()),
            "mean_entropy": float(binary_entropy(e007_predictions["probability"]).mean()),
            "mean_disagreement": float(e007_predictions["cbes_std"].mean()),
            "brier_score": float(
                np.mean((e007_predictions["probability"] - e007_predictions["endpoint"]) ** 2)
            ),
        },
        "stopped_unit_comparison": {
            "erroneous_stops": len(erroneous_stops),
            "correct_stops": len(correct_stops),
            "erroneous_mean_selective_confidence": float(
                erroneous_stops["selective_confidence"].mean()
            ),
            "correct_mean_selective_confidence": float(
                correct_stops["selective_confidence"].mean()
            ),
            "erroneous_mean_unit_ood_ratio": float(erroneous_stops["unit_ood_ratio"].mean()),
            "correct_mean_unit_ood_ratio": float(correct_stops["unit_ood_ratio"].mean()),
            "erroneous_mean_disagreement": float(erroneous_stops["ensemble_disagreement"].mean()),
            "correct_mean_disagreement": float(correct_stops["ensemble_disagreement"].mean()),
            "erroneous_stops_with_ood_ratio_at_least_one": int(
                (erroneous_stops["unit_ood_ratio"] >= 1).sum()
            ),
        },
        "diagnosis": {
            "primary": "conditional/concept shift or insufficient calibration representation",
            "confidence": "moderate; mechanism unidentified",
            "evidence": (
                "E007 has balanced conditions and near-average endpoint prevalence. Generic input "
                "distances are not extreme across experiments, all 15 erroneous stops have unit "
                "OOD ratios below one, and errors are mixed but false-negative dominant. The "
                "lowest-ranked early trajectory magnitude was detectable by the retrospective "
                "nested 5% gate, but not by generic covariate-shift scores."
            ),
            "not_supported": [
                "a specific biological mechanism",
                "label-prevalence shift as the main cause",
                "noisy ground truth",
                "pure covariate shift",
            ],
        },
    }
    (args.output / "e007_forensics.json").write_text(json.dumps(forensic, indent=2) + "\n")

    figure_dir = args.output / "figures"
    figure_dir.mkdir(exist_ok=True)
    phase2_ci = pd.read_csv(args.phase2 / "aggregate_group_bootstrap_ci.csv")
    labels = {
        "fixed_time": "Fixed time",
        "naive_confidence": "Naive confidence",
        "cbes_style": "CBES-style",
        "gates_without_ood": "GATES without OOD",
        "gates_full": "Full GATES",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label in labels.items():
        part = phase2_ci[phase2_ci["method"] == method].sort_values("observation_savings")
        ax.plot(part["observation_savings"] * 100, part["EESR"] * 100, "o-", label=label)
    gate_curve = comparison[
        comparison["method"] == "phase2_full_gates_plus_experiment_gate"
    ].sort_values("observation_savings")
    ax.plot(
        gate_curve["observation_savings"] * 100,
        gate_curve["EESR"] * 100,
        "D--",
        color="black",
        linewidth=2,
        label="Full GATES + experiment gate",
    )
    ax.axhline(5, color="black", linestyle=":", linewidth=1, label="Frozen 5% target")
    ax.axhline(10, color="grey", linestyle=":", linewidth=1, label="Frozen 10% target")
    ax.set(
        xlabel="Observation savings (%)",
        ylabel="Held-out EESR (%)",
        title="Retrospective nested experiment gate: risk versus savings",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "risk_vs_observation_savings.png", dpi=180)
    plt.close(fig)

    summary = {
        "protocol_hash": protocol_hash,
        "protocol_commit": "38fa90b29c08545cb6e15e1778f1cd110c78a8ff",
        "classification": "retrospective_nested_not_prospective",
        "external_validation": "not run; mTIPs failed availability/grouping acceptance",
        "independent_experiments": int(signals["experiment"].nunique()),
        "independent_units": int(dataset.independent_units),
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
