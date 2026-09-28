from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from gates.data.base import LongitudinalDataset
from gates.evaluation.uncertainty import exact_binomial_interval
from gates.features.generic import generic_feature_columns, generic_prefix_features
from gates.models.classifier import make_classifier
from gates.policy.cbes import confidence_bound_decision, sequential_calibrate_cbes
from gates.policy.group_calibration import calibrate_group_threshold
from gates.policy.stopping import calibrate_threshold
from gates.shift.mahalanobis import MahalanobisGate

DECISION_TIMES = (12.0, 24.0, 36.0, 48.0, 60.0, 72.0)
RISK_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
NAIVE_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)


def _bootstrap_probabilities(
    train: pd.DataFrame,
    prediction_frames: list[pd.DataFrame],
    columns: list[str],
    seed: int,
    estimators: int = 12,
    model_kind: str = "logistic",
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = train["endpoint"].to_numpy(int)
    predictions = [np.empty((estimators, len(frame)), dtype=float) for frame in prediction_frames]
    for estimator_index in range(estimators):
        for _ in range(100):
            sample = rng.integers(0, len(train), size=len(train))
            if len(np.unique(labels[sample])) == 2:
                break
        model = make_classifier(seed + estimator_index, model_kind)
        model.fit(train.iloc[sample][columns], labels[sample])
        for frame_index, frame in enumerate(prediction_frames):
            predictions[frame_index][estimator_index] = model.predict_proba(frame[columns])[:, 1]
    return predictions


def _metric_row(
    decisions: pd.DataFrame,
    dataset: str,
    held_out_group: str,
    method: str,
    operating_point: str,
    risk_target: float | None,
    seed: int,
    commit_hash: str,
) -> dict[str, object]:
    stopped = decisions.loc[decisions["decision"] == "STOP"]
    incorrect = stopped.loc[stopped["prediction"] != stopped["endpoint"]]
    false_positive = incorrect.loc[(incorrect["prediction"] == 1) & (incorrect["endpoint"] == 0)]
    false_negative = incorrect.loc[(incorrect["prediction"] == 0) & (incorrect["endpoint"] == 1)]
    n_stopped = len(stopped)
    n_units = len(decisions)
    return {
        "dataset": dataset,
        "held_out_group": held_out_group,
        "method": method,
        "operating_point": operating_point,
        "risk_target": risk_target,
        "independent_test_units": n_units,
        "stopped_early": n_stopped,
        "incorrect_early": len(incorrect),
        "EESR": len(incorrect) / n_stopped if n_stopped else 0.0,
        "FP_EESR": len(false_positive) / n_stopped if n_stopped else 0.0,
        "FN_EESR": len(false_negative) / n_stopped if n_stopped else 0.0,
        "coverage": n_stopped / n_units,
        "abstention_rate": float((decisions["decision"] == "ABSTAIN").mean()),
        "observation_savings": float(decisions["saved_fraction"].mean()),
        "observation_savings_stopped": float(stopped["saved_fraction"].mean())
        if n_stopped
        else 0.0,
        "mean_decision_time": float(decisions["terminal_time"].mean()),
        "median_decision_time": float(decisions["terminal_time"].median()),
        "calibration_metric": "empirical group EESR",
        "OOD_status": "enabled" if method == "gates_full" else "disabled",
        "seed_config": seed,
        "commit_hash": commit_hash,
    }


def _simulate(
    states: dict[float, pd.DataFrame],
    unit_frame: pd.DataFrame,
    final_time: float,
    method: str,
    parameter: float | None,
    cbes_can_stop: tuple[bool, ...] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    times = sorted(states)
    indexed = {time: state.set_index("unit_id", drop=False) for time, state in states.items()}
    for _, unit in unit_frame.iterrows():
        unit_id = unit["unit_id"]
        endpoint = int(unit["endpoint"])
        decision = "CONTINUE"
        prediction = endpoint
        candidate_ood_score = float("nan")
        candidate_ood_threshold = float("nan")
        stop_time = final_time
        for time_index, time in enumerate(times):
            if time >= final_time or unit_id not in indexed[time].index:
                continue
            row = indexed[time].loc[unit_id]
            probability = float(row["probability"])
            confidence = max(probability, 1 - probability)
            candidate_prediction = int(probability >= 0.5)
            should_stop = False
            if method == "fixed_time":
                should_stop = time >= float(parameter)
            elif method == "naive_confidence":
                should_stop = confidence >= float(parameter)
            elif method == "calibrated_non_group":
                should_stop = confidence >= float(row[f"pooled_threshold_{parameter:.2f}"])
            elif method in {"gates_without_ood", "gates_full"}:
                should_stop = confidence >= float(row[f"group_threshold_{parameter:.2f}"])
            elif method == "cbes_style":
                should_stop = (
                    bool(cbes_can_stop and cbes_can_stop[time_index])
                    and int(row["cbes_prediction"]) >= 0
                )
                candidate_prediction = int(row["cbes_prediction"])
            if not should_stop:
                continue
            candidate_ood_score = float(row["ood_score"])
            candidate_ood_threshold = float(row["ood_threshold"])
            if method == "gates_full" and bool(row["is_ood"]):
                decision = "ABSTAIN"
                prediction = candidate_prediction
                stop_time = final_time
            else:
                decision = "STOP"
                prediction = candidate_prediction
                stop_time = time
            break
        records.append(
            {
                "unit_id": unit_id,
                "group_id": unit["group_id"],
                "endpoint": endpoint,
                "method": method,
                "parameter": parameter,
                "decision": decision,
                "prediction": prediction,
                "decision_time": stop_time if decision == "STOP" else final_time,
                "terminal_time": stop_time if decision == "STOP" else final_time,
                "saved_fraction": (final_time - stop_time) / final_time
                if decision == "STOP"
                else 0.0,
                "ood_score": candidate_ood_score,
                "ood_threshold": candidate_ood_threshold,
            }
        )
    return pd.DataFrame(records)


def run_phase2_campaign(
    dataset: LongitudinalDataset,
    output_dir: Path,
    seed: int,
    commit_hash: str,
    bootstrap_estimators: int = 12,
    model_kind: str = "logistic",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_frames = {time: generic_prefix_features(dataset, time) for time in DECISION_TIMES}
    groups = sorted(dataset.frame["group_id"].unique())
    all_decisions: list[pd.DataFrame] = []
    result_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold, test_group in enumerate(groups):
        calibration_groups = [groups[(fold + 1) % len(groups)], groups[(fold + 2) % len(groups)]]
        train_groups = [group for group in groups if group not in {test_group, *calibration_groups}]
        unit_outcomes = dataset.frame.drop_duplicates("unit_id").set_index("unit_id")["endpoint"]
        cbes_calibration_ids = sorted(
            dataset.frame.loc[
                dataset.frame["group_id"] == calibration_groups[0], "unit_id"
            ].unique()
        )
        cbes_validation_ids = sorted(
            dataset.frame.loc[
                dataset.frame["group_id"] == calibration_groups[1], "unit_id"
            ].unique()
        )
        states: dict[float, pd.DataFrame] = {}
        cbes_calibration_decisions: list[np.ndarray] = []
        cbes_validation_decisions: list[np.ndarray] = []
        cbes_calibration_labels = unit_outcomes.loc[cbes_calibration_ids].to_numpy(int)
        cbes_validation_labels = unit_outcomes.loc[cbes_validation_ids].to_numpy(int)

        for time_index, time in enumerate(DECISION_TIMES):
            frame = prefix_frames[time]
            columns = generic_feature_columns(frame)
            train = frame.loc[frame["group_id"].isin(train_groups)].reset_index(drop=True)
            calibration = frame.loc[frame["group_id"].isin(calibration_groups)].reset_index(
                drop=True
            )
            calibration_one = calibration.loc[
                calibration["group_id"] == calibration_groups[0]
            ].reset_index(drop=True)
            validation_one = calibration.loc[
                calibration["group_id"] == calibration_groups[1]
            ].reset_index(drop=True)
            test = frame.loc[frame["group_id"] == test_group].reset_index(drop=True)

            model = make_classifier(seed + fold * 100 + time_index, model_kind)
            model.fit(train[columns], train["endpoint"].to_numpy(int))
            calibration_probability = model.predict_proba(calibration[columns])[:, 1]
            test_probability = model.predict_proba(test[columns])[:, 1]

            gate = MahalanobisGate(0.95).fit(train[columns].to_numpy(float))
            calibration_ood = gate.score(calibration[columns].to_numpy(float))
            per_group_ood = [
                np.quantile(
                    calibration_ood[calibration["group_id"].to_numpy() == calibration_group],
                    0.95,
                )
                for calibration_group in calibration_groups
            ]
            gate.threshold_ = float(max(per_group_ood))
            test_ood_score = gate.score(test[columns].to_numpy(float))

            bootstrap_calibration_one, bootstrap_validation_one, bootstrap_test = (
                _bootstrap_probabilities(
                    train,
                    [calibration_one, validation_one, test],
                    columns,
                    seed + fold * 1000 + time_index * 20,
                    estimators=bootstrap_estimators,
                    model_kind=model_kind,
                )
            )
            calibration_decision = pd.Series(
                confidence_bound_decision(
                    bootstrap_calibration_one.mean(axis=0),
                    bootstrap_calibration_one.std(axis=0),
                    1.96,
                ),
                index=calibration_one["unit_id"],
            )
            validation_decision = pd.Series(
                confidence_bound_decision(
                    bootstrap_validation_one.mean(axis=0),
                    bootstrap_validation_one.std(axis=0),
                    1.96,
                ),
                index=validation_one["unit_id"],
            )
            cbes_calibration_decisions.append(
                calibration_decision.reindex(cbes_calibration_ids, fill_value=-1).to_numpy(int)
            )
            cbes_validation_decisions.append(
                validation_decision.reindex(cbes_validation_ids, fill_value=-1).to_numpy(int)
            )
            cbes_test_prediction = confidence_bound_decision(
                bootstrap_test.mean(axis=0), bootstrap_test.std(axis=0), 1.96
            )
            state = test[
                [
                    "unit_id",
                    "group_id",
                    "endpoint",
                    "observations_available",
                    "last_observation_time",
                ]
            ].copy()
            state["fold"] = fold
            state["test_group"] = test_group
            state["decision_time"] = time
            state["probability"] = test_probability
            state["cbes_probability"] = bootstrap_test.mean(axis=0)
            state["cbes_std"] = bootstrap_test.std(axis=0)
            state["cbes_prediction"] = cbes_test_prediction
            state["ood_score"] = test_ood_score
            state["ood_threshold"] = gate.threshold_
            state["is_ood"] = test_ood_score > float(gate.threshold_)

            for risk in RISK_GRID:
                pooled = calibrate_threshold(
                    calibration_probability,
                    calibration["endpoint"].to_numpy(int),
                    risk,
                    min_stops=10,
                )
                grouped = calibrate_group_threshold(
                    calibration_probability,
                    calibration["endpoint"].to_numpy(int),
                    calibration["group_id"].to_numpy(),
                    risk,
                    minimum_stops_per_group=5,
                )
                state[f"pooled_threshold_{risk:.2f}"] = pooled.confidence_threshold
                state[f"group_threshold_{risk:.2f}"] = grouped.confidence_threshold
                calibration_rows.append(
                    {
                        "fold": fold,
                        "test_group": test_group,
                        "decision_time": time,
                        "risk_target": risk,
                        "pooled_threshold": pooled.confidence_threshold,
                        "pooled_errors": pooled.calibration_errors,
                        "pooled_stops": pooled.calibration_stops,
                        "group_threshold": grouped.confidence_threshold,
                        "group_errors": grouped.calibration_errors,
                        "group_stops": grouped.calibration_stops,
                        "worst_group_risk": grouped.worst_group_risk,
                        "group_upper_risk": grouped.upper_risk,
                        "group_certified": grouped.certified,
                        "ood_threshold": gate.threshold_,
                    }
                )
            states[time] = state
            prediction_rows.append(state)

        cbes_calibrations = {
            risk: sequential_calibrate_cbes(
                cbes_calibration_labels,
                cbes_calibration_decisions,
                cbes_validation_labels,
                cbes_validation_decisions,
                alpha=risk,
            )
            for risk in RISK_GRID
        }
        unit_frame = (
            dataset.frame.loc[dataset.frame["group_id"] == test_group]
            .drop_duplicates("unit_id")[["unit_id", "group_id", "endpoint"]]
            .sort_values("unit_id")
        )
        method_specs: list[tuple[str, float | None]] = [("full_protocol", None)]
        method_specs.extend(("fixed_time", time) for time in DECISION_TIMES[:-1])
        method_specs.extend(("naive_confidence", threshold) for threshold in NAIVE_GRID)
        for risk in RISK_GRID:
            method_specs.extend(
                [
                    ("calibrated_non_group", risk),
                    ("cbes_style", risk),
                    ("gates_without_ood", risk),
                    ("gates_full", risk),
                ]
            )
        for method, parameter in method_specs:
            cbes = cbes_calibrations.get(float(parameter)) if method == "cbes_style" else None
            decisions = _simulate(
                states,
                unit_frame,
                dataset.final_time,
                method,
                parameter,
                cbes.can_stop if cbes else None,
            )
            decisions["fold"] = fold
            decisions["test_group"] = test_group
            if cbes:
                decisions["cbes_validation_bound"] = cbes.validation_bound
            all_decisions.append(decisions)
            risk_target = (
                float(parameter)
                if method
                in {
                    "calibrated_non_group",
                    "cbes_style",
                    "gates_without_ood",
                    "gates_full",
                }
                else None
            )
            operating_point = "none" if parameter is None else str(parameter)
            result_rows.append(
                _metric_row(
                    decisions,
                    dataset.name,
                    str(test_group),
                    method,
                    operating_point,
                    risk_target,
                    seed,
                    commit_hash,
                )
            )
            if cbes:
                calibration_rows.append(
                    {
                        "fold": fold,
                        "test_group": test_group,
                        "decision_time": "sequential",
                        "risk_target": parameter,
                        "method": "cbes_style",
                        **asdict(cbes),
                    }
                )

    decisions_frame = pd.concat(all_decisions, ignore_index=True)
    results = pd.DataFrame(result_rows)
    calibrations = pd.DataFrame(calibration_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    decisions_frame.to_csv(output_dir / "unit_decisions.csv", index=False)
    results.to_csv(output_dir / "master_results.csv", index=False)
    calibrations.to_csv(output_dir / "calibration_audit.csv", index=False)
    predictions.to_csv(output_dir / "predictions_by_time.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for (method, operating_point), frame in decisions_frame.groupby(
        ["method", "parameter"], dropna=False
    ):
        stopped = frame.loc[frame["decision"] == "STOP"]
        errors = int((stopped["prediction"] != stopped["endpoint"]).sum())
        interval = exact_binomial_interval(errors, len(stopped))
        aggregate_rows.append(
            {
                **_metric_row(
                    frame,
                    dataset.name,
                    "ALL",
                    str(method),
                    "none" if pd.isna(operating_point) else str(operating_point),
                    float(operating_point)
                    if method
                    in {
                        "calibrated_non_group",
                        "cbes_style",
                        "gates_without_ood",
                        "gates_full",
                    }
                    and not pd.isna(operating_point)
                    else None,
                    seed,
                    commit_hash,
                ),
                "EESR_exact_lower": interval.lower,
                "EESR_exact_upper": interval.upper,
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "aggregate_results.csv", index=False)
    return {
        "dataset": dataset.name,
        "independent_units": dataset.independent_units,
        "independent_groups": dataset.independent_groups,
        "output_dir": str(output_dir),
    }
