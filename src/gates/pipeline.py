from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from gates.config import Config
from gates.data.organoid import audit_dataset, build_longitudinal_table
from gates.evaluation.metrics import stopping_metrics
from gates.features.prefix import feature_columns, prefix_features
from gates.models.classifier import make_classifier
from gates.policy.stopping import calibrate_threshold, decide
from gates.shift.mahalanobis import MahalanobisGate


def _partitions(replicates: list[int]) -> list[tuple[int, int, int]]:
    if len(replicates) != 3:
        raise ValueError("OrganoID endpoint table must expose exactly three endpoint replicates")
    return [
        (replicates[0], replicates[1], replicates[2]),
        (replicates[1], replicates[2], replicates[0]),
        (replicates[2], replicates[0], replicates[1]),
    ]


def _label(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    return (frame["viability"].to_numpy(float) < threshold).astype(int)


def run_experiment(root: Path, config: Config) -> dict[str, object]:
    raw_dir = root / "data" / "raw"
    evidence_dir = root / "artifacts" / "evidence"
    figure_dir = root / "artifacts" / "figures"
    model_dir = root / "artifacts" / "models"
    for path in (evidence_dir, figure_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)

    longitudinal = build_longitudinal_table(raw_dir)
    audit = audit_dataset(raw_dir)
    replicates = sorted(int(value) for value in longitudinal["replicate"].unique())
    prediction_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []

    for fold, (train_rep, calibration_rep, test_rep) in enumerate(_partitions(replicates)):
        for decision_time in config.decision_times:
            frame = prefix_features(longitudinal, decision_time)
            columns = feature_columns(frame)
            train = frame.loc[frame["replicate"] == train_rep].reset_index(drop=True)
            calibration = frame.loc[frame["replicate"] == calibration_rep].reset_index(drop=True)
            test = frame.loc[frame["replicate"] == test_rep].reset_index(drop=True)
            if set(train["unit_id"]) & set(calibration["unit_id"]):
                raise AssertionError("training and calibration units overlap")
            if set(train["unit_id"]) & set(test["unit_id"]):
                raise AssertionError("training and test units overlap")

            model = make_classifier(config.seed + fold + decision_time)
            model.fit(train[columns], _label(train, config.endpoint_threshold))
            calibration_probability = model.predict_proba(calibration[columns])[:, 1]
            threshold = calibrate_threshold(
                calibration_probability,
                _label(calibration, config.endpoint_threshold),
                config.target_risk,
            )
            gate = MahalanobisGate(config.ood_quantile).fit(train[columns].to_numpy(float))
            test_probability = model.predict_proba(test[columns])[:, 1]
            test_ood_score = gate.score(test[columns].to_numpy(float))
            test_ood = test_ood_score > float(gate.threshold_)

            calibration_rows.append(
                {
                    "fold": fold,
                    "decision_time": decision_time,
                    "train_replicate": train_rep,
                    "calibration_replicate": calibration_rep,
                    "test_replicate": test_rep,
                    **threshold.__dict__,
                    "ood_threshold": float(gate.threshold_),
                }
            )
            joblib.dump(
                {
                    "model": model,
                    "gate": gate,
                    "features": columns,
                    "threshold": threshold,
                    "decision_time": decision_time,
                    "fold": fold,
                },
                model_dir / f"fold_{fold}_time_{decision_time}.joblib",
            )
            for index, row in test.iterrows():
                probability = float(test_probability[index])
                is_ood = bool(test_ood[index])
                policy_decision = decide(probability, threshold.confidence_threshold, is_ood)
                prediction_rows.append(
                    {
                        "fold": fold,
                        "unit_id": row["unit_id"],
                        "dosage": float(row["dosage"]),
                        "replicate": int(row["replicate"]),
                        "viability": float(row["viability"]),
                        "label": int(row["viability"] < config.endpoint_threshold),
                        "decision_time": decision_time,
                        "probability_response": probability,
                        "prediction": int(probability >= 0.5),
                        "confidence": max(probability, 1 - probability),
                        "confidence_threshold": threshold.confidence_threshold,
                        "calibration_certified": threshold.certified,
                        "ood_score": float(test_ood_score[index]),
                        "ood_threshold": float(gate.threshold_),
                        "is_ood": is_ood,
                        "policy_decision": policy_decision,
                        "decision": "CONTINUE"
                        if decision_time < config.minimum_stop_time
                        else policy_decision,
                    }
                )

    predictions = pd.DataFrame(prediction_rows).sort_values(["unit_id", "decision_time"])
    calibrations = pd.DataFrame(calibration_rows).sort_values(["fold", "decision_time"])
    predictions.to_csv(evidence_dir / "predictions_by_time.csv", index=False)
    calibrations.to_csv(evidence_dir / "calibration_by_time.csv", index=False)

    decisions: list[dict[str, object]] = []
    final_time = max(config.decision_times)
    for _, trajectory in predictions.groupby("unit_id", sort=True):
        trajectory = trajectory.sort_values("decision_time")
        eligible = trajectory.loc[trajectory["decision"].isin(["STOP", "ABSTAIN"])]
        chosen = eligible.iloc[0] if len(eligible) else trajectory.iloc[-1]
        record = chosen.to_dict()
        if not len(eligible):
            record["decision"] = "CONTINUE"
        record["observations_saved"] = int((final_time - record["decision_time"]) / 4)
        decisions.append(record)
    decision_frame = pd.DataFrame(decisions).sort_values("unit_id")
    decision_frame.to_csv(evidence_dir / "decisions.csv", index=False)

    risk_savings_rows: list[dict[str, float]] = []
    for minimum_time in config.decision_times[:-1]:
        candidate_records: list[dict[str, object]] = []
        for _, trajectory in predictions.groupby("unit_id", sort=True):
            trajectory = trajectory.sort_values("decision_time")
            eligible = trajectory.loc[
                (trajectory["decision_time"] >= minimum_time)
                & trajectory["policy_decision"].isin(["STOP", "ABSTAIN"])
            ]
            chosen = eligible.iloc[0] if len(eligible) else trajectory.iloc[-1]
            record = chosen.to_dict()
            if not len(eligible):
                record["decision"] = "CONTINUE"
            else:
                record["decision"] = record["policy_decision"]
            candidate_records.append(record)
        point = stopping_metrics(candidate_records, final_time)
        risk_savings_rows.append(
            {
                "minimum_stop_time": float(minimum_time),
                "eesr": point["eesr"],
                "coverage": point["coverage"],
                "observation_savings": point["observation_savings"],
            }
        )
    risk_savings = pd.DataFrame(risk_savings_rows)
    risk_savings.to_csv(evidence_dir / "risk_savings.csv", index=False)

    time_rows: list[dict[str, float]] = []
    for decision_time, frame in predictions.groupby("decision_time"):
        labels = frame["label"].to_numpy(int)
        probabilities = frame["probability_response"].to_numpy(float)
        predicted = (probabilities >= 0.5).astype(int)
        time_rows.append(
            {
                "decision_time": float(decision_time),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "auroc": float(roc_auc_score(labels, probabilities)),
                "brier": float(brier_score_loss(labels, probabilities)),
            }
        )
    by_time = pd.DataFrame(time_rows).sort_values("decision_time")
    by_time.to_csv(evidence_dir / "performance_by_time.csv", index=False)

    metrics = stopping_metrics(decisions, final_time)
    metrics.update(
        {
            "target_risk": config.target_risk,
            "minimum_stop_time": config.minimum_stop_time,
            "endpoint_threshold": config.endpoint_threshold,
            "calibration_claim": "empirical_only"
            if not calibrations["certified"].any()
            else "confidence_bound_supported",
            "independent_endpoint_units": int(audit["shared_units"]),
            "evaluation_design": "cyclic replicate holdout: train/calibrate/test",
        }
    )
    (evidence_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (evidence_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    _make_figures(by_time, risk_savings, config.minimum_stop_time, figure_dir)
    return metrics


def _make_figures(
    by_time: pd.DataFrame,
    risk_savings: pd.DataFrame,
    selected_minimum_time: int,
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(by_time["decision_time"], by_time["balanced_accuracy"], marker="o")
    axes[0].plot(by_time["decision_time"], by_time["auroc"], marker="s")
    axes[0].set(xlabel="Hours observed", ylabel="Held-out score", ylim=(0, 1.03))
    axes[0].legend(["Balanced accuracy", "AUROC"], frameon=False)
    axes[1].plot(
        risk_savings["observation_savings"],
        risk_savings["eesr"],
        marker="o",
        color="#3155A4",
    )
    selected = risk_savings.loc[risk_savings["minimum_stop_time"] == selected_minimum_time]
    axes[1].scatter(
        selected["observation_savings"],
        selected["eesr"],
        s=130,
        color="#D62728",
        edgecolor="black",
        zorder=3,
        label=f"Selected policy ({selected_minimum_time} h)",
    )
    axes[1].set(
        xlabel="Mean fraction of observations saved",
        ylabel="Erroneous early-stop rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].set_title("Held-out risk–savings tradeoff")
    axes[1].legend(frameon=False)
    fig.suptitle("GATES feasibility evidence on held-out OrganoID replicates")
    fig.tight_layout()
    fig.savefig(output / "feasibility.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
