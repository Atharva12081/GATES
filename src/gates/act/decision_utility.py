from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from gates.act.benchmark_v2 import _seed_for
from gates.act.benchmark_v3 import V3Config, _load_sources
from gates.act.model_v3 import RiskSensitiveSurface
from gates.act.neural_mea import neural_coordinates

POLICIES = (
    "decision_voi",
    "max_variance",
    "maximin",
    "fixed_sentinel",
    "random",
    "risk_sensitive",
)
BUDGET_ROUNDS = (1, 2, 3)
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0


def reference_active(response: np.ndarray, threshold: float = 0.40) -> bool:
    """Final-assay activity call; this is not an in-vivo toxicity label."""
    treated_hits = np.abs(np.asarray(response).reshape(4, 8)[:, 1:]) >= threshold
    return bool(np.sum(treated_hits.sum(axis=1) >= 2) >= 2)


def _fold(group: str, namespace: str, folds: int = 5) -> int:
    digest = hashlib.sha256(f"{namespace}/{group}".encode()).hexdigest()[:8]
    return int(digest, 16) % folds


class AnalogDecisionModel:
    """Weighted analog posterior for the predeclared final-assay activity call."""

    def __init__(self, responses: np.ndarray, labels: np.ndarray, neighbors: int = 20):
        self.responses = np.asarray(responses, dtype=float)
        self.labels = np.asarray(labels, dtype=float)
        self.neighbors = min(int(neighbors), len(self.labels))
        self.center = np.median(self.responses, axis=0)
        self.scale = np.maximum(
            np.quantile(np.abs(self.responses - self.center), 0.75, axis=0), 0.08
        )
        self.prevalence = float(np.mean(self.labels))

    def weights(self, observed: list[int], values: np.ndarray) -> np.ndarray:
        standardized = (
            self.responses[:, np.asarray(observed)] - np.asarray(values)[None, :]
        ) / self.scale[np.asarray(observed)][None, :]
        distance = np.sqrt(np.mean(standardized**2, axis=1))
        keep = np.argsort(distance)[: self.neighbors]
        bandwidth = max(float(np.quantile(distance[keep], 0.35)), 0.10)
        weights = np.zeros(len(distance), dtype=float)
        weights[keep] = np.exp(-0.5 * (distance[keep] / bandwidth) ** 2)
        weights /= max(float(weights.sum()), 1e-12)
        return weights

    def probability(self, observed: list[int], values: np.ndarray) -> float:
        weights = self.weights(observed, values)
        analog = float(weights @ self.labels)
        # Mild shrinkage avoids unjustified 0/1 probabilities in small local neighborhoods.
        return float(np.clip(0.90 * analog + 0.10 * self.prevalence, 0.0, 1.0))

    @staticmethod
    def decision_risk(probability: float) -> float:
        return min(FALSE_NEGATIVE_COST * probability, FALSE_POSITIVE_COST * (1 - probability))

    def voi_score(
        self,
        observed: list[int],
        values: np.ndarray,
        candidate: int,
        batch_selected: list[int],
    ) -> float:
        weights = self.weights(observed, values)
        probability = float(weights @ self.labels)
        current_risk = self.decision_risk(probability)
        expected_risk = 0.0
        bandwidth = max(float(self.scale[candidate]) * 0.60, 0.04)
        candidate_values = self.responses[:, candidate]
        for value, predictive_weight in zip(candidate_values, weights, strict=True):
            if predictive_weight <= 1e-10:
                continue
            likelihood = np.exp(-0.5 * ((candidate_values - value) / bandwidth) ** 2)
            posterior = weights * likelihood
            posterior /= max(float(posterior.sum()), 1e-12)
            expected_risk += float(predictive_weight) * self.decision_risk(
                float(posterior @ self.labels)
            )
        reduction = max(current_risk - expected_risk, 0.0)
        if batch_selected:
            correlations = np.corrcoef(
                self.responses[:, [candidate, *batch_selected]], rowvar=False
            )[0, 1:]
            redundancy = float(np.nanmax(np.nan_to_num(correlations**2, nan=1.0)))
            reduction *= max(1.0 - redundancy, 0.02)
        return reduction


def _select_batch(
    policy: str,
    surface: RiskSensitiveSurface,
    decision_model: AnalogDecisionModel,
    observed: list[int],
    values: np.ndarray,
    time_rank: int,
    rng: np.random.Generator,
    config: V3Config,
) -> list[int]:
    candidates = np.arange(8 * time_rank, 8 * (time_rank + 1), dtype=int)
    if policy == "fixed_sentinel":
        return [8 * time_rank + rank for rank in (0, 2, 5, 7)]
    prediction = surface.fit_predict(observed, values, config.effect_threshold)
    selected = [int(candidates[0])]
    covariance = prediction.covariance.copy()
    for _ in range(3):
        available = np.asarray([i for i in candidates if i not in selected], dtype=int)
        if policy == "random":
            chosen = int(rng.choice(available))
        elif policy == "max_variance":
            chosen = int(available[np.argmax(np.diag(covariance)[available])])
        elif policy == "maximin":
            anchors = np.asarray(observed + selected, dtype=int)
            distances = np.min(
                np.linalg.norm(
                    surface.coordinates[available, None, :]
                    - surface.coordinates[anchors][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            chosen = int(available[np.argmax(distances)])
        elif policy == "risk_sensitive":
            targets = np.asarray(
                [i for i in range(32) if i not in observed + selected], dtype=int
            )
            scores = [
                surface.risk_sensitive_score(
                    prediction,
                    int(candidate),
                    targets,
                    config.effect_weight,
                    config.direct_bonus,
                )
                for candidate in available
            ]
            chosen = int(available[int(np.argmax(scores))])
        elif policy == "decision_voi":
            chosen = int(
                max(
                    available,
                    key=lambda candidate: decision_model.voi_score(
                        observed,
                        values,
                        int(candidate),
                        selected[1:],
                    ),
                )
            )
        else:
            raise ValueError(f"unknown policy: {policy}")
        selected.append(chosen)
        denominator = max(float(covariance[chosen, chosen] + surface.noise), 1e-12)
        covariance -= np.outer(covariance[:, chosen], covariance[chosen, :]) / denominator
    return selected


def _run_path(
    truth: np.ndarray,
    run_id: str,
    policy: str,
    surface: RiskSensitiveSurface,
    model: AnalogDecisionModel,
    config: V3Config,
    seed: int,
) -> list[dict[str, object]]:
    observed = list(range(8))
    rng = np.random.default_rng(_seed_for(f"decision-utility/{run_id}/{policy}", seed))
    rows = []
    for round_index in BUDGET_ROUNDS:
        if round_index > 1:
            observed.extend(
                _select_batch(
                    policy,
                    surface,
                    model,
                    observed,
                    truth[observed],
                    round_index - 1,
                    rng,
                    config,
                )
            )
        observed_effects = int(
            np.sum(np.abs(truth[np.asarray(observed)]) >= config.effect_threshold)
        )
        rows.append(
            {
                "round": round_index,
                "div": (5, 7, 9)[round_index - 1],
                "observations": len(observed),
                "raw_probability": model.probability(observed, truth[observed]),
                "observed_effects": observed_effects,
                "selected_indices": ";".join(map(str, observed)),
            }
        )
    return rows


class _Calibrator:
    def __init__(self, method: str):
        self.method = method
        self.model: object | None = None

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> _Calibrator:
        x = np.asarray(probability, dtype=float)
        y = np.asarray(labels, dtype=int)
        if self.method == "logistic":
            self.model = LogisticRegression(C=1.0, solver="lbfgs").fit(x[:, None], y)
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        x = np.asarray(probability, dtype=float)
        if self.method == "uncalibrated":
            return x
        if self.method == "logistic":
            return self.model.predict_proba(x[:, None])[:, 1]  # type: ignore[union-attr]
        return self.model.predict(x)  # type: ignore[union-attr]


def _choose_calibrator(probability: np.ndarray, labels: np.ndarray, groups: np.ndarray):
    methods = ("uncalibrated", "logistic", "isotonic")
    scores: dict[str, list[float]] = {method: [] for method in methods}
    split = np.asarray([_fold(str(group), "calibrator", 5) for group in groups])
    for held_out in range(5):
        train = split != held_out
        test = ~train
        if not np.any(test) or len(np.unique(labels[train])) < 2:
            continue
        for method in methods:
            calibrator = _Calibrator(method).fit(probability[train], labels[train])
            scores[method].append(
                brier_score_loss(labels[test], calibrator.predict(probability[test]))
            )
    mean_scores = {method: float(np.mean(value)) for method, value in scores.items() if value}
    selected = min(mean_scores, key=mean_scores.get)
    return _Calibrator(selected).fit(probability, labels), selected, mean_scores


def _choose_threshold(probability: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    rows = []
    for threshold in candidates:
        prediction = probability >= threshold
        tp = int(np.sum(prediction & (labels == 1)))
        fn = int(np.sum(~prediction & (labels == 1)))
        fp = int(np.sum(prediction & (labels == 0)))
        sensitivity = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        loss = (FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp) / len(labels)
        if sensitivity >= 0.90:
            rows.append((loss, -precision, -float(threshold), float(threshold)))
    return min(rows)[-1] if rows else 0.0


def _group_probabilities(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["compound_group", "policy", "round"], as_index=False)
        .agg(
            raw_probability=("raw_probability", "max"),
            label=("label", "max"),
            observed_effects=("observed_effects", "max"),
            div=("div", "first"),
            observations=("observations", "first"),
            selected_indices=("selected_indices", lambda values: "|".join(values)),
        )
        .reset_index(drop=True)
    )


def _evaluate_split(
    fit_meta: pd.DataFrame,
    eval_meta: pd.DataFrame,
    responses: dict[str, np.ndarray],
    config: V3Config,
    seed: int,
) -> pd.DataFrame:
    training = np.vstack([responses[run_id] for run_id in fit_meta["run_id"]])
    labels = np.asarray([reference_active(row) for row in training], dtype=int)
    surface = RiskSensitiveSurface(
        neural_coordinates(),
        training,
        config.covariance_blend,
        config.noise_fraction,
        config.analog_weight,
        config.neighbor_count,
    )
    decision_model = AnalogDecisionModel(training, labels)
    rows = []
    for record in eval_meta.itertuples(index=False):
        truth = responses[record.run_id]
        for policy in POLICIES:
            for result in _run_path(
                truth, record.run_id, policy, surface, decision_model, config, seed
            ):
                rows.append(
                    {
                        "run_id": record.run_id,
                        "compound_group": record.compound_group,
                        "source": record.source,
                        "policy": policy,
                        "label": int(reference_active(truth)),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    label = frame["label"].to_numpy(bool)
    prediction = frame["alert"].to_numpy(bool)
    tp = int(np.sum(label & prediction))
    tn = int(np.sum(~label & ~prediction))
    fp = int(np.sum(~label & prediction))
    fn = int(np.sum(label & ~prediction))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "groups": len(frame),
        "active_groups": int(label.sum()),
        "sensitivity": sensitivity,
        "precision": tp / max(tp + fp, 1),
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "false_alerts": fp,
        "false_negatives": fn,
        "decision_loss": (FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp) / len(frame),
        "brier": brier_score_loss(label.astype(int), frame["probability"]),
    }


def _group_labels(meta: pd.DataFrame, responses: dict[str, np.ndarray]) -> pd.Series:
    rows = pd.DataFrame(
        {
            "compound_group": meta["compound_group"].to_numpy(),
            "label": [int(reference_active(responses[run_id])) for run_id in meta["run_id"]],
        }
    )
    return rows.groupby("compound_group")["label"].max()


def _paired_loss_comparisons(
    predictions: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    primary = predictions.loc[predictions["round"].eq(2)].copy()
    voi = primary.loc[primary["policy"].eq("decision_voi")].set_index("compound_group")
    rows = []
    variance_improvement = np.asarray([], dtype=float)
    variance_aligned = pd.DataFrame()
    for baseline in primary["policy"].unique():
        if baseline == "decision_voi":
            continue
        other = primary.loc[primary["policy"].eq(baseline)].set_index("compound_group")
        aligned = voi.join(other, lsuffix="_voi", rsuffix="_baseline")
        loss_voi = (
            FALSE_NEGATIVE_COST * (aligned["label_voi"].eq(1) & ~aligned["alert_voi"])
            + FALSE_POSITIVE_COST * (aligned["label_voi"].eq(0) & aligned["alert_voi"])
        ).astype(float)
        loss_baseline = (
            FALSE_NEGATIVE_COST
            * (aligned["label_baseline"].eq(1) & ~aligned["alert_baseline"])
            + FALSE_POSITIVE_COST
            * (aligned["label_baseline"].eq(0) & aligned["alert_baseline"])
        ).astype(float)
        improvement = (loss_baseline - loss_voi).to_numpy()
        rng = np.random.default_rng(_seed_for(f"decision-comparison/{baseline}", seed))
        boot = np.asarray(
            [rng.choice(improvement, len(improvement), replace=True).mean() for _ in range(20_000)]
        )
        rows.append(
            {
                "baseline": baseline,
                "loss_improvement": float(np.mean(improvement)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "groups": len(improvement),
            }
        )
        if baseline == "max_variance":
            variance_improvement = improvement
            variance_aligned = aligned
    return pd.DataFrame(rows), variance_improvement, variance_aligned


def _alert_timing(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, part in predictions.groupby("policy"):
        active = part.loc[part["label"].eq(1)].copy()
        first = (
            active.loc[active["alert"]]
            .groupby("compound_group", as_index=False)
            .agg(first_alert_round=("round", "min"), first_alert_div=("div", "min"))
        )
        active_groups = active["compound_group"].nunique()
        rows.append(
            {
                "policy": policy,
                "active_groups": active_groups,
                "alerted_by_div7": int((first["first_alert_round"] <= 2).sum()),
                "alerted_by_div9": int((first["first_alert_round"] <= 3).sum()),
                "median_first_correct_alert_div": (
                    float(first["first_alert_div"].median()) if len(first) else np.nan
                ),
                "never_alerted_by_div9": int(active_groups - len(first)),
            }
        )
    return pd.DataFrame(rows)


def _metric_intervals(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    metric_names = (
        "sensitivity",
        "precision",
        "specificity",
        "balanced_accuracy",
        "decision_loss",
        "brier",
    )
    for (policy, round_index), part in predictions.groupby(["policy", "round"]):
        point = _metrics(part)
        rng = np.random.default_rng(
            _seed_for(f"decision-metric-interval/{policy}/{round_index}", seed)
        )
        indices = rng.integers(0, len(part), size=(5_000, len(part)))
        label = part["label"].to_numpy(bool)[indices]
        alert = part["alert"].to_numpy(bool)[indices]
        probability = part["probability"].to_numpy(float)[indices]
        tp = np.sum(label & alert, axis=1)
        tn = np.sum(~label & ~alert, axis=1)
        fp = np.sum(~label & alert, axis=1)
        fn = np.sum(label & ~alert, axis=1)
        sensitivity = tp / np.maximum(tp + fn, 1)
        specificity = tn / np.maximum(tn + fp, 1)
        values = {
            "sensitivity": sensitivity,
            "precision": tp / np.maximum(tp + fp, 1),
            "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "decision_loss": (
                FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp
            )
            / len(part),
            "brier": np.mean((probability - label) ** 2, axis=1),
        }
        for metric in metric_names:
            rows.append(
                {
                    "policy": policy,
                    "round": round_index,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_low": float(np.quantile(values[metric], 0.025)),
                    "ci_high": float(np.quantile(values[metric], 0.975)),
                }
            )
    return pd.DataFrame(rows)


def run_decision_utility_study(
    legacy_workbook: Path,
    phase1_workbook: Path,
    preregistration: Path,
    output_dir: Path,
) -> dict[str, object]:
    lock = json.loads(preregistration.read_text())
    config = V3Config(**lock["selected_config"])
    seed = int(lock["seed"])
    _, _, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    outer_eligible = meta.loc[meta["v3_split"].eq("train")].copy()
    support = meta.loc[meta["v3_split"].eq("development")].copy()
    outer_eligible["outer_fold"] = outer_eligible["compound_group"].map(
        lambda value: _fold(str(value), "decision-utility-outer", 5)
    )

    prediction_rows = []
    calibration_rows = []
    for outer_fold in range(5):
        outer_test = outer_eligible.loc[outer_eligible["outer_fold"].eq(outer_fold)].copy()
        outer_train = outer_eligible.loc[~outer_eligible["outer_fold"].eq(outer_fold)].copy()
        inner_parts = []
        inner_namespace = f"decision-utility-inner-{outer_fold}"
        for inner_fold in range(5):
            inner_test = outer_train.loc[
                outer_train["compound_group"].map(
                    lambda value, namespace=inner_namespace: _fold(
                        str(value), namespace, 5
                    )
                ).eq(inner_fold)
            ]
            inner_train = outer_train.loc[~outer_train.index.isin(inner_test.index)]
            inner_parts.append(
                _evaluate_split(
                    pd.concat([support, inner_train], ignore_index=True),
                    inner_test,
                    responses,
                    config,
                    seed,
                )
            )
        inner = _group_probabilities(pd.concat(inner_parts, ignore_index=True))
        outer_raw = _group_probabilities(
            _evaluate_split(
                pd.concat([support, outer_train], ignore_index=True),
                outer_test,
                responses,
                config,
                seed,
            )
        )
        for policy in POLICIES:
            for round_index in BUDGET_ROUNDS:
                inner_part = inner.loc[
                    inner["policy"].eq(policy) & inner["round"].eq(round_index)
                ].copy()
                outer_part = outer_raw.loc[
                    outer_raw["policy"].eq(policy) & outer_raw["round"].eq(round_index)
                ].copy()
                calibrator, method, scores = _choose_calibrator(
                    inner_part["raw_probability"].to_numpy(),
                    inner_part["label"].to_numpy(int),
                    inner_part["compound_group"].to_numpy(),
                )
                inner_probability = calibrator.predict(inner_part["raw_probability"].to_numpy())
                threshold = _choose_threshold(
                    inner_probability, inner_part["label"].to_numpy(int)
                )
                outer_probability = calibrator.predict(outer_part["raw_probability"].to_numpy())
                outer_part["probability"] = outer_probability
                outer_part["threshold"] = threshold
                outer_part["alert"] = outer_probability >= threshold
                outer_part["outer_fold"] = outer_fold
                outer_part["calibrator"] = method
                prediction_rows.append(outer_part)
                calibration_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "policy": policy,
                        "round": round_index,
                        "calibrator": method,
                        "threshold": threshold,
                        "calibrator_cv_brier": scores,
                        "inner_groups": int(inner_part["compound_group"].nunique()),
                    }
                )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    # Decision-only baselines use the common maximum-variance measurement path.
    max_variance = predictions.loc[predictions["policy"].eq("max_variance")].copy()
    simple_rows = []
    for name, minimum in (("any_observed_effect", 1), ("two_observed_effects", 2)):
        part = max_variance.copy()
        part["policy"] = name
        part["probability"] = part["observed_effects"].ge(minimum).astype(float)
        part["threshold"] = 0.5
        part["alert"] = part["observed_effects"].ge(minimum)
        part["calibrator"] = "deterministic_rule"
        simple_rows.append(part)
    predictions = pd.concat([predictions, *simple_rows], ignore_index=True)
    prior_rows = []
    cost_threshold = FALSE_POSITIVE_COST / (FALSE_NEGATIVE_COST + FALSE_POSITIVE_COST)
    for outer_fold in range(5):
        fit_meta = pd.concat(
            [
                support,
                outer_eligible.loc[~outer_eligible["outer_fold"].eq(outer_fold)],
            ],
            ignore_index=True,
        )
        prevalence = float(_group_labels(fit_meta, responses).mean())
        part = max_variance.loc[max_variance["outer_fold"].eq(outer_fold)].copy()
        part["policy"] = "prior_prevalence"
        part["probability"] = prevalence
        part["threshold"] = cost_threshold
        part["alert"] = prevalence >= cost_threshold
        part["calibrator"] = "training_prevalence"
        prior_rows.append(part)
    predictions = pd.concat([predictions, *prior_rows], ignore_index=True)

    metric_rows = []
    for (policy, round_index), part in predictions.groupby(["policy", "round"]):
        metric_rows.append({"policy": policy, "round": round_index, **_metrics(part)})
    metrics = pd.DataFrame(metric_rows)
    timing = _alert_timing(predictions)
    intervals = _metric_intervals(predictions, seed)

    primary = predictions.loc[predictions["round"].eq(2)].copy()
    voi = primary.loc[primary["policy"].eq("decision_voi")].set_index("compound_group")
    variance = primary.loc[primary["policy"].eq("max_variance")].set_index("compound_group")
    comparisons, improvement, aligned = _paired_loss_comparisons(predictions, seed)
    variance_comparison = comparisons.loc[comparisons["baseline"].eq("max_variance")].iloc[0]
    fold_improvement = (
        pd.DataFrame(
            {
                "fold": aligned["outer_fold_voi"].to_numpy(),
                "improvement": improvement,
            }
        )
        .groupby("fold")["improvement"]
        .mean()
    )
    voi_metrics = _metrics(voi.reset_index())
    variance_metrics = _metrics(variance.reset_index())
    conditions = {
        "sensitivity_at_least_0_90": voi_metrics["sensitivity"] >= 0.90,
        "sensitivity_noninferior": (
            voi_metrics["sensitivity"] - variance_metrics["sensitivity"] >= -0.03
        ),
        "precision_gain_at_least_0_10": (
            voi_metrics["precision"] - variance_metrics["precision"] >= 0.10
        ),
        "specificity_gain_at_least_0_10": (
            voi_metrics["specificity"] - variance_metrics["specificity"] >= 0.10
        ),
        "decision_loss_lower": voi_metrics["decision_loss"] < variance_metrics["decision_loss"],
        "loss_bootstrap_ci_excludes_zero": float(variance_comparison["ci_low"]) > 0,
        "four_of_five_folds_positive": int(np.sum(fold_improvement > 0)) >= 4,
    }
    go = bool(all(conditions.values()))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    metrics.to_csv(output_dir / "matched_budget_metrics.csv", index=False)
    timing.to_csv(output_dir / "alert_timing.csv", index=False)
    intervals.to_csv(output_dir / "metric_intervals.csv", index=False)
    comparisons.to_csv(output_dir / "paired_loss_comparisons.csv", index=False)
    pd.DataFrame(calibration_rows).assign(
        calibrator_cv_brier=lambda frame: frame["calibrator_cv_brier"].map(json.dumps)
    ).to_csv(output_dir / "fold_calibration.csv", index=False)
    summary = {
        "schema": "gates-act-decision-utility-study-v1",
        "evidence_level": "nested development; not external validation",
        "outer_evaluation_groups": int(outer_eligible["compound_group"].nunique()),
        "support_development_groups": int(support["compound_group"].nunique()),
        "sealed_groups_used": 0,
        "reference_call": (
            "final-assay NFA activity: >=2 treated cells at |log1p contrast| >=0.40 "
            "on each of >=2 DIVs; not independently established toxicity"
        ),
        "primary_budget": {"round": 2, "div": 7, "observations": 12},
        "candidate_metrics": voi_metrics,
        "max_variance_metrics": variance_metrics,
        "candidate_minus_max_variance": {
            "sensitivity": voi_metrics["sensitivity"] - variance_metrics["sensitivity"],
            "precision": voi_metrics["precision"] - variance_metrics["precision"],
            "specificity": voi_metrics["specificity"] - variance_metrics["specificity"],
            "decision_loss_improvement": float(np.mean(improvement)),
            "decision_loss_improvement_ci": [
                float(variance_comparison["ci_low"]),
                float(variance_comparison["ci_high"]),
            ],
        },
        "fold_loss_improvement": {
            str(key): float(value) for key, value in fold_improvement.items()
        },
        "go_conditions": conditions,
        "go": go,
        "canonical_action": (
            "eligible for independent verification before minimal integration"
            if go
            else "NO-GO: preserve canonical conditional maximum variance and presentation"
        ),
        "stopping": "disabled; ALERT is non-terminating",
        "config": asdict(config),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
