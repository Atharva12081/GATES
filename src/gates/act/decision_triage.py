from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gates.act.benchmark_v3 import V3Config, _load_sources
from gates.act.model_v3 import RiskSensitiveSurface
from gates.act.multivariate_mea import ENDPOINTS, load_multivariate_mea
from gates.act.neural_mea import neural_coordinates

BASE_FEATURES = (
    "abs_mean",
    "std",
    "effect_probability",
    "upper_effect_bound",
    "dose_rank",
    "time_rank",
)
CONSISTENCY_FEATURES = BASE_FEATURES + (
    "nearest_observed_distance",
    "same_div_neighbor_abs",
    "adjacent_div_same_dose_abs",
    "local_observed_mfr_abs",
    "observed_mfr_max_abs",
    "observed_mfr_effect_count",
    "observed_mfr_effect_divs",
)
FULL_FEATURES = CONSISTENCY_FEATURES + (
    "aux_local_mean_abs",
    "aux_local_max_abs",
    "aux_local_active_count",
    "aux_global_mean_abs",
    "aux_global_max_abs",
    "aux_global_active_count",
)
FEATURE_SETS = {
    "posterior": BASE_FEATURES,
    "consistent": CONSISTENCY_FEATURES,
    "multivariate": FULL_FEATURES,
}
MODEL_TYPES = ("logistic", "hist_gradient_boosting")


def _fold(group: str, namespace: str, folds: int = 5) -> int:
    digest = hashlib.sha256(f"{namespace}/{group}".encode()).hexdigest()[:8]
    return int(digest, 16) % folds


def _max_variance_batch(
    surface: RiskSensitiveSurface,
    observed: list[int],
    truth: np.ndarray,
    time_rank: int,
    effect_threshold: float,
) -> list[int]:
    prediction = surface.fit_predict(observed, truth[observed], effect_threshold)
    candidates = np.arange(8 * time_rank, 8 * (time_rank + 1), dtype=int)
    selected = [int(candidates[0])]
    covariance = prediction.covariance.copy()
    for _ in range(3):
        available = np.asarray([value for value in candidates if value not in selected])
        chosen = int(available[np.argmax(np.diag(covariance)[available])])
        selected.append(chosen)
        denominator = max(float(covariance[chosen, chosen] + surface.noise), 1e-12)
        covariance -= np.outer(covariance[:, chosen], covariance[chosen, :]) / denominator
    return selected


def _feature_rows(
    fit_meta: pd.DataFrame,
    eval_meta: pd.DataFrame,
    responses: dict[str, np.ndarray],
    multivariate: dict[str, np.ndarray],
    config: V3Config,
    conformal_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = np.vstack([responses[run_id] for run_id in fit_meta["run_id"]])
    surface = RiskSensitiveSurface(
        neural_coordinates(),
        training,
        config.covariance_blend,
        config.noise_fraction,
        config.analog_weight,
        config.neighbor_count,
    )
    aux_training = np.vstack([multivariate[run_id] for run_id in fit_meta["run_id"]])
    aux_scale = np.maximum(np.quantile(np.abs(aux_training), 0.75, axis=0), 0.05)
    coordinates = neural_coordinates()
    rows: list[dict[str, object]] = []
    direct_rows: list[dict[str, object]] = []
    for record in eval_meta.itertuples(index=False):
        truth = responses[record.run_id]
        endpoints = multivariate[record.run_id]
        observed = list(range(8))
        for round_index in (1, 2, 3):
            if round_index > 1:
                observed.extend(
                    _max_variance_batch(
                        surface,
                        observed,
                        truth,
                        round_index - 1,
                        config.effect_threshold,
                    )
                )
            prediction = surface.fit_predict(
                observed, truth[observed], config.effect_threshold
            )
            observed_array = np.asarray(observed, dtype=int)
            hidden = np.asarray([value for value in range(32) if value not in observed])
            observed_effect = np.abs(truth[observed_array]) >= config.effect_threshold
            direct_rows.append(
                {
                    "run_id": record.run_id,
                    "compound_group": record.compound_group,
                    "round": round_index,
                    "observations": len(observed),
                    "direct_effect_cells": int(observed_effect.sum()),
                    "total_effect_cells": int(
                        np.sum(np.abs(truth) >= config.effect_threshold)
                    ),
                }
            )
            observed_aux = np.abs(endpoints[observed_array, 1:] / aux_scale[1:])
            global_endpoint = np.max(observed_aux, axis=0)
            observed_mfr = truth[observed_array]
            effect_divs = len(
                {
                    index // 8
                    for index, hit in zip(observed, observed_effect, strict=True)
                    if hit
                }
            )
            for index in hidden:
                distance = np.linalg.norm(
                    coordinates[observed_array] - coordinates[index], axis=1
                )
                weights = np.exp(-3.0 * distance)
                weights /= max(float(weights.sum()), 1e-12)
                local_aux = np.sum(observed_aux * weights[:, None], axis=0)
                dose_rank = index % 8
                time_rank = index // 8
                same_div = [
                    candidate
                    for candidate in (index - 1, index + 1)
                    if 0 <= candidate < 32 and candidate // 8 == time_rank
                ]
                adjacent_div = [
                    candidate
                    for candidate in (index - 8, index + 8)
                    if 0 <= candidate < 32
                ]
                rows.append(
                    {
                        "run_id": record.run_id,
                        "compound_group": record.compound_group,
                        "source": record.source,
                        "round": round_index,
                        "observations": len(observed),
                        "cell": int(index),
                        "label": int(abs(truth[index]) >= config.effect_threshold),
                        "abs_mean": abs(float(prediction.mean[index])),
                        "std": float(prediction.std[index]),
                        "effect_probability": float(prediction.effect_probability[index]),
                        "upper_effect_bound": abs(float(prediction.mean[index]))
                        + conformal_q * float(prediction.std[index]),
                        "dose_rank": dose_rank / 7,
                        "time_rank": time_rank / 3,
                        "nearest_observed_distance": float(np.min(distance)),
                        "same_div_neighbor_abs": float(
                            np.max(np.abs(prediction.mean[same_div])) if same_div else 0.0
                        ),
                        "adjacent_div_same_dose_abs": float(
                            np.max(np.abs(prediction.mean[adjacent_div]))
                            if adjacent_div
                            else 0.0
                        ),
                        "local_observed_mfr_abs": float(weights @ np.abs(observed_mfr)),
                        "observed_mfr_max_abs": float(np.max(np.abs(observed_mfr))),
                        "observed_mfr_effect_count": int(observed_effect.sum()),
                        "observed_mfr_effect_divs": effect_divs,
                        "aux_local_mean_abs": float(np.mean(local_aux)),
                        "aux_local_max_abs": float(np.max(local_aux)),
                        "aux_local_active_count": int(np.sum(local_aux >= 1.0)),
                        "aux_global_mean_abs": float(np.mean(global_endpoint)),
                        "aux_global_max_abs": float(np.max(global_endpoint)),
                        "aux_global_active_count": int(np.sum(global_endpoint >= 1.0)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(direct_rows)


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("compound_group")["cell"].transform("count").to_numpy(float)
    weights = 1.0 / counts
    return weights / np.mean(weights)


def _make_model(model_type: str):
    if model_type == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000),
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=20261024,
        )
    raise ValueError(model_type)


def _fit_model(model, x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    if hasattr(model, "steps"):
        model.fit(x, y, logisticregression__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)
    return model


def _candidate_oof(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    model_type: str,
    namespace: str,
) -> np.ndarray:
    output = np.zeros(len(frame), dtype=float)
    folds = frame["compound_group"].map(lambda value: _fold(str(value), namespace, 5))
    for held_out in range(5):
        train = ~folds.eq(held_out)
        test = folds.eq(held_out)
        if not test.any():
            continue
        model = _make_model(model_type)
        _fit_model(
            model,
            frame.loc[train, list(features)].to_numpy(float),
            frame.loc[train, "label"].to_numpy(int),
            _sample_weights(frame.loc[train]),
        )
        output[test] = model.predict_proba(
            frame.loc[test, list(features)].to_numpy(float)
        )[:, 1]
    return output


class _ProbabilityCalibrator:
    def __init__(self, method: str):
        self.method = method
        self.model: object | None = None

    def fit(
        self, probability: np.ndarray, label: np.ndarray, weights: np.ndarray
    ) -> _ProbabilityCalibrator:
        if self.method == "platt":
            self.model = LogisticRegression(C=1.0, max_iter=2_000).fit(
                probability[:, None], label, sample_weight=weights
            )
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(
                probability, label, sample_weight=weights
            )
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.method == "uncalibrated":
            return probability
        if self.method == "platt":
            return self.model.predict_proba(probability[:, None])[:, 1]  # type: ignore[union-attr]
        return self.model.predict(probability)  # type: ignore[union-attr]


def _select_calibrator(
    frame: pd.DataFrame, raw: np.ndarray, namespace: str
) -> tuple[_ProbabilityCalibrator, np.ndarray, dict[str, float]]:
    methods = ("uncalibrated", "platt", "isotonic")
    fold = frame["compound_group"].map(lambda value: _fold(str(value), namespace, 5))
    weights = _sample_weights(frame)
    scores: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    for method in methods:
        out = np.zeros(len(frame), dtype=float)
        for held_out in range(5):
            train = ~fold.eq(held_out)
            test = fold.eq(held_out)
            if not test.any():
                continue
            calibrator = _ProbabilityCalibrator(method).fit(
                raw[train], frame.loc[train, "label"].to_numpy(int), weights[train]
            )
            out[test] = calibrator.predict(raw[test])
        predictions[method] = out
        scores[method] = brier_score_loss(
            frame["label"].to_numpy(int), out, sample_weight=weights
        )
    selected = min(methods, key=lambda method: (scores[method], methods.index(method)))
    calibrator = _ProbabilityCalibrator(selected).fit(
        raw, frame["label"].to_numpy(int), weights
    )
    return calibrator, predictions[selected], scores


def _binary_threshold(probability: np.ndarray, label: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    feasible = []
    for threshold in candidates:
        prediction = probability >= threshold
        tp = np.sum(prediction & (label == 1))
        fp = np.sum(prediction & (label == 0))
        fn = np.sum(~prediction & (label == 1))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        if recall >= 0.95:
            feasible.append((-precision, -threshold, threshold))
    return float(min(feasible)[-1]) if feasible else 0.0


def _triage_thresholds(frame: pd.DataFrame, probability: np.ndarray) -> tuple[float, float]:
    work = frame[["compound_group", "label"]].copy()
    work["probability"] = probability
    groups = work["compound_group"].nunique()
    grid = np.linspace(0.0, 1.0, 101)
    high_feasible = []
    low_feasible = []
    for threshold in grid:
        confirmed = work["probability"].ge(threshold)
        no_effect = work["probability"].le(threshold)
        fdp = []
        fnr = []
        for _, part in work.groupby("compound_group"):
            local_confirmed = confirmed.loc[part.index]
            local_no_effect = no_effect.loc[part.index]
            label = part["label"].to_numpy(bool)
            fdp.append(
                float(np.sum(local_confirmed.to_numpy() & ~label))
                / max(int(local_confirmed.sum()), 1)
            )
            fnr.append(
                float(np.sum(local_no_effect.to_numpy() & label)) / max(int(label.sum()), 1)
            )
        if (sum(fdp) + 1) / (groups + 1) <= 0.40:
            high_feasible.append(float(threshold))
        if (sum(fnr) + 1) / (groups + 1) <= 0.05:
            low_feasible.append(float(threshold))
    high = min(high_feasible) if high_feasible else 1.01
    low = max(low_feasible) if low_feasible else -0.01
    if low >= high:
        low = high - 0.01
    return low, high


def _binary_metrics(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    label = frame["label"].to_numpy(bool)
    prediction = frame[column].to_numpy(bool)
    tp = int(np.sum(label & prediction))
    fp = int(np.sum(~label & prediction))
    fn = int(np.sum(label & ~prediction))
    tn = int(np.sum(~label & ~prediction))
    return {
        "hidden_cells": len(frame),
        "effect_cells": int(label.sum()),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "specificity": tn / max(tn + fp, 1),
        "false_discoveries": fp,
        "false_negatives": fn,
        "wrong_per_100": 100 * (fp + fn) / len(frame),
        "utility_loss": (5 * fn + fp) / len(frame),
    }


def _triage_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    label = frame["label"].to_numpy(bool)
    confirmed = frame["triage"].eq("CONFIRMED_EFFECT").to_numpy()
    no_effect = frame["triage"].eq("NO_EFFECT_DETECTED").to_numpy()
    unresolved = ~(confirmed | no_effect)
    tp = int(np.sum(label & confirmed))
    fp = int(np.sum(~label & confirmed))
    fn = int(np.sum(label & no_effect))
    wrong = fp + fn
    return {
        "hidden_cells": len(frame),
        "effect_cells": int(label.sum()),
        "protected_recall": 1 - fn / max(int(label.sum()), 1),
        "confirmed_precision": tp / max(tp + fp, 1),
        "confirmed_effect_recall": tp / max(int(label.sum()), 1),
        "decisive_coverage": float(np.mean(~unresolved)),
        "unresolved_coverage": float(np.mean(unresolved)),
        "false_discoveries": fp,
        "missed_effects": fn,
        "correct_decisions_per_100": 100
        * int(np.sum((label & confirmed) | (~label & no_effect)))
        / len(frame),
        "wrong_decisions_per_100": 100 * wrong / len(frame),
        "utility_loss": (5 * fn + fp) / len(frame),
    }


def _bootstrap_comparison(
    frame: pd.DataFrame, seed: int, draws: int = 5_000
) -> dict[str, list[float] | float]:
    group_rows = []
    for _, part in frame.groupby("compound_group"):
        label = part["label"].to_numpy(bool)
        candidate = part["candidate_binary"].to_numpy(bool)
        canonical = part["canonical_binary"].to_numpy(bool)
        confirmed = part["triage"].eq("CONFIRMED_EFFECT").to_numpy()
        no_effect = part["triage"].eq("NO_EFFECT_DETECTED").to_numpy()
        group_rows.append(
            (
                np.sum(label & candidate),
                np.sum(~label & candidate),
                np.sum(label & ~candidate),
                np.sum(label & canonical),
                np.sum(~label & canonical),
                np.sum(label & ~canonical),
                np.sum(~label & confirmed) + np.sum(label & no_effect),
                np.sum(~label & canonical) + np.sum(label & ~canonical),
                len(part),
            )
        )
    counts = np.asarray(group_rows, dtype=float)
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(counts), size=(draws, len(counts)))
    totals = counts[picked].sum(axis=1)
    precision = totals[:, 0] / np.maximum(totals[:, 0] + totals[:, 1], 1)
    precision -= totals[:, 3] / np.maximum(totals[:, 3] + totals[:, 4], 1)
    recall = totals[:, 0] / np.maximum(totals[:, 0] + totals[:, 2], 1)
    recall -= totals[:, 3] / np.maximum(totals[:, 3] + totals[:, 5], 1)
    wrong = 100 * (totals[:, 6] - totals[:, 7]) / totals[:, 8]
    result: dict[str, list[float] | float] = {}
    for name, values in (
        ("precision_gain", precision),
        ("recall_gain", recall),
        ("wrong_decision_change", wrong),
    ):
        result[name] = float(np.mean(values))
        result[f"{name}_ci"] = [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
    return result


def run_decision_triage_study(
    legacy_workbook: Path,
    phase1_workbook: Path,
    preregistration: Path,
    v3_summary: Path,
    output_dir: Path,
) -> dict[str, object]:
    lock = json.loads(preregistration.read_text())
    config = V3Config(**lock["selected_config"])
    seed = int(lock["seed"])
    conformal_q = float(
        json.loads(v3_summary.read_text())["calibration_rules"]["max_variance"]["conformal_q"]
    )
    _, _, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    multi = load_multivariate_mea(legacy_workbook, phase1_workbook)
    if set(responses) != set(multi.responses):
        raise ValueError("multivariate and canonical run inventories differ")
    eligible = meta.loc[meta["v3_split"].eq("train")].copy()
    support = meta.loc[meta["v3_split"].eq("development")].copy()
    eligible["outer_fold"] = eligible["compound_group"].map(
        lambda value: _fold(str(value), "decision-utility-outer", 5)
    )
    all_predictions = []
    selections = []
    ablation_predictions = []
    direct_parts = []
    for outer_fold in range(5):
        outer_train = eligible.loc[~eligible["outer_fold"].eq(outer_fold)].copy()
        outer_test = eligible.loc[eligible["outer_fold"].eq(outer_fold)].copy()
        inner_frames = []
        inner_namespace = f"decision-triage-features-{outer_fold}"
        for inner_fold in range(5):
            held = outer_train["compound_group"].map(
                lambda value, namespace=inner_namespace: _fold(str(value), namespace, 5)
            ).eq(inner_fold)
            inner_test = outer_train.loc[held]
            inner_train = outer_train.loc[~held]
            features, _ = _feature_rows(
                pd.concat([support, inner_train], ignore_index=True),
                inner_test,
                responses,
                multi.responses,
                config,
                conformal_q,
            )
            inner_frames.append(features)
        inner = pd.concat(inner_frames, ignore_index=True)
        outer, direct = _feature_rows(
            pd.concat([support, outer_train], ignore_index=True),
            outer_test,
            responses,
            multi.responses,
            config,
            conformal_q,
        )
        direct["outer_fold"] = outer_fold
        direct_parts.append(direct)
        for round_index in (2, 3):
            inner_round = inner.loc[inner["round"].eq(round_index)].reset_index(drop=True)
            outer_round = outer.loc[outer["round"].eq(round_index)].reset_index(drop=True)
            candidate_rows = []
            candidate_outputs: dict[
                tuple[str, str], tuple[np.ndarray, np.ndarray, object, object]
            ] = {}
            for feature_name, feature_columns in FEATURE_SETS.items():
                for model_type in MODEL_TYPES:
                    namespace = (
                        f"decision-triage-model-{outer_fold}-{round_index}-{feature_name}-{model_type}"
                    )
                    raw_oof = _candidate_oof(
                        inner_round, feature_columns, model_type, namespace
                    )
                    calibrator, calibrated_oof, calibration_scores = _select_calibrator(
                        inner_round, raw_oof, f"{namespace}-calibration"
                    )
                    weights = _sample_weights(inner_round)
                    ap = average_precision_score(
                        inner_round["label"], calibrated_oof, sample_weight=weights
                    )
                    model = _make_model(model_type)
                    _fit_model(
                        model,
                        inner_round[list(feature_columns)].to_numpy(float),
                        inner_round["label"].to_numpy(int),
                        weights,
                    )
                    candidate_rows.append(
                        {
                            "feature_set": feature_name,
                            "model": model_type,
                            "average_precision": float(ap),
                            "calibrator": calibrator.method,
                            "calibration_brier": calibration_scores,
                        }
                    )
                    candidate_outputs[(feature_name, model_type)] = (
                        calibrated_oof,
                        np.asarray(feature_columns),
                        model,
                        calibrator,
                    )
            ranking = sorted(
                candidate_rows,
                key=lambda row: (
                    -row["average_precision"],
                    list(FEATURE_SETS).index(row["feature_set"]),
                    MODEL_TYPES.index(row["model"]),
                ),
            )
            selected = ranking[0]
            key = (str(selected["feature_set"]), str(selected["model"]))
            selected_oof, feature_columns, model, calibrator = candidate_outputs[key]
            inner_label = inner_round["label"].to_numpy(int)
            binary_threshold = _binary_threshold(selected_oof, inner_label)
            low, high = _triage_thresholds(inner_round, selected_oof)
            raw_outer = model.predict_proba(
                outer_round[list(feature_columns)].to_numpy(float)
            )[:, 1]
            probability = calibrator.predict(raw_outer)
            outer_round["probability"] = probability
            outer_round["candidate_binary"] = probability >= binary_threshold
            outer_round["canonical_binary"] = outer_round["upper_effect_bound"].ge(
                config.effect_threshold
            )
            outer_round["mean_binary"] = outer_round["abs_mean"].ge(config.effect_threshold)
            outer_round["probability_binary"] = outer_round["effect_probability"].ge(0.5)
            outer_round["dose_supported_binary"] = outer_round["abs_mean"].ge(
                config.effect_threshold
            ) & outer_round["same_div_neighbor_abs"].ge(config.effect_threshold)
            outer_round["triage"] = np.where(
                probability >= high,
                "CONFIRMED_EFFECT",
                np.where(
                    probability <= low, "NO_EFFECT_DETECTED", "NEEDS_MORE_EVIDENCE"
                ),
            )
            outer_round["outer_fold"] = outer_fold
            outer_round["binary_threshold"] = binary_threshold
            outer_round["low_threshold"] = low
            outer_round["high_threshold"] = high
            outer_round["selected_feature_set"] = selected["feature_set"]
            outer_round["selected_model"] = selected["model"]
            outer_round["selected_calibrator"] = selected["calibrator"]
            all_predictions.append(outer_round)
            selections.append(
                {
                    "outer_fold": outer_fold,
                    "round": round_index,
                    "selected": selected,
                    "ranking": ranking,
                    "binary_threshold": binary_threshold,
                    "low_threshold": low,
                    "high_threshold": high,
                }
            )
            for candidate in candidate_rows:
                candidate_key = (str(candidate["feature_set"]), str(candidate["model"]))
                candidate_oof, columns, fitted, candidate_calibrator = candidate_outputs[
                    candidate_key
                ]
                threshold = _binary_threshold(candidate_oof, inner_label)
                candidate_probability = candidate_calibrator.predict(
                    fitted.predict_proba(outer_round[list(columns)].to_numpy(float))[:, 1]
                )
                ablation_predictions.append(
                    pd.DataFrame(
                        {
                            "compound_group": outer_round["compound_group"],
                            "outer_fold": outer_fold,
                            "round": round_index,
                            "feature_set": candidate["feature_set"],
                            "model": candidate["model"],
                            "label": outer_round["label"],
                            "prediction": candidate_probability >= threshold,
                        }
                    )
                )
    predictions = pd.concat(all_predictions, ignore_index=True)
    direct = pd.concat(direct_parts, ignore_index=True)
    ablations = pd.concat(ablation_predictions, ignore_index=True)
    summaries = []
    for round_index, frame in predictions.groupby("round"):
        summaries.append(
            {
                "round": int(round_index),
                "observations": int(frame["observations"].iloc[0]),
                "candidate": _binary_metrics(frame, "candidate_binary"),
                "canonical": _binary_metrics(frame, "canonical_binary"),
                "posterior_mean": _binary_metrics(frame, "mean_binary"),
                "posterior_probability": _binary_metrics(frame, "probability_binary"),
                "dose_supported": _binary_metrics(frame, "dose_supported_binary"),
                "triage": _triage_metrics(frame),
                "comparison": _bootstrap_comparison(frame, seed + int(round_index)),
            }
        )
    primary = predictions.loc[predictions["round"].eq(2)].copy()
    candidate_primary = _binary_metrics(primary, "candidate_binary")
    canonical_primary = _binary_metrics(primary, "canonical_binary")
    triage_primary = _triage_metrics(primary)
    comparison = _bootstrap_comparison(primary, seed + 2)
    fold_precision = {}
    for fold, frame in primary.groupby("outer_fold"):
        candidate_fold = _binary_metrics(frame, "candidate_binary")
        canonical_fold = _binary_metrics(frame, "canonical_binary")
        fold_precision[str(fold)] = float(candidate_fold["precision"]) - float(
            canonical_fold["precision"]
        )
    conditions = {
        "binary_precision_at_least_0_60": candidate_primary["precision"] >= 0.60,
        "binary_recall_at_least_0_95": candidate_primary["recall"] >= 0.95,
        "precision_gain_at_least_0_20": (
            candidate_primary["precision"] - canonical_primary["precision"] >= 0.20
        ),
        "precision_gain_ci_above_zero": comparison["precision_gain_ci"][0] > 0,
        "recall_noninferior": (
            candidate_primary["recall"] - canonical_primary["recall"] >= -0.02
        ),
        "triage_protected_recall_at_least_0_95": triage_primary["protected_recall"]
        >= 0.95,
        "triage_confirmed_precision_at_least_0_60": triage_primary[
            "confirmed_precision"
        ]
        >= 0.60,
        "triage_decisive_coverage_at_least_0_50": triage_primary["decisive_coverage"]
        >= 0.50,
        "triage_wrong_decision_ci_below_zero": comparison["wrong_decision_change_ci"][1]
        < 0,
        "four_of_five_folds_precision_gain": sum(value > 0 for value in fold_precision.values())
        >= 4,
    }
    go = bool(all(conditions.values()))
    direct_summary = (
        direct.groupby("round", as_index=False)
        .agg(
            observations=("observations", "first"),
            direct_effect_cells=("direct_effect_cells", "sum"),
            total_effect_cells=("total_effect_cells", "sum"),
        )
        .assign(
            direct_effect_capture=lambda frame: frame["direct_effect_cells"]
            / frame["total_effect_cells"].clip(lower=1)
        )
    )
    ablation_rows = []
    for keys, frame in ablations.groupby(["round", "feature_set", "model"]):
        renamed = frame.rename(columns={"prediction": "candidate"})
        ablation_rows.append(
            {
                "round": keys[0],
                "feature_set": keys[1],
                "model": keys[2],
                **_binary_metrics(renamed, "candidate"),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "out_of_fold_cell_decisions.csv", index=False)
    direct.to_csv(output_dir / "direct_effects.csv", index=False)
    direct_summary.to_csv(output_dir / "direct_effect_summary.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(output_dir / "ablations.csv", index=False)
    (output_dir / "fold_selections.json").write_text(
        json.dumps(selections, indent=2) + "\n"
    )
    summary = {
        "schema": "gates-act-decision-triage-v1",
        "evidence_level": "repeated nested development; not fresh external validation",
        "outer_groups": int(eligible["compound_group"].nunique()),
        "support_groups": int(support["compound_group"].nunique()),
        "sealed_groups_used": 0,
        "endpoints": list(ENDPOINTS),
        "acquisition": "frozen conditional maximum variance",
        "primary_budget": {"round": 2, "observations": 12, "div": 7},
        "round_summaries": summaries,
        "fold_precision_gain": fold_precision,
        "go_conditions": conditions,
        "go": go,
        "canonical_action": (
            "eligible for shadow-mode integration and independent external validation"
            if go
            else "NO-GO: preserve canonical decision layer and presentation"
        ),
        "stopping": "disabled",
        "config": asdict(config),
        "multivariate_audit": multi.audit,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
