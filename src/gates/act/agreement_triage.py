from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("gates", "interpolation", "selected_residual")


def _wide(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    key = ["fold", "run_id", "compound_group", "context", "cell", "truth"]
    return (
        frame.loc[frame["method"].isin(METHODS)]
        .pivot(index=key, columns="method", values="prediction")
        .reset_index()
    )


def _score(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method == "agreement":
        return np.maximum(np.abs(frame["gates"]), np.abs(frame["selected_residual"]))
    return np.abs(frame[method]).to_numpy()


def _effect_call(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method == "agreement":
        gates = frame["gates"].to_numpy()
        residual = frame["selected_residual"].to_numpy()
        return (
            (np.abs(gates) >= 0.40)
            & (np.abs(residual) >= 0.40)
            & (np.sign(gates) == np.sign(residual))
        )
    return np.abs(frame[method].to_numpy()) >= 0.40


def _calibrate_margin(frame: pd.DataFrame, method: str) -> float:
    score = _score(frame, method)
    true_effect = np.abs(frame["truth"].to_numpy()) >= 0.40
    candidates = np.linspace(0, 0.20, 201)
    allowed = []
    for margin in candidates:
        work = frame[["compound_group"]].copy()
        work["effect"] = true_effect
        work["miss"] = true_effect & (score < margin)
        group = work.groupby("compound_group").agg(effect=("effect", "sum"), miss=("miss", "sum"))
        eligible = group["effect"] > 0
        false_negative_fraction = float(
            (group.loc[eligible, "miss"] / group.loc[eligible, "effect"]).mean()
        )
        if false_negative_fraction <= 0.05:
            allowed.append(float(margin))
    return max(allowed, default=0.0)


def _evaluate(frame: pd.DataFrame, method: str, margin: float) -> dict[str, float | int]:
    true_effect = np.abs(frame["truth"].to_numpy()) >= 0.40
    effect = _effect_call(frame, method)
    no_effect = _score(frame, method) < margin
    decisive = effect | no_effect
    correct = (effect & true_effect) | (no_effect & ~true_effect)
    tp = int((effect & true_effect).sum())
    fp = int((effect & ~true_effect).sum())
    fn = int((no_effect & true_effect).sum())
    return {
        "rows": len(frame),
        "effect_precision": tp / max(tp + fp, 1),
        "protected_recall": 1 - fn / max(int(true_effect.sum()), 1),
        "decisive_coverage": float(decisive.mean()),
        "decision_accuracy": float(correct[decisive].mean()) if decisive.any() else 0.0,
        "confirmed_effect": int(effect.sum()),
        "no_effect": int(no_effect.sum()),
        "unresolved": int((~decisive).sum()),
        "wrong_decisions": int((decisive & ~correct).sum()),
        "missed_effects": fn,
    }


def _crossfit(frame: pd.DataFrame, method: str) -> tuple[pd.DataFrame, dict[str, float]]:
    parts = []
    margins = {}
    for fold in sorted(frame["fold"].unique()):
        calibration = frame.loc[~frame["fold"].eq(fold)]
        margin = _calibrate_margin(calibration, method)
        part = frame.loc[frame["fold"].eq(fold)].copy()
        part["margin"] = margin
        part["method"] = method
        parts.append(part)
        margins[str(fold)] = margin
    return pd.concat(parts, ignore_index=True), margins


def _evaluate_crossfit(frame: pd.DataFrame, method: str) -> dict[str, object]:
    crossed, margins = _crossfit(frame, method)
    rows = []
    for fold, part in crossed.groupby("fold"):
        values = _evaluate(part, method, float(part["margin"].iloc[0]))
        rows.append({"fold": int(fold), **values})
    # Evaluation uses each row's fold-specific margin.
    truth = np.abs(crossed["truth"].to_numpy()) >= 0.40
    effect = _effect_call(crossed, method)
    no_effect = _score(crossed, method) < crossed["margin"].to_numpy()
    decisive = effect | no_effect
    correct = (effect & truth) | (no_effect & ~truth)
    tp, fp = int((effect & truth).sum()), int((effect & ~truth).sum())
    fn = int((no_effect & truth).sum())
    aggregate = {
        "rows": len(crossed),
        "effect_precision": tp / max(tp + fp, 1),
        "protected_recall": 1 - fn / max(int(truth.sum()), 1),
        "decisive_coverage": float(decisive.mean()),
        "decision_accuracy": float(correct[decisive].mean()),
        "confirmed_effect": int(effect.sum()),
        "no_effect": int(no_effect.sum()),
        "unresolved": int((~decisive).sum()),
        "wrong_decisions": int((decisive & ~correct).sum()),
        "missed_effects": fn,
    }
    return {"margins": margins, "aggregate": aggregate, "folds": rows}


def _source_summary(frame: pd.DataFrame, method: str) -> dict[str, dict[str, float | int]]:
    crossed, _ = _crossfit(frame, method)
    crossed["source"] = np.where(
        crossed["run_id"].str.startswith("P1::"), "phase1_external", "legacy_epa"
    )
    result = {}
    for source, part in crossed.groupby("source"):
        truth = np.abs(part["truth"].to_numpy()) >= 0.40
        effect = _effect_call(part, method)
        no_effect = _score(part, method) < part["margin"].to_numpy()
        decisive = effect | no_effect
        correct = (effect & truth) | (no_effect & ~truth)
        tp, fp = int((effect & truth).sum()), int((effect & ~truth).sum())
        fn = int((no_effect & truth).sum())
        result[source] = {
            "groups": int(part["compound_group"].nunique()),
            "effect_precision": tp / max(tp + fp, 1),
            "protected_recall": 1 - fn / max(int(truth.sum()), 1),
            "decisive_coverage": float(decisive.mean()),
            "decision_accuracy": float(correct[decisive].mean()),
            "wrong_decisions": int((decisive & ~correct).sum()),
        }
    return result


def _bootstrap_precision(frame: pd.DataFrame, seed: int = 20261027) -> dict[str, object]:
    crossed = {}
    for method in ("agreement", "gates"):
        data, _ = _crossfit(frame, method)
        data["effect_call"] = _effect_call(data, method)
        data["true_effect"] = np.abs(data["truth"]) >= 0.40
        crossed[method] = data
    groups = sorted(frame["compound_group"].unique())
    counts = {}
    for method, data in crossed.items():
        counts[method] = (
            data.assign(
                tp=data["effect_call"] & data["true_effect"],
                fp=data["effect_call"] & ~data["true_effect"],
            )
            .groupby("compound_group")[["tp", "fp"]]
            .sum()
            .reindex(groups)
        )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(10000):
        selected = rng.integers(0, len(groups), len(groups))
        values = []
        for method in ("agreement", "gates"):
            sample = counts[method].iloc[selected].sum()
            values.append(float(sample.tp / max(sample.tp + sample.fp, 1)))
        differences.append(values[0] - values[1])
    return {
        "difference": float(np.mean(differences)),
        "ci95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "groups": len(groups),
        "replicates": 10000,
        "seed": seed,
    }


def run_agreement_triage(
    predictions: Path,
    output: Path,
    max_variance_contexts: dict[int, tuple[int, int, int]] | None = None,
) -> dict[str, object]:
    frame = _wide(predictions)
    contexts = list(itertools.combinations(range(1, 8), 3))
    maximin = frame.loc[frame["context"].eq(contexts.index((1, 4, 7)))].copy()
    max_variance = None
    if max_variance_contexts is not None:
        mask = np.logical_or.reduce(
            [
                frame["fold"].eq(fold) & frame["context"].eq(contexts.index(context))
                for fold, context in max_variance_contexts.items()
            ]
        )
        max_variance = frame.loc[mask].copy()
    result = {
        "evidence_class": "exploratory nested-development; not independent confirmation",
        "measurement_budget": "16 of 32 dose-time response cells per scenario",
        "all_context_scenarios": {
            method: _evaluate_crossfit(frame, method)
            for method in ("agreement", "gates", "selected_residual", "interpolation")
        },
        "maximin_design_1_4_7": {
            method: _evaluate_crossfit(maximin, method)
            for method in ("agreement", "gates", "selected_residual", "interpolation")
        },
        "source_shift": {
            method: _source_summary(frame, method)
            for method in ("agreement", "gates", "selected_residual", "interpolation")
        },
        "agreement_minus_gates_precision_bootstrap": _bootstrap_precision(frame),
        "early_stopping": "disabled",
    }
    if max_variance is not None:
        policy = {
            method: _evaluate_crossfit(max_variance, method)
            for method in ("agreement", "gates", "selected_residual", "interpolation")
        }
        bootstrap = _bootstrap_precision(max_variance)
        agreement = policy["agreement"]["aggregate"]
        gates = policy["gates"]["aggregate"]
        gate = (
            bootstrap["ci95"][0] > 0
            and agreement["protected_recall"] >= gates["protected_recall"]
            and agreement["decisive_coverage"] >= gates["decisive_coverage"]
            and agreement["wrong_decisions"] < gates["wrong_decisions"]
        )
        result["conditional_max_variance_policy"] = {
            "contexts_by_fold": {
                str(fold): list(context) for fold, context in max_variance_contexts.items()
            },
            "methods": policy,
            "agreement_minus_gates_precision_bootstrap": bootstrap,
            "exploratory_gate": "GO" if gate else "NO-GO",
            "canonical_integration": "not authorized without independent confirmation",
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "h6_agreement_triage.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
