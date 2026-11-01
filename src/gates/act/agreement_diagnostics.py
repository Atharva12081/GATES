from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("agreement", "gates", "selected_residual", "interpolation")
RECALL_TARGETS = (0.99, 0.975, 0.95, 0.925, 0.90)
COVERAGE_TARGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
DIVS = (5, 7, 9, 12)


def load_selected(predictions: Path, triage: Path) -> pd.DataFrame:
    frame = pd.read_parquet(predictions)
    key = ["fold", "run_id", "compound_group", "context", "cell", "truth"]
    wide = (
        frame.loc[frame["method"].isin(METHODS[1:])]
        .pivot(index=key, columns="method", values="prediction")
        .reset_index()
    )
    saved = json.loads(triage.read_text())["conditional_max_variance_policy"]["contexts_by_fold"]
    contexts = list(itertools.combinations(range(1, 8), 3))
    mask = np.logical_or.reduce(
        [
            wide["fold"].eq(int(fold)) & wide["context"].eq(contexts.index(tuple(context)))
            for fold, context in saved.items()
        ]
    )
    selected = wide.loc[mask].copy()
    selected["dose_rank"] = selected["cell"] % 8
    selected["time_rank"] = selected["cell"] // 8
    selected["div"] = selected["time_rank"].map(dict(enumerate(DIVS)))
    selected["source"] = np.where(
        selected["run_id"].str.startswith("P1::"), "phase1_external", "legacy_epa"
    )
    selected["absolute_effect"] = np.abs(selected["truth"])
    selected["effect_bin"] = pd.cut(
        selected["absolute_effect"],
        bins=[0, 0.2, 0.4, 0.6, 1.0, np.inf],
        right=False,
        labels=["[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,1)", "[1,inf)"],
    )
    return selected


def effect_call(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method == "agreement":
        gates = frame["gates"].to_numpy()
        residual = frame["selected_residual"].to_numpy()
        return (
            (np.abs(gates) >= 0.40)
            & (np.abs(residual) >= 0.40)
            & (np.sign(gates) == np.sign(residual))
        )
    return np.abs(frame[method].to_numpy()) >= 0.40


def no_effect_score(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method == "agreement":
        return np.maximum(
            np.abs(frame["gates"].to_numpy()),
            np.abs(frame["selected_residual"].to_numpy()),
        )
    return np.abs(frame[method].to_numpy())


def disagreement_state(frame: pd.DataFrame) -> np.ndarray:
    gates = frame["gates"].to_numpy()
    residual = frame["selected_residual"].to_numpy()
    gates_effect = np.abs(gates) >= 0.40
    residual_effect = np.abs(residual) >= 0.40
    state = np.full(len(frame), "neither", dtype=object)
    state[gates_effect & ~residual_effect] = "gates_only"
    state[~gates_effect & residual_effect] = "residual_only"
    both = gates_effect & residual_effect
    state[both & (np.sign(gates) == np.sign(residual))] = "both_same_sign"
    state[both & (np.sign(gates) != np.sign(residual))] = "both_opposite_sign"
    return state


def _slice_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = frame.copy()
    truth = work["absolute_effect"] >= 0.40
    calls = {method: effect_call(work, method) for method in ("agreement", "gates")}
    work["true_effect"] = truth
    for method, call in calls.items():
        work[f"{method}_call"] = call
        work[f"{method}_tp"] = call & truth
        work[f"{method}_fp"] = call & ~truth
    grouped = work.groupby(columns, observed=True, as_index=False).agg(
        rows=("truth", "size"),
        true_effects=("true_effect", "sum"),
        gates_calls=("gates_call", "sum"),
        gates_tp=("gates_tp", "sum"),
        gates_fp=("gates_fp", "sum"),
        agreement_calls=("agreement_call", "sum"),
        agreement_tp=("agreement_tp", "sum"),
        agreement_fp=("agreement_fp", "sum"),
    )
    for method in ("gates", "agreement"):
        grouped[f"{method}_precision"] = grouped[f"{method}_tp"] / grouped[f"{method}_calls"].clip(
            lower=1
        )
    grouped["false_discoveries_removed"] = grouped["gates_fp"] - grouped["agreement_fp"]
    grouped["true_effect_calls_removed"] = grouped["gates_tp"] - grouped["agreement_tp"]
    return grouped


def mechanism_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    states = frame.copy()
    states["disagreement_type"] = disagreement_state(states)
    states["true_effect"] = states["absolute_effect"] >= 0.40
    states["gates_absolute_error"] = np.abs(states["gates"] - states["truth"])
    states["residual_absolute_error"] = np.abs(states["selected_residual"] - states["truth"])
    state_table = states.groupby("disagreement_type", as_index=False).agg(
        rows=("truth", "size"),
        true_effects=("true_effect", "sum"),
        mean_absolute_effect=("absolute_effect", "mean"),
        gates_mae=("gates_absolute_error", "mean"),
        residual_mae=("residual_absolute_error", "mean"),
    )
    state_table["true_effect_prevalence"] = state_table["true_effects"] / state_table["rows"]
    return {
        "state": state_table,
        "compound": _slice_summary(frame, ["compound_group", "source"]),
        "dose": _slice_summary(frame, ["dose_rank"]),
        "day": _slice_summary(frame, ["div"]),
        "effect_magnitude": _slice_summary(frame, ["effect_bin"]),
        "source": _slice_summary(frame, ["source"]),
    }


def _group_false_negative_fraction(frame: pd.DataFrame, score: np.ndarray, margin: float) -> float:
    truth = np.abs(frame["truth"].to_numpy()) >= 0.40
    work = frame[["compound_group"]].copy()
    work["effect"] = truth
    work["miss"] = truth & (score < margin)
    group = work.groupby("compound_group").agg(effect=("effect", "sum"), miss=("miss", "sum"))
    group = group.loc[group["effect"] > 0]
    return float((group["miss"] / group["effect"]).mean()) if len(group) else 0.0


def _calibrate_recall_margin(frame: pd.DataFrame, method: str, target: float) -> float:
    score = no_effect_score(frame, method)
    allowed = [
        float(margin)
        for margin in np.linspace(0, 0.40, 401)
        if _group_false_negative_fraction(frame, score, float(margin)) <= 1 - target
    ]
    return max(allowed, default=0.0)


def _calibrate_coverage_margin(frame: pd.DataFrame, method: str, target: float) -> float:
    effect = effect_call(frame, method)
    score = no_effect_score(frame, method)
    candidates = np.linspace(0, 0.40, 401)
    coverage = np.asarray([np.mean(effect | (score < margin)) for margin in candidates])
    return float(candidates[np.argmin(np.abs(coverage - target))])


def _metrics(frame: pd.DataFrame, method: str, margins: np.ndarray) -> dict[str, float | int]:
    truth = np.abs(frame["truth"].to_numpy()) >= 0.40
    effect = effect_call(frame, method)
    no_effect = no_effect_score(frame, method) < margins
    decisive = effect | no_effect
    correct = (effect & truth) | (no_effect & ~truth)
    tp, fp = int((effect & truth).sum()), int((effect & ~truth).sum())
    fn = int((no_effect & truth).sum())
    wrong = int((decisive & ~correct).sum())
    return {
        "rows": len(frame),
        "effect_precision": tp / max(tp + fp, 1),
        "protected_recall": 1 - fn / max(int(truth.sum()), 1),
        "decisive_coverage": float(decisive.mean()),
        "selective_risk": wrong / max(int(decisive.sum()), 1),
        "wrong_decisions": wrong,
        "missed_effects": fn,
        "unresolved": int((~decisive).sum()),
    }


def crossfit_curve(
    frame: pd.DataFrame, mode: str, targets: tuple[float, ...]
) -> tuple[pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    rows = []
    scored: dict[tuple[str, float], pd.DataFrame] = {}
    for method in METHODS:
        for target in targets:
            pieces = []
            for fold in sorted(frame["fold"].unique()):
                calibration = frame.loc[~frame["fold"].eq(fold)]
                if mode == "recall":
                    margin = _calibrate_recall_margin(calibration, method, target)
                elif mode == "coverage":
                    margin = _calibrate_coverage_margin(calibration, method, target)
                else:
                    raise ValueError(mode)
                held = frame.loc[frame["fold"].eq(fold)].copy()
                held["margin"] = margin
                pieces.append(held)
            combined = pd.concat(pieces, ignore_index=True)
            scored[(method, target)] = combined
            margins = combined["margin"].to_numpy()
            rows.append(
                {
                    "mode": mode,
                    "target": target,
                    "method": method,
                    "mean_margin": float(combined["margin"].mean()),
                    **_metrics(combined, method, margins),
                }
            )
    return pd.DataFrame(rows), scored


def _source_metrics(scored: dict[tuple[str, float], pd.DataFrame], target: float) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        frame = scored[(method, target)]
        for source, part in frame.groupby("source"):
            rows.append(
                {
                    "method": method,
                    "target_protected_recall": target,
                    "source": source,
                    "groups": int(part["compound_group"].nunique()),
                    **_metrics(part, method, part["margin"].to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_primary(
    scored: dict[tuple[str, float], pd.DataFrame], seed: int = 20261028
) -> dict[str, object]:
    target = 0.95
    by_method = {}
    for method in ("agreement", "gates"):
        frame = scored[(method, target)].copy()
        truth = np.abs(frame["truth"].to_numpy()) >= 0.40
        effect = effect_call(frame, method)
        no_effect = no_effect_score(frame, method) < frame["margin"].to_numpy()
        decisive = effect | no_effect
        correct = (effect & truth) | (no_effect & ~truth)
        frame["tp"] = effect & truth
        frame["fp"] = effect & ~truth
        frame["wrong"] = decisive & ~correct
        frame["decisive"] = decisive
        by_method[method] = frame.groupby("compound_group")[["tp", "fp", "wrong", "decisive"]].sum()
    groups = sorted(by_method["agreement"].index)
    rng = np.random.default_rng(seed)
    precision_difference = []
    risk_difference = []
    for _ in range(10000):
        sample = rng.integers(0, len(groups), len(groups))
        values = {}
        for method in ("agreement", "gates"):
            total = by_method[method].reindex(groups).iloc[sample].sum()
            values[method] = (
                float(total.tp / max(total.tp + total.fp, 1)),
                float(total.wrong / max(total.decisive, 1)),
            )
        precision_difference.append(values["agreement"][0] - values["gates"][0])
        risk_difference.append(values["agreement"][1] - values["gates"][1])
    return {
        "groups": len(groups),
        "replicates": 10000,
        "seed": seed,
        "precision_difference": float(np.mean(precision_difference)),
        "precision_difference_ci95": [
            float(np.quantile(precision_difference, 0.025)),
            float(np.quantile(precision_difference, 0.975)),
        ],
        "selective_risk_difference": float(np.mean(risk_difference)),
        "selective_risk_difference_ci95": [
            float(np.quantile(risk_difference, 0.025)),
            float(np.quantile(risk_difference, 0.975)),
        ],
    }


def run_agreement_diagnostics(predictions: Path, triage: Path, output: Path) -> dict[str, object]:
    frame = load_selected(predictions, triage)
    tables = mechanism_tables(frame)
    recall_curve, recall_scored = crossfit_curve(frame, "recall", RECALL_TARGETS)
    coverage_curve, _ = crossfit_curve(frame, "coverage", COVERAGE_TARGETS)
    source = _source_metrics(recall_scored, 0.95)
    primary = _bootstrap_primary(recall_scored)
    for name, table in tables.items():
        table.to_csv(output / f"mechanism_{name}.csv", index=False)
    recall_curve.to_csv(output / "risk_coverage_matched_recall.csv", index=False)
    coverage_curve.to_csv(output / "risk_coverage_matched_coverage.csv", index=False)
    source.to_csv(output / "risk_coverage_source_shift.csv", index=False)
    agreement = recall_curve.loc[
        recall_curve["method"].eq("agreement") & recall_curve["target"].eq(0.95)
    ].iloc[0]
    gates = recall_curve.loc[
        recall_curve["method"].eq("gates") & recall_curve["target"].eq(0.95)
    ].iloc[0]
    source_pivot = source.loc[source["method"].isin(["agreement", "gates"])].pivot(
        index="source", columns="method"
    )
    stable = bool(
        agreement["effect_precision"] > gates["effect_precision"]
        and agreement["selective_risk"] <= gates["selective_risk"]
        and agreement["protected_recall"] >= gates["protected_recall"]
        and agreement["wrong_decisions"] < gates["wrong_decisions"]
        and (
            source_pivot["effect_precision"]["agreement"]
            > source_pivot["effect_precision"]["gates"]
        ).all()
        and (
            source_pivot["selective_risk"]["agreement"] <= source_pivot["selective_risk"]["gates"]
        ).all()
        and (
            source_pivot["protected_recall"]["agreement"]
            >= source_pivot["protected_recall"]["gates"]
        ).all()
        and (
            source_pivot["wrong_decisions"]["agreement"] < source_pivot["wrong_decisions"]["gates"]
        ).all()
    )
    summary = {
        "schema": "gates-act-agreement-mechanism-v1",
        "evidence_class": "nested development; not independent confirmation",
        "rows": len(frame),
        "groups": int(frame["compound_group"].nunique()),
        "budget": "16 of 32 response cells",
        "primary_target_protected_recall": 0.95,
        "agreement": agreement.to_dict(),
        "gates": gates.to_dict(),
        "bootstrap": primary,
        "source_stability_gate": "GO" if stable else "NO-GO",
        "early_stopping": "disabled",
    }
    (output / "agreement_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
