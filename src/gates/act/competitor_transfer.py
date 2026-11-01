from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from gates.act.benchmark_v3 import _load_sources
from gates.act.model_v3 import RiskSensitiveSurface
from gates.act.neural_mea import neural_coordinates

CONTEXTS = tuple(itertools.combinations(range(1, 8), 3))
EFFECT_THRESHOLD = 0.40


@dataclass(frozen=True)
class FoldData:
    meta: pd.DataFrame
    responses: dict[str, np.ndarray]
    surface: RiskSensitiveSurface


def _bucket(group: str, salt: str, folds: int) -> int:
    digest = hashlib.sha256(f"{salt}/{group}".encode()).hexdigest()
    return int(digest[:12], 16) % folds


def _observed(context: tuple[int, ...]) -> np.ndarray:
    ranks = (0, *context)
    return np.asarray([8 * time + rank for time in range(4) for rank in ranks], dtype=int)


def _interpolate(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    result = np.empty(32, dtype=float)
    for time in range(4):
        local = observed[(observed // 8) == time]
        ranks = local % 8
        order = np.argsort(ranks)
        result[8 * time : 8 * (time + 1)] = np.interp(
            np.arange(8), ranks[order], values[local[order]]
        )
    result[observed] = values[observed]
    return result


def _features(
    truth: np.ndarray,
    observed: np.ndarray,
    anchor: np.ndarray,
    gates_mean: np.ndarray,
    gates_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.setdiff1d(np.arange(32), observed)
    rows: list[list[float]] = []
    for target in hidden:
        time, dose = divmod(int(target), 8)
        same = observed[(observed // 8) == time]
        ranks = same % 8
        order = np.argsort(ranks)
        ranks = ranks[order]
        vals = truth[same[order]]
        lower_pos = np.flatnonzero(ranks <= dose)[-1]
        upper_candidates = np.flatnonzero(ranks >= dose)
        upper_pos = upper_candidates[0] if len(upper_candidates) else len(ranks) - 1
        nearest = int(np.argmin(np.abs(ranks - dose)))
        span = max(int(ranks[-1] - ranks[0]), 1)
        global_vals = truth[observed]
        rows.append(
            [
                anchor[target],
                gates_mean[target],
                gates_std[target],
                anchor[target] - gates_mean[target],
                dose / 7.0,
                time / 3.0,
                (dose - ranks[lower_pos]) / 7.0,
                (ranks[upper_pos] - dose) / 7.0,
                vals[lower_pos],
                vals[upper_pos],
                vals[nearest],
                (vals[-1] - vals[0]) / span,
                float(np.mean(vals)),
                float(np.std(vals)),
                float(np.mean(global_vals)),
                float(np.std(global_vals)),
                float(np.max(np.abs(global_vals))),
                float(np.mean(np.abs(global_vals) >= EFFECT_THRESHOLD)),
            ]
        )
    return np.asarray(rows, dtype=float), truth[hidden] - anchor[hidden], hidden


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=127,
        min_samples_leaf=50,
        early_stopping=False,
        random_state=0,
    )


def _fold_data(
    meta: pd.DataFrame, responses: dict[str, np.ndarray], test_groups: set[str]
) -> FoldData:
    fit = meta.loc[
        meta["v3_split"].eq("development")
        | (meta["v3_split"].eq("train") & ~meta["compound_group"].isin(test_groups))
    ]
    matrix = np.vstack([responses[run_id] for run_id in fit["run_id"]])
    surface = RiskSensitiveSurface(neural_coordinates(), matrix)
    return FoldData(meta=meta, responses=responses, surface=surface)


def _examples(data: FoldData, run_ids: list[str], anchor_kind: str):
    xs, ys = [], []
    for run_id in run_ids:
        truth = data.responses[run_id]
        for context in CONTEXTS:
            observed = _observed(context)
            gates = data.surface.fit_predict(observed.tolist(), truth[observed], EFFECT_THRESHOLD)
            interp = _interpolate(truth, observed)
            anchor = interp if anchor_kind == "interpolation" else gates.mean
            x, y, _ = _features(truth, observed, anchor, gates.mean, gates.std)
            xs.append(x)
            ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def _predict_rows(
    data: FoldData,
    run_ids: list[str],
    model: HistGradientBoostingRegressor | None,
    anchor_kind: str,
    fold: int,
) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        truth = data.responses[run_id]
        group = data.meta.loc[data.meta["run_id"].eq(run_id), "compound_group"].iloc[0]
        for context_number, context in enumerate(CONTEXTS):
            observed = _observed(context)
            gates = data.surface.fit_predict(observed.tolist(), truth[observed], EFFECT_THRESHOLD)
            interp = _interpolate(truth, observed)
            anchors = {"interpolation": interp, "gates": gates.mean}
            for method, anchor in anchors.items():
                hidden = np.setdiff1d(np.arange(32), observed)
                prediction = anchor[hidden]
                for cell, actual, estimate in zip(hidden, truth[hidden], prediction, strict=True):
                    rows.append(
                        {
                            "fold": fold,
                            "run_id": run_id,
                            "compound_group": group,
                            "context": context_number,
                            "cell": int(cell),
                            "method": method,
                            "truth": float(actual),
                            "prediction": float(estimate),
                        }
                    )
            if model is not None:
                anchor = anchors[anchor_kind]
                x, _, hidden = _features(truth, observed, anchor, gates.mean, gates.std)
                prediction = anchor[hidden] + model.predict(x)
                for cell, actual, estimate in zip(hidden, truth[hidden], prediction, strict=True):
                    rows.append(
                        {
                            "fold": fold,
                            "run_id": run_id,
                            "compound_group": group,
                            "context": context_number,
                            "cell": int(cell),
                            "method": f"{anchor_kind}_residual",
                            "truth": float(actual),
                            "prediction": float(estimate),
                        }
                    )
    return pd.DataFrame(rows)


def _candidate_score(
    meta: pd.DataFrame,
    responses: dict[str, np.ndarray],
    outer_train_groups: set[str],
    anchor_kind: str,
    outer_fold: int,
) -> float:
    scores = []
    for inner_fold in range(3):
        val_groups = {
            g for g in outer_train_groups if _bucket(g, f"h1-inner-{outer_fold}", 3) == inner_fold
        }
        train_groups = outer_train_groups - val_groups
        data = _fold_data(
            meta,
            responses,
            val_groups
            | (set(meta.loc[meta["v3_split"].eq("train"), "compound_group"]) - outer_train_groups),
        )
        train_ids = meta.loc[
            meta["v3_split"].eq("train") & meta["compound_group"].isin(train_groups), "run_id"
        ].tolist()
        val_ids = meta.loc[
            meta["v3_split"].eq("train") & meta["compound_group"].isin(val_groups), "run_id"
        ].tolist()
        x, y = _examples(data, train_ids, anchor_kind)
        model = _model().fit(x, y)
        pred = _predict_rows(data, val_ids, model, anchor_kind, outer_fold)
        pred = pred.loc[pred["method"].eq(f"{anchor_kind}_residual")]
        scores.append(float(np.mean(np.abs(pred["prediction"] - pred["truth"]))))
    return float(np.mean(scores))


def _summarize(
    predictions: pd.DataFrame, selected: dict[int, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    for fold, anchor in selected.items():
        frame = predictions.loc[predictions["fold"].eq(fold)].copy()
        residual = frame.loc[frame["method"].eq(f"{anchor}_residual")].copy()
        residual["method"] = "selected_residual"
        selected_rows.append(residual)
    full = pd.concat([predictions, *selected_rows], ignore_index=True)
    full["absolute_error"] = np.abs(full["prediction"] - full["truth"])
    full["true_effect"] = np.abs(full["truth"]) >= EFFECT_THRESHOLD
    full["predicted_effect"] = np.abs(full["prediction"]) >= EFFECT_THRESHOLD
    group = full.groupby(["method", "compound_group"], as_index=False).agg(
        mae=("absolute_error", "mean"),
        effects=("true_effect", "sum"),
        true_positive=("true_effect", lambda x: 0),
    )
    # Recompute event counts without fragile multi-column aggregation lambdas.
    events = (
        full.assign(
            tp=full["true_effect"] & full["predicted_effect"],
            fp=~full["true_effect"] & full["predicted_effect"],
        )
        .groupby(["method", "compound_group"], as_index=False)
        .agg(tp=("tp", "sum"), fp=("fp", "sum"), effects=("true_effect", "sum"))
    )
    group = group.drop(columns=["effects", "true_positive"]).merge(
        events, on=["method", "compound_group"]
    )
    summary = group.groupby("method", as_index=False).agg(
        mae=("mae", "mean"), tp=("tp", "sum"), fp=("fp", "sum"), effects=("effects", "sum")
    )
    summary["effect_recall"] = summary["tp"] / summary["effects"].clip(lower=1)
    summary["effect_precision"] = summary["tp"] / (summary["tp"] + summary["fp"]).clip(lower=1)
    return full, summary


def _bootstrap(
    group: pd.DataFrame, candidate: str, baseline: str, seed: int = 20261026
) -> dict[str, float]:
    pivot = group.pivot(index="compound_group", columns="method", values="mae").dropna()
    difference = (pivot[candidate] - pivot[baseline]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(axis=1)
    return {
        "groups": int(len(difference)),
        "difference": float(np.mean(difference)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def conditional_max_variance_contexts(
    legacy_workbook: Path, phase1_workbook: Path
) -> dict[int, tuple[int, int, int]]:
    """Choose three full dose trajectories using fold-local covariance and no test outcomes."""
    _, _, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    groups = sorted(meta.loc[meta["v3_split"].eq("train"), "compound_group"].unique())
    result = {}
    for fold in range(5):
        test_groups = {group for group in groups if _bucket(group, "h1-outer", 5) == fold}
        surface = _fold_data(meta, responses, test_groups).surface
        observed = [0, 8, 16, 24]
        prediction = surface.fit_predict(observed, surface.prior_mean[observed], EFFECT_THRESHOLD)
        covariance = prediction.covariance.copy()
        selected: list[int] = []
        for _ in range(3):
            choices = []
            for rank in range(1, 8):
                if rank in selected:
                    continue
                batch = np.asarray([rank + 8 * time for time in range(4)])
                inverse = np.linalg.pinv(
                    covariance[np.ix_(batch, batch)] + np.eye(len(batch)) * surface.noise
                )
                remaining = np.asarray(
                    [
                        cell
                        for cell in range(32)
                        if cell not in observed and cell % 8 not in (*selected, rank)
                    ]
                )
                cross = covariance[np.ix_(remaining, batch)]
                reduction = float(np.trace(cross @ inverse @ cross.T))
                choices.append((reduction, rank, batch, inverse))
            _, rank, batch, inverse = max(choices, key=lambda item: item[0])
            selected.append(rank)
            covariance -= covariance[:, batch] @ inverse @ covariance[batch, :]
            observed.extend(batch.tolist())
        result[fold] = tuple(sorted(selected))
    return result


def run_h1(legacy_workbook: Path, phase1_workbook: Path, output: Path) -> dict[str, object]:
    _, _, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    train = meta.loc[meta["v3_split"].eq("train")].copy()
    groups = sorted(train["compound_group"].unique())
    predictions = []
    selections: dict[int, str] = {}
    inner_scores: dict[str, dict[str, float]] = {}
    fold_effects = []
    for fold in range(5):
        test_groups = {group for group in groups if _bucket(group, "h1-outer", 5) == fold}
        outer_train_groups = set(groups) - test_groups
        scores = {
            anchor: _candidate_score(meta, responses, outer_train_groups, anchor, fold)
            for anchor in ("interpolation", "gates")
        }
        selected = min(scores, key=scores.get)
        selections[fold] = selected
        inner_scores[str(fold)] = scores
        data = _fold_data(meta, responses, test_groups)
        fit_ids = train.loc[~train["compound_group"].isin(test_groups), "run_id"].tolist()
        test_ids = train.loc[train["compound_group"].isin(test_groups), "run_id"].tolist()
        x, y = _examples(data, fit_ids, selected)
        model = _model().fit(x, y)
        frame = _predict_rows(data, test_ids, model, selected, fold)
        predictions.append(frame)
        fold_effects.append(
            {
                "fold": fold,
                "test_groups": len(test_groups),
                "test_runs": len(test_ids),
                "selected": selected,
                **scores,
            }
        )
    raw = pd.concat(predictions, ignore_index=True)
    full, summary = _summarize(raw, selections)
    group = full.groupby(["method", "compound_group"], as_index=False).agg(
        mae=("absolute_error", "mean")
    )
    baseline_rows = summary.loc[summary["method"].isin(["interpolation", "gates"])]
    strongest = baseline_rows.sort_values("mae").iloc[0]["method"]
    boot = _bootstrap(group, "selected_residual", str(strongest))
    fold_mae = full.groupby(["fold", "method"], as_index=False).agg(mae=("absolute_error", "mean"))
    wins = 0
    for fold in range(5):
        values = fold_mae.loc[fold_mae["fold"].eq(fold)].set_index("method")["mae"]
        wins += int(values["selected_residual"] < values[str(strongest)])
    table = summary.set_index("method")
    relative = 1 - table.loc["selected_residual", "mae"] / table.loc[strongest, "mae"]
    recall_delta = (
        table.loc["selected_residual", "effect_recall"] - table.loc[strongest, "effect_recall"]
    )
    passed = bool(relative >= 0.05 and boot["ci_high"] < 0 and wins >= 4 and recall_delta >= -0.02)
    result = {
        "status": "GO" if passed else "NO-GO",
        "census": {
            "scored_runs": len(train),
            "scored_groups": len(groups),
            "support_runs": int(meta["v3_split"].eq("development").sum()),
        },
        "selected_by_fold": selections,
        "inner_scores": inner_scores,
        "folds": fold_effects,
        "strongest_baseline": str(strongest),
        "relative_mae_reduction": float(relative),
        "effect_recall_delta": float(recall_delta),
        "fold_wins": wins,
        "paired_bootstrap": boot,
        "summary": summary.to_dict(orient="records"),
    }
    output.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output / "h1_oof_predictions.parquet", index=False)
    full.to_parquet(output / "h1_oof_predictions_with_selected.parquet", index=False)
    group.to_csv(output / "h1_group_metrics.csv", index=False)
    fold_mae.to_csv(output / "h1_fold_metrics.csv", index=False)
    (output / "h1_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
