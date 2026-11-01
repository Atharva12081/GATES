from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.agreement_triage import _calibrate_margin
from gates.act.benchmark_v3 import _load_sources
from gates.act.competitor_transfer import (
    FoldData,
    _bucket,
    _examples,
    _features,
    _fold_data,
    _interpolate,
    _model,
)
from gates.act.direct_discovery import _bootstrap, _metrics

EFFECT_THRESHOLD = 0.40
CHANGE_THRESHOLD = 0.20


def chronological_design(truth: np.ndarray) -> tuple[list[int], list[dict[str, object]]]:
    observed: list[int] = []
    log: list[dict[str, object]] = []
    for time_rank in range(4):
        if time_rank == 0:
            ranks = (5, 6, 7)
            evidence_cells: list[int] = []
            trigger = "initial_high_dose_anchor"
        else:
            previous = [8 * (time_rank - 1) + rank for rank in (6, 7)]
            effect_trigger = any(abs(float(truth[cell])) >= EFFECT_THRESHOLD for cell in previous)
            change_trigger = False
            evidence_cells = previous.copy()
            if time_rank >= 2:
                earlier = [8 * (time_rank - 2) + rank for rank in (6, 7)]
                evidence_cells.extend(earlier)
                change_trigger = any(
                    abs(float(truth[current] - truth[prior])) >= CHANGE_THRESHOLD
                    for current, prior in zip(previous, earlier, strict=True)
                )
            adaptive_rank = 4 if effect_trigger or change_trigger else 5
            ranks = (adaptive_rank, 6, 7)
            trigger = (
                "prior_effect"
                if effect_trigger
                else ("prior_temporal_change" if change_trigger else "retain_high_dose")
            )
        cells = [8 * time_rank, *[8 * time_rank + rank for rank in ranks]]
        observed.extend(cells)
        log.append(
            {
                "time_rank": time_rank,
                "selected_cells": cells,
                "selected_treated_ranks": list(ranks),
                "evidence_cells": evidence_cells,
                "trigger": trigger,
            }
        )
    return observed, log


def _predict_adaptive(
    data: FoldData,
    run_ids: list[str],
    model: object,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    logs = []
    for run_id in run_ids:
        truth = data.responses[run_id]
        group = data.meta.loc[data.meta["run_id"].eq(run_id), "compound_group"].iloc[0]
        source = data.meta.loc[data.meta["run_id"].eq(run_id), "source"].iloc[0]
        observed, selection_log = chronological_design(truth)
        observed_array = np.asarray(observed, dtype=int)
        hidden = np.setdiff1d(np.arange(32), observed_array)
        gates = data.surface.fit_predict(observed, truth[observed_array], EFFECT_THRESHOLD)
        interpolation = _interpolate(truth, observed_array)
        features, _, feature_hidden = _features(
            truth, observed_array, interpolation, gates.mean, gates.std
        )
        if not np.array_equal(hidden, feature_hidden):
            raise RuntimeError("hidden-cell ordering changed")
        residual = interpolation[hidden] + model.predict(features)
        prediction_lookup = dict(zip(hidden.tolist(), residual.tolist(), strict=True))
        for cell in range(32):
            if cell % 8 == 0:
                continue
            direct = cell in observed
            rows.append(
                {
                    "fold": fold,
                    "run_id": run_id,
                    "compound_group": group,
                    "cell": cell,
                    "truth": float(truth[cell]),
                    "gates": np.nan if direct else float(gates.mean[cell]),
                    "selected_residual": np.nan if direct else float(prediction_lookup[cell]),
                    "policy": "high_anchor_temporal_boundary",
                    "directly_observed": direct,
                    "source": source,
                }
            )
        for entry in selection_log:
            logs.append(
                {
                    "fold": fold,
                    "run_id": run_id,
                    "compound_group": group,
                    "source": source,
                    **entry,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(logs)


def _finalize_decisions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    calibration_rows = []
    for fold in sorted(int(value) for value in predictions["fold"].unique()):
        calibration = predictions.loc[
            ~predictions["fold"].eq(fold) & ~predictions["directly_observed"]
        ].copy()
        calibration["interpolation"] = np.nan
        margin = _calibrate_margin(calibration, "agreement")
        held = predictions.loc[predictions["fold"].eq(fold)].copy()
        direct = held["directly_observed"].to_numpy(dtype=bool)
        truth = held["truth"].abs().to_numpy() >= EFFECT_THRESHOLD
        gates = held["gates"].to_numpy()
        residual = held["selected_residual"].to_numpy()
        agreement = (
            (np.abs(gates) >= EFFECT_THRESHOLD)
            & (np.abs(residual) >= EFFECT_THRESHOLD)
            & (np.sign(gates) == np.sign(residual))
        )
        score = np.maximum(np.abs(gates), np.abs(residual))
        held["effect_call"] = np.where(direct, truth, agreement)
        held["no_effect_call"] = np.where(direct, ~truth, score < margin)
        held["margin"] = margin
        pieces.append(held)
        calibration_rows.append(
            {
                "fold": fold,
                "calibration_groups": int(calibration["compound_group"].nunique()),
                "hidden_no_effect_margin": margin,
            }
        )
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(calibration_rows)


def run_chronological_high_dose(
    legacy_workbook: Path,
    phase1_workbook: Path,
    fixed_decisions: Path,
    output: Path,
) -> dict[str, object]:
    _, _, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    train = meta.loc[meta["v3_split"].eq("train")].copy()
    groups = sorted(train["compound_group"].unique())
    predictions = []
    selection_logs = []
    for fold in range(5):
        test_groups = {group for group in groups if _bucket(group, "h1-outer", 5) == fold}
        data = _fold_data(meta, responses, test_groups)
        fit_ids = train.loc[~train["compound_group"].isin(test_groups), "run_id"].tolist()
        test_ids = train.loc[train["compound_group"].isin(test_groups), "run_id"].tolist()
        x, y = _examples(data, fit_ids, "interpolation")
        model = _model().fit(x, y)
        fold_predictions, fold_log = _predict_adaptive(data, test_ids, model, fold)
        predictions.append(fold_predictions)
        selection_logs.append(fold_log)
    adaptive_raw = pd.concat(predictions, ignore_index=True)
    adaptive, calibration = _finalize_decisions(adaptive_raw)
    log = pd.concat(selection_logs, ignore_index=True)

    fixed = pd.read_parquet(fixed_decisions)
    comparators = fixed.loc[
        fixed["policy"].isin(
            [
                "conditional_max_variance",
                "highest_three_doses",
                "middle_high_sentinels",
                "maximin",
            ]
        )
    ].copy()
    all_decisions = pd.concat([comparators, adaptive], ignore_index=True)
    aggregate_rows = []
    source_rows = []
    for policy, part in all_decisions.groupby("policy"):
        aggregate_rows.append({"policy": policy, **_metrics(part)})
        for source, source_part in part.groupby("source"):
            source_rows.append({"policy": policy, "source": source, **_metrics(source_part)})
    aggregate = pd.DataFrame(aggregate_rows)
    by_source = pd.DataFrame(source_rows)
    metrics = aggregate.set_index("policy")
    candidate = metrics.loc["high_anchor_temporal_boundary"]
    baseline = metrics.loc["highest_three_doses"]
    time_checks = []
    for record in log.itertuples(index=False):
        selected_time = int(record.time_rank)
        evidence = list(record.evidence_cells)
        time_checks.append(not evidence or max(int(cell) // 8 for cell in evidence) < selected_time)
    per_run = log.groupby(["fold", "run_id", "compound_group"]).agg(
        days=("time_rank", "size"),
        selected_cells=("selected_cells", lambda values: sum(len(value) for value in values)),
    )
    chronology_pass = bool(
        all(time_checks) and per_run["days"].eq(4).all() and per_run["selected_cells"].eq(16).all()
    )
    bootstrap = _bootstrap(all_decisions, "high_anchor_temporal_boundary", "highest_three_doses")
    gate = bool(
        candidate["total_discovery_recall"] - baseline["total_discovery_recall"] >= 0.03
        and candidate["run_group_sensitivity"] >= baseline["run_group_sensitivity"]
        and candidate["confirmed_effect_precision"] >= baseline["confirmed_effect_precision"]
        and candidate["protected_recall"] >= baseline["protected_recall"]
        and candidate["selective_risk"] <= baseline["selective_risk"]
        and chronology_pass
    )
    result = {
        "schema": "gates-act-chronological-high-dose-v1",
        "evidence_class": "repeated nested development on previously examined groups",
        "policy": (
            "future-only rank-7/rank-6 anchors; rank 4 replaces rank 5 after a prior effect or "
            "absolute anchor change >=0.20"
        ),
        "budget": "one vehicle plus three treated cells at each of four assay days",
        "chronology_audit": {
            "status": "PASS" if chronology_pass else "FAIL",
            "all_evidence_precedes_selected_batch": bool(all(time_checks)),
            "four_batches_per_run": bool(per_run["days"].eq(4).all()),
            "sixteen_physical_cells_per_run": bool(per_run["selected_cells"].eq(16).all()),
        },
        "aggregate": aggregate.to_dict(orient="records"),
        "by_source": by_source.to_dict(orient="records"),
        "bootstrap_candidate_minus_fixed_high_dose": bootstrap,
        "research_gate": "GO" if gate else "NO-GO",
        "canonical_integration": "NO-GO without independent confirmation",
        "early_stopping": "disabled",
    }
    output.mkdir(parents=True, exist_ok=True)
    adaptive.to_parquet(output / "chronological_adaptive_decisions.parquet", index=False)
    log.to_json(output / "chronological_selection_log.json", orient="records", indent=2)
    calibration.to_csv(output / "chronological_calibration.csv", index=False)
    aggregate.to_csv(output / "chronological_policy_metrics.csv", index=False)
    by_source.to_csv(output / "chronological_policy_by_source.csv", index=False)
    (output / "chronological_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
