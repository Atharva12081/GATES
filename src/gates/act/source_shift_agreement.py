from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.agreement_diagnostics import _calibrate_recall_margin, _metrics
from gates.act.benchmark_v3 import _load_sources
from gates.act.competitor_transfer import (
    FoldData,
    _bucket,
    _examples,
    _features,
    _interpolate,
    _model,
    _observed,
)
from gates.act.model_v3 import RiskSensitiveSurface
from gates.act.neural_mea import neural_coordinates


def _select_context(surface: RiskSensitiveSurface) -> tuple[int, int, int]:
    observed = [0, 8, 16, 24]
    prediction = surface.fit_predict(observed, surface.prior_mean[observed], 0.40)
    covariance = prediction.covariance.copy()
    selected: list[int] = []
    for _ in range(3):
        candidates = []
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
            candidates.append((float(np.trace(cross @ inverse @ cross.T)), rank, batch, inverse))
        _, rank, batch, inverse = max(candidates, key=lambda item: item[0])
        selected.append(rank)
        covariance -= covariance[:, batch] @ inverse @ covariance[batch, :]
        observed.extend(batch.tolist())
    return tuple(sorted(selected))


def _predict_context(
    data: FoldData,
    run_ids: list[str],
    model: object,
    context: tuple[int, int, int],
) -> pd.DataFrame:
    rows = []
    observed = _observed(context)
    hidden = np.setdiff1d(np.arange(32), observed)
    for run_id in run_ids:
        truth = data.responses[run_id]
        group = data.meta.loc[data.meta["run_id"].eq(run_id), "compound_group"].iloc[0]
        gates = data.surface.fit_predict(observed.tolist(), truth[observed], 0.40)
        interpolation = _interpolate(truth, observed)
        features, _, feature_hidden = _features(
            truth, observed, interpolation, gates.mean, gates.std
        )
        if not np.array_equal(hidden, feature_hidden):
            raise RuntimeError("hidden-cell ordering changed")
        residual = interpolation[hidden] + model.predict(features)
        rows.extend(
            {
                "run_id": run_id,
                "compound_group": group,
                "cell": int(cell),
                "truth": float(actual),
                "gates": float(gates.mean[cell]),
                "selected_residual": float(residual_prediction),
            }
            for cell, actual, residual_prediction in zip(
                hidden, truth[hidden], residual, strict=True
            )
        )
    return pd.DataFrame(rows)


def _evaluate(test: pd.DataFrame, calibration: pd.DataFrame, method: str) -> dict[str, object]:
    margin = _calibrate_recall_margin(calibration, method, 0.95)
    return {"calibration_margin": margin, **_metrics(test, method, np.full(len(test), margin))}


def _direction(
    meta: pd.DataFrame,
    responses: dict[str, np.ndarray],
    train_source: str,
    target_source: str,
) -> dict[str, object]:
    source_train = meta.loc[meta["v3_split"].eq("train") & meta["source"].eq(train_source)].copy()
    source_groups = set(source_train["compound_group"])
    calibration_groups = {
        group
        for group in source_groups
        if _bucket(group, f"agreement-source-cal-v1/{train_source}", 5) == 0
    }
    fit_groups = source_groups - calibration_groups
    prior_rows = meta.loc[
        meta["source"].eq(train_source)
        & (
            meta["v3_split"].eq("development")
            | (meta["v3_split"].eq("train") & meta["compound_group"].isin(fit_groups))
        )
    ]
    matrix = np.vstack([responses[run_id] for run_id in prior_rows["run_id"]])
    surface = RiskSensitiveSurface(neural_coordinates(), matrix)
    data = FoldData(meta=meta, responses=responses, surface=surface)
    fit_ids = source_train.loc[source_train["compound_group"].isin(fit_groups), "run_id"].tolist()
    calibration_ids = source_train.loc[
        source_train["compound_group"].isin(calibration_groups), "run_id"
    ].tolist()
    target_ids = meta.loc[
        meta["v3_split"].eq("train") & meta["source"].eq(target_source), "run_id"
    ].tolist()
    target_groups = set(meta.loc[meta["run_id"].isin(target_ids), "compound_group"].astype(str))
    x, y = _examples(data, fit_ids, "interpolation")
    model = _model().fit(x, y)
    context = _select_context(surface)
    calibration = _predict_context(data, calibration_ids, model, context)
    target = _predict_context(data, target_ids, model, context)
    methods = {method: _evaluate(target, calibration, method) for method in ("agreement", "gates")}
    methods["agreement_minus_gates"] = {
        "effect_precision": (
            methods["agreement"]["effect_precision"] - methods["gates"]["effect_precision"]
        ),
        "protected_recall": (
            methods["agreement"]["protected_recall"] - methods["gates"]["protected_recall"]
        ),
        "decisive_coverage": (
            methods["agreement"]["decisive_coverage"] - methods["gates"]["decisive_coverage"]
        ),
        "selective_risk": (
            methods["agreement"]["selective_risk"] - methods["gates"]["selective_risk"]
        ),
        "wrong_decisions": (
            methods["agreement"]["wrong_decisions"] - methods["gates"]["wrong_decisions"]
        ),
    }
    return {
        "train_source": train_source,
        "target_source": target_source,
        "fit_groups": len(fit_groups),
        "calibration_groups": len(calibration_groups),
        "target_groups": int(target["compound_group"].nunique()),
        "train_target_compound_group_overlap": len(source_groups & target_groups),
        "context": list(context),
        "methods": methods,
    }


def run_source_shift_agreement(
    legacy_workbook: Path, phase1_workbook: Path, output: Path
) -> dict[str, object]:
    _, _, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    directions = [
        _direction(meta, responses, "legacy_epa", "phase1_external"),
        _direction(meta, responses, "phase1_external", "legacy_epa"),
    ]
    strict = all(
        direction["methods"]["agreement_minus_gates"]["effect_precision"] > 0
        and direction["methods"]["agreement_minus_gates"]["protected_recall"] >= 0
        and direction["methods"]["agreement_minus_gates"]["selective_risk"] <= 0
        and direction["methods"]["agreement_minus_gates"]["wrong_decisions"] < 0
        for direction in directions
    )
    result = {
        "schema": "gates-act-leave-source-out-agreement-v1",
        "evidence_class": "locked development distribution-shift stress test; not confirmation",
        "budget": "16 of 32 response cells",
        "directions": directions,
        "strict_bidirectional_gate": "GO" if strict else "NO-GO",
        "canonical_integration": "not authorized",
        "early_stopping": "disabled",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "agreement_leave_source_out.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
