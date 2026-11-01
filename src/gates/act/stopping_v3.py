from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.benchmark_v3 import (
    POLICIES,
    V3Config,
    _apply_conformal,
    _conformal_quantile,
    _load_sources,
    _trace,
    _training_matrix,
)
from gates.act.model_v3 import RiskSensitiveSurface
from gates.act.neural_mea import neural_coordinates


def _risk(row: dict[str, object], weights: dict[str, float], interval_width: float) -> float:
    return float(
        weights["uncertainty"] * float(row["uncertainty"])
        + weights["shift"] * float(row["shift_score"])
        + weights["ambiguity"] * float(row["effect_ambiguity"])
        + weights["loo"] * float(row["loo_nrmse"])
        + weights["drift"] * float(row["prediction_drift"])
        + weights["interval"] * interval_width
    )


def _threshold(frame: pd.DataFrame, minimum_groups: int = 6) -> dict[str, float]:
    best = {"risk_threshold": -1.0, "stop_groups": 0, "violation_rate": 0.0}
    for threshold in sorted(frame.loc[frame["certificate"], "risk_score"].unique()):
        decisions = []
        for group, path in frame.groupby("compound_group"):
            eligible = path.loc[path["certificate"] & path["risk_score"].le(threshold)].sort_values(
                "round"
            )
            if len(eligible):
                decisions.append((group, bool(eligible.iloc[0]["safe"])))
        if len(decisions) < minimum_groups or any(not safe for _, safe in decisions):
            continue
        if len(decisions) > int(best["stop_groups"]):
            best = {
                "risk_threshold": float(threshold),
                "stop_groups": len(decisions),
                "violation_rate": 0.0,
            }
    return best


def lock_stopping_rule(
    legacy_workbook: Path,
    phase1_workbook: Path,
    model_lock_path: Path,
    output_path: Path,
) -> dict[str, object]:
    lock = json.loads(model_lock_path.read_text())
    config = V3Config(**lock["selected_config"])
    seed = int(lock["seed"])
    _, _, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    training = _training_matrix(meta, responses)
    coordinates = neural_coordinates()
    raw: dict[tuple[str, str, int], dict[str, object]] = {}
    subset = meta.loc[meta["v3_split"].isin(["development", "calibration"])]
    for record in subset.itertuples(index=False):
        truth = responses[record.run_id]
        surface = RiskSensitiveSurface(
            coordinates,
            training,
            config.covariance_blend,
            config.noise_fraction,
            config.analog_weight,
            config.neighbor_count,
        )
        for policy in POLICIES:
            for row in _trace(surface, truth, policy, record.run_id, seed, config):
                raw[(record.run_id, policy, int(row["round"]))] = row

    q_by_policy: dict[str, float] = {}
    calibration = meta.loc[meta["v3_split"].eq("calibration")]
    for policy in POLICIES:
        maxima = []
        for _group, group in calibration.groupby("compound_group"):
            values = []
            for run_id in group["run_id"]:
                for round_index in (2, 3):
                    values.extend(
                        np.asarray(raw[(run_id, policy, round_index)]["required_q"], float)
                        .clip(min=0)
                        .tolist()
                    )
            if values:
                maxima.append(max(values))
        q_by_policy[policy] = _conformal_quantile(maxima, config.conformal_coverage)

    prepared = []
    for record in subset.itertuples(index=False):
        truth = responses[record.run_id]
        for policy in POLICIES:
            for round_index in (2, 3):
                row = raw[(record.run_id, policy, round_index)]
                band = _apply_conformal(row, truth, q_by_policy[policy], config.effect_threshold)
                prepared.append(
                    {
                        "run_id": record.run_id,
                        "compound_group": record.compound_group,
                        "v3_split": record.v3_split,
                        "policy": policy,
                        "round": round_index,
                        "certificate": row["observed_effect_divs"] >= config.minimum_effect_divs
                        or band["max_effect_upper"] < config.effect_threshold,
                        "safe": row["nrmse"] <= config.target_nrmse
                        and band["effect_recall"] >= config.target_effect_recall,
                        "interval_width": band["mean_interval_width"],
                        **{
                            key: float(row[key])
                            for key in (
                                "uncertainty",
                                "shift_score",
                                "effect_ambiguity",
                                "loo_nrmse",
                                "prediction_drift",
                            )
                        },
                    }
                )
    frame = pd.DataFrame(prepared)
    search_rows = []
    development = frame.loc[
        frame["v3_split"].eq("development") & frame["policy"].eq("risk_sensitive")
    ].copy()
    for shift, ambiguity, loo, drift, interval in itertools.product(
        (0.0, 0.05, 0.10, 0.20),
        (0.0, 0.20, 0.50),
        (0.5, 1.0, 2.0, 4.0),
        (0.0, 0.5, 1.0, 2.0),
        (0.0, 0.05, 0.10),
    ):
        weights = {
            "uncertainty": 1.0,
            "shift": shift,
            "ambiguity": ambiguity,
            "loo": loo,
            "drift": drift,
            "interval": interval,
        }
        development["risk_score"] = [
            _risk(row._asdict(), weights, row.interval_width)
            for row in development.itertuples(index=False)
        ]
        result = _threshold(development, minimum_groups=5)
        search_rows.append({**weights, **result})
    search = pd.DataFrame(search_rows).sort_values(
        ["stop_groups", "risk_threshold"], ascending=[False, False]
    )
    best = search.iloc[0]
    if int(best["stop_groups"]) < 5:
        raise RuntimeError("no zero-violation development stopping rule reached five groups")
    weights = {
        key: float(best[key])
        for key in ("uncertainty", "shift", "ambiguity", "loo", "drift", "interval")
    }
    rules = {}
    for policy in POLICIES:
        policy_frame = frame.loc[
            frame["v3_split"].eq("calibration") & frame["policy"].eq(policy)
        ].copy()
        policy_frame["risk_score"] = [
            _risk(row._asdict(), weights, row.interval_width)
            for row in policy_frame.itertuples(index=False)
        ]
        result = _threshold(policy_frame, minimum_groups=3)
        result["conformal_q"] = q_by_policy[policy]
        result["ood_threshold"] = float(
            policy_frame.loc[policy_frame["round"].eq(2), "shift_score"].quantile(
                config.ood_quantile
            )
        )
        rules[policy] = result
    diagnostic = {
        "risk_weights": weights,
        "rules": rules,
        "development_stop_groups": int(best["stop_groups"]),
    }
    output_path.with_name("stopping_lock_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    )
    if rules["risk_sensitive"]["risk_threshold"] < 0:
        raise RuntimeError("risk-sensitive policy failed zero-violation calibration acceptance")
    locked = {
        "schema": "gates-act-v3-stopping-lock-v1",
        "model_lock_sha256": __import__("hashlib").sha256(model_lock_path.read_bytes()).hexdigest(),
        "risk_weights": weights,
        "rules": rules,
        "acceptance": (
            "weights selected on development groups; policy thresholds calibrated with zero "
            "observed unsafe stops; no legacy v2 test or final tipping-point confirmation groups"
        ),
    }
    output_path.write_text(json.dumps(locked, indent=2, sort_keys=True) + "\n")
    search.to_csv(output_path.with_name("stopping_development_search.csv"), index=False)
    return locked
