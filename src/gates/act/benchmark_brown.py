from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.benchmark_v2 import _seed_for
from gates.act.benchmark_v3 import (
    POLICIES,
    V3Config,
    _conformal_quantile,
    _load_sources,
    _paired_sign_flip,
)
from gates.act.brown_mea import (
    brown_surface_arrays,
    load_brown_mea,
    seven_dose_coordinates,
    seven_dose_surface,
)
from gates.act.model_v3 import RiskPrediction, RiskSensitiveSurface
from gates.act.neural_mea import DIVS


def _select(
    surface: RiskSensitiveSurface,
    prediction: RiskPrediction,
    observed: list[int],
    time_rank: int,
    policy: str,
    rng: np.random.Generator,
    config: V3Config,
) -> list[int]:
    candidates = np.arange(7 * time_rank, 7 * (time_rank + 1), dtype=int)
    if policy == "fixed_sentinel":
        return [int(candidates[i]) for i in (0, 2, 4, 6)]
    selected = [int(candidates[0])]
    covariance = prediction.covariance.copy()
    while len(selected) < 4:
        available = np.asarray([i for i in candidates if i not in selected], dtype=int)
        if policy == "random":
            chosen = int(rng.choice(available))
        elif policy == "max_variance":
            chosen = int(available[np.argmax(np.diag(covariance)[available])])
        elif policy == "maximin":
            anchors = np.asarray(observed + selected, dtype=int)
            distance = np.min(
                np.linalg.norm(
                    surface.coordinates[available, None, :]
                    - surface.coordinates[anchors][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            chosen = int(available[np.argmax(distance)])
        else:
            targets = np.asarray(
                [i for i in range(len(surface.coordinates)) if i not in observed + selected]
            )
            scores = [
                surface.risk_sensitive_score(
                    prediction, int(i), targets, config.effect_weight, config.direct_bonus
                )
                for i in available
            ]
            chosen = int(available[int(np.argmax(scores))])
        selected.append(chosen)
        denominator = max(float(covariance[chosen, chosen] + surface.noise), 1e-12)
        covariance -= np.outer(covariance[:, chosen], covariance[chosen, :]) / denominator
    return selected


def _trace(surface, truth, policy, run_id, config, seed):
    observed = list(range(7))
    rng = np.random.default_rng(_seed_for(f"brown/{run_id}/{policy}", seed))
    rows = []
    for round_index, div in enumerate(DIVS, start=1):
        if round_index > 1:
            before = surface.fit_predict(observed, truth[observed], config.effect_threshold)
            observed.extend(
                _select(surface, before, observed, round_index - 1, policy, rng, config)
            )
        prediction = surface.fit_predict(observed, truth[observed], config.effect_threshold)
        hidden = np.asarray([i for i in range(len(truth)) if i not in observed], dtype=int)
        effects = np.abs(truth) >= config.effect_threshold
        observed_effect = effects[np.asarray(observed)]
        required = (
            (config.effect_threshold - np.abs(prediction.mean[hidden][effects[hidden]]))
            / prediction.std[hidden][effects[hidden]]
            if np.any(effects[hidden])
            else np.asarray([])
        )
        error = prediction.mean[hidden] - truth[hidden]
        rows.append(
            {
                "round": round_index,
                "div": div,
                "selected_cells": len(observed),
                "selected_indices": ";".join(map(str, observed)),
                "nrmse": float(np.sqrt(np.mean(error**2)) / surface.response_scale)
                if len(hidden)
                else 0.0,
                "uncertainty": prediction.uncertainty_score,
                "shift_score": prediction.shift_score,
                "effect_ambiguity": float(
                    np.mean(prediction.effect_probability * (1 - prediction.effect_probability))
                ),
                "mean": prediction.mean,
                "std": prediction.std,
                "hidden": hidden,
                "required_q": required,
                "true_effect_cells": int(effects.sum()),
                "observed_effect_cells": int(observed_effect.sum()),
                "observed_effect_divs": int(
                    len({i // 7 for i, hit in zip(observed, observed_effect, strict=True) if hit})
                ),
                "direct_effect_capture": float(observed_effect.sum() / max(effects.sum(), 1)),
            }
        )
    return rows


def _band(row, truth, q, threshold):
    hidden = np.asarray(row["hidden"], dtype=int)
    effect = np.abs(truth[hidden]) >= threshold
    predicted = np.abs(row["mean"][hidden]) + q * row["std"][hidden] >= threshold
    tp = int((effect & predicted).sum())
    fn = int((effect & ~predicted).sum())
    fp = int((~effect & predicted).sum())
    protected = int(row["observed_effect_cells"]) + tp
    total = int(row["true_effect_cells"])
    return {
        "effect_recall": float(protected / total) if total else 1.0,
        "effect_precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "missed_effect_cells": fn,
        "compound_effect_missed": int(total > 0 and protected == 0),
        "max_effect_upper": float(np.max(np.abs(row["mean"][hidden]) + q * row["std"][hidden]))
        if len(hidden)
        else 0.0,
        "mean_interval_width": float(np.mean(2 * q * row["std"][hidden])) if len(hidden) else 0.0,
    }


def run_brown_confirmation(
    legacy_workbook: Path,
    phase1_workbook: Path,
    brown_csv: Path,
    preregistration: Path,
    output_dir: Path,
):
    lock = json.loads(preregistration.read_text())
    config = V3Config(**lock["selected_config"])
    seed = int(lock["seed"])
    _, _, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    training = np.vstack(
        [seven_dose_surface(responses[x]) for x in meta.loc[meta.v3_split.eq("train"), "run_id"]]
    )
    calibration = meta.loc[meta.v3_split.eq("calibration")].copy()
    brown = load_brown_mea(brown_csv)
    brown_meta, brown_values = brown_surface_arrays(brown)
    coordinates = seven_dose_coordinates()
    raw: dict[tuple[str, str, int], dict[str, object]] = {}
    for record in calibration.itertuples(index=False):
        truth = seven_dose_surface(responses[record.run_id])
        surface = RiskSensitiveSurface(
            coordinates,
            training,
            config.covariance_blend,
            config.noise_fraction,
            config.analog_weight,
            config.neighbor_count,
        )
        for policy in POLICIES:
            for row in _trace(surface, truth, policy, record.run_id, config, seed):
                raw[(record.run_id, policy, row["round"])] = row | {"truth": truth}

    q_by_policy = {}
    for policy in POLICIES:
        maxima = []
        for _group, group in calibration.groupby("compound_group"):
            values = []
            for run_id in group.run_id:
                for round_index in (2, 3):
                    values.extend(
                        np.asarray(raw[(run_id, policy, round_index)]["required_q"])
                        .clip(min=0)
                        .tolist()
                    )
            if values:
                maxima.append(max(values))
        q_by_policy[policy] = _conformal_quantile(maxima, config.conformal_coverage)

    rules = {}
    calibration_rows = []
    for (run_id, policy, round_index), row in raw.items():
        band = _band(row, row["truth"], q_by_policy[policy], config.effect_threshold)
        risk = (
            row["uncertainty"]
            + 0.08 * row["shift_score"]
            + 0.35 * row["effect_ambiguity"]
            + 0.05 * band["mean_interval_width"]
        )
        record = calibration.loc[calibration.run_id.eq(run_id)].iloc[0]
        calibration_rows.append(
            {
                "run_id": run_id,
                "compound_group": record.compound_group,
                "policy": policy,
                "round": round_index,
                "risk_score": risk,
                "shift_score": row["shift_score"],
                "certificate": row["observed_effect_divs"] >= config.minimum_effect_divs
                or band["max_effect_upper"] < config.effect_threshold,
                "safe": row["nrmse"] <= config.target_nrmse
                and band["effect_recall"] >= config.target_effect_recall,
            }
        )
    cal = pd.DataFrame(calibration_rows)
    for policy in POLICIES:
        frame = cal.loc[cal.policy.eq(policy) & cal["round"].isin([2, 3])]
        best = (-1.0, 0, 0.0)
        for threshold in sorted(frame.loc[frame.certificate, "risk_score"].unique()):
            decisions = []
            for group, path in frame.groupby("compound_group"):
                eligible = path.loc[path.certificate & path.risk_score.le(threshold)].sort_values(
                    "round"
                )
                if len(eligible):
                    decisions.append((group, bool(eligible.iloc[0].safe)))
            if len(decisions) < 8:
                continue
            violation = np.mean([not safe for _, safe in decisions])
            if violation <= 0.10 and len(decisions) > best[1]:
                best = (float(threshold), len(decisions), float(violation))
        rules[policy] = {
            "risk_threshold": best[0],
            "calibration_stop_groups": best[1],
            "calibration_violation_rate": best[2],
            "conformal_q": q_by_policy[policy],
            "ood_threshold": float(
                frame.loc[frame["round"].eq(2), "shift_score"].quantile(config.ood_quantile)
            ),
        }

    test_rows = []
    decisions = []
    for record in brown_meta.itertuples(index=False):
        truth = brown_values[record.run_id]
        surface = RiskSensitiveSurface(
            coordinates,
            training,
            config.covariance_blend,
            config.noise_fraction,
            config.analog_weight,
            config.neighbor_count,
        )
        for policy in POLICIES:
            rule = rules[policy]
            path = []
            for row in _trace(surface, truth, policy, record.run_id, config, seed):
                band = _band(row, truth, rule["conformal_q"], config.effect_threshold)
                risk = (
                    row["uncertainty"]
                    + 0.08 * row["shift_score"]
                    + 0.35 * row["effect_ambiguity"]
                    + 0.05 * band["mean_interval_width"]
                )
                serial = (
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"mean", "std", "hidden", "required_q"}
                    }
                    | band
                    | {"risk_score": risk}
                )
                path.append(serial)
                test_rows.append(
                    {
                        "run_id": record.run_id,
                        "compound": record.compound,
                        "compound_group": record.compound_group,
                        "policy": policy,
                        **serial,
                    }
                )
            path_frame = pd.DataFrame(path)
            terminal = path_frame.iloc[-1].copy()
            decision = "COMPLETE"
            for _, row in path_frame.loc[path_frame["round"].isin([2, 3])].iterrows():
                certificate = (
                    row["observed_effect_divs"] >= config.minimum_effect_divs
                    or row["max_effect_upper"] < config.effect_threshold
                )
                if row["round"] == 2 and row["shift_score"] > rule["ood_threshold"]:
                    decision = "ABSTAIN"
                    break
                if (
                    rule["risk_threshold"] >= 0
                    and certificate
                    and row["risk_score"] <= rule["risk_threshold"]
                ):
                    decision = "STOP"
                    terminal = row
                    break
            if decision in {"COMPLETE", "ABSTAIN"}:
                terminal = path_frame.iloc[-1].copy()
                terminal["round"] = 4
                terminal["div"] = 12
                terminal["nrmse"] = 0.0
                terminal["effect_recall"] = 1.0
                terminal["missed_effect_cells"] = 0
                terminal["compound_effect_missed"] = 0
            decisions.append(
                {
                    "run_id": record.run_id,
                    "compound": record.compound,
                    "compound_group": record.compound_group,
                    "policy": policy,
                    "decision": decision,
                    "decision_round": int(terminal["round"]),
                    "decision_div": int(terminal["div"]),
                    "recording_sessions": int(terminal["round"]),
                    "instrument_well_recordings": 24 * int(terminal["round"]),
                    "terminal_nrmse": float(terminal["nrmse"]),
                    "effect_recall": float(terminal["effect_recall"]),
                    "missed_effect_cells": int(terminal["missed_effect_cells"]),
                    "stop_safe": bool(
                        terminal["nrmse"] <= config.target_nrmse
                        and terminal["effect_recall"] >= config.target_effect_recall
                    ),
                }
            )

    traces = pd.DataFrame(test_rows)
    decision_frame = pd.DataFrame(decisions)
    group = (
        traces.loc[traces["round"].isin([2, 3])]
        .groupby(["compound_group", "policy"], as_index=False)
        .agg(effect_recall=("effect_recall", "mean"), nrmse=("nrmse", "mean"))
    )
    rs = group.loc[group.policy.eq("risk_sensitive")].set_index("compound_group")
    comparison_rows = []
    for baseline in POLICIES[1:]:
        other = group.loc[group.policy.eq(baseline)].set_index("compound_group")
        common = rs.index.intersection(other.index)
        recall = (rs.loc[common, "effect_recall"] - other.loc[common, "effect_recall"]).to_numpy()
        comparison_rows.append(
            {
                "baseline": baseline,
                "effect_recall_advantage": float(recall.mean()),
                "effect_recall_sign_flip_p": _paired_sign_flip(recall, seed),
                "test_groups": len(common),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces.to_csv(output_dir / "brown_fresh_traces.csv", index=False)
    decision_frame.to_csv(output_dir / "brown_fresh_decisions.csv", index=False)
    comparisons.to_csv(output_dir / "brown_fresh_comparisons.csv", index=False)
    (output_dir / "brown_data_audit.json").write_text(
        json.dumps(brown.audit, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "brown_rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    risk = decision_frame.loc[decision_frame.policy.eq("risk_sensitive")]
    stops = risk.loc[risk.decision.eq("STOP")]
    summary = {
        "dataset": "Brown et al. 2016 EPA neural network ontogeny MEA training set",
        "source_doi": "10.23719/1412682",
        "fresh_compound_groups": int(risk.compound_group.nunique()),
        "fresh_runs": int(risk.run_id.nunique()),
        "risk_sensitive_decisions": risk.decision.value_counts().to_dict(),
        "risk_sensitive_stops": int(len(stops)),
        "risk_sensitive_unsafe_stops": int((~stops.stop_safe).sum()),
        "risk_sensitive_stop_effect_recall": float(stops.effect_recall.mean())
        if len(stops)
        else None,
        "recording_session_reduction": float(1 - risk.recording_sessions.mean() / 4),
        "calibration_violation_budget": 0.10,
        "measurement_cost_contract": "24 wells allocated; savings count only avoided DIV sessions",
    }
    (output_dir / "brown_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
