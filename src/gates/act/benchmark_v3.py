from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.benchmark_v2 import _seed_for
from gates.act.model_v3 import RiskPrediction, RiskSensitiveSurface
from gates.act.neural_mea import DIVS, load_neural_mea, neural_coordinates, surface_arrays
from gates.act.neural_mea_phase1 import load_phase1_mea, phase1_surface_arrays

POLICIES = ("risk_sensitive", "max_variance", "maximin", "fixed_sentinel", "random")


@dataclass(frozen=True)
class V3Config:
    covariance_blend: float
    noise_fraction: float
    effect_weight: float
    direct_bonus: float
    analog_weight: float = 0.50
    neighbor_count: int = 15
    effect_threshold: float = 0.40
    target_nrmse: float = 0.20
    target_effect_recall: float = 0.90
    conformal_coverage: float = 0.90
    ood_quantile: float = 0.90
    minimum_stop_round: int = 2
    minimum_effect_divs: int = 2


def _development_split(group: str) -> str:
    bucket = int(hashlib.sha256(f"v3-development/{group}".encode()).hexdigest()[:8], 16) % 5
    return "development" if bucket == 0 else "train"


def _load_sources(legacy_workbook: Path, phase1_workbook: Path):
    legacy = load_neural_mea(legacy_workbook)
    legacy_meta, legacy_values = surface_arrays(legacy)
    legacy_meta = legacy_meta.copy()
    legacy_meta["source"] = "legacy_epa"
    legacy_meta["v3_split"] = legacy_meta.apply(
        lambda row: (
            _development_split(row["compound_group"])
            if row["split"] == "train"
            else ("calibration" if row["split"] == "calibration" else "legacy_frozen")
        ),
        axis=1,
    )
    phase1 = load_phase1_mea(phase1_workbook, set(legacy.surfaces["compound_group"]))
    phase1_meta, phase1_values = phase1_surface_arrays(phase1)
    phase1_meta = phase1_meta.copy()
    phase1_meta["source"] = "phase1_external"
    phase1_meta["v3_split"] = phase1_meta["split"]
    meta = pd.concat([legacy_meta, phase1_meta], ignore_index=True)
    responses = {**legacy_values, **phase1_values}
    return legacy, phase1, meta, responses


def _training_matrix(meta: pd.DataFrame, responses: dict[str, np.ndarray]) -> np.ndarray:
    ids = meta.loc[meta["v3_split"].eq("train"), "run_id"]
    return np.vstack([responses[run_id] for run_id in ids])


def _select_batch(
    surface: RiskSensitiveSurface,
    prediction: RiskPrediction,
    observed: list[int],
    time_rank: int,
    policy: str,
    rng: np.random.Generator,
    config: V3Config,
    size: int = 4,
) -> list[int]:
    candidates = np.arange(8 * time_rank, 8 * (time_rank + 1), dtype=int)
    if policy == "fixed_sentinel":
        return [8 * time_rank + rank for rank in (0, 2, 5, 7)][:size]
    selected = [int(candidates[0])]  # vehicle anchor required in every batch
    covariance = prediction.covariance.copy()
    for _ in range(size - 1):
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
                [i for i in range(len(surface.coordinates)) if i not in observed + selected],
                dtype=int,
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
        else:
            raise ValueError(f"unknown v3 policy {policy}")
        selected.append(chosen)
        denominator = max(float(covariance[chosen, chosen] + surface.noise), 1e-12)
        covariance -= np.outer(covariance[:, chosen], covariance[chosen, :]) / denominator
    return selected


def _raw_metrics(
    prediction: RiskPrediction,
    truth: np.ndarray,
    observed: list[int],
    scale: float,
    threshold: float,
) -> dict[str, object]:
    hidden = np.asarray([i for i in range(len(truth)) if i not in observed], dtype=int)
    observed_array = np.asarray(observed, dtype=int)
    true_effect_all = np.abs(truth) >= threshold
    true_effect_hidden = true_effect_all[hidden]
    observed_effect = true_effect_all[observed_array]
    error = prediction.mean[hidden] - truth[hidden]
    return {
        "nrmse": float(np.sqrt(np.mean(error**2)) / scale) if len(hidden) else 0.0,
        "mae": float(np.mean(np.abs(error))) if len(hidden) else 0.0,
        "true_effect_cells": int(true_effect_all.sum()),
        "true_hidden_effect_cells": int(true_effect_hidden.sum()),
        "observed_effect_cells": int(observed_effect.sum()),
        "direct_effect_capture": float(observed_effect.sum() / max(true_effect_all.sum(), 1)),
        "observed_effect_divs": int(
            len({i // 8 for i, hit in zip(observed, observed_effect, strict=True) if hit})
        ),
        "hidden_indices": hidden,
        "required_q": (
            (threshold - np.abs(prediction.mean[hidden][true_effect_hidden]))
            / prediction.std[hidden][true_effect_hidden]
            if np.any(true_effect_hidden)
            else np.asarray([], dtype=float)
        ),
    }


def _trace(
    surface: RiskSensitiveSurface,
    truth: np.ndarray,
    policy: str,
    run_id: str,
    seed: int,
    config: V3Config,
) -> list[dict[str, object]]:
    observed = list(range(8))
    rng = np.random.default_rng(_seed_for(f"v3/{run_id}/{policy}", seed))
    rows: list[dict[str, object]] = []
    previous_mean: np.ndarray | None = None
    for round_index, div in enumerate(DIVS, start=1):
        if round_index > 1:
            before = surface.fit_predict(observed, truth[observed], config.effect_threshold)
            observed.extend(
                _select_batch(
                    surface,
                    before,
                    observed,
                    round_index - 1,
                    policy,
                    rng,
                    config,
                )
            )
        prediction = surface.fit_predict(observed, truth[observed], config.effect_threshold)
        cross_validation_errors = []
        for held_out in observed:
            retained = [index for index in observed if index != held_out]
            if not retained:
                continue
            leave_one_out = surface.fit_predict(retained, truth[retained], config.effect_threshold)
            cross_validation_errors.append(leave_one_out.mean[held_out] - truth[held_out])
        loo_nrmse = (
            float(
                np.sqrt(np.mean(np.asarray(cross_validation_errors) ** 2)) / surface.response_scale
            )
            if cross_validation_errors
            else 0.0
        )
        prediction_drift = (
            float(np.sqrt(np.mean((prediction.mean - previous_mean) ** 2)) / surface.response_scale)
            if previous_mean is not None
            else 0.0
        )
        metrics = _raw_metrics(
            prediction,
            truth,
            observed,
            surface.response_scale,
            config.effect_threshold,
        )
        rows.append(
            {
                "round": round_index,
                "div": div,
                "selected_cells": len(observed),
                "selected_indices": ";".join(map(str, observed)),
                "uncertainty": prediction.uncertainty_score,
                "shift_score": prediction.shift_score,
                "loo_nrmse": loo_nrmse,
                "prediction_drift": prediction_drift,
                "effect_ambiguity": float(
                    np.mean(prediction.effect_probability * (1 - prediction.effect_probability))
                ),
                "mean": prediction.mean,
                "std": prediction.std,
                **metrics,
            }
        )
        previous_mean = prediction.mean.copy()
    return rows


def _conformal_quantile(values: list[float], coverage: float) -> float:
    if not values:
        return 0.0
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(math.ceil((len(ordered) + 1) * coverage), len(ordered))
    return float(max(ordered[rank - 1], 0.0))


def _apply_conformal(row: dict[str, object], truth: np.ndarray, q: float, threshold: float):
    hidden = np.asarray(row["hidden_indices"], dtype=int)
    mean = np.asarray(row["mean"], dtype=float)
    std = np.asarray(row["std"], dtype=float)
    true_effect = np.abs(truth[hidden]) >= threshold
    predicted_effect = np.abs(mean[hidden]) + q * std[hidden] >= threshold
    tp = int((true_effect & predicted_effect).sum())
    fn = int((true_effect & ~predicted_effect).sum())
    fp = int((~true_effect & predicted_effect).sum())
    observed_effect = int(row["observed_effect_cells"])
    total_effect = int(row["true_effect_cells"])
    protected = observed_effect + tp
    return {
        "effect_recall": float(protected / total_effect) if total_effect else 1.0,
        "hidden_effect_recall": float(tp / true_effect.sum()) if true_effect.sum() else 1.0,
        "effect_precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "missed_effect_cells": fn,
        "compound_effect_missed": int(total_effect > 0 and protected == 0),
        "max_effect_upper": float(np.max(np.abs(mean[hidden]) + q * std[hidden]))
        if len(hidden)
        else 0.0,
        "mean_interval_width": float(np.mean(2 * q * std[hidden])) if len(hidden) else 0.0,
    }


def _evaluate_config(
    config: V3Config,
    meta: pd.DataFrame,
    responses: dict[str, np.ndarray],
    training: np.ndarray,
    seed: int,
) -> dict[str, float]:
    coordinates = neural_coordinates()
    policy_rows: dict[str, list[dict[str, float]]] = {policy: [] for policy in POLICIES}
    for record in meta.loc[meta["v3_split"].eq("development")].itertuples(index=False):
        surface = RiskSensitiveSurface(
            coordinates,
            training,
            config.covariance_blend,
            config.noise_fraction,
            config.analog_weight,
            config.neighbor_count,
        )
        truth = responses[record.run_id]
        for policy in POLICIES:
            for row in _trace(surface, truth, policy, record.run_id, seed, config):
                if row["round"] not in (2, 3):
                    continue
                band = _apply_conformal(row, truth, q=1.64, threshold=config.effect_threshold)
                policy_rows[policy].append(
                    {
                        "recall": band["effect_recall"],
                        "precision": band["effect_precision"],
                        "nrmse": float(row["nrmse"]),
                        "direct": float(row["direct_effect_capture"]),
                    }
                )
    summaries = {
        policy: pd.DataFrame(rows).mean(numeric_only=True).to_dict()
        for policy, rows in policy_rows.items()
    }
    rs = summaries["risk_sensitive"]
    strongest_recall = max(summaries[p]["recall"] for p in POLICIES if p != "risk_sensitive")
    strongest_direct = max(summaries[p]["direct"] for p in POLICIES if p != "risk_sensitive")
    objective = (
        rs["recall"]
        + 0.35 * rs["direct"]
        + 0.20 * rs["precision"]
        - 0.30 * rs["nrmse"]
        + 0.50 * (rs["recall"] - strongest_recall)
        + 0.20 * (rs["direct"] - strongest_direct)
    )
    return {
        "objective": float(objective),
        "risk_recall": float(rs["recall"]),
        "risk_precision": float(rs["precision"]),
        "risk_direct_capture": float(rs["direct"]),
        "risk_nrmse": float(rs["nrmse"]),
        "recall_advantage_vs_best_baseline": float(rs["recall"] - strongest_recall),
        "direct_advantage_vs_best_baseline": float(rs["direct"] - strongest_direct),
    }


def select_v3_config(
    legacy_workbook: Path,
    phase1_workbook: Path,
    output_dir: Path,
    seed: int = 20261023,
) -> dict[str, object]:
    legacy, phase1, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    training = _training_matrix(meta, responses)
    rows = []
    for blend, noise, weight, bonus, analog, neighbors in itertools.product(
        (0.55, 0.75, 0.90),
        (0.03, 0.08),
        (1.0, 3.0, 6.0),
        (0.5, 1.5, 3.0),
        (0.25, 0.50, 0.75),
        (10, 18),
    ):
        config = V3Config(blend, noise, weight, bonus, analog, neighbors)
        rows.append({**asdict(config), **_evaluate_config(config, meta, responses, training, seed)})
    search = pd.DataFrame(rows).sort_values(
        ["objective", "recall_advantage_vs_best_baseline", "risk_recall"], ascending=False
    )
    best = search.iloc[0]
    if (
        best["risk_recall"] < 0.90
        or best["recall_advantage_vs_best_baseline"] <= 0
        or best["direct_advantage_vs_best_baseline"] < 0.02
    ):
        raise RuntimeError(
            "development acceptance gate failed: require >=90% effect recall, positive recall "
            "margin versus every baseline, and >=2 percentage-point direct-capture margin"
        )
    config = V3Config(
        covariance_blend=float(best["covariance_blend"]),
        noise_fraction=float(best["noise_fraction"]),
        effect_weight=float(best["effect_weight"]),
        direct_bonus=float(best["direct_bonus"]),
        analog_weight=float(best["analog_weight"]),
        neighbor_count=int(best["neighbor_count"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(output_dir / "development_model_search.csv", index=False)
    lock = {
        "schema": "gates-act-v3-preregistration-v1",
        "seed": seed,
        "selected_config": asdict(config),
        "selection_evidence": {
            key: float(best[key])
            for key in (
                "objective",
                "risk_recall",
                "risk_precision",
                "risk_direct_capture",
                "risk_nrmse",
                "recall_advantage_vs_best_baseline",
                "direct_advantage_vs_best_baseline",
            )
        },
        "fresh_test_boundary": (
            "Phase-1 DTXSID groups assigned to test by _v3_split; their outcomes are ingested "
            "for schema-complete loading but never used by configuration selection, fitting, "
            "calibration, or acceptance gates"
        ),
        "legacy_test_boundary": "all legacy v2 test groups excluded from v3 selection and fitting",
        "source_audits": {"legacy": legacy.audit, "phase1": phase1.audit},
    }
    (output_dir / "preregistered_config.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n"
    )
    return lock


def _paired_sign_flip(differences: np.ndarray, seed: int, draws: int = 100_000) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(differences)))
    if len(differences) <= 20:
        values = [
            abs(float(np.mean(differences * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
        return float(np.mean(np.asarray(values) >= observed - 1e-12))
    exceed = 0
    for _ in range(draws):
        signed = differences * rng.choice((-1.0, 1.0), size=len(differences))
        exceed += abs(float(signed.mean())) >= observed - 1e-12
    return float((exceed + 1) / (draws + 1))


def _cluster_comparisons(traces: pd.DataFrame, seed: int) -> pd.DataFrame:
    test = traces.loc[traces["v3_split"].eq("test") & traces["round"].isin([2, 3])].copy()
    group = test.groupby(["compound_group", "policy"], as_index=False).agg(
        effect_recall=("effect_recall", "mean"), nrmse=("nrmse", "mean")
    )
    rs = group.loc[group["policy"].eq("risk_sensitive")].set_index("compound_group")
    rng = np.random.default_rng(seed)
    rows = []
    for baseline in POLICIES[1:]:
        other = group.loc[group["policy"].eq(baseline)].set_index("compound_group")
        common = rs.index.intersection(other.index)
        recall = (rs.loc[common, "effect_recall"] - other.loc[common, "effect_recall"]).to_numpy()
        nrmse = (other.loc[common, "nrmse"] - rs.loc[common, "nrmse"]).to_numpy()
        recall_boot = np.asarray(
            [rng.choice(recall, len(recall), replace=True).mean() for _ in range(20_000)]
        )
        nrmse_boot = np.asarray(
            [rng.choice(nrmse, len(nrmse), replace=True).mean() for _ in range(20_000)]
        )
        rows.append(
            {
                "baseline": baseline,
                "effect_recall_advantage": float(recall.mean()),
                "effect_recall_ci_low": float(np.quantile(recall_boot, 0.025)),
                "effect_recall_ci_high": float(np.quantile(recall_boot, 0.975)),
                "effect_recall_sign_flip_p": _paired_sign_flip(recall, seed),
                "nrmse_advantage": float(nrmse.mean()),
                "nrmse_ci_low": float(np.quantile(nrmse_boot, 0.025)),
                "nrmse_ci_high": float(np.quantile(nrmse_boot, 0.975)),
                "nrmse_sign_flip_p": _paired_sign_flip(nrmse, seed),
                "test_groups": len(common),
            }
        )
    return pd.DataFrame(rows)


def run_v3_benchmark(
    legacy_workbook: Path,
    phase1_workbook: Path,
    preregistration: Path,
    output_dir: Path,
) -> dict[str, object]:
    lock = json.loads(preregistration.read_text())
    config = V3Config(**lock["selected_config"])
    seed = int(lock["seed"])
    legacy, phase1, meta, responses = _load_sources(legacy_workbook, phase1_workbook)
    training = _training_matrix(meta, responses)
    coordinates = neural_coordinates()

    all_rows: list[dict[str, object]] = []
    raw_by_key: dict[tuple[str, str, int], dict[str, object]] = {}
    eligible_meta = meta.loc[meta["v3_split"].isin(["calibration", "test"])].copy()
    for record in eligible_meta.itertuples(index=False):
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
                raw_by_key[(record.run_id, policy, int(row["round"]))] = row
                all_rows.append(
                    {
                        "run_id": record.run_id,
                        "compound": record.compound,
                        "compound_group": record.compound_group,
                        "source": record.source,
                        "v3_split": record.v3_split,
                        "policy": policy,
                        **row,
                    }
                )

    conformal_q: dict[str, float] = {}
    rules: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        group_scores = []
        calibration_meta = meta.loc[meta["v3_split"].eq("calibration")]
        for _group, group_meta in calibration_meta.groupby("compound_group"):
            scores = []
            for run_id in group_meta["run_id"]:
                for round_index in (2, 3):
                    scores.extend(
                        np.asarray(raw_by_key[(run_id, policy, round_index)]["required_q"], float)
                        .clip(min=0)
                        .tolist()
                    )
            if scores:
                group_scores.append(max(scores))
        conformal_q[policy] = _conformal_quantile(group_scores, config.conformal_coverage)

    serial_rows = []
    for row in all_rows:
        truth = responses[str(row["run_id"])]
        band = _apply_conformal(
            row,
            truth,
            conformal_q[str(row["policy"])],
            config.effect_threshold,
        )
        risk_score = (
            float(row["uncertainty"])
            + 0.08 * float(row["shift_score"])
            + 0.35 * float(row["effect_ambiguity"])
            + 0.05 * band["mean_interval_width"]
        )
        serial_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"mean", "std", "hidden_indices", "required_q"}
            }
            | band
            | {"risk_score": risk_score, "conformal_q": conformal_q[str(row["policy"])]}
        )
    traces = pd.DataFrame(serial_rows)

    for policy in POLICIES:
        cal = traces.loc[
            traces["v3_split"].eq("calibration")
            & traces["policy"].eq(policy)
            & traces["round"].isin([2, 3])
        ].copy()
        cal["biology_certificate"] = cal["observed_effect_divs"].ge(
            config.minimum_effect_divs
        ) | cal["max_effect_upper"].lt(config.effect_threshold)
        cal["safe"] = cal["nrmse"].le(config.target_nrmse) & cal["effect_recall"].ge(
            config.target_effect_recall
        )
        group_cal = cal.groupby(["compound_group", "round"], as_index=False).agg(
            risk_score=("risk_score", "max"),
            shift_score=("shift_score", "max"),
            biology_certificate=("biology_certificate", "all"),
            safe=("safe", "all"),
        )
        candidates = group_cal.loc[group_cal["biology_certificate"]].copy()
        best_threshold = -1.0
        best_coverage = 0.0
        for threshold in sorted(candidates["risk_score"].unique()):
            selected = candidates.loc[candidates["risk_score"].le(threshold)]
            if selected["compound_group"].nunique() < 8 or not selected["safe"].all():
                continue
            coverage = selected["compound_group"].nunique() / max(
                candidates["compound_group"].nunique(), 1
            )
            if coverage > best_coverage:
                best_threshold = float(threshold)
                best_coverage = float(coverage)
        rules[policy] = {
            "risk_threshold": best_threshold,
            "calibration_coverage": best_coverage,
            "calibration_violation_rate": 0.0,
            "ood_threshold": float(
                group_cal.loc[group_cal["round"].eq(2), "shift_score"].quantile(config.ood_quantile)
            ),
            "conformal_q": conformal_q[policy],
        }

    decisions = []
    for policy in POLICIES:
        rule = rules[policy]
        for run_id, path in traces.loc[
            traces["v3_split"].eq("test") & traces["policy"].eq(policy)
        ].groupby("run_id"):
            path = path.sort_values("round")
            terminal = path.iloc[-1].copy()
            decision = "COMPLETE"
            for _, row in path.loc[path["round"].isin([2, 3])].iterrows():
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
                terminal = path.iloc[-1].copy()
                terminal["round"] = 4
                terminal["div"] = 12
                terminal["selected_cells"] = 32
                terminal["nrmse"] = 0.0
                terminal["effect_recall"] = 1.0
                terminal["missed_effect_cells"] = 0
                terminal["compound_effect_missed"] = 0
            safe = bool(
                terminal["nrmse"] <= config.target_nrmse
                and terminal["effect_recall"] >= config.target_effect_recall
            )
            decisions.append(
                {
                    "run_id": run_id,
                    "compound": terminal["compound"],
                    "compound_group": terminal["compound_group"],
                    "policy": policy,
                    "decision": decision,
                    "decision_round": int(terminal["round"]),
                    "decision_div": int(terminal["div"]),
                    "selected_cells_used": int(terminal["selected_cells"]),
                    "physical_wells_allocated": 24,
                    "instrument_well_recordings": 24 * int(terminal["round"]),
                    "recording_sessions": int(terminal["round"]),
                    "terminal_nrmse": float(terminal["nrmse"]),
                    "effect_recall": float(terminal["effect_recall"]),
                    "missed_effect_cells": int(terminal["missed_effect_cells"]),
                    "compound_effect_missed": int(terminal["compound_effect_missed"]),
                    "stop_safe": safe,
                    **rule,
                }
            )
    decisions_frame = pd.DataFrame(decisions)
    comparisons = _cluster_comparisons(traces, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces.to_csv(output_dir / "fresh_acquisition_traces.csv", index=False)
    decisions_frame.to_csv(output_dir / "fresh_decisions.csv", index=False)
    comparisons.to_csv(output_dir / "fresh_paired_comparisons.csv", index=False)
    (output_dir / "phase1_data_audit.json").write_text(
        json.dumps(phase1.audit, indent=2, sort_keys=True) + "\n"
    )
    rs = decisions_frame.loc[decisions_frame["policy"].eq("risk_sensitive")]
    stops = rs.loc[rs["decision"].eq("STOP")]
    summary = {
        "schema": "gates-act-v3-results-v1",
        "dataset": "NTP/EPA Phase-1 DNT MEA Network Formation curated release",
        "source_doi": "10.22427/NTP-DATA-500-018-001-000-5",
        "fresh_test_groups": int(rs["compound_group"].nunique()),
        "fresh_test_runs": int(rs["run_id"].nunique()),
        "selected_config": asdict(config),
        "calibration_rules": rules,
        "risk_sensitive_decisions": rs["decision"].value_counts().to_dict(),
        "risk_sensitive_stop_rounds": stops["decision_round"].value_counts().to_dict(),
        "risk_sensitive_unsafe_stops": int((~stops["stop_safe"]).sum()),
        "risk_sensitive_stop_effect_recall": float(stops["effect_recall"].mean())
        if len(stops)
        else None,
        "risk_sensitive_missed_effect_cells_at_stop": int(stops["missed_effect_cells"].sum()),
        "mean_recording_sessions": float(rs["recording_sessions"].mean()),
        "recording_session_reduction": float(1 - rs["recording_sessions"].mean() / 4),
        "physical_wells_per_run": 24,
        "measurement_cost_contract": (
            "24 wells are physically allocated; only avoided future whole-plate DIV recording "
            "sessions count as laboratory savings"
        ),
        "legacy_v2_evidence_status": "preserved and excluded from v3 fitting/selection",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
