from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.model import AdaptiveResponseSurface
from gates.act.neural_mea import DIVS, load_neural_mea, neural_coordinates, surface_arrays

POLICIES = ("act", "max_variance", "maximin", "random", "fixed_sentinel")


def _seed_for(text: str, seed: int) -> int:
    return seed + int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 1_000_000


def _select_batch(
    surface: AdaptiveResponseSurface,
    prediction,
    observed: list[int],
    time_rank: int,
    policy: str,
    rng: np.random.Generator,
    size: int = 4,
) -> list[int]:
    candidates = np.array(
        [index for index in range(8 * time_rank, 8 * (time_rank + 1)) if index not in observed]
    )
    if policy == "fixed_sentinel":
        return [8 * time_rank + rank for rank in (0, 2, 5, 7)][:size]
    # Every laboratory batch includes its matched vehicle control. Remaining ACT
    # slots minimize integrated posterior variance under the chronological constraint.
    selected: list[int] = [8 * time_rank]
    covariance = prediction.covariance.copy()
    for chosen in selected:
        denom = max(float(covariance[chosen, chosen]), 1e-12)
        covariance -= np.outer(covariance[:, chosen], covariance[chosen, :]) / denom
    for _ in range(min(size, len(candidates)) - len(selected)):
        available = np.array([index for index in candidates if index not in selected])
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
        elif policy == "act":
            unobserved = np.array(
                [
                    index
                    for index in range(len(surface.coordinates))
                    if index not in observed + selected
                ]
            )
            scores = []
            for candidate in available:
                denom = max(float(covariance[candidate, candidate]), 1e-12)
                scores.append(float(np.sum(covariance[unobserved, candidate] ** 2) / denom))
            chosen = int(available[int(np.argmax(scores))])
        else:
            raise ValueError(f"unknown policy {policy}")
        selected.append(chosen)
        denom = max(float(covariance[chosen, chosen]), 1e-12)
        covariance -= np.outer(covariance[:, chosen], covariance[chosen, :]) / denom
    return selected


def _metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    observed: list[int],
    scale: float,
    effect_threshold: float,
) -> dict[str, float | int]:
    hidden = np.array([index for index in range(len(truth)) if index not in observed])
    if not len(hidden):
        observed_effect_mask = np.abs(truth[np.asarray(observed, dtype=int)]) >= effect_threshold
        return {
            "nrmse": 0.0,
            "mae": 0.0,
            "effect_recall": 1.0,
            "missed_effect_cells": 0,
            "true_effect_cells": 0,
            "compound_effect_missed": 0,
            "observed_effect_detected": int(np.any(observed_effect_mask)),
            "observed_effect_cells": int(observed_effect_mask.sum()),
            "observed_effect_divs": int(
                len(
                    {
                        index // 8
                        for index, hit in zip(observed, observed_effect_mask, strict=True)
                        if hit
                    }
                )
            ),
        }
    error = prediction[hidden] - truth[hidden]
    true_effect = np.abs(truth[hidden]) >= effect_threshold
    predicted_effect = np.abs(prediction[hidden]) >= effect_threshold
    observed_effect = np.any(np.abs(truth[np.asarray(observed, dtype=int)]) >= effect_threshold)
    observed_effect_mask = np.abs(truth[np.asarray(observed, dtype=int)]) >= effect_threshold
    true_count = int(true_effect.sum())
    missed = int((true_effect & ~predicted_effect).sum())
    return {
        "nrmse": float(np.sqrt(np.mean(error**2)) / scale),
        "mae": float(np.mean(np.abs(error))),
        "effect_recall": float(1 - missed / true_count) if true_count else 1.0,
        "missed_effect_cells": missed,
        "true_effect_cells": true_count,
        "compound_effect_missed": int(
            np.any(np.abs(truth) >= effect_threshold)
            and not observed_effect
            and not np.any(predicted_effect)
        ),
        "observed_effect_detected": int(observed_effect),
        "observed_effect_cells": int(observed_effect_mask.sum()),
        "observed_effect_divs": int(
            len(
                {
                    index // 8
                    for index, hit in zip(observed, observed_effect_mask, strict=True)
                    if hit
                }
            )
        ),
    }


def _trace(
    surface: AdaptiveResponseSurface,
    truth: np.ndarray,
    policy: str,
    run_id: str,
    seed: int,
    effect_threshold: float,
) -> list[dict[str, object]]:
    observed = list(range(8))  # complete DIV5 batch; no non-chronological look-ahead
    rng = np.random.default_rng(_seed_for(f"{run_id}/{policy}", seed))
    rows = []
    for round_index, div in enumerate(DIVS, start=1):
        if round_index > 1:
            before = surface.fit_predict(observed, truth[observed])
            observed.extend(
                _select_batch(surface, before, observed, round_index - 1, policy, rng, size=4)
            )
        prediction = surface.fit_predict(observed, truth[observed])
        metrics = _metrics(
            prediction.mean, truth, observed, surface.response_scale, effect_threshold
        )
        rows.append(
            {
                "round": round_index,
                "div": div,
                "selected_cells": len(observed),
                "selected_indices": ";".join(map(str, observed)),
                "uncertainty": prediction.uncertainty_score,
                "misfit": prediction.misfit_score,
                "risk_score": prediction.uncertainty_score + 0.025 * prediction.misfit_score,
                **metrics,
            }
        )
    return rows


def _calibrate_rule(
    calibration: pd.DataFrame,
    target_nrmse: float,
    min_effect_divs: int,
    calibration_violation_limit: float,
    ood_quantile: float,
) -> dict[str, float]:
    candidates = (
        calibration.loc[
            calibration["round"].isin([2, 3])
            & calibration["observed_effect_divs"].ge(min_effect_divs)
        ]
        .drop_duplicates(["compound_group", "round"])
        .copy()
    )
    # A stop is calibration-safe only if reconstruction meets tolerance and no strong
    # hidden effect is completely missed. Threshold choice never sees test compounds.
    candidates["safe"] = (candidates["nrmse"] <= target_nrmse) & (
        candidates["compound_effect_missed"] == 0
    )
    best_threshold = -1.0
    best_coverage = -1.0
    best_violation = 1.0
    for threshold in sorted(candidates["risk_score"].unique()):
        eligible = candidates.loc[candidates["risk_score"] <= threshold]
        if len(eligible) < 5:
            continue
        violation = float((~eligible["safe"]).mean())
        coverage = float(len(eligible) / len(candidates))
        if violation <= calibration_violation_limit and coverage > best_coverage:
            best_threshold = float(threshold)
            best_coverage = coverage
            best_violation = violation
    return {
        "risk_threshold": best_threshold,
        "calibration_eligible_fraction": max(best_coverage, 0.0),
        "calibration_violation_rate": best_violation if best_threshold >= 0 else 0.0,
        "ood_threshold": float(
            calibration.loc[calibration["round"] == 2, "misfit"].quantile(ood_quantile)
        ),
    }


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


def _comparison_table(traces: pd.DataFrame, seed: int) -> pd.DataFrame:
    test = traces.loc[traces["split"] == "test"].copy()
    auc = test.groupby(["compound_group", "policy"], as_index=False)["nrmse"].mean()
    act = auc.loc[auc["policy"] == "act"].set_index("compound_group")["nrmse"]
    rows = []
    rng = np.random.default_rng(seed)
    for baseline in [policy for policy in POLICIES if policy != "act"]:
        other = auc.loc[auc["policy"] == baseline].set_index("compound_group")["nrmse"]
        differences = (other - act).dropna().to_numpy()
        bootstrap = np.array(
            [rng.choice(differences, len(differences), replace=True).mean() for _ in range(20_000)]
        )
        rows.append(
            {
                "baseline": baseline,
                "act_nrmse_advantage": float(differences.mean()),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "paired_sign_flip_p": _paired_sign_flip(differences, seed),
                "test_runs": len(differences),
            }
        )
    return pd.DataFrame(rows)


def run_neural_mea_benchmark(
    workbook: Path,
    output_dir: Path,
    target_nrmse: float = 0.20,
    effect_threshold: float = 0.40,
    min_effect_divs: int = 2,
    calibration_violation_limit: float = 0.0,
    ood_quantile: float = 0.90,
    seed: int = 20261023,
) -> dict[str, object]:
    dataset = load_neural_mea(workbook)
    meta, responses = surface_arrays(dataset)
    coordinates = neural_coordinates()
    train_ids = meta.loc[meta["split"] == "train", "run_id"].tolist()
    training_matrix = np.vstack([responses[run_id] for run_id in train_ids])
    trace_rows = []
    for record in meta.itertuples(index=False):
        surface = AdaptiveResponseSurface(
            coordinates, training_matrix, seed=_seed_for(record.run_id, seed)
        )
        for policy in POLICIES:
            for row in _trace(
                surface,
                responses[record.run_id],
                policy,
                record.run_id,
                seed,
                effect_threshold,
            ):
                trace_rows.append(
                    {
                        "run_id": record.run_id,
                        "compound": record.compound,
                        "compound_group": record.compound_group,
                        "split": record.split,
                        "policy": policy,
                        **row,
                    }
                )
    traces = pd.DataFrame(trace_rows)
    decisions = []
    calibration_rules = {}
    for policy in POLICIES:
        rule = _calibrate_rule(
            traces.loc[(traces["split"] == "calibration") & (traces["policy"] == policy)],
            target_nrmse,
            min_effect_divs,
            calibration_violation_limit,
            ood_quantile,
        )
        calibration_rules[policy] = rule
        for run_id, path in traces.loc[
            (traces["split"] == "test") & (traces["policy"] == policy)
        ].groupby("run_id"):
            path = path.sort_values("round")
            terminal = path.iloc[-1].copy()
            decision = "COMPLETE"
            for _, row in path.loc[path["round"].isin([2, 3])].iterrows():
                if row["round"] == 2 and row["misfit"] > rule["ood_threshold"]:
                    decision = "ABSTAIN"
                    terminal = path.iloc[-1]
                    break
                if (
                    rule["risk_threshold"] >= 0
                    and row["observed_effect_divs"] >= min_effect_divs
                    and row["risk_score"] <= rule["risk_threshold"]
                ):
                    decision = "STOP"
                    terminal = row
                    break
            if decision in {"COMPLETE", "ABSTAIN"}:
                # A completed multiwell MEA run has recorded every dose at every DIV.
                terminal = terminal.copy()
                terminal["round"] = 4
                terminal["div"] = 12
                terminal["selected_cells"] = 32
                terminal["nrmse"] = 0.0
                terminal["effect_recall"] = 1.0
                terminal["missed_effect_cells"] = 0
                terminal["compound_effect_missed"] = 0
            sessions = int(terminal["round"])
            decision_safe = bool(
                terminal["nrmse"] <= target_nrmse and int(terminal["compound_effect_missed"]) == 0
            )
            decisions.append(
                {
                    "run_id": run_id,
                    "compound": terminal["compound"],
                    "compound_group": terminal["compound_group"],
                    "policy": policy,
                    "decision": decision,
                    "decision_round": sessions,
                    "decision_div": int(terminal["div"]),
                    "selected_cells_used": int(terminal["selected_cells"]),
                    "physical_wells_allocated": 24,
                    "instrument_well_recordings": 24 * sessions,
                    "recording_sessions": sessions,
                    "culture_days": int(terminal["div"]),
                    "terminal_nrmse": float(terminal["nrmse"]),
                    "effect_recall": float(terminal["effect_recall"]),
                    "missed_effect_cells": int(terminal["missed_effect_cells"]),
                    "compound_effect_missed": int(terminal["compound_effect_missed"]),
                    "stop_safe": decision_safe,
                    **rule,
                }
            )
    decisions_frame = pd.DataFrame(decisions)
    comparisons = _comparison_table(traces, seed)
    fixed_run_rows = []
    for record in meta.loc[meta["split"] == "test"].itertuples(index=False):
        surface = AdaptiveResponseSurface(
            coordinates, training_matrix, seed=_seed_for(record.run_id, seed)
        )
        truth = responses[record.run_id]
        for round_index, div in enumerate(DIVS, start=1):
            observed = list(range(8 * round_index))
            prediction = surface.fit_predict(observed, truth[observed])
            fixed_run_rows.append(
                {
                    "run_id": record.run_id,
                    "compound_group": record.compound_group,
                    "round": round_index,
                    "div": div,
                    "selected_cells": len(observed),
                    "instrument_well_recordings": 24 * round_index,
                    **_metrics(
                        prediction.mean,
                        truth,
                        observed,
                        surface.response_scale,
                        effect_threshold,
                    ),
                }
            )
    fixed_by_run = pd.DataFrame(fixed_run_rows)
    fixed = (
        fixed_by_run.groupby(["round", "div", "selected_cells", "instrument_well_recordings"])
        .agg(
            mean_nrmse=("nrmse", "mean"),
            mean_effect_recall=("effect_recall", "mean"),
            compound_effect_miss_rate=("compound_effect_missed", "mean"),
        )
        .reset_index()
    )
    fixed.insert(0, "policy", "fixed_full_panel")
    output_dir.mkdir(parents=True, exist_ok=True)
    traces.to_csv(output_dir / "neural_acquisition_traces.csv", index=False)
    decisions_frame.to_csv(output_dir / "neural_decisions.csv", index=False)
    comparisons.to_csv(output_dir / "neural_paired_comparisons.csv", index=False)
    fixed.to_csv(output_dir / "neural_fixed_budget.csv", index=False)
    fixed_by_run.to_csv(output_dir / "neural_fixed_budget_by_run.csv", index=False)
    (output_dir / "neural_data_audit.json").write_text(
        json.dumps(dataset.audit, indent=2, sort_keys=True) + "\n"
    )
    act_decisions = decisions_frame.loc[decisions_frame["policy"] == "act"]
    act_stops = act_decisions.loc[act_decisions["decision"] == "STOP"]
    summary = {
        "dataset": "EPA cortical-neuron microelectrode-array network formation assay",
        "source_doi": "10.23719/1503191",
        "publication_doi": "10.1093/toxsci/kfz052",
        "endpoint": "mean firing rate",
        "response": dataset.audit["normalization"],
        "validation": (
            "SHA-256 stable compound-group split; training prior, calibration rules, "
            "test reporting"
        ),
        "train_calibration_test_groups": dataset.audit["split_groups"],
        "eligible_complete_runs": dataset.audit["eligible_complete_runs"],
        "target_nrmse": target_nrmse,
        "effect_threshold_log1p_contrast": effect_threshold,
        "minimum_stop_round": 2,
        "minimum_effect_divs_for_stop": min_effect_divs,
        "calibration_violation_limit": calibration_violation_limit,
        "ood_quantile": ood_quantile,
        "act_test_decisions": act_decisions["decision"].value_counts().to_dict(),
        "act_stop_rounds": act_stops["decision_round"].value_counts().sort_index().to_dict(),
        "act_unsafe_stops": int((~act_stops["stop_safe"]).sum()),
        "act_compound_effect_misses": int(act_stops["compound_effect_missed"].sum()),
        "act_mean_instrument_well_recordings": float(
            act_decisions["instrument_well_recordings"].mean()
        ),
        "full_instrument_well_recordings": 96,
        "act_mean_recording_sessions": float(act_decisions["recording_sessions"].mean()),
        "full_recording_sessions": 4,
        "physical_wells_allocated_all_policies": 24,
        "cost_interpretation": (
            "dose subsampling saves analysis inputs, not plate allocation; only avoided future "
            "DIV sessions are counted as laboratory savings"
        ),
        "calibration_rules": calibration_rules,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
