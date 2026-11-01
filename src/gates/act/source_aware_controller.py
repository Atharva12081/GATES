from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.agreement_diagnostics import load_selected

POLICIES = (
    "gates_global",
    "agreement_global",
    "agreement_source_calibrated",
    "source_blind_controller",
    "source_aware_controller",
)


def _calls(frame: pd.DataFrame, mode: str) -> np.ndarray:
    gates = frame["gates"].to_numpy()
    residual = frame["selected_residual"].to_numpy()
    if mode == "gates":
        return np.abs(gates) >= 0.40
    if mode == "agreement":
        return (
            (np.abs(gates) >= 0.40)
            & (np.abs(residual) >= 0.40)
            & (np.sign(gates) == np.sign(residual))
        )
    raise ValueError(mode)


def _score(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "gates":
        return np.abs(frame["gates"].to_numpy())
    if mode == "agreement":
        return np.maximum(
            np.abs(frame["gates"].to_numpy()),
            np.abs(frame["selected_residual"].to_numpy()),
        )
    raise ValueError(mode)


def _activation(calibration: pd.DataFrame) -> tuple[str, dict[str, int]]:
    truth = calibration["truth"].abs().to_numpy() >= 0.40
    gates = _calls(calibration, "gates")
    agreement = _calls(calibration, "agreement")
    removed = gates & ~agreement
    true_deferred = int((removed & truth).sum())
    false_removed = int((removed & ~truth).sum())
    mode = "agreement" if int(removed.sum()) < 5 or false_removed > true_deferred else "gates"
    return mode, {
        "gates_only_calls": int(removed.sum()),
        "false_discoveries_removed": false_removed,
        "true_effect_calls_deferred": true_deferred,
    }


def _group_false_negative_fraction(frame: pd.DataFrame, score: np.ndarray, margin: float) -> float:
    truth = frame["truth"].abs().to_numpy() >= 0.40
    work = frame[["compound_group"]].copy()
    work["effect"] = truth
    work["miss"] = truth & (score < margin)
    grouped = work.groupby("compound_group")[["effect", "miss"]].sum()
    grouped = grouped.loc[grouped["effect"] > 0]
    if grouped.empty:
        return 0.0
    return float((grouped["miss"] / grouped["effect"]).mean())


def _recall_margin(frame: pd.DataFrame, mode: str, target: float = 0.95) -> float:
    score = _score(frame, mode)
    allowed = [
        float(value)
        for value in np.linspace(0, 0.40, 401)
        if _group_false_negative_fraction(frame, score, float(value)) <= 1 - target
    ]
    return max(allowed, default=0.0)


def _coverage_margin(frame: pd.DataFrame, mode: str, target: float = 0.30) -> float:
    effect = _calls(frame, mode)
    score = _score(frame, mode)
    candidates = np.linspace(0, 0.40, 401)
    coverage = np.asarray([np.mean(effect | (score < value)) for value in candidates])
    return float(candidates[np.argmin(np.abs(coverage - target))])


def _calibration_subset(frame: pd.DataFrame, fold: int, source: str | None) -> pd.DataFrame:
    calibration = frame.loc[~frame["fold"].eq(fold)]
    if source is not None:
        calibration = calibration.loc[calibration["source"].eq(source)]
    return calibration


def _policy_decisions(
    frame: pd.DataFrame, policy: str, calibration_mode: str
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pieces = []
    calibration_receipt = []
    for fold in sorted(frame["fold"].unique()):
        held_fold = frame.loc[frame["fold"].eq(fold)]
        sources = sorted(held_fold["source"].unique())
        global_calibration = _calibration_subset(frame, int(fold), None)
        global_controller_mode, global_counts = _activation(global_calibration)
        for source in sources:
            held = held_fold.loc[held_fold["source"].eq(source)].copy()
            source_calibration = _calibration_subset(frame, int(fold), str(source))
            if policy == "gates_global":
                mode, calibration = "gates", global_calibration
            elif policy == "agreement_global":
                mode, calibration = "agreement", global_calibration
            elif policy == "agreement_source_calibrated":
                mode, calibration = "agreement", source_calibration
            elif policy == "source_blind_controller":
                mode, calibration = global_controller_mode, global_calibration
            elif policy == "source_aware_controller":
                mode, counts = _activation(source_calibration)
                calibration = source_calibration
            else:
                raise ValueError(policy)
            if policy != "source_aware_controller":
                counts = global_counts if "controller" in policy else {}
            margin = (
                _recall_margin(calibration, mode)
                if calibration_mode == "recall"
                else _coverage_margin(calibration, mode)
            )
            effect = _calls(held, mode)
            no_effect = _score(held, mode) < margin
            held["policy"] = policy
            held["calibration_mode"] = calibration_mode
            held["controller_mode"] = mode
            held["margin"] = margin
            held["effect_call"] = effect
            held["no_effect_call"] = no_effect
            pieces.append(held)
            calibration_receipt.append(
                {
                    "policy": policy,
                    "calibration_mode": calibration_mode,
                    "fold": int(fold),
                    "source": str(source),
                    "calibration_groups": int(calibration["compound_group"].nunique()),
                    "controller_mode": mode,
                    "margin": margin,
                    **counts,
                }
            )
    return pd.concat(pieces, ignore_index=True), calibration_receipt


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["truth"].abs().to_numpy() >= 0.40
    effect = frame["effect_call"].to_numpy(dtype=bool)
    no_effect = frame["no_effect_call"].to_numpy(dtype=bool)
    decisive = effect | no_effect
    correct = (effect & truth) | (no_effect & ~truth)
    tp = int((effect & truth).sum())
    fp = int((effect & ~truth).sum())
    fn = int((no_effect & truth).sum())
    tn = int((no_effect & ~truth).sum())
    wrong = int((decisive & ~correct).sum())
    return {
        "rows": len(frame),
        "effect_precision": tp / max(tp + fp, 1),
        "effect_call_recall": tp / max(int(truth.sum()), 1),
        "protected_recall": 1 - fn / max(int(truth.sum()), 1),
        "decisive_specificity": tn / max(tn + fp, 1),
        "decisive_coverage": float(decisive.mean()),
        "selective_risk": wrong / max(int(decisive.sum()), 1),
        "confirmed_effects": int(effect.sum()),
        "no_effect_decisions": int(no_effect.sum()),
        "wrong_decisions": wrong,
        "missed_effects": fn,
        "unresolved": int((~decisive).sum()),
    }


def _summarize(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = []
    source = []
    for (mode, policy), part in decisions.groupby(["calibration_mode", "policy"]):
        aggregate.append({"calibration_mode": mode, "policy": policy, **_metrics(part)})
        for source_name, source_part in part.groupby("source"):
            source.append(
                {
                    "calibration_mode": mode,
                    "policy": policy,
                    "source": source_name,
                    "groups": int(source_part["compound_group"].nunique()),
                    **_metrics(source_part),
                }
            )
    return pd.DataFrame(aggregate), pd.DataFrame(source)


def _bootstrap(
    decisions: pd.DataFrame, candidate: str, baseline: str, seed: int = 20261029
) -> dict[str, object]:
    recall = decisions.loc[decisions["calibration_mode"].eq("recall")]
    grouped = {}
    for policy in (candidate, baseline):
        part = recall.loc[recall["policy"].eq(policy)].copy()
        truth = part["truth"].abs().to_numpy() >= 0.40
        effect = part["effect_call"].to_numpy(dtype=bool)
        no_effect = part["no_effect_call"].to_numpy(dtype=bool)
        decisive = effect | no_effect
        correct = (effect & truth) | (no_effect & ~truth)
        part["tp"] = effect & truth
        part["fp"] = effect & ~truth
        part["wrong"] = decisive & ~correct
        part["decisive"] = decisive
        grouped[policy] = part.groupby("compound_group")[["tp", "fp", "wrong", "decisive"]].sum()
    groups = sorted(set(grouped[candidate].index) & set(grouped[baseline].index))
    rng = np.random.default_rng(seed)
    precision_difference = []
    risk_difference = []
    wrong_difference = []
    for _ in range(10000):
        sample = rng.integers(0, len(groups), len(groups))
        values = {}
        for policy in (candidate, baseline):
            total = grouped[policy].reindex(groups).iloc[sample].sum()
            values[policy] = (
                float(total.tp / max(total.tp + total.fp, 1)),
                float(total.wrong / max(total.decisive, 1)),
                int(total.wrong),
            )
        precision_difference.append(values[candidate][0] - values[baseline][0])
        risk_difference.append(values[candidate][1] - values[baseline][1])
        wrong_difference.append(values[candidate][2] - values[baseline][2])

    def interval(values: list[float | int]) -> dict[str, object]:
        return {
            "difference": float(np.mean(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }

    return {
        "groups": len(groups),
        "replicates": 10000,
        "seed": seed,
        "precision": interval(precision_difference),
        "selective_risk": interval(risk_difference),
        "wrong_decisions": interval(wrong_difference),
    }


def run_source_aware_controller(predictions: Path, triage: Path, output: Path) -> dict[str, object]:
    frame = load_selected(predictions, triage)
    decision_frames = []
    receipts = []
    for calibration_mode in ("recall", "coverage"):
        for policy in POLICIES:
            decisions, receipt = _policy_decisions(frame, policy, calibration_mode)
            decision_frames.append(decisions)
            receipts.extend(receipt)
    all_decisions = pd.concat(decision_frames, ignore_index=True)
    aggregate, source = _summarize(all_decisions)
    recall_aggregate = aggregate.loc[aggregate["calibration_mode"].eq("recall")].set_index("policy")
    recall_source = source.loc[source["calibration_mode"].eq("recall")]
    baseline = recall_aggregate.loc["gates_global"]
    candidate = recall_aggregate.loc["source_aware_controller"]
    baseline_sources = recall_source.loc[recall_source["policy"].eq("gates_global")]
    candidate_sources = recall_source.loc[recall_source["policy"].eq("source_aware_controller")]
    gate = bool(
        candidate["effect_precision"] > baseline["effect_precision"]
        and candidate["wrong_decisions"] < baseline["wrong_decisions"]
        and candidate["decisive_coverage"] >= 0.25
        and candidate_sources["protected_recall"].min()
        >= baseline_sources["protected_recall"].min()
        and candidate_sources["selective_risk"].max() <= baseline_sources["selective_risk"].max()
    )
    comparisons = {
        "source_aware_controller_minus_gates_global": _bootstrap(
            all_decisions, "source_aware_controller", "gates_global"
        ),
        "agreement_global_minus_gates_global": _bootstrap(
            all_decisions, "agreement_global", "gates_global"
        ),
        "agreement_source_calibrated_minus_gates_global": _bootstrap(
            all_decisions, "agreement_source_calibrated", "gates_global"
        ),
        "agreement_source_calibrated_minus_agreement_global": _bootstrap(
            all_decisions, "agreement_source_calibrated", "agreement_global"
        ),
    }
    strongest_policy = "agreement_source_calibrated"
    strongest = recall_aggregate.loc[strongest_policy]
    output.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output / "source_aware_controller_aggregate.csv", index=False)
    source.to_csv(output / "source_aware_controller_by_source.csv", index=False)
    pd.DataFrame(receipts).to_csv(output / "source_aware_controller_calibration.csv", index=False)
    all_decisions.to_parquet(output / "source_aware_controller_decisions.parquet", index=False)
    result = {
        "schema": "gates-act-source-aware-controller-v1",
        "evidence_class": "nested development; not independent confirmation",
        "groups": int(frame["compound_group"].nunique()),
        "rows_per_policy": len(frame),
        "budget": "16 of 32 response cells",
        "primary_gate": "GO" if gate else "NO-GO",
        "candidate": candidate.to_dict(),
        "baseline": baseline.to_dict(),
        "candidate_worst_source": {
            "protected_recall": float(candidate_sources["protected_recall"].min()),
            "selective_risk": float(candidate_sources["selective_risk"].max()),
        },
        "baseline_worst_source": {
            "protected_recall": float(baseline_sources["protected_recall"].min()),
            "selective_risk": float(baseline_sources["selective_risk"].max()),
        },
        "bootstrap_comparisons": comparisons,
        "strongest_observed_policy": strongest_policy,
        "strongest_observed_metrics": strongest.to_dict(),
        "incremental_controller_judgment": (
            "NO-GO: source-aware activation is inferior to always-on agreement on precision, "
            "selective risk, and wrong decisions in the locked comparison"
        ),
        "canonical_integration": "not authorized without independent confirmation",
        "early_stopping": "disabled",
    }
    (output / "source_aware_controller_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
