from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/act/research/competitor_transfer"
PREDICTIONS = ARTIFACT / "h1_oof_predictions_with_selected.parquet"
TRIAGE = ARTIFACT / "h6_agreement_triage.json"
SUMMARY = ARTIFACT / "agreement_mechanism_summary.json"
OUTPUT = ARTIFACT / "agreement_mechanism_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_predictions() -> pd.DataFrame:
    raw = pd.read_parquet(PREDICTIONS)
    key = ["fold", "run_id", "compound_group", "context", "cell", "truth"]
    wide = (
        raw.loc[raw["method"].isin(["gates", "selected_residual", "interpolation"])]
        .pivot(index=key, columns="method", values="prediction")
        .reset_index()
    )
    contexts = list(itertools.combinations(range(1, 8), 3))
    saved = json.loads(TRIAGE.read_text())["conditional_max_variance_policy"]["contexts_by_fold"]
    pieces = []
    for fold, context in saved.items():
        context_index = contexts.index(tuple(context))
        pieces.append(wide.loc[wide["fold"].eq(int(fold)) & wide["context"].eq(context_index)])
    return pd.concat(pieces, ignore_index=True)


def group_miss_fraction(frame: pd.DataFrame, score: np.ndarray, margin: float) -> float:
    truth = frame["truth"].abs().to_numpy() >= 0.40
    work = frame[["compound_group"]].copy()
    work["truth"] = truth
    work["miss"] = truth & (score < margin)
    group = work.groupby("compound_group")[["truth", "miss"]].sum()
    group = group.loc[group["truth"] > 0]
    return float((group["miss"] / group["truth"]).mean())


def margin(frame: pd.DataFrame, method: str) -> float:
    if method == "agreement":
        score = np.maximum(frame["gates"].abs(), frame["selected_residual"].abs()).to_numpy()
    else:
        score = frame[method].abs().to_numpy()
    allowed = [
        candidate
        for candidate in np.linspace(0, 0.40, 401)
        if group_miss_fraction(frame, score, float(candidate)) <= 0.05
    ]
    return float(max(allowed, default=0.0))


def metrics(frame: pd.DataFrame, method: str) -> dict[str, float | int]:
    fold_margin = {
        int(fold): margin(frame.loc[~frame["fold"].eq(fold)], method)
        for fold in sorted(frame["fold"].unique())
    }
    margins = frame["fold"].map(fold_margin).to_numpy()
    truth = frame["truth"].abs().to_numpy() >= 0.40
    if method == "agreement":
        gates = frame["gates"].to_numpy()
        residual = frame["selected_residual"].to_numpy()
        effect = (
            (np.abs(gates) >= 0.40)
            & (np.abs(residual) >= 0.40)
            & (np.sign(gates) == np.sign(residual))
        )
        score = np.maximum(np.abs(gates), np.abs(residual))
    else:
        score = frame[method].abs().to_numpy()
        effect = score >= 0.40
    no_effect = score < margins
    decisive = effect | no_effect
    correct = (effect & truth) | (no_effect & ~truth)
    tp = int((effect & truth).sum())
    fp = int((effect & ~truth).sum())
    missed = int((no_effect & truth).sum())
    wrong = int((decisive & ~correct).sum())
    return {
        "effect_precision": tp / (tp + fp),
        "protected_recall": 1 - missed / int(truth.sum()),
        "decisive_coverage": float(decisive.mean()),
        "selective_risk": wrong / int(decisive.sum()),
        "wrong_decisions": wrong,
        "missed_effects": missed,
        "unresolved": int((~decisive).sum()),
    }


def main() -> None:
    frame = selected_predictions()
    reported = json.loads(SUMMARY.read_text())
    recomputed = {method: metrics(frame, method) for method in ("agreement", "gates")}
    tolerances = {}
    for method in recomputed:
        for key, value in recomputed[method].items():
            reference = reported[method][key]
            tolerances[f"{method}.{key}"] = bool(
                value == reference
                if isinstance(value, int)
                else np.isclose(value, reference, atol=1e-12)
            )

    gates = frame["gates"].to_numpy()
    residual = frame["selected_residual"].to_numpy()
    truth = frame["truth"].abs().to_numpy() >= 0.40
    gates_call = np.abs(gates) >= 0.40
    agreement_call = gates_call & (np.abs(residual) >= 0.40) & (np.sign(gates) == np.sign(residual))
    removed = gates_call & ~agreement_call
    mechanism = {
        "gates_calls_removed": int(removed.sum()),
        "false_discoveries_removed": int((removed & ~truth).sum()),
        "true_effect_calls_deferred": int((removed & truth).sum()),
        "opposite_sign_joint_calls": int(
            (gates_call & (np.abs(residual) >= 0.40) & (np.sign(gates) != np.sign(residual))).sum()
        ),
    }
    checks = {
        "99_unique_groups": int(frame["compound_group"].nunique()) == 99,
        "1696_hidden_responses": len(frame) == 1696,
        "all_reported_primary_metrics_match": all(tolerances.values()),
        "ten_false_discoveries_removed": mechanism["false_discoveries_removed"] == 10,
        "seven_true_calls_deferred": mechanism["true_effect_calls_deferred"] == 7,
        "wrong_decisions_reduced_by_ten": (
            recomputed["gates"]["wrong_decisions"] - recomputed["agreement"]["wrong_decisions"]
            == 10
        ),
    }
    payload = {
        "schema": "gates-act-agreement-mechanism-audit-v1",
        "evidence_class": "independent code-path audit of nested development evidence",
        "inputs": {
            str(PREDICTIONS.relative_to(ROOT)): sha256(PREDICTIONS),
            str(TRIAGE.relative_to(ROOT)): sha256(TRIAGE),
            str(SUMMARY.relative_to(ROOT)): sha256(SUMMARY),
        },
        "recomputed": recomputed,
        "mechanism": mechanism,
        "metric_checks": tolerances,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
