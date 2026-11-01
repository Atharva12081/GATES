from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/act/research/competitor_transfer"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    source = ARTIFACT / "h1_oof_predictions_with_selected.parquet"
    frame = pd.read_parquet(source)
    assert frame[["truth", "prediction"]].notna().all().all()
    frame["absolute_error"] = np.abs(frame["truth"] - frame["prediction"])
    frame["true_effect"] = np.abs(frame["truth"]) >= 0.40
    frame["predicted_effect"] = np.abs(frame["prediction"]) >= 0.40
    frame["tp"] = frame["true_effect"] & frame["predicted_effect"]
    frame["fp"] = ~frame["true_effect"] & frame["predicted_effect"]
    group = frame.groupby(["method", "compound_group"], as_index=False).agg(
        mae=("absolute_error", "mean"),
        tp=("tp", "sum"),
        fp=("fp", "sum"),
        effects=("true_effect", "sum"),
    )
    summary = group.groupby("method", as_index=False).agg(
        mae=("mae", "mean"), tp=("tp", "sum"), fp=("fp", "sum"), effects=("effects", "sum")
    )
    summary["effect_recall"] = summary["tp"] / summary["effects"]
    summary["effect_precision"] = summary["tp"] / (summary["tp"] + summary["fp"])
    table = summary.set_index("method")
    strongest = table.loc[["gates", "interpolation"], "mae"].idxmin()
    pivot = group.pivot(index="compound_group", columns="method", values="mae").dropna()
    difference = (pivot["selected_residual"] - pivot[strongest]).to_numpy()
    rng = np.random.default_rng(20261026)
    draws = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(axis=1)
    fold = frame.groupby(["fold", "method"], as_index=False).agg(mae=("absolute_error", "mean"))
    wins = sum(
        int(values["selected_residual"] < values[strongest])
        for _, values in fold.pivot(index="fold", columns="method", values="mae").iterrows()
    )
    relative = 1 - table.loc["selected_residual", "mae"] / table.loc[strongest, "mae"]
    recall_delta = (
        table.loc["selected_residual", "effect_recall"] - table.loc[strongest, "effect_recall"]
    )
    passed = bool(
        relative >= 0.05 and np.quantile(draws, 0.975) < 0 and wins >= 4 and recall_delta >= -0.02
    )
    stored = json.loads((ARTIFACT / "h1_summary.json").read_text())
    np.testing.assert_allclose(relative, stored["relative_mae_reduction"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(recall_delta, stored["effect_recall_delta"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        [difference.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)],
        [
            stored["paired_bootstrap"]["difference"],
            stored["paired_bootstrap"]["ci_low"],
            stored["paired_bootstrap"]["ci_high"],
        ],
        rtol=0,
        atol=1e-12,
    )
    assert wins == stored["fold_wins"]
    assert ("GO" if passed else "NO-GO") == stored["status"]
    receipt = {
        "status": "passed",
        "source_sha256": digest(source),
        "rows": len(frame),
        "groups": len(pivot),
        "strongest_baseline": strongest,
        "relative_mae_reduction": float(relative),
        "effect_recall_delta": float(recall_delta),
        "fold_wins": int(wins),
        "difference_ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "gate": "GO" if passed else "NO-GO",
    }
    (ARTIFACT / "h1_audit.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))
