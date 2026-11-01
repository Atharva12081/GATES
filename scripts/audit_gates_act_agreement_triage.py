from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/act/research/competitor_transfer"


if __name__ == "__main__":
    source = ARTIFACT / "h1_oof_predictions_with_selected.parquet"
    stored = json.loads((ARTIFACT / "h6_agreement_triage.json").read_text())
    context_by_fold = {
        int(fold): tuple(context)
        for fold, context in stored["conditional_max_variance_policy"]["contexts_by_fold"].items()
    }
    assert context_by_fold == {
        0: (3, 5, 6),
        1: (3, 5, 6),
        2: (3, 5, 6),
        3: (2, 5, 6),
        4: (3, 5, 6),
    }
    frame = pd.read_parquet(source)
    key = ["fold", "run_id", "compound_group", "context", "cell", "truth"]
    wide = (
        frame.loc[frame["method"].isin(["gates", "selected_residual"])]
        .pivot(index=key, columns="method", values="prediction")
        .reset_index()
    )
    contexts = list(itertools.combinations(range(1, 8), 3))
    mask = np.logical_or.reduce(
        [
            wide["fold"].eq(fold) & wide["context"].eq(contexts.index(context))
            for fold, context in context_by_fold.items()
        ]
    )
    selected = wide.loc[mask].copy()
    assert len(selected) == 1696
    truth = np.abs(selected["truth"].to_numpy()) >= 0.40
    calls = {
        "agreement": (
            (np.abs(selected["gates"]) >= 0.40)
            & (np.abs(selected["selected_residual"]) >= 0.40)
            & (np.sign(selected["gates"]) == np.sign(selected["selected_residual"]))
        ).to_numpy(),
        "gates": (np.abs(selected["gates"]) >= 0.40).to_numpy(),
    }
    counts = {}
    precision = {}
    for method, call in calls.items():
        table = selected[["compound_group"]].copy()
        table["tp"] = call & truth
        table["fp"] = call & ~truth
        counts[method] = table.groupby("compound_group")[["tp", "fp"]].sum()
        precision[method] = float((call & truth).sum() / max(call.sum(), 1))
    groups = sorted(counts["agreement"].index)
    rng = np.random.default_rng(20261027)
    differences = []
    for _ in range(10000):
        sampled = rng.integers(0, len(groups), len(groups))
        values = []
        for method in ("agreement", "gates"):
            total = counts[method].reindex(groups).iloc[sampled].sum()
            values.append(float(total.tp / (total.tp + total.fp)))
        differences.append(values[0] - values[1])
    expected = stored["conditional_max_variance_policy"]
    np.testing.assert_allclose(
        [precision["agreement"], precision["gates"]],
        [
            expected["methods"]["agreement"]["aggregate"]["effect_precision"],
            expected["methods"]["gates"]["aggregate"]["effect_precision"],
        ],
        rtol=0,
        atol=1e-12,
    )
    bootstrap = expected["agreement_minus_gates_precision_bootstrap"]
    np.testing.assert_allclose(
        [np.mean(differences), np.quantile(differences, 0.025), np.quantile(differences, 0.975)],
        [bootstrap["difference"], *bootstrap["ci95"]],
        rtol=0,
        atol=1e-12,
    )
    receipt = {
        "status": "passed",
        "prediction_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(selected),
        "groups": len(groups),
        "agreement_precision": precision["agreement"],
        "gates_precision": precision["gates"],
        "precision_difference_ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "exploratory_gate": expected["exploratory_gate"],
        "canonical_integration": expected["canonical_integration"],
    }
    (ARTIFACT / "h6_audit.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))
