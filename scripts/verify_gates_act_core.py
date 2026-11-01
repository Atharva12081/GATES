from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "act" / "canonical"


def main() -> None:
    evidence = json.loads((ARTIFACT / "core_evidence.json").read_text())
    for relative, expected in evidence["source_hashes"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}")

    groups = pd.read_csv(ARTIFACT / "group_metrics.csv")
    if groups[["dataset", "compound_group"]].drop_duplicates().shape[0] != 76:
        raise ValueError("external group count changed")
    if groups.groupby(["dataset", "compound_group"])["policy"].nunique().ne(5).any():
        raise ValueError("policy coverage is incomplete")

    independent = groups.groupby("policy")[
        ["effect_recall", "effect_precision", "nrmse", "direct_effect_capture"]
    ].mean()
    canonical = independent.loc["max_variance"]
    recorded = evidence["pooled_group_weighted"]
    for metric in recorded:
        if not np.isclose(float(canonical[metric]), float(recorded[metric]), atol=1e-12):
            raise ValueError(f"canonical {metric} does not independently reproduce")
    if canonical.effect_recall < independent.effect_recall.max() - 1e-12:
        raise ValueError("canonical policy no longer has the highest pooled effect recall")
    if canonical.nrmse > independent.nrmse.min() + 1e-12:
        raise ValueError("canonical policy no longer has the lowest pooled NRMSE")

    stopping = evidence["stopping"]
    interval = binomtest(
        int(stopping["candidate_unsafe_stops"]),
        int(stopping["candidate_validation_stops"]),
    ).proportion_ci(confidence_level=0.90, method="exact")
    if not np.isclose(interval.high, stopping["one_sided_95pct_unsafe_rate_upper_bound"]):
        raise ValueError("unsafe-stop risk bound does not reproduce")
    if stopping["authorized"] or stopping["actual_session_savings"] != 0:
        raise ValueError("unsafe early stopping must remain disabled")
    print("GATES-ACT core evidence independently verified; early stopping remains disabled.")


if __name__ == "__main__":
    main()
