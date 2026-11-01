from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gates.act.benchmark_v3 import _paired_sign_flip
from gates.act.final_policy import RiskLimit, minimum_zero_failure_stops

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "act_v3"
OUTPUT = ROOT / "artifacts" / "act" / "canonical"
TRACE_FILES = {
    "phase1": SOURCE / "fresh_acquisition_traces.csv",
    "brown": SOURCE / "brown_fresh_traces.csv",
    "tipping": SOURCE / "tipping_fresh_traces.csv",
}
DECISION_FILES = {
    "phase1": SOURCE / "fresh_decisions.csv",
    "brown": SOURCE / "brown_fresh_decisions.csv",
    "tipping": SOURCE / "tipping_fresh_decisions.csv",
}
POLICIES = ("max_variance", "risk_sensitive", "maximin", "fixed_sentinel", "random")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_metrics(path: Path, dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "compound_group",
        "policy",
        "round",
        "effect_recall",
        "effect_precision",
        "nrmse",
        "direct_effect_capture",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    if "v3_split" in frame:
        frame = frame.loc[frame["v3_split"].eq("test")].copy()
    selected = frame.loc[frame["round"].isin([2, 3])].copy()
    grouped = (
        selected.groupby(["compound_group", "policy"], as_index=False)[
            ["effect_recall", "effect_precision", "nrmse", "direct_effect_capture"]
        ]
        .mean()
        .assign(dataset=dataset)
    )
    counts = grouped.groupby("compound_group")["policy"].nunique()
    if not counts.eq(len(POLICIES)).all():
        raise ValueError(f"{path} does not contain all policies for every group")
    return grouped


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(20_000)]
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group_metrics = pd.concat(
        [_group_metrics(path, dataset) for dataset, path in TRACE_FILES.items()],
        ignore_index=True,
    )
    group_metrics.to_csv(OUTPUT / "group_metrics.csv", index=False)

    canonical = group_metrics.loc[group_metrics["policy"].eq("max_variance")]
    policy_summary = (
        group_metrics.groupby("policy", as_index=False)
        .agg(
            groups=("compound_group", "size"),
            effect_recall=("effect_recall", "mean"),
            effect_precision=("effect_precision", "mean"),
            nrmse=("nrmse", "mean"),
            direct_effect_capture=("direct_effect_capture", "mean"),
        )
        .sort_values("effect_recall", ascending=False)
    )
    policy_summary.to_csv(OUTPUT / "policy_summary.csv", index=False)

    comparisons = []
    canonical_index = canonical.set_index(["dataset", "compound_group"])
    for baseline in POLICIES:
        if baseline == "max_variance":
            continue
        other = group_metrics.loc[group_metrics["policy"].eq(baseline)].set_index(
            ["dataset", "compound_group"]
        )
        aligned = canonical_index.join(other, lsuffix="_canonical", rsuffix="_baseline")
        for metric in ("effect_recall", "nrmse", "direct_effect_capture"):
            values = (
                aligned[f"{metric}_canonical"] - aligned[f"{metric}_baseline"]
            ).to_numpy()
            low, high = _bootstrap(values, seed=20261023 + len(comparisons))
            comparisons.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "difference": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "paired_sign_flip_p": _paired_sign_flip(values, 20261023),
                    "groups": len(values),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(OUTPUT / "paired_comparisons.csv", index=False)

    candidate_stops = 0
    candidate_unsafe = 0
    for path in DECISION_FILES.values():
        frame = pd.read_csv(path)
        stops = frame.loc[
            frame["policy"].eq("max_variance") & frame["decision"].eq("STOP")
        ]
        candidate_stops += len(stops)
        candidate_unsafe += int((~stops["stop_safe"].astype(bool)).sum())
    risk_limit = RiskLimit(candidate_stops, candidate_unsafe)

    largest = canonical.loc[canonical["dataset"].eq("tipping")]
    evidence = {
        "schema": "gates-act-canonical-core-v1",
        "canonical_acquisition": "cost-aware conditional maximum variance",
        "selection_basis": (
            "adopt the strongest frozen acquisition baseline; no external outcomes were used "
            "to tune a new acquisition score"
        ),
        "evaluation_groups": int(
            canonical[["dataset", "compound_group"]].drop_duplicates().shape[0]
        ),
        "pooled_group_weighted": {
            "effect_recall": float(canonical["effect_recall"].mean()),
            "effect_precision": float(canonical["effect_precision"].mean()),
            "nrmse": float(canonical["nrmse"].mean()),
            "direct_effect_capture": float(canonical["direct_effect_capture"].mean()),
        },
        "largest_independent_confirmation": {
            "dataset": "tipping",
            "groups": int(largest["compound_group"].nunique()),
            "effect_recall": float(largest["effect_recall"].mean()),
            "nrmse": float(largest["nrmse"].mean()),
            "direct_effect_capture": float(largest["direct_effect_capture"].mean()),
        },
        "stopping": {
            "candidate_validation_stops": candidate_stops,
            "candidate_unsafe_stops": candidate_unsafe,
            "one_sided_95pct_unsafe_rate_upper_bound": risk_limit.upper_bound,
            "authorized": risk_limit.authorized,
            "minimum_zero_failure_validation_stops_for_5pct_limit": minimum_zero_failure_stops(),
            "released_action": "COMPLETE_OR_ABSTAIN",
            "actual_session_savings": 0.0,
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): _sha(path)
            for path in [*TRACE_FILES.values(), *DECISION_FILES.values()]
        },
        "limitations": [
            "No current evidence authorizes early stopping at a 5% unsafe-stop risk limit.",
            "Protected effect recall is conservative and has low precision under group shift.",
            "Measurement-count efficiency is not equivalent to saved physical wells or sessions.",
            "The Vassallo public archive lacks phase-resolved processed responses needed "
            "for a compatible sequential replay.",
        ],
    }
    (OUTPUT / "core_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
