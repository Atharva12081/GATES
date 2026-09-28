from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gates.evaluation.uncertainty import exact_binomial_interval, group_bootstrap_interval

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "evidence" / "decisions.csv"
OUTPUT = ROOT / "artifacts" / "phase2" / "organoid_feasibility_uncertainty"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(SOURCE)
    decisions["incorrect_early"] = (
        (decisions["decision"] == "STOP") & (decisions["prediction"] != decisions["label"])
    ).astype(int)
    decisions["saved_fraction"] = decisions.apply(
        lambda row: 1 - float(row["decision_time"]) / 72 if row["decision"] == "STOP" else 0.0,
        axis=1,
    )
    decisions["saved_fraction_stopped"] = decisions.apply(
        lambda row: (
            1 - float(row["decision_time"]) / 72 if row["decision"] == "STOP" else float("nan")
        ),
        axis=1,
    )
    decisions["terminal_time"] = decisions.apply(
        lambda row: float(row["decision_time"]) if row["decision"] == "STOP" else 72.0,
        axis=1,
    )
    stopped = decisions.loc[decisions["decision"] == "STOP"].copy()
    errors = int(stopped["incorrect_early"].sum())
    n_stopped = len(stopped)

    fold_rows: list[dict[str, float | int]] = []
    for fold, frame in decisions.groupby("fold"):
        fold_stopped = frame.loc[frame["decision"] == "STOP"]
        fold_rows.append(
            {
                "fold": int(fold),
                "test_replicate": int(frame["replicate"].iloc[0]),
                "independent_units": int(len(frame)),
                "stopped": int(len(fold_stopped)),
                "incorrect_early": int((fold_stopped["prediction"] != fold_stopped["label"]).sum()),
                "EESR": float((fold_stopped["prediction"] != fold_stopped["label"]).mean())
                if len(fold_stopped)
                else 0.0,
                "coverage": float(len(fold_stopped) / len(frame)),
                "observation_savings_all": float(frame["saved_fraction"].mean()),
                "observation_savings_stopped": float(fold_stopped["saved_fraction_stopped"].mean())
                if len(fold_stopped)
                else 0.0,
                "mean_decision_time": float(frame["terminal_time"].mean()),
            }
        )
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "fold_results.csv", index=False)

    report = {
        "snapshot": "gates-organoid-feasibility-v0",
        "independent_biological_units": int(len(decisions)),
        "independent_groups": int(decisions["replicate"].nunique()),
        "stopped_early": n_stopped,
        "incorrect_early": errors,
        "EESR_exact_95": exact_binomial_interval(errors, n_stopped).as_dict(),
        "observation_savings_all_group_bootstrap_95": group_bootstrap_interval(
            decisions,
            "saved_fraction",
            "replicate",
            samples=10_000,
            seed=20260929,
        ).as_dict(),
        "observation_savings_stopped_group_bootstrap_95": group_bootstrap_interval(
            stopped,
            "saved_fraction_stopped",
            "replicate",
            samples=10_000,
            seed=20260929,
        ).as_dict(),
        "decision_time_group_bootstrap_95": group_bootstrap_interval(
            decisions,
            "terminal_time",
            "replicate",
            samples=10_000,
            seed=20260929,
        ).as_dict(),
        "failure_sensitivity": {
            "observed": errors / n_stopped,
            "one_additional_failure": (errors + 1) / n_stopped,
            "two_additional_failures": (errors + 2) / n_stopped,
        },
        "claim": "empirical_only",
        "warning": "Only three replicate groups exist; group-bootstrap intervals are exploratory.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
