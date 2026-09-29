from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a class-specific appendix from frozen unit decisions."
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("artifacts/phase2/retinal_rpe_final/unit_decisions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/final/class_specific_audit.csv"),
    )
    args = parser.parse_args()

    decisions = pd.read_csv(args.decisions)
    frozen = decisions[
        (decisions["method"] == "gates_full")
        & np.isclose(decisions["parameter"], 0.05)
    ].copy()
    if len(frozen) != 988 or frozen["unit_id"].nunique() != 988:
        raise ValueError("Expected one frozen Full GATES decision for each of 988 organoids")

    rows: list[dict[str, float | int | str]] = []
    for endpoint, label in ((0, "RPE-negative"), (1, "RPE-positive")):
        part = frozen[frozen["endpoint"] == endpoint]
        stopped = part[part["decision"] == "STOP"]
        refused = part[part["decision"] == "ABSTAIN"]
        errors = stopped[stopped["prediction"] != stopped["endpoint"]]
        rows.append(
            {
                "endpoint_class": label,
                "organoids": len(part),
                "early_stops": len(stopped),
                "early_decision_coverage": len(stopped) / len(part),
                "refusals": len(refused),
                "refusal_rate": len(refused) / len(part),
                "continued_without_refusal": int((part["decision"] == "CONTINUE").sum()),
                "erroneous_early_stops": len(errors),
                "class_specific_eesr": len(errors) / len(stopped),
                "false_positive_early_stops": int(
                    ((errors["prediction"] == 1) & (errors["endpoint"] == 0)).sum()
                ),
                "false_negative_early_stops": int(
                    ((errors["prediction"] == 0) & (errors["endpoint"] == 1)).sum()
                ),
            }
        )

    output = pd.DataFrame(rows)
    if output["organoids"].sum() != 988 or output["early_stops"].sum() != 439:
        raise ValueError("Class-specific audit does not reconcile with the frozen headline")
    if output["erroneous_early_stops"].sum() != 25 or output["refusals"].sum() != 38:
        raise ValueError("Class-specific errors/refusals do not reconcile with frozen evidence")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
