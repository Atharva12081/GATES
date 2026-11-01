from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRECT = ROOT / "artifacts/act/research/direct_discovery/direct_discovery_decisions.parquet"
ARTIFACT = ROOT / "artifacts/act/research/scientific_opportunity"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary_path = ARTIFACT / "opportunity_summary.json"
    group_path = ARTIFACT / "opportunity_group_patterns.csv"
    legal_path = ARTIFACT / "opportunity_legal_identifiability.csv"
    missed_path = ARTIFACT / "opportunity_missed_effect_cells.csv"
    summary = json.loads(summary_path.read_text())
    groups = pd.read_csv(group_path)
    legal = pd.read_csv(legal_path)
    missed = pd.read_csv(missed_path)
    decisions = pd.read_parquet(DIRECT)
    high = decisions.loc[decisions["policy"].eq("highest_three_doses")].copy()
    high["day_index"] = high["cell"] // 8
    high["effect"] = high["truth"].abs() >= 0.40
    oracle = int(
        high.groupby(["fold", "run_id", "compound_group", "day_index"])["effect"]
        .sum()
        .clip(upper=3)
        .sum()
    )
    checks = {
        "one_row_per_run_group": len(groups) == 106,
        "ninety_nine_compound_groups": groups["compound_group"].nunique() == 99,
        "eighty_three_active_run_groups": int(groups["reference_active"].sum()) == 83,
        "exactly_two_lower_dose_only_groups": int(groups["lower_dose_only"].sum()) == 2,
        "missed_effect_cells_are_hidden_rank4": bool(
            len(missed) == 2
            and missed["dose_rank"].eq(4).all()
            and not missed["directly_observed"].any()
        ),
        "legal_table_is_high_dose_negative_only": bool(
            len(legal) == 25 and ~legal["direct_high_dose_active"].all()
        ),
        "legal_scores_are_finite": bool(
            np.isfinite(
                legal[["observed_max_absolute_response", "hidden_agreement_score"]].to_numpy()
            ).all()
        ),
        "oracle_direct_effect_count_recomputes": (
            oracle == summary["oracle_upper_bound"]["direct_effect_cells"] == 446
        ),
        "algorithmic_search_stopped": (
            summary["new_method_gate"]["decision"] == "STOP_ALGORITHMIC_SEARCH"
        ),
        "canonical_unchanged": summary["canonical_change"] == "NONE",
        "early_stopping_disabled": summary["early_stopping"] == "disabled",
    }
    checks = {key: bool(value) for key, value in checks.items()}
    receipt = {
        "schema": "gates-act-scientific-opportunity-independent-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sha256": {
            path.name: _sha256(path) for path in (summary_path, group_path, legal_path, missed_path)
        },
    }
    (ARTIFACT / "opportunity_audit.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
