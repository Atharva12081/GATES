from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gates.act.benchmark_v2 import run_neural_mea_benchmark

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data" / "neural_mea_raw" / "All_Compound_Summary.xlsx"
OUTPUT = ROOT / "artifacts" / "act_v2" / "ablations"

CONFIGS = {
    "primary": {},
    "single_div_hit": {"min_effect_divs": 1},
    "no_abstention": {"ood_quantile": 1.0},
    "allow_10pct_calibration_error": {"calibration_violation_limit": 0.10},
    "strict_nrmse_015": {"target_nrmse": 0.15},
    "lenient_nrmse_025": {"target_nrmse": 0.25},
    "effect_threshold_030": {"effect_threshold": 0.30},
    "effect_threshold_050": {"effect_threshold": 0.50},
}


def main() -> None:
    rows = []
    for name, parameters in CONFIGS.items():
        summary = run_neural_mea_benchmark(WORKBOOK, OUTPUT / name, **parameters)
        decisions = pd.read_csv(OUTPUT / name / "neural_decisions.csv")
        act = decisions.loc[decisions["policy"] == "act"]
        stops = act.loc[act["decision"] == "STOP"]
        rows.append(
            {
                "ablation": name,
                "target_nrmse": summary["target_nrmse"],
                "effect_threshold": summary["effect_threshold_log1p_contrast"],
                "min_effect_divs": summary["minimum_effect_divs_for_stop"],
                "calibration_violation_limit": summary["calibration_violation_limit"],
                "ood_quantile": summary["ood_quantile"],
                "stops": len(stops),
                "unsafe_stops": int((~stops["stop_safe"]).sum()),
                "compound_effect_misses": int(stops["compound_effect_missed"].sum()),
                "abstentions": int((act["decision"] == "ABSTAIN").sum()),
                "mean_recording_sessions": float(act["recording_sessions"].mean()),
                "session_reduction": float(1 - act["recording_sessions"].mean() / 4),
            }
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "artifacts" / "act_v2" / "neural_ablations.csv", index=False)
    (ROOT / "artifacts" / "act_v2" / "ablation_manifest.json").write_text(
        json.dumps(CONFIGS, indent=2, sort_keys=True) + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
