from __future__ import annotations

import json
from pathlib import Path

from gates.act.decision_triage import run_decision_triage_study

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = run_decision_triage_study(
        ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx",
        ROOT / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx",
        ROOT / "artifacts/act_v3/preregistered_config.json",
        ROOT / "artifacts/act_v3/summary.json",
        ROOT / "artifacts/act/research/decision_triage",
    )
    print(json.dumps(result, indent=2))
