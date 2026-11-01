from __future__ import annotations

import json
from pathlib import Path

from gates.act.chronological_high_dose import run_chronological_high_dose

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = run_chronological_high_dose(
        ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx",
        ROOT / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx",
        ROOT / "artifacts/act/research/direct_discovery/direct_discovery_decisions.parquet",
        ROOT / "artifacts/act/research/chronological_high_dose",
    )
    print(json.dumps(result, indent=2))
