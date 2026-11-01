from __future__ import annotations

import json
from pathlib import Path

from gates.act.competitor_transfer import run_h1

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = run_h1(
        ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx",
        ROOT / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx",
        ROOT / "artifacts/act/research/competitor_transfer",
    )
    print(json.dumps(result, indent=2))
