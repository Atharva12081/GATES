from __future__ import annotations

import json
from pathlib import Path

from gates.act.source_shift_agreement import run_source_shift_agreement

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = run_source_shift_agreement(
        ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx",
        ROOT / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx",
        ROOT / "artifacts/act/research/competitor_transfer",
    )
    print(json.dumps(result, indent=2))
