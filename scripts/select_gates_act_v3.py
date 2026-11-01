from __future__ import annotations

from pathlib import Path

from gates.act.benchmark_v3 import select_v3_config

if __name__ == "__main__":
    select_v3_config(
        Path("data/neural_mea_raw/All_Compound_Summary.xlsx"),
        Path("data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx"),
        Path("artifacts/act_v3"),
    )
