from __future__ import annotations

from pathlib import Path

from gates.act.benchmark_v3 import run_v3_benchmark

if __name__ == "__main__":
    run_v3_benchmark(
        Path("data/neural_mea_raw/All_Compound_Summary.xlsx"),
        Path("data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx"),
        Path("artifacts/act_v3/preregistered_config.json"),
        Path("artifacts/act_v3"),
    )
