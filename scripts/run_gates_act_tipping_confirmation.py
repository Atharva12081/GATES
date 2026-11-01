from pathlib import Path

from gates.act.benchmark_tipping import run_tipping_confirmation

if __name__ == "__main__":
    run_tipping_confirmation(
        Path("data/neural_mea_raw/All_Compound_Summary.xlsx"),
        Path("data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx"),
        Path("data/neural_mea_brown_raw/brown_mea_2016.csv"),
        [
            Path("data/neural_mea_tipping_raw/DNTRef_all.csv"),
            Path("data/neural_mea_tipping_raw/AllData_SA3_NoDNTRef.csv"),
        ],
        Path("artifacts/act_v3/preregistered_config.json"),
        Path("artifacts/act_v3/stopping_lock.json"),
        Path("artifacts/act_v3"),
    )
