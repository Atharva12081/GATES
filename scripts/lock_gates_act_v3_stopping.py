from pathlib import Path

from gates.act.stopping_v3 import lock_stopping_rule

if __name__ == "__main__":
    lock_stopping_rule(
        Path("data/neural_mea_raw/All_Compound_Summary.xlsx"),
        Path("data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx"),
        Path("artifacts/act_v3/preregistered_config.json"),
        Path("artifacts/act_v3/stopping_lock.json"),
    )
