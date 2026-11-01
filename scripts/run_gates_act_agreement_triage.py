from __future__ import annotations

import json
from pathlib import Path

from gates.act.agreement_triage import run_agreement_triage
from gates.act.competitor_transfer import conditional_max_variance_contexts

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    artifact = ROOT / "artifacts/act/research/competitor_transfer"
    legacy = ROOT / "data/neural_mea_raw/All_Compound_Summary.xlsx"
    phase1 = ROOT / "data/neural_mea_phase1_raw/DNT_NTP2021_EPA_MEA_NFA_2022-08-16_Output.xlsx"
    contexts = conditional_max_variance_contexts(legacy, phase1)
    result = run_agreement_triage(
        artifact / "h1_oof_predictions_with_selected.parquet", artifact, contexts
    )
    print(json.dumps(result, indent=2))
