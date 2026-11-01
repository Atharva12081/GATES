from __future__ import annotations

import argparse
import json
from pathlib import Path

from gates.act.competitor_nfa import run_nfa_head_to_head

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("competitor_checkout", type=Path)
    args = parser.parse_args()
    result = run_nfa_head_to_head(
        args.competitor_checkout / ".cache/neurochip_twin/nfa_tasks.npz",
        args.competitor_checkout / "chip_forecast_result.csv",
        ROOT / "artifacts/act/research/competitor_transfer",
    )
    print(json.dumps(result, indent=2))
