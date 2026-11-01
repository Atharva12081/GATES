from __future__ import annotations

import argparse
from pathlib import Path

from gates.act.benchmark_v2 import run_neural_mea_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the chronological GATES-ACT v2 benchmark")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT / "data" / "neural_mea_raw" / "All_Compound_Summary.xlsx",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "act_v2")
    parser.add_argument("--target-nrmse", type=float, default=0.20)
    parser.add_argument("--effect-threshold", type=float, default=0.40)
    args = parser.parse_args()
    print(
        run_neural_mea_benchmark(
            args.workbook,
            args.output,
            target_nrmse=args.target_nrmse,
            effect_threshold=args.effect_threshold,
        )
    )


if __name__ == "__main__":
    main()
