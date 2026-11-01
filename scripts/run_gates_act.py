from __future__ import annotations

import argparse
from pathlib import Path

from gates.act import run_act_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the leakage-free GATES-ACT benchmark")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "act")
    parser.add_argument("--max-budget", type=int, default=30)
    parser.add_argument("--target-nrmse", type=float, default=0.12)
    args = parser.parse_args()
    print(
        run_act_benchmark(
            args.raw,
            args.output,
            max_budget=args.max_budget,
            target_nrmse=args.target_nrmse,
        )
    )


if __name__ == "__main__":
    main()
