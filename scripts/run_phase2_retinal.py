from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gates.data.retinal import load_retinal_morphometrics
from gates.evaluation.phase2 import run_phase2_campaign

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=["RPE_Final", "Lens_Final"], default="RPE_Final")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "phase2_raw" / "orgainoid_morphometrics.csv",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dataset = load_retinal_morphometrics(
        args.input,
        endpoint=args.endpoint,
    )
    output = args.output or ROOT / "artifacts" / "phase2" / f"retinal_{args.endpoint.lower()}"
    result = run_phase2_campaign(dataset, output, args.seed, commit)
    (output / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
