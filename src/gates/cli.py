from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gates.config import Config
from gates.data.organoid import audit_dataset
from gates.evaluation.metrics import stopping_metrics
from gates.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="gates")
    parser.add_argument("command", choices=["audit", "run", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/organoid.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    if args.command == "audit":
        print(json.dumps(audit_dataset(root / "data" / "raw"), indent=2))
    elif args.command == "run":
        print(json.dumps(run_experiment(root, Config.load(config_path)), indent=2))
    else:
        summary_path = root / "artifacts" / "evidence" / "summary.json"
        if not summary_path.exists():
            raise SystemExit("missing evidence; run `gates run` first")
        summary = json.loads(summary_path.read_text())
        decisions_path = root / "artifacts" / "evidence" / "decisions.csv"
        if not decisions_path.exists():
            raise SystemExit("missing decisions.csv")
        decisions = pd.read_csv(decisions_path).to_dict(orient="records")
        recomputed = stopping_metrics(decisions, max(Config.load(config_path).decision_times))
        for key in ("eesr", "coverage", "observation_savings", "mean_decision_time"):
            if abs(float(summary[key]) - float(recomputed[key])) > 1e-12:
                raise SystemExit(
                    f"evidence mismatch for {key}: saved={summary[key]} "
                    f"recomputed={recomputed[key]}"
                )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
