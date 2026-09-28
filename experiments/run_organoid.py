from pathlib import Path

from gates.config import Config
from gates.pipeline import run_experiment

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    print(run_experiment(ROOT, Config.load(ROOT / "configs" / "organoid.json")))
