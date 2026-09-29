from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for endpoint in ("retinal_rpe_final", "retinal_lens_final"):
        path = ROOT / "artifacts" / "phase2" / endpoint / "run_summary.json"
        summary = json.loads(path.read_text())
        summary["output_dir"] = f"artifacts/phase2/{endpoint}"
        path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
