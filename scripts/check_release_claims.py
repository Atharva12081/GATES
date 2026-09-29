from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").rglob("*.md")),
    ROOT / "app/streamlit_app.py",
]


def main() -> None:
    master = pd.read_csv(ROOT / "artifacts/final/master_evidence.csv")
    row = master[
        (master["dataset"] == "orgAInoid_RPE_Final")
        & (master["experiment"] == "ALL")
        & (master["method"] == "gates_full")
        & np.isclose(master["risk_target"], 0.05)
    ].iloc[0]
    expected = {
        "988": int(row["test_units"]),
        "439": int(row["early_stops"]),
        "25": int(row["early_stop_errors"]),
        "5.69%": f"{row['eesr']:.2%}",
        "44.43%": f"{row['coverage']:.2%}",
        "12.26%": f"{row['observation_savings']:.2%}",
    }
    assert all(str(key) == str(value) for key, value in expected.items())

    core = [
        ROOT / "README.md",
        ROOT / "docs/final/TECHNICAL_REPORT.md",
        ROOT / "docs/final/KAGGLE_WRITEUP.md",
        ROOT / "docs/final/VIDEO_SCRIPT.md",
    ]
    for path in core:
        text = path.read_text()
        for token in expected:
            assert token in text, f"{path.relative_to(ROOT)} omits frozen claim {token}"

    local_link = re.compile(r"\[[^]]+\]\((?!https?://|#)([^)]+)\)")
    for path in PUBLIC_FILES:
        text = path.read_text()
        for target in local_link.findall(text):
            target_path = target.split("#", 1)[0]
            if target_path:
                assert (path.parent / target_path).exists(), (
                    f"broken local link in {path.relative_to(ROOT)}: {target}"
                )
    forbidden = ("/" + "Users/", "/" + "tmp/", "file" + "://")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    for relative in tracked:
        try:
            text = (ROOT / relative).read_text()
        except UnicodeDecodeError:
            continue
        for marker in forbidden:
            assert marker not in text, f"local path leak in {relative}: {marker}"
    print("Release claims, local links, and path hygiene verified.")


if __name__ == "__main__":
    main()
