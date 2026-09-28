import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "evidence"


@pytest.mark.skipif(not (EVIDENCE / "decisions.csv").exists(), reason="evidence not generated")
def test_every_unit_is_evaluated_once() -> None:
    decisions = pd.read_csv(EVIDENCE / "decisions.csv")
    assert len(decisions) == 21
    assert decisions["unit_id"].is_unique


@pytest.mark.skipif(not (EVIDENCE / "summary.json").exists(), reason="evidence not generated")
def test_summary_does_not_overclaim_certification() -> None:
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    assert summary["calibration_claim"] in {"empirical_only", "confidence_bound_supported"}
