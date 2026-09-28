from pathlib import Path

import pytest

from gates.data.organoid import build_longitudinal_table

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "data" / "raw" / "MTSAssay.csv").exists(), reason="data not fetched"
)
def test_join_has_independent_endpoint_units() -> None:
    frame = build_longitudinal_table(ROOT / "data" / "raw")
    assert frame["unit_id"].nunique() == 21
    assert frame.groupby("unit_id")["viability"].nunique().max() == 1
