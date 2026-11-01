from pathlib import Path

from gates.act.brainsphere_audit import audit_brainsphere


def test_public_brainsphere_supplements_fail_row_level_eligibility(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    result = audit_brainsphere(
        root / "data/research/carstens_2025_brainsphere_supplement", tmp_path
    )
    assert result["eligibility"] == "FAIL"
    assert result["raw_data_status"].startswith("not possessed")
    assert result["checks"]["xlsx_has_no_row_level_dose_time_identity_table"]
