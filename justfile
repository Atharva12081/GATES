set shell := ["zsh", "-cu"]

setup:
    uv sync --frozen --extra dev
    uv run python scripts/fetch_data.py

data-organoid:
    uv run python scripts/fetch_data.py

data-retinal:
    uv run python scripts/fetch_retinal_data.py

run:
    uv run gates run

verify:
    uv run ruff check .
    uv run pytest
    uv run gates verify

phase3-finalize:
    uv run python scripts/run_phase3_experiment_gate.py --data data/phase2_raw/orgainoid_morphometrics.csv --phase2 artifacts/phase2/retinal_rpe_final --protocol protocols/phase3/orgainoid_experiment_gate_protocol.json --output artifacts/phase3/orgainoid_experiment_gate
    uv run python scripts/finalize_phase3_submission.py

reproduce: data-retinal phase3-finalize verify verify-final

verify-final:
    uv run python scripts/build_class_specific_audit.py
    uv run python scripts/verify_final_evidence.py
    uv run python scripts/check_release_claims.py

demo:
    uv run streamlit run app/streamlit_app.py

act: data-organoid
    uv run python scripts/run_gates_act.py

data-neural:
    uv run python scripts/fetch_neural_mea_data.py

act-v2: data-neural
    uv run python scripts/run_gates_act_v2.py
    uv run python scripts/run_gates_act_v2_ablations.py
    uv run python scripts/build_act_v2_figures.py

act-report:
    uv run python scripts/build_act_report.py

act-v3:
    uv run python scripts/verify_gates_act_v3_sources.py
    uv run python scripts/select_gates_act_v3.py
    uv run python scripts/run_gates_act_v3.py
    uv run python scripts/run_gates_act_brown_confirmation.py
    uv run python scripts/lock_gates_act_v3_stopping.py
    uv run python scripts/run_gates_act_tipping_confirmation.py
    uv run python scripts/audit_gates_act_v3.py

act-core:
    uv run python scripts/verify_gates_act_v3_sources.py
    uv run python scripts/audit_gates_act_core.py
    uv run python scripts/verify_gates_act_core.py
    uv run pytest

act-vision-audit:
    uv run python scripts/audit_gates_act_vision.py

act-final-verify: act-core act-vision-audit
    uv run ruff check .

act-decision-study:
    uv run python scripts/run_gates_act_decision_utility.py
    uv run python scripts/audit_gates_act_decision_utility.py
    uv run pytest -q tests/test_decision_utility.py

act-decision-triage:
    uv run python scripts/run_gates_act_decision_triage.py
    uv run python scripts/audit_gates_act_decision_triage.py
    uv run pytest -q tests/test_decision_triage.py
