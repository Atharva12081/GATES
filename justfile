set shell := ["zsh", "-cu"]

setup:
    uv sync --frozen --extra dev
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
    uv run python scripts/verify_final_evidence.py

demo:
    uv run streamlit run app/streamlit_app.py
