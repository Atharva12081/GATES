set shell := ["zsh", "-cu"]

setup:
    uv sync --extra dev
    uv run python scripts/fetch_data.py

run:
    uv run gates run

verify:
    uv run ruff check .
    uv run pytest
    uv run gates verify

demo:
    uv run streamlit run app/streamlit_app.py

