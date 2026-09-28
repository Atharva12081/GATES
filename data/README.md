# Data

Raw files are not committed. Run `uv run python scripts/fetch_data.py` to download the three
published OrganoID Figure 3 tables into `data/raw/`. The source manifest records URLs and
SHA-256 checksums after download.

The experimental unit is a dosage–replicate well. All observations from one unit stay in one
partition. The endpoint is the independently measured MTS viability value. The default binary
endpoint is viability below 0.5, chosen before evaluation as a coarse response threshold.

The public tables are small. Results are a feasibility demonstration, not evidence of clinical
or laboratory deployment readiness.

