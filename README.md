# GATES

**Group-Aware Time-Efficient Stopping for Longitudinal In-Vitro Experiments**

GATES turns endpoint prediction into a sequential laboratory decision: **STOP** when a held-out
trajectory is confident enough under a calibration rule, **CONTINUE** when more observation is
needed, and **ABSTAIN** when a simple distribution-shift gate says the sample is unfamiliar.

This repository is an executable feasibility study, not a deployment claim. It uses the public
OrganoID gemcitabine time course: measurements every four hours from 0 to 72 h and independently
measured endpoint viability. All time points from an experimental unit stay together. The default
evaluation rotates the three endpoint replicates through train, calibration, and test roles.

## Reproduce

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra dev
uv run python scripts/fetch_data.py
just data-retinal
uv run gates audit
uv run gates run
uv run gates verify
uv run pytest
```

The run writes every headline number to `artifacts/evidence/`, the main figure to
`artifacts/figures/feasibility.png`, and fold/time-specific models to `artifacts/models/`.

## Demo

```bash
uv run streamlit run app/streamlit_app.py
```

The app does not retrain or invent results. It reads the frozen held-out predictions and decisions
from `artifacts/evidence/` and exposes the confidence threshold, shift score, and reason for each
decision.

## Scientific question

How much of a longitudinal experiment can be skipped while keeping premature endpoint decisions
within a predefined empirical risk tolerance on unseen experimental units?

The primary metrics are:

- erroneous early-stop rate: wrong final decisions among cases stopped early;
- early-stop coverage: fraction of held-out units receiving an early automated decision;
- observation savings: scheduled observations avoided after a stop;
- fixed-time balanced accuracy, AUROC, and Brier score;
- refusal behavior under the fitted feature-distance gate.

## Repository map

```text
app/                       evidence-only Streamlit demo
artifacts/evidence/        machine-readable held-out results
artifacts/figures/         publication-ready figures
configs/                   frozen experiment configuration
data/manifests/            source URLs and checksums
docs/                      report, data statement, limitations, defense notes
experiments/               reproducible entry points
scripts/                   data retrieval and verification helpers
src/gates/                 data, features, models, calibration, shift, policy, evaluation
tests/                     leakage, prefix, policy, and evidence tests
```

## What is and is not claimed

GATES demonstrates an auditable decision layer on a very small public biological dataset. The
21 endpoint-labelled units are enough to test the pipeline and expose failure modes, but not enough
to establish a 5% finite-sample risk guarantee. The software records whether a calibrated threshold
is supported by the conservative confidence bound; when it is not, the result is labelled
`empirical_only`. See [limitations](docs/limitations.md) and the [data statement](docs/data_statement.md).

## Data and licensing

The download script retrieves three Figure 3 tables from the upstream
[OrganoID repository](https://github.com/jono-m/OrganoID). The upstream repository has no
GitHub-detected license, so this repository does not redistribute the raw tables. Confirm reuse
terms with the dataset authors before a public competition release.

## Safety

This is research software for retrospective analysis. It does not prescribe wet-lab actions,
medical treatment, or clinical decisions. STOP means "the model's endpoint call meets the configured
research threshold under this evaluation," not "terminate a real experiment without human review."

## Frozen retinal-organoid result

The competition-facing result uses 988 organoids from 11 independent experiments and holds out
entire experiments. At the frozen 5% operating point, Full GATES produced 5.69% observed EESR at
44.43% early-decision coverage while saving 12.26% of scheduled observations. This is empirical,
not certified risk control. See the [claim freeze](docs/phase3/CLAIM_FREEZE.md),
[technical report](docs/final/TECHNICAL_REPORT.md), and
[master evidence table](artifacts/final/master_evidence.csv).

Rebuild and verify the final retrospective package with:

```bash
uv sync --frozen --extra dev
just data-retinal
just phase3-finalize
just verify
just verify-final
```

`just data-retinal` downloads the exact Zenodo record and filename pinned in
`data/manifests/retinal_sources.json`, then checks its 400,059,241-byte size, MD5, and SHA-256.
It never resolves a mutable “latest” record. See the [frozen-evidence erratum](docs/final/ERRATUM.md)
for the corrected descriptive label audit; the model and frozen headline did not change.
