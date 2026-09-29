# GATES

## Group-Aware Time-Efficient Stopping

**Most experimental AI asks what can be learned from collected data. GATES asks whether another
measurement is still worth collecting.**

GATES is a group-calibrated selective stopping framework for longitudinal biological experiments.
It makes an early endpoint call only when the available trajectory passes an independently
calibrated rule, and refuses early automation when the trajectory is unfamiliar.

```text
new observation -> CONTINUE -> STOP
                         \----> ABSTAIN and complete the protocol
```

| Frozen RPE evaluation | Result |
|---|---:|
| Medaka retinal organoids | 988 |
| Independent held-out experiments | 11 |
| Observed erroneous-early-stop rate | 5.69% (25/439) |
| Unit-level exact 95% CI | 3.72%-8.29% |
| Early-decision coverage | 44.43% |
| Retrospectively estimated observation savings | 12.26% |

![Risk versus retrospectively estimated observation savings](artifacts/final/figures/01_rpe_risk_savings_frontier.png)

**[Demo](#interactive-evidence-demo) · [Technical report](docs/final/TECHNICAL_REPORT.md) ·
[Reproduce](#reproduce-the-frozen-evidence)**

Developed by **Atharva Parande**.

## Why this is not ordinary prediction

The source orgAInoid study established that future tissue outcomes can be predicted from early
images. GATES addresses the downstream decision: whether a particular partial trajectory is
reliable enough to justify terminating observation. Entire experiments—not individual images or
organoids—form the held-out generalization boundary. Experimental groups are therefore part of the
reliability problem, not merely a train/test bookkeeping variable.

At the frozen operating point, GATES permits early automation for 44.43% of organoids and
deliberately continues observation for the remainder rather than forcing a premature decision.
Removing the refusal layer increased observed EESR from 5.69% to 8.60%, while retrospectively
estimated savings changed from 12.26% to 13.65%.

## Interactive evidence demo

```bash
uv sync --frozen --extra dev
uv run streamlit run app/streamlit_app.py
```

The CPU-only app reads frozen held-out predictions and decisions; it never retrains. It exposes
three canonical cases: a correct early stop, a refused trajectory, and an erroneous early stop.
For each case it shows the experiment, available observation time, endpoint probability,
confidence, OOD score, decision, ground truth after 72 h, estimated savings, and provenance.

## Evaluation design

The frozen analysis uses longitudinal bright-field morphometrics from 988 *Oryzias latipes*
(medaka) retinal organoids across 11 experiments. The hierarchy is:

```text
114,510 published morphometric observations
             ↓
      988 organoid wells
             ↓
   11 independent experiments
```

The article reports 117,249 acquired images. GATES reads the publication's pinned
`Extended_Data_2.csv` morphometrics table directly, which contains 114,510 rows; its loader removes
no rows. The 2,739-image difference therefore arose upstream, between image acquisition and the
published morphometrics table. The released table and article do not provide per-image exclusion
reasons, so none are inferred here.

## Results in context

| Frozen 5% operating point | EESR | Coverage | Estimated savings |
|---|---:|---:|---:|
| Non-group calibration | 9.01% | 65.18% | 19.79% |
| CBES-style adaptation | 15.51% | 45.04% | 15.32% |
| GATES without refusal | 8.60% | 48.28% | 13.65% |
| **Full GATES** | **5.69%** | **44.43%** | **12.26%** |

These are different risk-coverage-savings operating points, so the full frontier—not a single bar
chart—is the primary comparison. The CBES-style result is a documented binary adaptation of the
original continuous Gaussian-process method, not a reproduction of every CBES formulation.

## Failure evidence

The aggregate result is not a universal guarantee. E007 produced 15 errors among 49 early stops
(30.61% EESR), Lens endpoint transfer reached 17.82% EESR with 6.43% savings, and irregular
sampling reached 10.61% EESR. The exploratory experiment-level gate helped at the frozen 5% point
but failed at 10%; it is not part of the principal method claim.

The 5.69% result is an empirical property of the complete frozen sequential policy. It is not an
anytime-valid, distribution-free, group-conditional, or deployment guarantee.

## Reproduce the frozen evidence

Requirements: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Git, and sufficient disk space for
the pinned 400,059,241-byte source table.

```bash
uv sync --frozen --extra dev
just reproduce
```

`just reproduce` downloads the exact Zenodo record and filename specified in
`data/manifests/retinal_sources.json`, verifies its size, MD5, and SHA-256, rebuilds the final
retrospective evidence, and runs lint, tests, evidence verification, and release-claim checks.

For a verification-only pass when the pinned dataset is already present:

```bash
just verify
just verify-final
```

## Repository map

```text
app/                       frozen-evidence Streamlit demo
artifacts/final/           headline tables, appendices, figures, and manifests
artifacts/phase2/          experiment-held-out predictions and decisions
docs/final/                report, Kaggle copy, video script, defense, erratum
protocols/                 pre-specified evaluation protocols
scripts/                   acquisition, analysis, audit, and verification commands
src/gates/                 features, models, calibration, refusal, policy, evaluation
tests/                     leakage, policy, evidence, audit, and integrity tests
```

## Scope, data, and licensing

This is retrospective research software, not a wet-lab or clinical instruction. STOP means that
the frozen research rule passed on a held-out trajectory; it does not mean an experiment should be
terminated without prospective validation and human oversight. Retinal organoids are used as a
longitudinal biological testbed and are not organ-on-chip experiments. Direct organ-on-chip and
independent-laboratory validation remain future work.

The primary source table is published by Afting et al. under CC BY 4.0 and is downloaded from the
pinned Zenodo record rather than redistributed in Git. Project code is released under the MIT
License. See the [technical report](docs/final/TECHNICAL_REPORT.md),
[CBES adaptation note](docs/phase2/CBES_ADAPTATION.md), and transparent
[frozen-evidence erratum](docs/final/ERRATUM.md).
