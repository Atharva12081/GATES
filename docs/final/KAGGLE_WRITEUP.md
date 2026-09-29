# GATES: Group-Aware Time-Efficient Stopping

**Category: END-TO-END SYSTEM**

**Author:** Atharva Parande

## Demo Video

The exact 4:45 recording script is complete in `docs/final/VIDEO_SCRIPT.md`, but the recording is
not. The public video URL is not yet available; this is a submission blocker, and the Kaggle entry
must not be finalized until the recording is uploaded and verified without login.

## Code Repository

<https://github.com/Atharva12081/GATES>

## Project Summary

Most experimental AI asks what can be learned from measurements already collected. GATES asks a
different question: **is another measurement still necessary?** It converts longitudinal
biological endpoint prediction into a selective stopping decision. At each candidate time, the
system returns STOP when independently calibrated evidence supports an early endpoint call,
CONTINUE when more observation is needed, or ABSTAIN when the trajectory appears unfamiliar.

GATES was evaluated retrospectively on 988 medaka retinal organoids from 11 independent
experiments. All observations from each held-out experiment were excluded from preprocessing,
model fitting, calibration, and refusal-threshold fitting. At the frozen operating point, Full
GATES made 439 early decisions with 25 errors: **5.69% observed held-out erroneous-early-stop rate**
(unit-level exact 95% CI 3.72%-8.29%), **44.43% early-decision coverage**, and **12.26%
retrospectively estimated observation savings**.

Refusal was operational rather than decorative. Removing it increased observed error from 5.69%
to 8.60%, while estimated savings changed from 12.26% to 13.65%. The result is nevertheless not a
5% guarantee: uncertainty across only 11 experiments is wide, E007 reached 30.61% error among
early stops, Lens transfer failed, and irregular sampling degraded performance. GATES is validated
on retinal organoids as a longitudinal biological testbed, not directly on organ-on-chip systems.

The practical contribution is an auditable workflow for deciding whether continued acquisition is
justified. It integrates pinned public-data acquisition, experiment-disjoint evaluation,
calibration, refusal, sequential decisions, evidence logging, an interactive frozen-evidence demo,
and one-command reproduction. Prospective independent-laboratory and direct organ-on-chip
validation are the next experiments.

## What Is Distinct

Prediction is not the novelty: the source orgAInoid study already demonstrated early outcome
prediction. GATES converts longitudinal biological prediction into an experiment-disjoint
selective stopping problem, combining independent-group calibration with refusal and evaluating
the resulting policy through explicit risk-versus-observation-cost tradeoffs.

Experimental groups are not merely a train/test bookkeeping variable. E007 shows why: acceptable
aggregate performance can coexist with severe experiment-specific failure. The complete
experiment is therefore the fitting, calibration, evaluation, and resampling boundary.

## Main Result: Risk Versus Observation Savings

![Risk versus observation savings](../../artifacts/final/figures/01_rpe_risk_savings_frontier.png)

| Frozen 5% operating point | EESR | Coverage | Estimated savings |
|---|---:|---:|---:|
| Non-group calibration | 9.01% | 65.18% | 19.79% |
| CBES-style adaptation | 15.51% | 45.04% | 15.32% |
| GATES without refusal | 8.60% | 48.28% | 13.65% |
| **Full GATES** | **5.69%** | **44.43%** | **12.26%** |

Methods occupy different coverage and savings points; the frontier is the primary comparison.
At the frozen point, GATES permits early automation for 44.43% of organoids and deliberately
continues observation for the remainder rather than forcing a premature decision.

## Why Refusal Matters

![Refusal ablation](../../artifacts/final/figures/06_ood_refusal_ablation.png)

Under the frozen RPE evaluation, refusal reduced observed premature errors from 41/477 to 25/439,
while retrospectively estimated observation savings changed by 1.39 percentage points. The gate
is a training-derived feature-distance diagnostic, not proof that every shifted experiment will be
detected; it missed the important E007 and E012 failures.

## Failure Evidence

E007 produced 15 errors among 49 early stops (30.61%). Its erroneous stops did not cross the
frozen OOD threshold, supporting only a cautious diagnosis of possible conditional/concept shift.
Lens transfer reached 17.82% EESR with 6.43% savings. Irregular sampling reached 10.61%; feature
noise, half-feature removal, and altered class balance reached 7.37%, 7.92%, and 8.47%.

The experiment-level refusal extension prevented 17 errors at the 5% operating point, but at 10%
it accepted every unsafe experiment, refused every safe experiment, and worsened EESR to 16.34%.
It remains an optional research extension, not part of the principal claim.

## Reproducibility

```bash
git clone https://github.com/Atharva12081/GATES.git
cd GATES
uv sync --frozen --extra dev
just reproduce
```

The source record, filename, byte size, MD5, SHA-256, protocol hashes, environment lock, decisions,
confidence intervals, figure sources, and release artifacts are recorded. The Streamlit demo reads
frozen evidence and never retrains.

## Technical Report

The self-contained report source is `docs/final/TECHNICAL_REPORT.md`; the
[16-page submission PDF](https://github.com/Atharva12081/GATES/blob/main/docs/final/GATES_Technical_Report.pdf)
is publicly accessible in the repository.

## Optional Demo

The demo runs locally on CPU with `just demo`. Public hosting is optional and is not yet available;
the required video provides a complete demonstration and backup once uploaded.
