# GATES feasibility writeup — superseded for submission

Use the current [Kaggle writeup](final/KAGGLE_WRITEUP.md). This file remains as historical
21-unit feasibility material.

## The problem

Longitudinal in-vitro experiments often collect every scheduled observation even after the final
endpoint has become predictable. A fixed early cutoff can reduce cost, but it treats every sample as
equally easy and can hide premature wrong decisions.

## Our solution

GATES watches a trajectory as observations arrive and returns one of three actions:

- **CONTINUE** when the endpoint call is not yet confident enough;
- **STOP** when a threshold learned on a separate calibration replicate passes;
- **ABSTAIN** when a feature-distance gate marks the sample as unfamiliar.

The classifier is deliberately simple. The scientific contribution is the group-aware sequential
decision workflow, transparent refusal behavior, and an explicit risk–observation tradeoff.

## Real-data feasibility result

We evaluated GATES on the public OrganoID gemcitabine time course: 21 endpoint-labelled units,
19 scheduled observations from 0 to 72 hours, and independently measured MTS viability. Complete
replicates rotate through train, calibration, and test roles. No frame or prefix from a held-out unit
is used to fit its model or threshold.

With automated stopping disabled before hour 32, the frozen held-out evidence shows:

- 20 of 21 units receive an early STOP;
- 10% observed error among early stops;
- 53.6% of scheduled observations saved on average among stopped units;
- mean decision time of 33.4 hours instead of 72 hours.

These figures are an empirical feasibility result, not a certified low-risk guarantee. Seven units
per calibration role are too few to support a strong conservative confidence bound, and GATES says
so in the saved artifacts and interface.

## Why the evaluation matters

At hour 0 the model is near chance. Predictive quality rises materially after 20 hours and reaches
perfect held-out AUROC by hour 44 in this small dataset. The stopping boundary is therefore tied to
the observed predictive regime rather than chosen to maximize an attractive savings number.

## Reproducibility

The repository contains a locked environment, exact data URLs and checksums, deterministic cyclic
splits, model artifacts, per-time predictions, calibration records, decision logs, leakage tests,
and an evidence-only Streamlit demo. Every headline number can be recomputed from CSV/JSON outputs.

## Limitations and next experiment

The dataset is small, shares dose conditions across replicates, and does not represent an independent
laboratory or donor. The next gate is a larger longitudinal organoid dataset with many independent
experiments, allowing stronger calibration, a sealed test set, and prospective shift stress tests.
