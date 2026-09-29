# GATES feasibility report — superseded for submission

Use the [final technical report](final/TECHNICAL_REPORT.md). This file remains as historical
21-unit feasibility material.

## Abstract

GATES is a model-agnostic decision layer for longitudinal in-vitro experiments. At each scheduled
observation it estimates a final endpoint, evaluates confidence against a threshold learned on a
separate replicate, checks feature-space shift, and returns STOP, CONTINUE, or ABSTAIN. The initial
feasibility study uses the public OrganoID gemcitabine time course and a strict cyclic
train/calibration/test split by endpoint replicate. All reported results are regenerated from
machine-readable held-out evidence. Because only 21 endpoint-labelled units are available, the
study reports empirical stopping risk and explicitly withholds a strong finite-sample control claim.

## Problem and practical motivation

Longitudinal assays often acquire every scheduled observation even when the eventual endpoint is
already predictable. A fixed early cutoff can save time but does not adapt to easy, difficult, or
shifted cases. GATES treats observation as a sequential decision: stop only when the endpoint call
meets a calibrated criterion, keep observing when it does not, and refuse early automation when the
sample lies outside the fitted feature distribution.

## Dataset

The feasibility dataset contains propidium iodide fluorescence and image-derived organoid
morphology sampled every four hours from 0 to 72 h. MTS viability is the independently measured
endpoint. The join shared by the trajectories and endpoint table contains 21 dosage–replicate
units: seven doses and three endpoint replicates per dose. No individual frame is an evaluation
unit.

## Method

For every decision time, leakage-safe prefix features summarize only observations at or before that
time. Each signal contributes its latest value, mean, change from baseline, and linear slope. A
regularized logistic model predicts whether endpoint viability is below 0.5. The three replicates
rotate through train, calibration, and test roles so each unit is tested once. The confidence
threshold is selected on the calibration replicate. A standardized nearest-neighbor distance gate
fitted on the training replicate marks shifted test prefixes. Its cutoff is derived from
leave-one-training-unit-out distances with a conservative margin.

The policy does not permit automated stopping before hour 32, the predefined feasibility boundary
at which the held-out fixed-time curve enters a useful predictive regime. After that boundary, the
system stops when the predicted-class probability exceeds the calibrated threshold and the shift
gate passes. It continues otherwise. A shifted prefix produces ABSTAIN for early automation.
The threshold procedure records both empirical calibration risk and a one-sided exact binomial
upper bound. When the bound cannot support the requested target, the artifact is labelled empirical
only rather than certified.

## Evaluation

The primary safety metric is erroneous early-stop rate, measured among held-out units that stop
early. Efficiency is captured by early-stop coverage, mean decision time, and scheduled observations
saved. Fixed-time endpoint quality is reported using balanced accuracy, AUROC, and Brier score.
Every headline number is computed from `artifacts/evidence/decisions.csv`; per-time predictions and
calibration details remain available for audit.

## Interpretation

This run is a 24-hour feasibility gate. A positive result means the real-data pipeline yields a
nontrivial performance-versus-time or risk-versus-savings signal without frame leakage. It does not
establish generalization across laboratories, instruments, donors, or biological systems. The next
scientific gate is a larger second dataset with many independent experiments and enough calibration
units to support a conservative risk statement.

## Reproducibility

The environment is locked by `uv.lock`. Source URLs and hashes live in the data manifest. The run
configuration is JSON, splits are deterministic, models and evidence are serialized by fold and
decision time, and tests cover unit separation, future-information exclusion, stopping behavior,
and artifact consistency.
