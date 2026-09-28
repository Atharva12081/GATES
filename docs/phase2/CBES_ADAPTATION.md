# CBES-style baseline adaptation

## Original decision principle

CBES uses an ensemble/Gaussian-process predictive mean and uncertainty to make a
one-sided early decision only when a confidence bound clears the endpoint boundary.
Candidate observation times are screened on an independent calibration set, then
the earliest candidates are pruned until the *sequential* calibration false-stop
rate meets the requested alpha. A separate validation set supplies the reported
Hoeffding uncertainty term.

Primary reference: Liu et al., *Confidence-bound early stopping for efficient
bioprocess development*, Chemical Engineering Research and Design (2026), DOI
`10.1016/j.cherd.2026.05.013`. Reference implementation:
<https://github.com/IroayX/CBES>.

## What is reproduced directly

- predictive mean and standard deviation from an ensemble;
- two-sided confidence-bound decision around the binary boundary at 0.5;
- per-time calibration eligibility;
- sequential pruning of early eligible times;
- an independent validation experiment and Hoeffding term;
- the same target-risk grid used for GATES.

## Necessary adaptation

The public retinal data provide a binary final morphology endpoint rather than the
continuous bioprocess quality variable and Gaussian-process setup in the original
paper. We therefore use bootstrap regularized-logistic ensembles and symmetric
confidence bounds: predict positive when `mean - 1.96 * std >= 0.5`, predict
negative when `mean + 1.96 * std < 0.5`, otherwise continue. Organoids not yet
observed at a candidate time are ineligible rather than silently removed from the
sequential denominator.

## Comparison parity

For every held-out experiment, CBES-style and GATES receive the identical source
features and candidate observation times. They use the same test experiment, base
training experiments, and predefined risk grid. The two calibration experiments
are split for CBES into sequential calibration and independent validation because
that is required by its procedure. GATES uses both as group-aware calibration
groups. Neither method uses the held-out experiment for preprocessing, fitting,
threshold selection, OOD thresholding, or stopping-rule selection.

This is a serious binary adaptation, not a claim of byte-for-byte reproduction of
the original continuous GP algorithm. Its validation bound is reported separately
from empirical held-out EESR and must not be described as a clinical or universal
guarantee.
