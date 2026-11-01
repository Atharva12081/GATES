# GATES-ACT competitor-transfer research protocol

Locked: 2026-10-26, before fitting or scoring any transferred model.

## Objective and evidence boundary

Determine whether mechanisms from the three leading public AI4S systems materially improve
GATES-ACT on matched neural-MEA forecasting and laboratory-decision tasks. The stable canonical
release remains unchanged while this protocol runs.

**Pre-score census amendment (2026-10-26).** The locked text initially said 99 `train` compound
groups. A loader census performed before any transferred model was fitted or scored found 106 eligible
`train` runs (not 99 groups): 70 legacy-EPA and 36 phase-1 runs. Grouping remains by canonical compound
identity, so replicate/source runs of the same compound never cross a fold. These 106 runs are the only
scored GATES development population. The 27 existing `development` runs may support fold-local priors
but are not scored. Original calibration,
test, Brown, Tipping Point and all 76 previously examined confirmation groups remain sealed during
model development and selection. Any later use of those groups is retrospective confirmation, never
fresh evidence, and cannot rescue a failed nested-development gate.

The public competitor repositories are inspected at these commits:

- Three-Pointer, `logxio/ooc-evaluation-audit`: `6c5007237cf29031cb670ef952ec2974c5f3235e`;
- NeuroChip Twin, `Agnuxo1/neurochip-twin`: `f9848800dfab66a8bc005e6b3087153eeaabe9ac`;
- NeuroForecast, `SaltyTaro/neuroforecast`: `3c2f55281d06703a4726fe349c59b6419611dd54`.

All three codebases are MIT licensed. Their published results are reproduced from their own evidence
before transfer. Competitor results, code and scientific contributions retain explicit attribution.
Transferred mechanisms are independently implemented unless a copied component is explicitly
identified with its copyright and licence notice.

## Public-result reproduction

Run each repository's published-number or bundle verifier. Reproduction establishes that the public
artifact is internally consistent; it does not independently validate its data-generating process or
turn its public labels into a blind test.

The matched public reference is Three-Pointer's k=3 NFA benchmark. Preserve its task definition,
folds, chemical weighting and paired identity-group bootstrap. Do not compare unmatched metrics or
call a different normalization a leaderboard win.

## H1: exhaustive sparse-context residual forecasting

**Mechanism.** Three-Pointer attributes part of AnchorBoost's gain to training on every sparse
concentration design. Test that mechanism on the GATES MFR surface without copying its source.

Each context contains vehicle plus three treated dose ranks measured at all four DIVs: 16 observed
dose-time cells. All `C(7,3)=35` treated-dose contexts are training examples. Targets are the remaining
16 treated dose-time cells. Test groups are scored across all contexts and averaged first within run,
then within compound group.

Prespecified methods:

1. log-dose interpolation independently at each DIV;
2. frozen GATES covariance/analog surface;
3. interpolation-anchored histogram-gradient-boosted residual model;
4. GATES-anchored residual model with the same learner and context summaries.

Model choice between methods 3 and 4 uses inner compound-group cross-validation only. Hyperparameters
are fixed from the public AnchorBoost recipe, reduced only if required by memory limits and recorded.

H1 passes only if the selected candidate, on outer out-of-fold groups:

- reduces group-weighted MAE by at least 5% relative to the strongest prespecified baseline;
- has a paired group-bootstrap 95% interval for the MAE difference wholly below zero;
- improves in at least four of five outer folds;
- does not reduce hidden-effect recall by more than two percentage points at the comparator's operating
  threshold.

## H2: chronological residual forecasting and decisions

Run only if H1 passes. Retain the frozen conditional maximum-variance acquisition path at 12 and 16
measurements. Training contexts enumerate every physically legal choice of three treated wells at the
current DIV after all earlier acquired cells. No later-DIV value may enter a feature.

Compare the accepted H1 learner, GATES alone, interpolation/carry-forward, and Extra Trees inspired by
NeuroForecast. Decision thresholds and conformal scales are learned on inner groups. Report hidden-cell
MAE/NRMSE, effect sensitivity, precision, specificity, false discoveries, missed effects, three-way
coverage, correct and wrong decisions per 100 cells, and direct observation capture.

H2 passes only if, at 12 measurements, the candidate preserves at least 95% hidden-effect recall while
improving precision by at least 10 percentage points over both GATES and the strongest simple rule, with
a paired group-bootstrap interval above zero. It must also lower five-to-one false-negative-weighted
decision loss with an interval below zero and improve at least four of five folds.

## H1b: identical-task public NFA transfer

H1's residual learner failed its gate. Before pursuing a new GATES mechanism, perform a diagnostic
head-to-head on Three-Pointer's own public k=3 NFA benchmark. Restrict prospectively to chemicals with
exactly seven concentration levels so a rank-grid GATES surface is well-defined without extrapolating
or fabricating levels. Preserve the public folds, five deterministic contexts per chemical, transformed
outputs and curve MAE. Fit the frozen GATES covariance/analog estimator on other-fold chemicals only;
missing training outputs are imputed by fold-local cell medians. Compare against the released
per-chemical AnchorBoost and interpolation results with paired chemical bootstrap intervals.

This diagnostic cannot change the canonical release: it tests task transferability, not GATES-ACT's
chronological laboratory workflow. A win requires a lower mean curve MAE than AnchorBoost with a paired
95% interval below zero and improvement in at least four held-out folds. A loss establishes that
AnchorBoost should remain the preferred method for this public sparse-concentration task.

## H3: disagreement acquisition and calibrated selective release

Run only if H2 improves forecasting. Use disagreement among the accepted residual model, GATES and
interpolation as a value-of-information feature. Compare at identical chronological budgets with
conditional maximum variance, maximin, fixed sentinels and seeded random acquisition. Acquisition may
be adopted only if it lowers group decision loss with a paired 95% interval below zero and is not worse
in effect recall or direct-effect capture.

Selective release uses group-disjoint conformal calibration. `NEEDS_MORE_EVIDENCE` is an abstention,
not a correct decision. Report risk, decisive coverage and unresolved cases together. Early stopping
remains disabled regardless of result.

## H4: batch/source shift

For GATES data, treat the two EPA releases as source-shift strata and keep compound identities disjoint.
For public competitor reproduction, preserve each repository's stated batch/chemical split. Report
source-specific calibration, worst-group loss and interval coverage. Do not describe release shift as
independent laboratory validation.

## H6: cross-fitted model-agreement triage (exploratory amendment)

Added after H1 and H1b failed, and therefore labelled exploratory rather than preregistered. The H1
out-of-fold predictions showed complementary behavior: the residual model increased effect recall but
lost MAE, while frozen GATES had lower MAE and higher effect-call precision. Test a minimal three-way
rule without refitting either point model. `CONFIRMED EFFECT` requires both models to exceed the fixed
0.40 biological threshold with the same sign. `NO EFFECT DETECTED` requires both absolute predictions
to fall below a low-signal margin. The margin is learned for each held-out outer fold from the other
compound groups only, choosing the largest value whose mean group-level false-negative fraction is at
most 5%. All other cells are `NEEDS MORE EVIDENCE`.

Compare with single-model GATES, residual and interpolation triage under the same cross-fitted 95%
protected-recall calibration, both across all 35 virtual sparse designs and at the prespecified maximin
design (treated ranks 1, 4 and 7). Report effect precision, protected recall, decisive coverage,
decision accuracy, wrong decisions and unresolved cases with compound bootstrap intervals. Repeated
virtual designs are scenarios, not additional biological replicates. Every scenario costs 16 of 32
dose-time cells; this is a 50% response-cell budget, not a claim of 50% wall-clock or plate savings.
Early stopping remains disabled.

**Post-result acquisition diagnostic.** After the maximin analysis, apply the already-frozen GATES
conditional maximum-variance rule to select three entire dose trajectories from fold-local training
covariance, without test responses. This extension was motivated after inspecting H6 and is therefore
post-result exploratory. It can support a future confirmation protocol but cannot by itself authorize
canonical integration. Require a paired compound-bootstrap precision gain above zero, no protected-
recall loss, no coverage loss and fewer wrong decisions before considering prospective confirmation.

## H5: compatible biological datasets

An additional organ-on-chip or organoid dataset is eligible only if it supplies public legal access,
independent specimen or experiment identifiers, compound dose/time responses, compatible biological
effect truth, and a defensible physical measurement-cost unit. Image-quality, segmentation, terminal
omics, five/six-drug demonstrations and archives without paired MEA outcomes cannot validate this task.

## Sequential decision rule

Stop a hypothesis when its gate fails, preserve its evidence, and proceed to the next mechanism only
when the protocol permits. Do not tune on outer-fold outcomes. A candidate changes the canonical release
only after it passes its own gate, survives the shift audit, and is independently recomputed from saved
out-of-fold predictions. Otherwise the canonical model and all presentation artifacts remain unchanged.

