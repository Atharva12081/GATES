# GATES-ACT direct-discovery benchmark protocol

Locked: 2026-10-26 before evaluating no-effect decisions, selective risk, coverage, or uncertainty
for the direct-discovery policies. Direct-effect enrichment by the highest-dose context was already
observed during post-hoc mechanism diagnosis and is therefore explicitly development evidence.

## Evidence boundary

Use the 99 previously examined GATES-ACT development compound groups only. The 76 confirmation groups,
Brown and Tipping evaluation outcomes, and Saavedra acute outcomes are excluded. This benchmark cannot
become independent confirmation. It tests whether a simple acquisition principle is worth future
prospective validation; it cannot authorize canonical integration.

The final-assay reference is the frozen per-cell absolute matched-control log1p contrast >= 0.40. It is
an assay response, not independently established biological toxicity. Early stopping remains disabled.

## Matched acquisition policies

Every policy measures exactly 16 of 32 response cells: four shared vehicle controls plus three treated
dose ranks at all four assay days. The decision tables contain the 28 treated response cells only, so
their matched accounting is 12 observed and 16 hidden treated cells per run; the four vehicle cells are
common calibration inputs and are excluded from the effect-decision denominator. No policy may choose a
different number of cells.

1. `conditional_max_variance`: frozen fold-specific GATES-ACT baseline contexts.
2. `highest_three_doses`: fixed treated ranks `(5, 6, 7)`; primary hypothesis.
3. `maximin`: fixed ranks `(1, 4, 7)`.
4. `middle_high_sentinels`: fixed ranks `(3, 5, 7)` as a prespecified robustness comparator.
5. `random`: exact enumeration of all 35 three-of-seven contexts; report the mean and range, never the
   best context as a selected method.

Three-Pointer is not an exact comparator here: its released AnchorBoost benchmark removes concentrations
from an acute/sparse curve and does not model chronological dose-time acquisition. Its direct-measurement
accounting is transferred and attributed, but no head-to-head superiority claim is permitted.

## Decisions and calibration

- Every observed cell is decided directly: threshold crossings are confirmed effects and non-crossings
  are observed no-effect responses.
- Hidden cells use the already frozen same-sign GATES/residual agreement rule for effect confirmation.
- Hidden no-effect margins are calibrated separately for each outer fold and acquisition context using
  only other outer folds. The group-average hidden false-negative fraction target remains 5%.
- All other hidden cells abstain as `NEEDS_MORE_EVIDENCE`.

This makes measurement cost, direct observations, inferred decisions, and unresolved cells explicit.
Calibration may not use the held fold, source label, or policy's held outcomes.

## Primary metrics and gate

Report cell-level confirmed-effect precision, total discovery recall (direct plus correct inferred
effects divided by all reference effects), protected recall, decisive coverage, selective risk, wrong
decisions, direct-effect capture, and observed/hidden treated-decision counts. Also report compound-run activity
sensitivity and precision and every metric by assay source.

Use a 10,000-replicate compound-group cluster bootstrap for `highest_three_doses` minus frozen conditional
maximum variance. The research gate requires all of:

1. total discovery recall improves by at least 3 percentage points and its bootstrap 95% interval is
   entirely above zero;
2. confirmed-effect precision is no lower;
3. selective risk is no higher;
4. compound-run activity sensitivity is higher;
5. discovery recall is no lower in either assay source; and
6. the observed response-cell budget is exactly identical.

A pass establishes only a strong nested-development hypothesis. Canonical integration remains `NO-GO`
until a genuinely independent, provenance-complete assay reproduces the benefit.

## Locked H2: risk-budgeted abstention

Locked after H1 evaluation, before H2 execution. H1 showed that `highest_three_doses` had lower selective
risk but lower decisive coverage than conditional maximum variance. H2 changes no measurements and no
effect calls. For each held outer fold, use only the other folds to:

1. compute the frozen conditional-maximum-variance policy's recall-calibrated selective risk and
   protected recall; then
2. choose the highest-coverage `highest_three_doses` no-effect margin whose calibration selective risk
   does not exceed that baseline risk and whose protected recall is not lower.

Apply that margin once to the held fold. The H2 gate requires no lower total discovery recall or
confirmed precision than H1, held-out selective risk no higher than conditional maximum variance,
protected recall no lower than conditional maximum variance, and decisive coverage improved by at least
10 percentage points over H1. This is repeated nested development and cannot rescue the missing
independent confirmation.
