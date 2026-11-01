# GATES-ACT high-dose mechanistic falsification protocol

Locked: 2026-10-28 before computing the direct-only, threshold-rule, or chronological-policy outcomes.

## Evidence boundary

Use only the 99 previously examined development compound groups and their existing five outer folds.
This is repeated nested development evidence, never fresh validation. The 76 exposed confirmation groups,
Brown and Tipping outcomes, and every external assay are excluded. The canonical acquisition policy and
stopping prohibition remain unchanged.

The assay-reference effect remains the frozen per-cell absolute matched-control log1p contrast of at
least 0.40. It is an assay response, not independently established toxicity. Every comparison uses 12
treated measurements plus four same-day vehicle controls per run: 16/32 physical assay cells.

## Falsification layers on the fixed high-dose design

The acquisition design is treated ranks `(5, 6, 7)` at each of four assay days. Compare:

1. `direct_only`: observed effect crossings are confirmed effects and observed non-crossings are direct
   no-effect decisions. Every hidden cell abstains.
2. `adjacent_boundary_rule`: direct decisions plus a hidden rank-4 effect call when observed ranks 5 and
   6 on the same day both cross 0.40 with the same sign. Other hidden cells abstain.
3. `strict_high_dose_rule`: the boundary effect rule plus hidden ranks 1--4 called no-effect only when
   all observed ranks 5--7 on that day have absolute response below 0.20, half the frozen effect
   threshold. No outcome-dependent threshold selection is allowed.
4. `gates_agreement_release`: the already frozen same-sign GATES/residual agreement effect call and
   outer-fold recall-calibrated no-effect margin, with direct observations decided exactly.

Report confirmed-effect precision, total effect recall, protected recall, decisive coverage, selective
risk, wrong decisions, missed effects, direct and inferred true effects, active-run sensitivity and
precision, and results by source.

The incremental AI contribution is layer 4 minus layer 1 and, separately, layer 4 minus the strongest
simple threshold layer. Report absolute count differences as well as percentage-point differences.
The AI discovery gate requires at least +3 percentage points total effect recall, at least one additional
correctly identified active run group, no lower precision, and no higher selective risk than direct-only.
Failure means the development gain is attributable primarily to dose placement rather than modeling.

## Chronologically feasible acquisition hypothesis

Evaluate `high_anchor_temporal_boundary` without selecting any elapsed cell:

- Day 0 measures vehicle and treated ranks `(5, 6, 7)`.
- Each later day always measures vehicle and ranks 6 and 7.
- The third treated rank is 4 when either rank 6 or 7 crossed the 0.40 effect threshold on the immediately
  preceding day, or when the absolute change at either common anchor since the preceding day is at least
  0.20. Otherwise it is rank 5.
- At day 1, only the preceding-day effect condition is available. At days 2 and 3, change is computed
  solely from measurements already made at the two preceding days.

The selector may use only measurements with time rank strictly lower than the batch it selects. It must
record the selection time and evidence cells. Each day consumes exactly one vehicle plus three treated
measurements, matching fixed-policy timing and total cost.

Compare with frozen conditional maximum variance, fixed `(5,6,7)`, middle-high `(3,5,7)`, and maximin
`(1,4,7)`. Fit fold-local GATES and residual models without the held compound groups. For the adaptive
pattern, calibrate hidden no-effect margins on other outer folds only. Apply the same direct-observation,
agreement, and abstention rules to all policies.

The chronological candidate gate requires total effect recall at least 3 points above fixed high-dose,
no lower active-run sensitivity, no lower precision or protected recall, no higher selective risk, and
strict per-day and total budget parity. A failed gate preserves the fixed high-dose design as the stronger
development hypothesis and cannot change the canonical release.
