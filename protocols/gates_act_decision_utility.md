# GATES-ACT decision-utility study protocol

Locked: 2026-10-24, before fitting or evaluating the proposed policy.

## Question and evidence boundary

Can a decision-aware acquisition rule issue a correct chemical-level neural-network-formation
assay warning earlier than strong acquisition baselines, with fewer measured dose-time cells and
fewer false alarms?

This is a nested development study, not a new external validation. It uses only compound groups
previously assigned to the legacy or Phase-1 training/development partitions. **Outer evaluation is
restricted to groups originally assigned to `train`**; groups previously assigned to `development`
may support fold-local fitting but are never scored because they influenced the frozen v3 model
configuration. Original calibration,
original test, Brown, Tipping Point, and the 76 previously examined confirmation groups are sealed
and excluded from model selection, calibration, threshold selection, and primary evaluation. No
result from this study will be described as independently externally validated.

## Biological estimand

The reference is a **final-assay NFA activity call**, not independently established biological or
in-vivo developmental neurotoxicity. A compound run is reference-active when treated responses
contain at least two cells with absolute matched-vehicle log1p contrast >= 0.40 on each of at least
two distinct DIVs. A compound group is active if any replicate run is active. Vehicle cells are
never counted as effects.

The early-warning action is `ALERT` or `NO_ALERT`; it never terminates the assay. All deployment
stopping gates remain disabled.

## Chronology and budgets

DIV 5 is the mandatory initialization batch (vehicle plus seven treated doses; 8 observations).
At DIV 7, DIV 9, and DIV 12, each policy receives the vehicle anchor plus three treated-dose
measurements. Primary evaluation is after DIV 7 at a matched budget of 12 observations. DIV 9
(16 observations) is secondary. Full-assay calls use all 32 cells only to construct references.
No policy may acquire a future-DIV cell.

## Validation

Five deterministic outer folds are assigned among original-training groups by canonical
compound/DTXSID, keeping every repeat, plate, dose, and timepoint together. For each outer fold,
the prior-development groups and the other four original-training folds may be used for fitting;
priors, analogs, probability calibration, alert threshold, and any candidate-policy hyperparameters
are fit without the outer-evaluation groups.
Threshold selection is by inner grouped cross-validation and cannot inspect the outer fold.

All reported uncertainty intervals resample compound groups, not cells or runs. Multiple runs are
aggregated to one group-level decision before scoring.

## Methods

Candidate: posterior analog ensemble with a cost-sensitive value-of-information (VOI) acquisition.
The score is expected reduction in asymmetric Bayes decision risk for the final-assay activity call,
subject to the chronological batch constraint. False negatives cost 5 and false positives cost 1.
Probability calibration is selected inside each outer-training fold from isotonic regression,
logistic calibration, and the uncalibrated posterior using grouped inner validation and Brier score.

Matched-budget acquisition baselines:

1. conditional maximum posterior variance (current canonical policy),
2. fixed sentinels,
3. maximin space filling,
4. random acquisition with deterministic seeds,
5. the prior effect-weighted risk-sensitive policy.

Decision baselines:

1. any-observed-effect early rule,
2. two-observed-effect early rule,
3. prior-prevalence prediction.

Every acquisition policy uses the same fold-specific decision model and calibration machinery; only
the selected dose cells differ.

## Primary metrics and go/no-go rule

At 12 observations report group-level sensitivity, precision, specificity, balanced accuracy,
asymmetric decision loss, Brier score, and false alerts. Secondary metrics are the same at 16
observations plus time to first correct alert for reference-active groups.

The candidate is a `GO` only if all conditions hold on pooled out-of-fold predictions:

1. sensitivity is at least 0.90 and no more than 0.03 below conditional maximum variance;
2. precision improves by at least 0.10 and specificity by at least 0.10 versus conditional maximum
   variance at 12 observations;
3. asymmetric decision loss is lower than conditional maximum variance and the 95% grouped
   bootstrap interval for the paired loss improvement excludes zero;
4. the improvement is directionally consistent in at least four of five outer folds;
5. no reported gain depends on counting physical-well or recording-session savings that the replay
   does not establish.

If any condition fails, the canonical acquisition system and presentation centerpiece remain
unchanged. Negative and null findings are retained as research evidence.

## Multiplicity and interpretation

The comparison with conditional maximum variance at 12 observations is primary. Other baselines,
the 16-observation budget, alternative calibrators, and acquisition ablations are secondary and
cannot rescue a failed primary gate. This protocol does not authorize early stopping and does not
claim clinical, in-vivo, or regulatory toxicity classification.
