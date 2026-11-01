# GATES-ACT calibrated decision-triage protocol

Locked: 2026-10-24, before fitting or scoring the decision models.

## Objective and evidence boundary

Test whether a calibrated decision layer can convert the **fixed, frozen conditional
maximum-variance acquisition path** into substantially more precise biological-effect decisions at
the same 8/12/16-cell chronological budgets.

This is repeated nested development evidence, not fresh external validation. Outer evaluation is
restricted to the 99 groups assigned to `train` by the existing v3 loaders. The 27 groups assigned
to `development` may support fold-local response priors but are never scored. Original calibration,
original test, Brown, Tipping Point, and all 76 previously examined confirmation groups are sealed.
No decision threshold or model may use those outcomes.

The full-series mean-firing-rate (MFR) cell call, absolute matched-vehicle log1p contrast >= 0.40,
is the assay reference. It is not independently established developmental neurotoxicity.

## Transfer from Three-Pointer

The public Kaggle entry “More screening decisions from fewer measurements” was inspected at its
2026-10-24 public repository state. Transferable ideas are independently reimplemented, not copied:

1. prediction, selective release and acquisition cost are evaluated separately;
2. directly measured threshold crossings are confirmed by construction;
3. training-only score models and group-level calibration choose release thresholds;
4. unresolved cases remain unresolved rather than being forced into a binary call;
5. utility is compared at matched measurement budget and matched biological recall.

GATES-ACT retains its existing dose-time model and acquisition rule. It does not adopt AnchorBoost,
its feature code, its presentation, or its activity definition.

## Inputs and chronology

Each measured MEA well yields MFR plus seven simultaneously recorded, complete endpoints in both
EPA releases: burst rate, active electrodes, bursting electrodes, percent spikes in bursts,
network-spike count, percent spikes in network spikes, and correlation. Duration/interval endpoints
and AB/LDH are excluded because development-pool completeness is only 52–74% and about 25%,
respectively.

All endpoints are plate/DIV matched-control log1p contrasts except correlation, which is a raw
matched-control difference because it can be negative. Endpoint scales are learned from fold-local
training runs only. No value from an unmeasured dose-time cell is used as a feature.

Budgets are the existing chronological maximum-variance paths:

- DIV 5: all eight dose ranks, 8 cells;
- DIV 7: vehicle plus three selected treated doses, 12 cumulative cells;
- DIV 9: vehicle plus three selected treated doses, 16 cumulative cells.

The primary budget is 12 cells. Sixteen cells are secondary. The policy never stops an assay.

## Candidate models and features

Every hidden cell receives fold-local features available at decision time:

- posterior MFR mean, standard deviation and effect probability;
- dose/time coordinates and distance to the nearest observed cell;
- same-DIV and adjacent-DIV predicted dose consistency;
- observed MFR magnitude, effect count and longitudinal support;
- local and compound-level standardized magnitude/agreement across the seven auxiliary endpoints.

Two prespecified classifiers are compared by inner grouped cross-validation: L2 logistic regression
and histogram gradient boosting. Three feature ablations are evaluated: posterior only;
posterior plus dose/longitudinal consistency; and the full multivariate feature set. Inner grouped
average precision selects one candidate; ties favor the simpler feature set and then logistic
regression. Probability calibration is selected from uncalibrated, Platt and isotonic mappings by
inner grouped Brier score.

## Decisions

The binary primary rule chooses a probability threshold on inner out-of-fold groups to maximize
precision subject to hidden-cell effect recall >= 0.95. It is evaluated against the frozen canonical
conformal rule and simple posterior-mean/probability rules at the same cells.

The operational three-way rule uses two thresholds:

- `CONFIRMED_EFFECT`: directly observed MFR threshold crossing, or hidden-cell probability >= high;
- `NO_EFFECT_DETECTED`: hidden-cell probability <= low;
- `NEEDS_MORE_EVIDENCE`: all remaining hidden cells.

The high threshold is the lowest grid value whose group-average false-discovery proportion passes
the finite-sample corrected 40% risk target. The low threshold is the highest value whose
group-average missed-effect proportion passes the corrected 5% risk target. If either feasible set
is empty, that action is disabled. Thresholds are chosen on inner out-of-fold groups only.

## Metrics

At each matched budget, report group-clustered point estimates and 95% bootstrap intervals for:

- hidden-effect recall, precision, specificity and false discoveries;
- three-way protected recall (an effect is protected unless called `NO_EFFECT_DETECTED`);
- confirmed-effect precision;
- decisive coverage and unresolved coverage;
- correct and wrong decisive decisions per 100 hidden cells;
- directly observed effect cells and direct-effect capture;
- group-average decision utility, with false negatives costing five and false positives one.

Baselines are frozen canonical conformal protection, posterior mean thresholding, posterior
probability thresholding, direct-observed-only, and simple dose-supported confirmation.

## GO/NO-GO rule

The 12-cell result is a `GO` only if every condition holds out of fold:

1. binary hidden-effect precision >= 0.60 and recall >= 0.95;
2. precision improves by at least 0.20 over the canonical conformal rule, with the 95% grouped
   bootstrap interval for the paired gain entirely above zero;
3. recall is no more than 0.02 below the canonical rule;
4. the three-way rule has protected recall >= 0.95, confirmed precision >= 0.60, and decisive
   coverage >= 0.50;
5. wrong decisive decisions per 100 cells are lower than canonical binary decisions, with a paired
   grouped-bootstrap interval entirely below zero;
6. precision improves directionally in at least four of five outer folds;
7. no gain relies on physical-session savings, external toxicity labels, or the sealed 76 groups.

Secondary 16-cell results, feature ablations, alternate thresholds and organ-on-chip analyses cannot
rescue a failed primary gate. A passing nested result may justify shadow-mode integration but is not
an external-validation claim. A failed gate leaves the canonical system and all presentation files
unchanged.

## Organ-on-chip boundary

An organ-on-chip dataset is eligible only if it exposes per-drug longitudinal or dose-series
responses, independent drug/specimen identifiers, a legal public source, and enough groups for a
separate decision-level test. Imaging quality labels are not paired biological-effect truth. Small
five- or six-drug liver-chip cohorts and digitized data with unresolved redistribution terms may be
audited but cannot validate or rescue the MEA decision claim.
