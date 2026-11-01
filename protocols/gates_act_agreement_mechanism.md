# GATES-ACT agreement-mechanism protocol

Locked: 2026-10-26, before running the decomposition and risk-coverage analysis.

## Evidence boundary

This study explains the already observed H6 development result. It does not create a new independent
test set. Use only the 99 compound-group out-of-fold predictions produced by the locked H1 pipeline and
only the fold-local conditional maximum-variance contexts already saved in
`h6_agreement_triage.json`. Do not refit either point model or change the biological effect threshold
of 0.40. All results are nested-development evidence. Early stopping remains disabled.

## Mechanism decomposition

For each hidden response cell, classify model state before examining truth:

1. neither model calls an effect;
2. GATES-only effect;
3. residual-only effect;
4. both call an effect with the same sign;
5. both call an effect with opposite signs.

Report counts, true-effect prevalence, false discoveries, signed error and absolute error by state.
Decompose the difference between agreement and GATES by compound group, dose rank, DIV, source release
and fixed absolute-effect bins `[0, .2), [.2, .4), [.4, .6), [.6, 1), [1, infinity)`. Repeated response
cells and virtual designs are not independent biological replicates. Compound group is the bootstrap
unit.

## Reliability and risk-coverage

Every method uses the same 16/32 response-cell context selected without test outcomes. Effect calls
remain fixed at absolute prediction >= 0.40. `NO EFFECT DETECTED` uses a low-signal margin learned from
other outer folds; all remaining cells are unresolved.

Prespecified protected-recall targets are 99%, 97.5%, 95%, 92.5% and 90%. For each held-out fold and
method, choose the largest margin on other compound groups whose mean group-level false-negative
fraction does not exceed `1 - target`. Prespecified decisive-coverage targets are 10%, 20%, 30%, 40%
and 50%; choose the calibration margin whose calibration coverage is closest to the target. Evaluate
the resulting margin once on the held-out fold. Report actual protected recall, effect precision,
decisive coverage, selective decision risk, wrong decisions and unresolved cases.

The primary mechanism comparison is agreement versus GATES at the 95% protected-recall target. A
mechanism is considered stable only if agreement has higher effect precision, no higher selective risk,
no lower protected recall and fewer wrong decisions in both EPA release strata. Release strata are a
source-shift stress test, not independent laboratories.

## Independent dataset eligibility

Audit candidate archives before any outcome testing. Record provenance, licence, plate and specimen
identity, compound overlap, concentration mapping, controls, treatment sequence, repeated time points
and accessible outcome grain.

The Kapucu archive (`10.12751/g-node.wvr3jf`) is eligible for a separate maturation-trajectory study
only: its hPSC_MEA1/hPSC_MEA2 and Rat_MEA1 plates are untreated longitudinal cultures. Its pharmacology
plates are acute single-DIV experiments and are ineligible for longitudinal compound-screening
confirmation. Do not pool untreated maturation and pharmacology or call plate wells independent
biological batches.

## Locked leave-source-out stress test

Added before running this stress test on 2026-10-26. This is a mechanistic robustness analysis on the
same development corpus, not new evidence. Run two directions: legacy EPA to phase-1 and phase-1 to
legacy EPA. In each direction, train the GATES prior and the fixed interpolation-residual model using
only the source domain. Reserve a deterministic one-fifth of source-domain training compound groups
for calibration; do not use their outcomes for either model. Select three complete dose trajectories
by conditional maximum variance using the source-only prior and no target outcomes. Evaluate all
target-source training groups once at 16/32 response cells.

The residual anchor is fixed to interpolation because all five pre-existing H1 outer folds selected
it. The effect threshold remains 0.40. Calibrate each method's no-effect margin to 95% group-level
protected recall on the source-domain calibration groups. Report precision, protected recall,
coverage, risk and wrong decisions. No direction-specific threshold or model search is permitted.
