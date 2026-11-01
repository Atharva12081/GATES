# GATES-ACT source-shift controller protocol

Locked: 2026-10-26, before inspecting numerical values in the HESI supplementary workbooks or
running the controller experiment.

## Evidence boundaries

The canonical release stays unchanged and early stopping stays disabled. The 99-group H1 out-of-fold
prediction corpus is already examined development evidence. It may support a nested mechanistic study,
but cannot become independent confirmation. The 76 previously examined confirmation groups are not
used. No controller choice may be tuned on its held-out fold or source.

The Roberts et al. HESI article supplements are independently sourced, but are not eligible for the
controller benchmark. Publisher files `mmc2.xlsx` and `mmc3.xlsx` expose anonymous site-level means,
five concentrations, 39 endpoints and across-site mean/SD columns for 11 compounds. They omit plate,
well, replicate, culture batch, measurement time, baseline-normalization and site-to-laboratory mapping.
The human workbook is nominally five-site but its linopirdine sheet labels the fifth column `Site 4`
again instead of identifying a fifth site; the rat workbook has six. `mmc1.pptx` lists six named human
contributors and six named rat contributors while stating that seven human laboratories contributed.
The workbooks contain 11 compounds although the article reports 10 pro-convulsants and three negative
controls. Do not read numerical outcomes for modeling. Use the files only for a reproducible metadata,
integrity and missingness audit. The official EPA Clowder DOI record is a Shafer/EPA laboratory record,
not a verified seven-laboratory archive.

## Development hypotheses

Use only the locked conditional-maximum-variance context for each H1 outer fold at the same 16/32
response-cell budget and the saved GATES/interpolation-residual predictions. The biological effect
threshold remains absolute response >= 0.40.

Evaluate five prespecified policies:

1. `gates_global`: frozen GATES effect calls and a source-blind no-effect margin.
2. `agreement_global`: always require same-sign GATES/residual effect agreement and use a source-blind
   no-effect margin based on the maximum model magnitude.
3. `agreement_source_calibrated`: the same effect rule, but calibrate the no-effect margin only on other
   folds from the same assay source.
4. `source_blind_controller`: on other folds pooled across sources, activate agreement only when the
   GATES-only calls contain more false discoveries than true effects; otherwise retain GATES. Calibrate
   the no-effect margin on the same pooled calibration rows.
5. `source_aware_controller`: make the same activation decision separately by source and calibrate the
   no-effect margin only on other folds from that source. If a source/fold has fewer than five GATES-only
   calls, abstain on those calls rather than extrapolate a reliability estimate.

For every held-out fold, policy activation and margins are determined without that fold's outcomes.
No threshold grid is searched. The protected-recall target is 95% using the existing group-mean
false-negative definition. A policy passes the primary gate only if, relative to `gates_global`, it has:

- no lower worst-source protected recall;
- higher aggregate precision;
- no higher worst-source selective risk;
- fewer aggregate wrong decisions; and
- at least 25% decisive coverage.

Report aggregate and per-source effect precision, biological effect-call recall, protected recall,
specificity among decisive calls, decisive coverage, selective risk, wrong decisions, missed effects
and unresolved responses. Also evaluate a 30% matched-coverage calibration and compound-group bootstrap
intervals for precision, risk and wrong-decision differences. The primary result is development-only,
regardless of outcome.

## Operational interpretation

These rows are hidden dose-time response cells, not saved laboratory sessions. A lower wrong-decision
count demonstrates safer selective inference at a matched response-cell budget; it does not establish
chronological session savings. HESI is an acute concentration-response study and cannot validate the
chronological GATES-ACT workflow from the available supplements.
