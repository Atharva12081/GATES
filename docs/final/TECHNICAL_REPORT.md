# GATES: Group-Aware Time-Efficient Stopping for Longitudinal Retinal Organoids

## Abstract

Longitudinal biological experiments are commonly observed on a fixed schedule even when the final
outcome becomes predictable earlier. GATES asks when the available evidence is reliable enough to
stop observing. On 988 retinal organoids from 11 independent experiments (114,510 audited
observations), evaluated by holding out entire experiments, Full GATES produced 439 early decisions
with 25 errors: 5.69% observed erroneous-early-stop rate (EESR; exact 95% CI 3.72–8.29%), 44.43%
coverage, and 12.26% observation savings. Removing refusal increased EESR to 8.60%; a tested
CBES-style adaptation reached 15.51%. The result is empirical, not a universal guarantee. Several
experiments remained difficult, Lens transfer failed, and a retrospective experiment-level gate
was unstable across operating points.

## 1. Problem

Endpoint prediction asks what outcome will occur. A stopping system must also decide whether the
current evidence is reliable enough to end observation without creating an unacceptable premature
error. GATES therefore returns STOP, CONTINUE, or ABSTAIN and reports both error among early stops
and observations avoided.

## 2. Why fixed-duration observation is inefficient

A fixed schedule treats easy and difficult trajectories identically. Conversely, a fixed early
cutoff forces a decision even when evidence is weak or shifted. The empirical risk–savings frontier
shows that neither time nor confidence alone defines a uniformly safe cutoff across held-out
experiments.

## 3. Related work and CBES context

Early time-series classification and confidence-based early stopping motivate the sequential
decision framing. The comparison here is specifically to the frozen CBES-style adaptation in the
repository; it is not a claim against every CBES formulation. GATES differs by separating
experiment-held-out model fitting and calibration, using group-calibrated stopping thresholds, and
allowing explicit OOD refusal.

## 4. Data

The retrospective dataset is the CC-BY-4.0 orgAInoid retinal-morphometrics source table. It contains
988 organoid wells from 11 experiments, sampled up to 72 h at a nominal 0.5 h cadence. The audit
contains 405 RPE-positive and 583 RPE-negative organoids; the Lens endpoint contains 450 positive
and 538 negative organoids. These are dataset prevalence counts, distinct from the 25 errors among
439 Full GATES early stops reported below. The audit found 114,510 rows, no duplicate
experiment–well–loop rows, no missing final labels, and substantial
experiment-dependent trajectory incompleteness. The endpoint is the published final RPE morphology
label. No new biological data were collected.

## 5. Independent experiment splitting

The independent evaluation boundary is the experiment. In each rotation, one complete experiment
is held out; its wells, frames, prefixes, and labels are absent from preprocessing, predictor
fitting, calibration, and OOD-threshold fitting. The 11 rotations produce one held-out decision for
each of 988 organoids. Availability columns, identifiers, and final labels are excluded from
predictors.

## 6. GATES method

Leakage-safe prefix features summarize only measurements available by each decision time. A frozen
regularized classifier produces endpoint probabilities. Group-calibrated thresholds determine
whether confidence is sufficient for STOP. Otherwise the policy continues. A training-derived
feature-distance gate can refuse early automation; refused units are observed through 72 h.

## 7. Calibration and refusal

Calibration uses independent development experiments, never the held-out experiment. Reported risk
targets are operating-point labels, not certified guarantees. At 5%, the unit-level refusal layer
changed 477 early stops and 41 errors without refusal to 439 early stops and 25 errors with refusal.
EESR fell from 8.60% to 5.69%; savings fell from 13.65% to 12.26%.

## 8. Evaluation protocol and metrics

The primary metric is EESR: incorrect final endpoint decisions among units stopped early. Secondary
metrics are coverage, abstention, observation savings, decision time, and false-positive/negative
error direction. EESR receives a two-sided Clopper–Pearson interval. Variation across experiments
is summarized by a 10,000-draw percentile bootstrap over the 11 held-out experiments.

## 9. Baselines

At the 5% operating point, the safest tested fixed-time point had 13.82% EESR and 16.60% savings;
naive confidence had 10.48% EESR and 14.17% savings; non-group calibration had 9.01% EESR and
19.79% savings; CBES-style had 15.51% EESR and 15.32% savings; GATES without refusal had 8.60% EESR
and 13.65% savings; Full GATES had 5.69% EESR and 12.26% savings. Baselines that did not meet the
target are reported rather than suppressed.

## 10. RPE result and confidence intervals

Full GATES stopped 439/988 organoids early and made 25 errors. EESR was 5.69% (exact 95% CI
3.72–8.29%), coverage was 44.43%, abstention was 3.85%, and savings were 12.26%. Experiment-bootstrap
intervals were 0.68–12.60% for EESR, 27.06–61.52% for coverage, and 6.74–17.98% for savings. The
wide group interval is central to interpretation: observed aggregate performance does not establish
a 5% guarantee on a new experiment.

## 11. Risk–savings and risk–coverage tradeoff

The complete tested frontier is stored in `artifacts/final/risk_savings_frontier.csv`. More
aggressive operating points raise coverage and savings but also raise observed EESR. At the frozen
10% operating point, Full GATES reached 58.70% coverage and 23.63% savings but 13.10% EESR.

## 12. Per-experiment variation

At 5%, seven experiments had zero early-stop errors, while E007, E008, E009, and E012 had at least
one. E007 had 15 errors among 49 early stops (30.61%); E012 had 6 among 43 (13.95%). This heterogeneity
dominates the aggregate uncertainty and prevents a universal risk claim.

## 13. E007 forensic analysis

E007 contained 93 units with 50.54% positive endpoints and balanced available condition counts.
Errors included five false positives and ten false negatives, occurring at 24, 48, and 60 h.
Generic input-distance statistics were not extreme across experiments; none of the 15 erroneous
stops crossed the unit OOD threshold. Error confidence was lower and entropy/disagreement higher on
average than for correct stops, but not enough for the frozen gate to refuse them. The supported
classification is **possible conditional/concept shift**. The precise mechanism is undetermined;
no biological explanation is inferred.

## 14. E012 forensic analysis

E012 contained 77 units with 54.55% positive endpoints. All six errors were false positives at
48 h. Their mean confidence was 95.69%, slightly higher than correct stops, while mean ensemble
disagreement was lower. At 12 h the mean predicted probability was 0.247 versus 54.55% observed
positive prevalence, and the Brier score was 0.362. With non-extreme generic input-shift scores, the
primary classification is **calibration failure**; concurrent conditional shift cannot be excluded.

## 15. Experiment-level refusal extension

The nested retrospective extension uses only experiment summaries available by 12 h. At 5%, it
refused E007 and E008, prevented 17 errors, reduced EESR from 5.69% to 2.54% (exact 95% CI
1.10–4.94%), and retained 7.88% savings. E009 and E012 were missed. At 10%, it accepted all seven
unsafe experiments, refused all four safe experiments, prevented no errors, and worsened EESR from
13.10% to 16.34%. It is therefore an **OPTIONAL RESEARCH EXTENSION**, not a general safety mechanism.

## 16. Candidate failure scores

Post-hoc experiment-level AUROCs are descriptive because only 11 experiments and four 5%-risk
events are available. Early trajectory-slope magnitude had the highest oriented descriptive AUROC
(0.893). At the unit level, entropy/negative confidence reached 0.803, OOD ratio 0.766, and ensemble
disagreement 0.765. These diagnostics did not justify changing the frozen method.

## 17. Robustness and negative results

The 5% configuration degraded under irregular sampling (10.61% EESR), 0.25-SD feature noise
(7.37%), half-feature removal (7.92%), and altered class balance (8.47%). A decision-tree predictor
reached 15.77%. Missing 20% of observations produced 6.16%, and a reduced 2 h cadence produced
5.64%. Lens endpoint transfer failed at 17.82% EESR with only 6.43% savings. These results are part
of the evidence, not excluded sensitivity runs.

## 18. Limitations

This is retrospective evidence from one published dataset. Eleven experiments give limited power
for group-level calibration and wide uncertainty for new experiments. The RPE label and observation
schedule are dataset-specific. Refusal is a diagnostic, not a proof of OOD detection. There is no
direct organ-on-chip validation, no independent laboratory, no prospective intervention, and no
successful Lens transfer. The work does not prescribe wet-lab or clinical action.

## 19. Reproducibility

`uv.lock` freezes the environment. Dataset, protocol, evidence, and figure hashes are recorded in
`artifacts/final/MANIFEST.json`. Every headline row is in `artifacts/final/master_evidence.csv`;
figure-to-table mappings are in `artifacts/final/figure_sources.csv`. Reproduce with:

```bash
uv sync --frozen --extra dev
uv run python scripts/run_phase3_experiment_gate.py --data data/phase2_raw/orgainoid_morphometrics.csv --phase2 artifacts/phase2/retinal_rpe_final --protocol protocols/phase3/orgainoid_experiment_gate_protocol.json --output artifacts/phase3/orgainoid_experiment_gate
uv run python scripts/finalize_phase3_submission.py
uv run python scripts/verify_final_evidence.py
```

## 20. Conclusion

GATES demonstrates that strict experiment-held-out selective stopping can retain measurable
observation savings while reducing premature errors relative to the tested alternatives. Its value
is the auditable decision contract and honest refusal behavior, not a universal guarantee. The
strongest next evidence would be frozen prospective evaluation in an independent laboratory.
