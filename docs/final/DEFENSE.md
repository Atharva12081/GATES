# GATES Defense Questions

**Author:** Atharva Parande

## 1. Why isn't this just CBES?

The tested CBES-style adaptation supplies one comparator, not the entire method. GATES adds strict
experiment-held-out calibration, group-specific thresholding, an explicit OOD/refusal action, and
auditable STOP/CONTINUE/ABSTAIN decisions. At the frozen 5% point, CBES-style produced 15.51% EESR
and 15.32% savings; Full GATES produced 5.69% and 12.26%.

## 2. Why not stop at a fixed time?

Fixed time cannot adapt to easy, difficult, or shifted trajectories. The safest tested fixed-time
point still had 13.82% EESR at 16.60% savings. Full GATES stopped only 44.43% of units and continued
or refused the rest, reducing observed EESR to 5.69%.

## 3. Why does Full GATES outperform the tested CBES-style adaptation?

The evidence supports an empirical answer: group-calibrated thresholds plus refusal select a
smaller, safer set of early decisions. It does not isolate a universal causal advantage. Full GATES
made 439 stops/25 errors; CBES-style made 445/69 at the same nominal target.

## 4. Why does the OOD/refusal layer help?

Compared with GATES without refusal, it prevented 16 errors, reduced EESR from 8.60% to 5.69%, and
reduced savings from 13.65% to 12.26%. It helps empirically, but it missed E007/E012 errors and is
not proof of OOD detection.

## 5. What happened in E007?

E007 had 93 units and 15 errors among 49 early stops (30.61% EESR): five false positives and ten
false negatives. Errors occurred at 24, 48, and 60 h. Generic shift scores were not extreme, and no
erroneous stopped unit crossed the OOD threshold. The supported label is possible
conditional/concept shift; the mechanism is undetermined.

## 6. Why can't you claim a universal 5% risk bound?

Observed aggregate EESR is 5.69%, already above 5%. Its exact 95% CI is 3.72–8.29%, while the
experiment-bootstrap interval is 0.68–12.60%. Eleven experiments are insufficient for a tight
new-experiment guarantee, and the 10% operating point also misses its nominal target.

## 7. Why does Lens fail?

Lens transfer reaches 17.82% EESR with 6.43% savings. The available evidence establishes endpoint
transfer failure but does not identify a biological mechanism. I report it as a negative result.

## 8. Why isn't there direct organ-on-chip validation?

No accessible dataset passed the pre-specified availability, endpoint, grouping, and leakage-safety
requirements. Direct OoC validation was therefore not run. **NOT PURSUED — outside current
computational scope** for any new experimental data generation.

## 9. What does 12.26% savings mean practically?

It is the mean fraction of scheduled post-decision observations avoided across all 988 held-out
organoids. It is not 12.26% shorter wall-clock duration for every experiment and does not include
labor, batching, or instrument-utilization assumptions.

## 10. Why is coverage only 44.43%?

The method refuses to force early decisions. It stopped 439/988 units; the others continued or
abstained because confidence or familiarity criteria did not pass. Coverage is the cost of the
observed risk reduction.

## 11. Why not force predictions for every organoid?

Forced decisions erase the safety/efficiency distinction. The fixed-time comparator effectively
decides for 99.60% of units but produces 13.82% EESR. A selective method should expose uncertainty,
not hide it.

## 12. What is the independent experimental unit?

An organoid well is the prediction unit; the complete experiment is the generalization and
resampling boundary. There are 988 wells across 11 experiments.

## 13. How did you prevent leakage?

Each held-out experiment is excluded from preprocessing, model fitting, calibration, and OOD
threshold fitting. Repeated frames and all prefixes from a well remain in the same experiment.
Identifiers, availability columns, and final labels are excluded from predictors.

## 14. What was frozen before testing?

Phase 2 code is commit `69b1b20`; the immutable evidence freeze is `02ecb77`, tagged
`gates-phase2-evidence-v1`. The Phase 3 protocol hash is
`35f7642203792bdde9190fbdcd6c7fd0fa1ebd3c74068197cfe71f1fbb0f75f3` and was committed before
the nested retrospective gate results were generated.

## 15. How reproducible is the result?

The environment is locked, seeds are fixed, source/protocol/evidence/figure hashes are recorded,
and final verification recomputes Phase 3 outputs, checks headline rows, verifies both freeze
manifests, and runs lint/tests. A clean clone downloads the pinned source and reproduces the package
using README commands only.

## 16. What happens under missing observations?

With 20% observations removed, the frozen 5% configuration produced 6.16% EESR and 12.28% savings.
Under irregular sampling it produced 10.61% EESR. Missingness is therefore not uniformly benign.

## 17. Does GATES generalize to another laboratory?

Unknown. Every evaluation here is within one published dataset, although complete experiments are
held out. Independent-laboratory validation is required before that claim.

## 18. What would be required for deployment?

A pre-registered endpoint, prospective independent laboratories, larger calibration populations,
validated missingness and shift monitoring, operational cost analysis, human oversight, and an
application-specific safety review. The current software is retrospective research code.

## 19. Why didn't you use a transformer?

The evidence bottleneck is 11 independent experiments, not representational capacity. A more
complex model would add tuning degrees of freedom without increasing independent information. A
decision tree stress test was worse (15.77% EESR), illustrating that architecture substitution is
not automatically beneficial.

## 20. What is actually novel?

The contribution is the auditable combination of experiment-held-out sequential prediction,
group-calibrated stopping, explicit refusal, risk–savings accounting, and visible negative results
on retinal-organoid trajectories. It is not a claim of a new universal classifier or certified risk
controller.

## 21. Was frozen evidence changed after the label-audit bug was found?

No. `gates-science-freeze-v1` remains at `d701487`. V2 corrects only descriptive audit metadata and
adds deterministic data acquisition. A full isolated recomputation found zero changed results.

## 22. Why trust the data acquisition path?

It pins Zenodo record 18198347, filename, 400,059,241-byte size, MD5, and SHA-256. The downloader
never resolves “latest” and exposes the file only after verification.

## 23. Are 114,510 frames the sample size?

No. They are repeated observations. The prediction unit is an organoid well (988), and the
generalization boundary is the experiment (11). Frames are not treated as independent samples.

## 24. Does the experiment-level gate rescue the headline?

No. It is outside the core headline. It looks useful at 5% but reverses safe and unsafe experiments
at 10%, so it remains an optional retrospective extension with its failure reported.

## 25. What would falsify the practical value of GATES?

Prospective multi-laboratory evidence showing no reproducible savings at an acceptable error and
coverage tradeoff would falsify it. So would a comparator that dominates the complete frontier
under the same group-held-out protocol.

## 26. Why does the paper report 117,249 images but GATES reports 114,510 observations?

The paper reports the upstream acquisition count. GATES reads the pinned published morphometrics
table, `Extended_Data_2.csv`, which contains exactly 114,510 rows; the loader removes no rows. The
2,739-image difference therefore arose before GATES, between acquisition and the published table.
Neither the article nor table supplies per-image exclusion reasons, so I do not invent one.

## 27. Are these human retinal organoids?

No. They are medaka retinal organoids derived from *Oryzias latipes*. The project makes no human or
mammalian performance claim.

## 28. Why use medaka?

The public dataset provides unusually dense, experiment-identified longitudinal trajectories with
final tissue labels. That makes it a useful biological testbed for selective stopping. Faster
development also enabled the source study's 72 h acquisition window. Transfer to other species is
unknown.

## 29. Why is there no direct organ-on-chip validation?

No accessible OoC dataset passed the pre-specified requirements for longitudinal prefixes, final
endpoint labels, complete experimental grouping, and leakage-safe evaluation. Retinal organoids are
not represented as OoC experiments. Direct OoC validation is a priority future experiment.

## 30. Does refusal disproportionately reject one endpoint class?

Yes, in the frozen audit. Refusal was 5.93% for RPE-positive organoids and 2.40% for RPE-negative
organoids. Early-decision coverage was 35.56% and 50.60%, respectively. This appendix result was
computed from frozen decisions without changing the method and motivates larger class-conditional
validation.

## 31. Is 5.69% an anytime-valid guarantee under repeated looks?

No. It is the empirical held-out result of the complete frozen sequential policy. The current method
does not provide anytime-valid distribution-free risk control; that is future work.

## 32. Why is experiment-bootstrap uncertainty wider than the unit-level interval?

The binomial interval treats 439 early-stop decisions as its denominator. The bootstrap resamples
only 11 experiments, preserving the generalization boundary and the observed heterogeneity. Its
wider interval reflects uncertainty about transfer across experiments.

## 33. Is 12.26% savings prospective?

No. It is retrospectively estimated by replaying the frozen policy over recorded trajectories. It
counts observations scheduled after an early stop that would not have been needed under that policy.
No microscope time, labor, money, or wall-clock duration was physically saved in this study.

## 34. Could future information leak into prefixes?

Prefix features use only measurements available at the decision time. Final labels, identifiers,
and availability variables are excluded from predictors. The complete test experiment is absent
from preprocessing, fitting, calibration, and OOD-threshold estimation.

## 35. Was the CBES-style comparison fair?

It used identical test experiments, source features, candidate times, and risk grid. The binary
endpoint required replacing the original continuous Gaussian-process setup with bootstrap logistic
ensembles and symmetric confidence bounds. CBES also reserves separate calibration and validation
experiments as its procedure requires. It is a documented adaptation, not a claim to reproduce every
original implementation detail.

## 36. Why not use a vision foundation model?

The frozen public artifact is a morphometrics table, and the independent-information bottleneck is
11 experiments. A larger image encoder would add training and tuning degrees of freedom without
creating more independent experiments. It is not required to test the stopping-policy contribution.

## 37. What happens when the model sees an unfamiliar experiment?

The unit-level distance gate can disable early automation when a prefix crosses its fitted
threshold. However, E007 demonstrates that unfamiliar failure can occur without extreme generic
input distance. Refusal reduces risk empirically but does not solve all experiment-level shift.

## 38. What is the single most important limitation?

Only 11 experiments from one medaka dataset define the independent evidence base. That makes the
aggregate result promising but insufficient for a reliable new-laboratory guarantee.

## 39. What is the immediate next experiment?

A preregistered prospective shadow evaluation across independent laboratories: freeze the endpoint,
policy, and analysis; log every candidate decision without acting on it; then assess risk, coverage,
savings, class asymmetry, and shift at the experiment level. Direct OoC data should follow the same
protocol.

## 40. Why should this project win?

GATES tackles a practical question most predictive systems leave unanswered: whether another
measurement is necessary. It delivers a complete, reproducible decision workflow, validates it at
the experiment boundary, demonstrates that refusal materially changes outcomes, and exposes the
cases where the system fails. Its value is defensible selective termination, not inflated accuracy.
