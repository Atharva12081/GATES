# GATES Claim Freeze

This file freezes the scientific claims supported by the immutable Phase 2 evidence and the
retrospective Phase 3 analyses. Percentages below are derived from
`artifacts/final/master_evidence.csv`; they are empirical results, not prospective guarantees.

## SUPPORTED CLAIMS

- On longitudinal retinal-organoid RPE prediction, group-calibrated selective stopping with
  refusal reduced premature early-stop error relative to the tested non-group, CBES-style, and
  no-refusal alternatives while retaining measurable observation savings under strict
  experiment-held-out evaluation.
- The source evidence contains 988 organoids in 11 independent experiments and 114,510 audited
  longitudinal observations. Entire experiments were held out from preprocessing, model fitting,
  calibration, and OOD-threshold estimation.
- At the frozen 5% operating point, Full GATES made 439 early decisions with 25 errors: 5.69% EESR
  (exact 95% CI 3.72–8.29%), 44.43% coverage, and 12.26% observation savings. The 11-experiment
  bootstrap intervals were 0.68–12.60% for EESR, 27.06–61.52% for coverage, and 6.74–17.98% for
  savings.
- At the same operating point, GATES without refusal had 8.60% EESR, and the tested CBES-style
  adaptation had 15.51% EESR. Unit-level refusal prevented 16 errors and cost 1.38 percentage
  points of observation savings relative to GATES without refusal.
- The experiment-level gate is an **OPTIONAL RESEARCH EXTENSION**, not part of the core method.
  At the pre-specified 5% target it refused E007 and E008, prevented 17 errors, reduced EESR from
  5.69% to 2.54%, and retained 7.88% savings. This is a nested retrospective result.

## UNSUPPORTED CLAIMS

- A universal 5% risk guarantee or certified finite-sample error control.
- General stopping performance across life-science experiments or endpoints.
- Generalization to another laboratory, donor, instrument, or acquisition campaign.
- Direct organ-on-chip validation.
- Successful Lens endpoint transfer.
- Reliable detection of every experiment-level shift.
- A universally reliable experiment-level gate.
- A biological explanation for E007 or E012.

## NEGATIVE RESULTS

- Lens transfer: 17.82% EESR and 6.43% observation savings at the 5% operating point.
- Irregular sampling: 10.61% EESR.
- Feature noise (0.25 SD): 7.37% EESR.
- Half-feature removal: 7.92% EESR.
- Altered class balance: 8.47% EESR.
- Decision-tree predictor: 15.77% EESR.
- At the 10% operating point, the experiment gate accepted all seven unsafe experiments, refused
  all four safe experiments, prevented no errors, and worsened EESR from 13.10% to 16.34%.
- E007 remained a severe held-out failure for core Full GATES (15/49 errors; 30.61% EESR).
- E012 remained a held-out failure (6/43 errors; 13.95% EESR); all six were high-confidence false
  positives at 48 h.

The frozen interpretation is: “Experiment-level refusal showed useful behavior at one
pre-specified operating point but was unstable across risk targets.”
