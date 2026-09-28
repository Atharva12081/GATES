# GATES Phase 2 scientific evidence report

Evidence code commit: `69b1b200c29ca42626fc4cdc7d32551c9524e8f9`  
Frozen feasibility tag: `gates-organoid-feasibility-v0`  
Evaluation status: empirical held-out evidence, not a formal or clinical guarantee.

## 1. Datasets and independent units

- Frozen OrganoID gemcitabine feasibility: 21 biological units in 3 replicate
  groups. This remains feasibility evidence only.
- orgAInoid retinal morphometrics: 988 organoids in 11 independent experiments,
  114,510 rows, 144 possible half-hour time points through 72 h. RPE and lens
  final morphology are evaluated as separate endpoints. The experiment, never the
  frame, is the held-out generalization boundary.
- Direct organ-on-chip: four candidates audited; none supports honest chip/group
  held-out early-to-final evaluation with the available public metadata.

## 2. Frozen protocol

`protocols/phase2_protocol.json` was committed before held-out retinal results were
fit or inspected. The aspirational primary EESR target is 5%; the secondary
empirical target is 10%. An erroneous early stop is a terminal call before 72 h
that differs from the final endpoint. Coverage, abstention, savings, decision time,
FP-EESR, and FN-EESR are defined there. CONTINUE and ABSTAIN receive zero savings.

Each rotation holds out one experiment as test, reserves the next two experiments
for calibration, and fits on the remaining eight. Test data contribute nothing to
preprocessing, fitting, feature choice, thresholds, OOD thresholds, or rules.

## 3. CBES implementation status

A serious binary CBES adaptation uses bootstrap predictive mean/standard deviation,
symmetric 95% confidence-bound calls around 0.5, per-time calibration, sequential
pruning, an independent validation experiment, and its Hoeffding term. It sees the
same features, candidate times, training data, held-out test experiment, and risk
grid as GATES. The binary logistic ensemble is a necessary adaptation of the
original continuous Gaussian-process setting; it is not represented as an exact
reproduction. See `docs/phase2/CBES_ADAPTATION.md`.

## 4. Per-method flagship results

Aggregate RPE results at the frozen 5% target (988 organoids, 11 experiments):

| Method | Stops | Errors | EESR | Coverage | Abstention | Savings | Mean terminal time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated, non-group | 644 | 58 | 9.01% | 65.18% | 0% | 19.79% | 57.75 h |
| CBES-style | 445 | 69 | 15.51% | 45.04% | 0% | 15.32% | 60.97 h |
| GATES without OOD | 477 | 41 | 8.60% | 48.28% | 0% | 13.65% | 62.17 h |
| Full GATES | 439 | 25 | 5.69% | 44.43% | 3.85% | 12.26% | 63.17 h |

No fixed-time or naive-confidence operating point reached 5% aggregate EESR. Even
the strictest naive point stopped 146 units at 13.70% EESR. At the nominal 10%
target, full GATES observed 13.10% EESR with 23.63% savings, so that target did not
control held-out risk.

The independent lens endpoint is a failure: at the 5% target, full GATES stopped
101/988 with 18 errors (17.82% EESR), 10.22% coverage, and 6.43% savings. CBES-style
was worse (33.33% EESR), but relative advantage does not make the GATES result safe.

## 5. Per-experiment results

At the 5% RPE target, full-GATES error/stops and all-unit savings were:

| Experiment | Error/stops | EESR | Savings |
|---|---:|---:|---:|
| E001 | 0/49 | 0% | 16.67% |
| E002 | 0/25 | 0% | 4.53% |
| E004 | 0/52 | 0% | 9.03% |
| E005 | 0/0 | no stops | 0% |
| E006 | 0/0 | no stops | 0% |
| E007 | 15/49 | 30.61% | 30.29% |
| E008 | 2/75 | 2.67% | 16.67% |
| E009 | 2/88 | 2.27% | 27.22% |
| E010 | 0/34 | 0% | 6.16% |
| E011 | 0/24 | 0% | 8.89% |
| E012 | 6/43 | 13.95% | 17.53% |

The aggregate therefore conceals major experiment instability. Complete RPE and
lens rotation tables are in each `master_results.csv` and the combined matrix.

## 6. Risk-savings comparison

The full frontier, not a selected point, is in
`retinal_rpe_final/figures/risk_vs_observation_savings.png`. Full GATES is the only
method near the 5% line while retaining double-digit savings at the frozen 5%
operating point, but experiment-bootstrap uncertainty overlaps materially and the
curve must be described as observed empirical behavior.

## 7. OOD ablation

At the RPE 5% target, OOD refusal prevented 16 erroneous early stops, introduced 38
additional abstentions, reduced EESR from 8.60% to 5.69%, and reduced savings only
from 13.65% to 12.26% (1.38 percentage points). At 10%, it prevented 13 errors, added
50 abstentions, changed EESR 14.13% to 13.10%, and sacrificed 2.45 savings points.
The OOD component therefore materially changes outcomes, but does not eliminate
the E007 failure. Lens shows the same direction but remains unsafe.

## 8. Uncertainty

- Frozen 21-unit feasibility result: 10% EESR, exact 95% CI 1.23%-31.70%; all-unit
  savings 51.06%, exploratory three-group bootstrap 46.03%-55.56%; decision time
  35.24 h, bootstrap 32.00-38.86 h. One additional failure raises EESR to 15%; two
  raise it to 20%.
- Retinal RPE full GATES at 5% target: 5.69% EESR, exact 95% CI 3.72%-8.29%, and
  11-experiment bootstrap 0.68%-12.60%. Savings 12.26%, experiment-bootstrap
  6.74%-17.98%; coverage 44.43%, 27.06%-61.52%.
- Lens full GATES at 5% target: 17.82% EESR, exact 95% CI 10.92%-26.70% and
  experiment-bootstrap 12.90%-21.37%.

## 9. Direct organ-on-chip audit

The lung-on-chip tuberculosis archive is scientifically plausible but roughly
27 GB and lightweight metadata do not establish chip IDs, independent experiments,
or a frozen terminal endpoint. The 3,072-image bright-field dataset lacks chip
linkage and linked final outcomes; the EIS dataset has one device; the OCT dataset
lacks independent biological groups and a terminal endpoint. No direct OoC result
is claimed. Details and source DOIs are in `evidence/phase2/direct_ooc_audit.json`.

## 10. Failures discovered

- Experiment E007 breaks the nominal 5% RPE operating point (30.61% EESR).
- The nominal 10% target does not control aggregate RPE risk.
- Lens transfer fails badly, showing endpoint specificity.
- Group calibration is data-starved: only 2/66 RPE time/rotation calibrations at
  the 5% target pass their finite-sample upper-bound check; none do for lens.
- Stress tests worsen risk under irregular sampling, noise, and feature removal;
  early corruption reduces attainable savings. At the 5% operating point: 20%
  missingness produced 6.16% EESR, irregular sampling 10.61%, 0.25-SD feature
  noise 7.37%, half-feature removal 7.92%, early corruption 6.25% with only 6.97%
  savings, 2 h cadence 5.64%, and altered class balance 8.47%. The alternate seed
  reproduced 5.69%; a depth-4 decision tree failed at 15.77%.
- No direct OoC dataset passed the scientific-validity gate.

## 11. Exact evidence paths

- `snapshots/gates-organoid-feasibility-v0/`
- `protocols/phase2_protocol.json`
- `evidence/phase2/retinal_dataset_audit.json`
- `evidence/phase2/direct_ooc_audit.json`
- `artifacts/phase2/master_results.csv`
- `artifacts/phase2/organoid_feasibility_uncertainty/`
- `artifacts/phase2/retinal_rpe_final/`
- `artifacts/phase2/retinal_lens_final/`
- `artifacts/phase2/retinal_rpe_stress/`
- `artifacts/phase2/MANIFEST.json`

## 12. Tests and reproduction status

`uv run pytest -q` passes 9 tests, including a claim-lint check; `uv run ruff check .`
passes. Dataset ingestion, leave-one-experiment-out evaluation, analysis figures,
stress runs, and manifest generation are scripted. Raw 382 MiB source data remain
excluded from git and are identified by URL, license, and SHA-256 in the audit.

## 13. Verdict

**CONDITIONAL GO.** Preserve the narrower contribution: shift-aware refusal clearly
prevents failures and full GATES dominates the tested strong baselines on RPE. Do
not advance a broad empirical-risk-control, cross-endpoint, or direct-OoC claim.
Risk calibration is unstable across experiments, Lens fails, stress robustness is
limited, and direct OoC transfer is unvalidated. This is not a STRONG GO.

## 14. Single highest-value next action

Build and prospectively freeze an **experiment-level shift/refusal gate**, calibrated
on more independent retinal experiments, then test it on at least one entirely new
experiment batch. The gate must identify E007-like failures before any unit-level
early calls. More organoids inside the existing 11 experiments will not resolve the
dominant uncertainty.
