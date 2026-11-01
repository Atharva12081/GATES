# GATES-ACT preregistered external-confirmation bundle

Version: 1.0, frozen without access to any new HESI or Carstens BrainSphere row-level outcomes.

## Eligibility gate

A dataset is eligible only if all of the following are available and auditable:

- stable laboratory, plate/experiment, well/specimen and chemical identifiers;
- concentration values with units and an unambiguous within-chemical dose ordering;
- at least four repeated recording times on the same experimental units;
- vehicle controls matched to plate/experiment and recording time;
- numeric per-well endpoint values rather than fitted curves or digitized figures;
- at least two biological experiments per chemical and a documented license/data-use basis;
- no overlap with the 76 canonical external groups or the 99 exposed development compound groups
  unless the new records are demonstrably new experiments and analyzed as such.

Failure of any identity, control, timing or outcome requirement makes the dataset ineligible for a
sequential confirmation claim. Acute single-time recordings may be used only for a separately
labeled non-longitudinal stress test.

## Frozen split and preprocessing

Before outcomes are inspected, hash the raw files and freeze laboratory -> experiment/plate ->
chemical -> well counts. Hold out complete laboratories when more than one laboratory is present;
otherwise hold out complete chemicals and experiments. No well, plate, replicate or chemical may
cross train/calibration/test boundaries. Normalize each endpoint to its matched vehicle on the same
plate and day using the preregistered log-ratio or standardized contrast justified by the assay.
Missing controls are not imputed across plates.

The effect threshold is fixed before test inspection from assay controls and the development-scale
mapping. If normalized responses are commensurate with the existing benchmark, use absolute 0.40.
Otherwise estimate a control-derived biological threshold using training controls only and freeze
the mapping before calibration/test evaluation.

## Matched task and budget

The primary longitudinal task uses four prespecified recording times and eight ordered cells per
time: one matched vehicle plus seven treated concentrations. If the experiment has a different
grid, freeze a one-to-one mapping before outcomes; do not interpolate identities or claim exact
task parity. Every method receives one vehicle plus three treated measurements per time (16/32
physical cells). No method may select an elapsed time or reuse a control unavailable to a baseline.
No retrospective omission is described as a saved physical well or laboratory session.

## Frozen methods

1. direct-only highest-three-dose design;
2. highest-three-dose GATES agreement/release;
3. canonical cost-aware conditional maximum variance;
4. middle-high sentinels;
5. maximin;
6. random acquisition with at least 1,000 seeded draws when computationally feasible.

Early stopping remains disabled. All methods finish the full scheduled assay and may output
CONFIRMED EFFECT, NO EFFECT DETECTED or NEEDS MORE EVIDENCE at the cell level.

## Endpoints and success criterion

Primary endpoint: active-chemical sensitivity at matched 16/32 budget, where activity is defined by
the full-assay reference call. This is an assay-reference decision, not independently established
human neurotoxicity.

Key safety and utility endpoints: effect-call precision, specificity, protected recall, wrong
decisions, decisive coverage, directly observed effect capture, hidden-effect recall, and per-source
worst-case performance. Report chemical-level counts as well as pooled response-cell metrics.

The experimental high-dose GATES result is confirmed only if it improves active-chemical
sensitivity by at least five percentage points or at least two chemicals versus canonical
conditional maximum variance, while effect precision is at least 95%, protected recall is at least
95%, and wrong decisions do not increase. It must also be no worse than direct-only high dose in
active-chemical sensitivity. With only ten chemicals, the count requirement governs when five
percentage points is less than one chemical.

## Uncertainty and audit

- Report exact Clopper-Pearson intervals for chemical-level sensitivity/specificity and hierarchical
  cluster-bootstrap intervals over chemicals, then experiments, for response-cell differences.
- Preserve every unresolved case; do not convert abstention into a correct negative.
- Produce a row-level decision ledger, selection chronology, raw-file hashes, frozen split manifest,
  calibration record and independent recomputation audit.
- Label the result independent confirmation only if the dataset was genuinely unexamined when this
  protocol was frozen and every eligibility and success gate passes.
