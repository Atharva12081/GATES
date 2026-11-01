# Carstens 2025 human 3D BrainSphere MEA eligibility and confirmation protocol

Locked: 2026-10-28 before downloading or inspecting numerical supplemental outcomes for
DOI `10.1007/s00204-025-04043-x`.

## Evidence question

Determine whether the public article and supplements expose a reconstructable, independently sourced
chemical-by-concentration-by-recording-day MEA panel with stable experiment/well identities and matched
controls. The full raw dataset is described as available from the authors on request and must not be
claimed as possessed unless it is actually received and integrity checked.

Publication independence does not imply task compatibility. The dataset is eligible for independent
decision testing only if all of the following are available without inference:

1. chemical identity and ordered concentration with units;
2. stable experiment, culture batch, plate and/or well identifiers sufficient to prevent pseudoreplication;
3. explicit mapping showing that repeated recording days belong to the same biological/technical unit;
4. named recording days and treatment sequence;
5. same-day vehicle or untreated controls with solvent identity;
6. per-well or per-experiment numerical MEA endpoints rather than figure pixels or only means/SEM; and
7. enough ordered treated concentrations to compare policies at identical daily measurement cost.

Absent or ambiguous identifiers cause rejection for confirmation. Figure digitization, duplicated group
means, invented well identities, and assumptions that equal column positions denote the same well are
prohibited.

## Frozen confirmation task if eligible

- Primary endpoint: per-well mean firing rate if supplied; otherwise the first article-designated global
  activity endpoint with a complete same-well series, documented before numerical scoring.
- Response: same-day matched-control `log1p(treated) - log1p(control)` using only source-provided links.
- Reference effect: absolute response at least 0.40, retained from GATES-ACT without dataset tuning.
- Group unit: chemical. All concentrations, days, experiments, wells and declared repeats for one
  chemical remain together.
- Evaluation: leave-one-chemical-out descriptive predictions with calibration excluding the held
  chemical. With only ten chemicals, report every chemical, exact counts, chemical-cluster bootstrap
  intervals, and leave-one-chemical sensitivity analysis; do not claim broad population generalization.
- Acquisition: vehicle plus three treated concentrations per available day, comparing fixed highest
  three, fixed middle-high, maximin, and the forward-only temporal-boundary policy from the locked
  falsification protocol. If the source has fewer than four ordered treated concentrations, the adaptive
  policy is not identifiable and the dataset cannot confirm its advantage.
- Decisions: direct observed-effect detection is the primary comparator. GATES inference is evaluated
  incrementally and must improve correct decisions beyond the fixed design without reducing precision,
  protected recall, or worst-chemical safety.

No threshold, endpoint, concentration mapping, policy trigger or inclusion rule may be changed after
row-level outcomes are inspected. If public supplements fail eligibility, record the exact blocker and
perform no outcome-driven model selection.
