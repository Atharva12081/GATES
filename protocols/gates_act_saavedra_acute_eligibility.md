# Saavedra 2021 acute MEA eligibility protocol

Locked: 2026-10-26, before downloading or inspecting numerical workbook outcomes.

## Evidence boundary

The official EPA dataset DOI is `10.23719/1518437`, associated with Saavedra et al.,
*Comparison of Acute Effects of Neurotoxic Compounds on Network Activity in Human and Rodent Neural
Cultures* (Toxicological Sciences, 2021). This source has not been found in the existing GATES-ACT
manifests. It is a candidate for a new, independently sourced **acute concentration-response decision
benchmark**, not for longitudinal dose-time screening or laboratory-session savings.

The public catalog states that workbook rows are concentrations and columns are experimental
replicates. The article states that compounds were tested at seven concentrations, with six replicates
per concentration distributed across separate MEA plates, and that compound measurements were taken
after a baseline recording. Those claims must be verified against the downloaded workbook and article
metadata before any model is fit.

## Eligibility checks before outcome use

1. Download only from the official EPA distribution URL and record bytes and SHA-256.
2. Verify workbook type, sheet names, dimensions, headers, compound and species identifiers,
   concentration units, replicate columns, vehicle/control representation, missingness and formulas.
3. Determine whether replicate columns retain stable identities and whether they can be mapped to
   separate plates. Do not infer plate or well identifiers that are not supplied.
4. Verify the assay schedule: baseline and acute post-dose window, with no claim that it is repeated
   longitudinal observation of the same treatment trajectory.
5. Verify license and source provenance from the EPA catalog and article.

## Frozen prospective decision task if eligible

- Unit of biological decision: compound × species.
- Primary measured response: mean firing rate normalized to each well's pre-treatment baseline, if the
  workbook exposes it at replicate grain.
- Information budget: number of measured concentration × replicate observations; laboratory sessions
  are not a supported cost unit.
- Controls: retain the three prespecified negative-control compounds and the corresponding solvent
  controls described by the article. Do not manufacture unreported vehicle rows.
- Split: whole-compound grouped partitions. Human/rat measurements of the same compound must remain in
  the same partition for any cross-species model; replicate columns may never cross partitions.
- Ground truth: distinguish the article's assay-response reference from independently established human
  seizure liability. Do not relabel assay activity as biological toxicity.
- Primary decision metrics: compound-level precision, recall, specificity, false discoveries,
  unresolved coverage and observations consumed at matched selective risk.

The dataset is rejected for modeling if the public file contains only plotted aggregates, lacks
replicate-level concentration responses, cannot identify species/compound/control structure, or has
ambiguous outcome provenance. Eligibility does not authorize inspection-driven threshold tuning.

## Frozen transfer test addendum

Locked: 2026-10-26 after the structural/header audit, before complete numerical extraction or policy
scoring. The highest-dose sentinel hypothesis was generated on the previously exposed GATES-ACT NFA
development corpus before downloading this workbook. This addendum therefore tests transfer of that
fixed policy to an independently sourced acute assay; it does not make the acute assay longitudinal.

The workbook exposes 12 compounds for rat and human cultures, seven treated concentrations
(`0.03`, `0.1`, `0.3`, `1`, `3`, `10`, `30` µM), and five MEA endpoints. Plate/well IDs and stable
cross-concentration replicate identities are absent. Full inspection found 2--6 finite treatment values
per row and 2--18 vehicle values, with multiple nonidentical vehicle vectors within each species. This
fails the frozen eligibility gate. The policy calculation below is retained only as quarantined
preliminary description; it is not independent confirmation and cannot support model selection.

### Endpoint and reference

- Primary endpoint: workbook `wMFR`, expressed as percent of the same well's baseline.
- Concentration response: arithmetic mean of all available finite treatment replicates. This follows
  the article's mean/SEM reporting convention; no outlier removal or imputation is allowed.
- Full-curve assay-reference effect: any of the seven treated concentration means has
  `abs(log(mean / 100)) >= 0.40`. This reuses the frozen GATES-ACT log-effect threshold. It is not an
  independently established neurotoxicity label.
- Evaluation unit: compound × species; bootstrap clusters the paired species rows by compound.

### Fixed three-concentration policies

1. `top3`: `3`, `10`, `30` µM. This is the transferred highest-dose sentinel and primary candidate.
2. `maximin`: `0.03`, `1`, `30` µM, the evenly spaced rank baseline.
3. `low3`: `0.03`, `0.1`, `0.3` µM, a negative control for dose targeting.
4. `random`: exact enumeration of all 35 three-of-seven subsets; report the mean policy result and
   the full range, without selecting the best subset as a deployable method.

A compound-species effect is confirmed only when a directly measured selected concentration crosses
the frozen effect threshold. Otherwise the result is `NEEDS_MORE_EVIDENCE`; no no-effect call is made.
Consequently all policies have the same zero false-discovery risk relative to the full-curve reference,
and the comparison is discovery recall at matched measurement cost: three concentration conditions and
all available replicate measurements in those rows. Missing replicate cells reduce actual observation
count and must be reported per policy.

### Decision rule

The transfer result is a development `GO` only if `top3` has higher aggregate discovery recall than
`maximin`, no lower recall in either species, and a compound-cluster bootstrap 95% interval for the
paired recall difference entirely above zero. Regardless of the outcome, this dataset cannot authorize
canonical longitudinal integration, early stopping, laboratory-session savings, or a claim of superior
forecasting. A positive result may justify retaining `top3` as an acute direct-discovery baseline for a
future named-plate prospective study.
