# Frozen-evidence erratum: retinal label prevalence audit

**Author:** Atharva Parande

Discovered: 2026-09-29T12:00:09+05:30  
Scope: descriptive dataset-audit metadata only  
Status: corrected with no change to method, model inputs, or reported scientific results

## What was wrong

The frozen `gates-science-freeze-v1` audit converted the strings `yes` and `no` with
`astype(bool)`. In Python, both are non-empty strings and therefore both evaluate as true. The
per-experiment positive counts in `evidence/phase2/retinal_dataset_audit.json` were consequently
wrong. The source file, unit counts, row counts, model loader, predictions, stopping decisions,
and evaluation metrics were not affected.

The modeling loader independently mapped `yes` to 1 and `no` to 0. The defect was in the
descriptive audit script only.

## Correction

Labels are now stripped, normalized case-insensitively, decoded through an explicit `yes`/`no`
mapping, and rejected if missing or unknown. Tests cover `yes`, `no`, `YES`, `No`, and an
unexpected value.

Across 988 independent organoids, the corrected totals are:

| Endpoint | Positive | Negative | Prevalence |
|---|---:|---:|---:|
| RPE_Final | 405 | 583 | 40.99% |
| Lens_Final | 450 | 538 | 45.55% |

The corrected audit includes positive and negative counts plus prevalence for every experiment.

## Independent impact check

Starting from the pinned 400,059,241-byte Zenodo source file, the project independently recomputed both
endpoint campaigns, the OOD ablation, risk–savings frontier, robustness scenarios, nested
experiment gate, final evidence tables, and all 12 final PNG figures in an isolated temporary
directory. Numeric tables matched within `1e-12`, categorical data matched exactly, and the PNG
files were byte-identical. Provenance-only commit/path fields were excluded from the scientific
comparison.

Result: zero artifacts were classified `RESULT_CHANGED`. The headline remains 25 errors among
439 early stops (5.69% observed EESR; exact 95% CI 3.72–8.29%), 44.43% coverage, and 12.26%
observation savings across 988 organoids from 11 held-out experiments.

Machine-readable details are in `artifacts/final/erratum.json` and
`artifacts/final/erratum_impact_report.json`.

## Freeze lineage

`gates-science-freeze-v1` is preserved unchanged.

- v1 commit: `d701487df78595f835215b60523a95c154e0b16d`
- Phase 2 evidence commit: `02ecb77976801b38228d4f09717b9f6593b7dc1d`
- v1 manifest SHA-256: `2def9352762ddfbb9735fb8c01e3e71d9cf4dd15cbc8b9bf83da97652cdb95d7`
- frozen audit SHA-256: `074192eb98219c1f5c2b6de819c8903018f6bc2031291c8fb27f10a62bfb3bcb`

`gates-science-freeze-v2` supersedes v1 only for corrected audit metadata and deterministic source
acquisition. It does not supersede or revise the frozen method or scientific results.
