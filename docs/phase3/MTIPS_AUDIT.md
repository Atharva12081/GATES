# mTIPs external retinal-organoid dataset audit

Audit date: 2026-09-29

## Decision

**Unusable for Phase 3 external validation under the acceptance requirements.**

The paper is real and describes a potentially valuable independent system, but an
auditable public dataset could not be located or downloaded. The official IEEE page
exposes one supplemental PDF, yet both the article full text and supplement require
institutional/member access. No official IEEE DataPort record, dataset DOI/accession,
project repository, GitHub code, license, file inventory, or download size was
discoverable from the authoritative metadata or repository searches below.

The audit therefore cannot establish individual longitudinal IDs, differentiation
batch IDs, endpoint construction, leakage safety, or a group-respecting split. The
abstract's statement that a dataset is publicly available is not sufficient to
invent those fields.

## Authoritative sources checked

- IEEE article: <https://doi.org/10.1109/JBHI.2026.3668206>
- IEEE Xplore record 11411936 and Supplemental Items page
- IEEE supplement DOI: <https://doi.org/10.1109/JBHI.2026.3668206/mm1>
- PubMed PMID 41740113
- Crossref, OpenAlex, Semantic Scholar, and Unpaywall metadata
- IEEE DataPort web/domain search
- GitHub repository and code search by title, DOI, acronym, and authors
- University of Science and Technology of China author/institution pages

## What can be verified

- The work uses hESC-derived retinal organoids.
- Imaging is multimodal: traditional 2-D bright-field microscopy and 3-D OCT are
  explicitly described.
- The system is longitudinal and reports useful signal as early as Day 6.
- The target concerns later retinal-organoid differentiation/quality, but the exact
  operational endpoint and its annotation time cannot be recovered from accessible
  authoritative material.
- The official graphical abstract describes sequential early imaging followed by a
  prediction of differentiation fate.

## Acceptance-requirement assessment

| Requirement | Status | Evidence |
|---|---|---|
| Individual longitudinal units identifiable | Unknown / failed | No downloadable schema or file inventory |
| Independent batches identifiable | Unknown / failed | No group field or batch count available |
| Scientifically meaningful endpoint | Plausible, not operationally specified | Abstract only |
| Endpoint free of future leakage | Unknown / failed | Endpoint construction unavailable |
| Group-respecting evaluation possible | Unknown / failed | Batch structure unavailable |

## Requested audit fields

- Organoid count: not verifiable.
- Independent differentiation batches: not verifiable.
- Modalities: 2-D bright-field and 3-D OCT.
- Longitudinal times/cadence: Day 6 is reported; complete schedule is unavailable.
- Endpoint and early target: differentiation fate/quality is described, exact label
  definition unavailable.
- Cell line: hESC-derived; exact line and whether multiple lines exist unavailable.
- Author split: unavailable.
- Repeated-unit grouping: unavailable.
- Dataset license: unavailable. IEEE article copyright terms are not a dataset
  redistribution license.
- Download size and accessibility: unavailable; no downloadable dataset located.
- Feasibility before October 10: no, unless the authors or IEEE release an accessible
  dataset record with unit IDs, batch IDs, labels, and license immediately.

No external performance number is produced.
