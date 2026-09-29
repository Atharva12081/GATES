# Final judge simulation and red-team review

This review tests the frozen submission as a skeptical judge would. It does not assign an
"objective" score and does not change the science. Remaining scientific limitations are kept
visible; presentation issues that could be repaired without changing evidence have been repaired
in the README, report, writeup, demo, and defense.

## Rubric A: Kaggle weighting

### Impact (30%)

- **Strongest evidence:** GATES makes a concrete acquisition decision—whether another longitudinal
  measurement remains necessary—rather than analyzing only after collection. The frozen replay
  estimates 12.26% observation savings while allowing early automation for 44.43% of organoids.
- **Likely deductions:** savings are retrospective, not demonstrated microscope hours or cost; the
  study has no prospective independent-laboratory or direct organ-on-chip deployment.
- **Confusion/weak wording checked:** all primary surfaces say "retrospectively estimated
  observation savings" and avoid unsupported money or time conversions.
- **Missing proof:** a prospective resource-utilization study remains future work and cannot be
  repaired by presentation.

### Technical approach and innovation (30%)

- **Strongest evidence:** experiment-disjoint calibration and evaluation, selective
  STOP/CONTINUE/ABSTAIN decisions, explicit refusal, and risk-versus-cost evaluation are integrated
  into one auditable longitudinal-biological workflow.
- **Likely deductions:** the components are established ideas; GATES does not invent early
  classification, abstention, OOD detection, or calibration and offers no formal anytime-valid
  guarantee.
- **Confusion/unsupported claims checked:** novelty is stated as the combination and problem
  formulation, prediction is explicitly not claimed as novel, and CBES receives fair treatment.
- **Missing proof:** formal group-conditional or anytime-valid control remains future work.

### Results and validation (20%)

- **Strongest evidence:** 988 medaka retinal organoids from 11 fully held-out experiments; frozen
  25/439 observed erroneous early stops; exact unit-level interval, experiment-level uncertainty,
  baselines, refusal ablation, class audit, negative transfer, and robustness evidence.
- **Likely deductions:** only 11 independent experiments; E007 reaches 30.61% error among early
  stops; Lens and irregular-sampling results are weak; unit-level confidence intervals do not
  describe all between-experiment uncertainty.
- **Confusion checked:** the 114,510 observations → 988 organoids → 11 experiments hierarchy and the
  117,249 acquisition-count discrepancy are explicit.
- **Missing proof:** external lab, mammalian/human organoid, and organ-on-chip validation.

### Reproducibility and implementation quality (10%)

- **Strongest evidence:** public MIT repository, pinned CC BY 4.0 source, hashes, `uv.lock`, automated
  tests, CI, frozen decision records, one-command rebuild, release manifest, and CPU-only demo.
- **Likely deductions:** data retrieval is approximately 400 MB and depends on Zenodo availability.
- **Presentation issue repaired:** generated SVG timestamps and randomized IDs were made
  deterministic and tested byte-for-byte.

### Presentation quality (10%)

- **Strongest evidence:** concise above-the-fold story, central risk–savings frontier, 16-page
  report, frozen-evidence demo, exact 4:45 script, and failure cases shown rather than buried.
- **Current blocker:** no public recording exists. Until a readable sub-five-minute demonstration
  is recorded and checked without login, presentation is incomplete and submission should not be
  finalized.

## Rubric B: Pazhou weighting

### Technical innovation (30%)

The strongest case is safe selective termination under whole-experiment heterogeneity. Expected
deductions are the use of established components and the absence of a formal risk guarantee. The
report now makes the contribution boundary and CBES differences precise.

### Completion and results (25%)

The software, evidence pipeline, report, audit, and demo are complete and verified. Expected
deductions remain E007, failed Lens transfer, the exploratory gate's failure at 10%, and the lack
of prospective/direct-OoC results. These are disclosed, not reframed as successes.

### Practical value (20%)

The workflow targets acquisition burden, microscope occupancy, storage, analysis load, and
longitudinal measurement burden without inventing dollar or hour savings. The principal weakness
is that practical value is estimated by retrospective policy replay.

### Completeness (15%)

Data acquisition, auditing, modeling, calibration, refusal, sequential decisions, evidence logs,
reproduction, report, and demo form an end-to-end system. The required public video is the only
known submission-surface blocker; a hosted demo is optional and remains unavailable.

### Interpretability and trustworthiness (10%)

Decision-level evidence, refusal behavior, provenance, hierarchy-aware uncertainty, an erratum,
and explicit negative results are strong. Trustworthiness is limited by only 11 independent
experiments and failure to detect the important E007/E012 shifts. The solo team does not qualify
for the cross-disciplinary team bonus.

## Hostile reviewer findings

| Reviewer | Criticism | Class | Disposition |
|---|---|---|---|
| ML | This combines known methods rather than introducing a new learner. | Already addressed | The contribution is framed as an experiment-disjoint selective-stopping formulation and integrated workflow. |
| Statistician | Repeated looks do not yield an anytime-valid 5% guarantee. | Major | Explicitly disclosed; 5.69% is an empirical held-out property of the complete policy. |
| Statistician | Eleven groups are too few for precise biological-generalization claims. | Major | Experiment bootstrap uncertainty is shown; no broader guarantee is claimed. |
| Biologist | The independent unit is obscured by a six-figure observation count. | Already addressed | Every primary surface shows observations nested in organoids nested in 11 experiments. |
| Biologist | The project could be mistaken for human-organoid evidence. | Already addressed | Species is consistently medaka (*Oryzias latipes*); human/mammalian validation is future work. |
| OoC researcher | Retinal organoids are not organ-on-chip validation. | Major | Direct OoC validation is named as the next experiment and never implied as completed. |
| ML | Aggregate performance conceals catastrophic group failure. | Major | E007's 30.61% is prominent and motivates group-respecting evaluation; the current refusal limitation remains visible. |
| ML | Refusal may simply improve the metric by rejecting difficult cases. | Already addressed | Coverage, savings, refusals, and no-refusal ablation are reported together, including class-specific rates. |
| Reviewer | CBES was weakened into a straw-man. | Minor | The binary adaptation and its difference from the original formal method are documented; no claim of reproducing every CBES formulation is made. |
| Reproducibility | The source count differs from the analyzed table. | Already addressed | 117,249 is acquisition count; the pinned published morphometrics table has 114,510 rows and loaders remove none. Per-image reasons are unavailable and not invented. |
| Reproducibility | A clean checkout may not reproduce presentation bytes. | Already addressed | SVG date metadata, IDs, and trailing whitespace are deterministic and covered by a repeat-export test. |
| Competition judge | The submission lacks a required system video. | Fatal until fixed | Record, upload, and verify the scripted 4:45 demonstration before final submission. |

## Final interpretation

No remaining wording or organization defect obviously misstates the frozen evidence. The major
scientific weaknesses—small group count, E007, retrospective savings, failed transfer, absence of
formal anytime validity, and no direct organ-on-chip validation—are genuine limitations, not
documentation bugs. The sole known fatal submission-readiness issue is the missing public video.
