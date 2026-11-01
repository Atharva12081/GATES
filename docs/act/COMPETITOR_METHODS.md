# Public-method red team

This review compares three relevant public method families, not named competition entrants. There
is no public evidence that these are the competition's three highest-ranked teams.

## 1 Multi-output Gaussian processes

**Strength:** joint covariance across related dose-response curves; probabilistic interpolation.

**Weakness for this task:** prediction does not define a chronological acquisition policy, a
laboratory cost model, or a calibrated STOP/ABSTAIN decision.

**GATES-ACT response:** use a small-data population prior with shrinkage covariance, then evaluate
explicit chronological policies and keep stopping calibration separate from model fitting.

Reference: Giraldo Gutierrez et al., *npj Precision Oncology* 2024,
<https://doi.org/10.1038/s41698-024-00685-1>.

## 2 Active kernelized matrix completion

**Strength:** connects missing-response reconstruction to experiment selection and stopping.

**Weakness for this task:** a matrix view does not automatically encode continuous dose-time
geometry, chronological eligibility, biological-repeat grouping, or an abstention boundary.

**GATES-ACT response:** treat dose and DIV as coordinates, bind repeat labels into one split group,
and require a calibrated biological-persistence guard before stopping.

Reference: D'Souza et al., *BMC Bioinformatics* 2015,
<https://doi.org/10.1186/s12859-015-0713-8>.

## 3 Expected integrated variance reduction

**Strength:** selects a candidate for its global covariance reduction rather than its isolated
marginal variance.

**Weakness for this task:** posterior variance is not biological utility. It can miss an effect and
does not prove physical cost savings on a batched instrument.

**GATES-ACT response:** compare integrated variance reduction against maximum variance, maximin,
random, fixed sentinels, and fixed full panels; report effect misses separately; count only avoided
future MEA sessions as laboratory savings.

Reference: Kontoudis and Stilwell, ACC 2023,
<https://doi.org/10.23919/ACC55779.2023.10155994>.

## Competition evidence boundary

Official 2025 World AI4S material reports 30 finalists across five applied science tracks. The 2025
Pazhou final reports 183 finalists from 8,131 teams and describes a combined technical and
application assessment. Public finalist implementations are incomplete, so GATES-ACT does not name
or reverse-engineer an unsupported "top three." The defensible lesson is the repeated emphasis on
real data, external evidence, laboratory workflow, and translation.

Sources: <https://competition.ai4s.com.cn/> and
<https://www.aicompetition-pz.com/4th/event_detail/105>.

## Independently verified result

Across 76 protected compound groups, conditional maximum variance achieves 97.87% effect recall and
0.334 hidden NRMSE. It significantly improves recall over fixed sentinels by 2.46 percentage points
(95% grouped-bootstrap CI 0.94 to 4.08; paired sign-flip p=0.00187) and random acquisition by 1.67
points (CI 0.48 to 2.86; p=0.00318). Recall differences versus maximin and the rejected
effect-weighted acquisition are unresolved. Maximum variance has significantly lower NRMSE than all
four comparators, so it is the canonical policy.
