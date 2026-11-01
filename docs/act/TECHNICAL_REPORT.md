# GATES-ACT: adaptive neural-assay measurement under explicit risk limits

Atharva Parande  
24 October 2026

## Executive result

GATES-ACT predicts unmeasured dose-by-time responses in longitudinal neural microelectrode-array
(MEA) assays and selects the next physically eligible measurement batch. The final system uses
conditional maximum posterior variance because it was the strongest reproducible acquisition rule.
The earlier effect-weighted rule was rejected: its estimated effect probabilities were not precise
enough to justify exploitation under assay shift.

Across 76 independent external compound groups from three public neural-MEA datasets, the canonical
policy achieved 97.87% group-weighted protected effect recall and the lowest pooled hidden-response
NRMSE (0.334). It exceeded fixed sentinels by 2.46 percentage points of recall (95% bootstrap CI
0.94 to 4.08; paired sign-flip p=0.00187) and random acquisition by 1.67 points (CI 0.48 to 2.86;
p=0.00318). Recall differences versus maximin and the rejected effect-weighted rule were not
significant. NRMSE was significantly lower than every comparator.

Early stopping is disabled. Three external maximum-variance candidate stops included two unsafe
outcomes. The exact one-sided 95% upper confidence bound on unsafe-stop risk is 98.3%, far above the
5% release limit. With zero failures, at least 59 independent validation stops would be required to
bring that bound below 5%. The released action is therefore COMPLETE or ABSTAIN, with zero claimed
physical-session savings.

Two independent imaging studies tested whether tissue images could safely extend that workflow.
They did not justify an imaging-informed MEA acquisition claim. On the protected chronological
test, a 3,072-image organ-on-chip quality model improved average precision from 0.595 to 0.673 over
a frozen representation, but no nonzero automatic-pass threshold met the preregistered 5% risk bound. A
Lab-A-only brain-organoid segmenter reached 0.968 Dice in Lab A and 0.942 in Lab B, where it lost to
the published multi-lab SegFormer by 0.018 Dice (exact paired p=0.000031). The final system therefore
keeps imaging as an independent, fail-closed audit module and never invents paired MEA/image labels.

## Scientific question and estimand

For each compound run, let `y[d,t]` be the vehicle-normalized response at dose rank `d` and recording
day `t`. At each scheduled day the system must select a vehicle anchor and a feasible batch of dose
wells without observing future days. The primary estimands are:

1. recall of biologically meaningful affected cells, defined by
   `abs(log1p treatment contrast) >= 0.40`;
2. normalized RMSE on unmeasured cells;
3. direct capture of affected cells by selected measurements;
4. actual avoided physical recording sessions; and
5. the unsafe-stop rate among proposed early stops.

Measurement-count efficiency is reported separately from laboratory savings. A well already
allocated to a plate is not treated as saved, and no session is counted as saved unless the entire
future recording session can safely be omitted.

## Data and leakage control

| Source | Independent groups | Role |
|---|---:|---|
| NTP/EPA Phase-1 DNT MEA network formation | 17 | held-out confirmation |
| Brown et al. neural-network ontogeny MEA | 6 | external compatibility test |
| EPA Tipping Point determinations | 53 | largest shifted confirmation |
| **Total MEA evidence** | **76** | group-weighted final evidence |

Compound identity is the grouping boundary. Biological repeats, technical wells, doses, and all
timepoints from the same compound group remain in one split. Vehicle normalization is plate matched.
Replicates are aggregated only after normalization. Model selection uses training/development groups;
conformal quantities use separate calibration groups; reported external groups are never fitted.

The Vassallo multi-laboratory acute-MEA archive was downloaded and hash-pinned but excluded from
sequential replay. Its processed workbook provides baseline summaries rather than a validated
phase-resolved dose sequence, and the raw timestamps do not retain a trustworthy phase-to-dose map.
Pooling it would invent the acquisition order. Imaging datasets remain separate because image
quality and organoid area are different estimands with different cost units.

## Probabilistic response model

The response surface uses a robust median prior, Ledoit-Wolf shrinkage covariance, and a structural
Matern-like dose-time kernel. Conditioning on observed cells `O` gives:

$$
\mu_{\mathrm{post}} = \mu + K_{:,O}
\left(K_{O,O}+\sigma_n^2 I\right)^{-1}(y_O-\mu_O)
$$

$$
K_{\mathrm{post}} = K - K_{:,O}
\left(K_{O,O}+\sigma_n^2 I\right)^{-1}K_{O,:}
$$

A local biological-analog ensemble supplies a conservative covariance contribution from nearby
training surfaces. This component belongs to the frozen model shared by all policies; no external
result was used to retune it.

## Canonical acquisition policy

The selected batch greedily maximizes conditional posterior variance per unit measurement cost:

$$
s(c)=\frac{K_{\mathrm{post}}(c,c)}{\operatorname{cost}(c)}
$$

After selecting candidate `c`, covariance is conditioned before selecting the next member. Equal
costs reduce exactly to conditional maximum variance. Every batch includes the required vehicle
anchor and only current-day wells are eligible.

The rejected effect-weighted acquisition multiplied covariance reduction by estimated effect
probability and added a direct-effect bonus. On the largest confirmation it achieved 96.48% recall,
while maximum variance achieved 97.90%. The added exploitation term increased complexity without a
reliable benefit and was removed from the canonical policy.

## Uncertainty, shift, and stopping

Conformal upper bands protect potential effects and Mahalanobis-style shift scores can trigger
ABSTAIN. These are uncertainty displays and refusal signals, not proof that stopping is safe under
arbitrary group shift.

Stopping release uses an exact finite-sample risk limit. If `x` of `n` independent validation stops
are unsafe, the one-sided Clopper-Pearson upper bound is:

$$
U(n,x,\alpha)=F^{-1}_{\operatorname{Beta}(x+1,n-x)}(1-\alpha)
$$

STOP is authorized only when this bound is at most 0.05 and the current group is inside the
calibrated shift boundary. Otherwise candidate STOP becomes COMPLETE; shifted groups become
ABSTAIN. Current evidence has `n=3`, `x=2`, upper bound 0.983, so stopping remains locked.

## Independent results

| Policy | Protected recall | Effect precision | Hidden NRMSE | Direct capture |
|---|---:|---:|---:|---:|
| **Conditional maximum variance** | **97.87%** | 32.04% | **0.334** | 42.68% |
| Effect-weighted acquisition | 97.47% | 31.97% | 0.342 | **44.16%** |
| Maximin | 97.36% | 33.56% | 0.353 | 41.43% |
| Random | 96.20% | **35.96%** | 0.362 | 39.50% |
| Fixed sentinels | 95.41% | 35.44% | 0.348 | 39.09% |

Maximum variance improved recall over fixed sentinels and random acquisition. Its recall advantage
over maximin was 0.51 points (CI -0.44 to 1.51; p=0.3458) and over effect-weighted acquisition was
0.40 points (CI -0.60 to 1.32; p=0.4272), so neither is claimed as a recall win. Its NRMSE advantage
was significant against all four comparators, including -0.0075 versus effect-weighted acquisition
(CI -0.0152 to -0.00066; p=0.0464).

On 53 Tipping Point groups, maximum variance achieved 98.01% protected recall, NRMSE 0.406, and
43.28% direct capture across decision rounds 2 and 3. This is the primary group-shift stress test.

The precision-recall tradeoff is explicit: conformal protection raises recall while marking many
unaffected cells as potentially affected. The final system optimizes false-negative protection, not
positive-call efficiency.

## Independent imaging stress tests

### Organ-on-chip microscopy quality

The public OOC archive contains 3,072 expert-labelled images across six cell types and 59 acquisition
dates (Zenodo 10.5281/zenodo.10203721). The record states CC BY 4.0; the associated paper states CC
BY-SA, so GATES-ACT records the ambiguity and redistributes no source images. Acquisition date is the
strict grouping boundary: 39 dates train, five select models, five calibrate release, and the last ten
form one protected chronological test of 1,373 images.

The benchmark compared HOG/LBP logistic regression, a random forest, an ImageNet MobileNetV3 frozen
linear probe, and class-weighted MobileNetV3 fine-tuning. Fine-tuning won development selection and
reached protected AUROC 0.647 and average precision 0.673. The strongest classical average precision
was 0.650. Calibration shifted severely: at a nominal 0.5 threshold, bad-image recall was only 9.4%.
The independent release dates authorized no automatic PASS region under the exact 5% risk limit.
Operational coverage is therefore 0%, with every image sent to RETAKE_OR_REVIEW. This module exposes
a failure mode and supplies a future benchmark; it does not currently save review effort.

### Cross-laboratory brain-organoid segmentation

The second archive contains 1,407 images of 64 organoids, four clones, ten days, and two laboratories
(Zenodo 10.5281/zenodo.10301912). A deterministic clone-balanced split assigns 36 organoids to
training, 12 to calibration, and 16 to protected testing. The LR-ASPP model trains only on Lab A.
The same protected organoids are evaluated in Lab A and unseen Lab B.

| Protected result | Lab A | Lab B |
|---|---:|---:|
| LR-ASPP mean Dice | 0.968 | 0.942 |
| LR-ASPP mean area error | 4.08% | 9.87% |
| Images with area error above 10% | 6.25% | 22.73% |
| Published SegFormer mean Dice | 0.960 | 0.960 |

LR-ASPP beats SegFormer Dice in Lab A by 0.0079 across the 16 organoids (14 wins; exact paired
p=0.00049) and loses in Lab B by 0.0181 (0 wins; p=0.000031). Pixel/TTA uncertainty accepted 99.4%
of Lab-B images and therefore failed as a shift safeguard. A post-hoc acquisition-fingerprint gate
flagged all Lab-B images and released none; because that detector followed the observed failure, it
is secondary evidence rather than an untouched claim. GATES-ACT adopts the stronger published
multi-lab SegFormer for segmentation and refuses automated release in an uncalibrated laboratory.

These experiments establish a boundary, not a multimodal claim: neither archive contains paired MEA
responses, so imaging does not change the active-acquisition result.

## Competition-method study

Public implementation detail for the leading 2025 World AI4S and Pazhou finalists is incomplete, so
no competitor code or rank is invented. Official material shows the recurring evaluation pattern:
real industrial data, a scientifically important decision, deployable workflow, and evidence that
survives outside the development setting. The World AI4S final had 30 teams across five applied
tracks; the Pazhou final reported 183 finalists from 8,131 teams and explicitly combined technical
and application assessment. GATES-ACT applies those lessons through frozen group-level tests,
laboratory constraints, executable refusal, reproducible audits, and explicit negative results.

Methodologically, the closest public baselines are multi-output Gaussian processes, active
kernelized matrix completion, and expected integrated variance reduction. GATES-ACT directly tests
the missing pieces in those families: chronological feasibility, effect recall, actual session cost,
group shift, and stopping safety. Conditional maximum variance remains canonical because the more
elaborate biological-utility score did not improve external recall.

## Release decision and limitations

The acquisition system is released for retrospective replay and prospective shadow-mode
recommendations. Early stopping is not released. Complete plates must be recorded while logging
recommendations until at least 59 independent zero-failure candidate stops are available; any
failure increases the evidence requirement.

Remaining limitations:

- no validated physical-session savings;
- low effect precision under external group shift;
- no prospective wet-lab intervention study;
- OOC image-quality automation has zero authorized coverage under chronological shift;
- brain-organoid automation is not released for an uncalibrated laboratory;
- exact risk control assumes validation stops are independent and exchangeable with deployment.

## Reproducibility

Public archives are pinned by byte count and SHA-256. Frozen external traces are hashed inside the
canonical evidence file. Reproduce the focused audit with:

```bash
uv sync --frozen --extra dev
uv run python scripts/verify_gates_act_v3_sources.py
uv run python scripts/audit_gates_act_core.py
uv run python scripts/verify_gates_act_core.py
uv run python scripts/audit_gates_act_vision.py
uv run pytest
```

Original GATES and earlier GATES-ACT evidence remain unchanged and auditable.

## Sources

- NTP/EPA Phase-1 DNT MEA: https://doi.org/10.22427/NTP-DATA-500-018-001-000-5
- Brown et al. MEA: https://doi.org/10.23719/1412682
- EPA Tipping Point data: https://doi.org/10.23719/1407690
- Vassallo multi-laboratory archive: https://doi.org/10.23719/1517583
- Shafer et al., network formation assay: https://doi.org/10.1093/toxsci/kfz052
- Vassallo et al., acute neurotoxicity testing: https://doi.org/10.1016/j.neuro.2016.03.019
- OOC Image Dataset: https://doi.org/10.5281/zenodo.10203721
- OOC quality-classification paper: https://doi.org/10.3390/data9020028
- Cross-laboratory brain-organoid images: https://doi.org/10.5281/zenodo.10301912
- Brain-organoid monitoring paper: https://doi.org/10.1038/s41597-024-03330-z
- World AI4S Prize: https://competition.ai4s.com.cn/
- 2025 Pazhou winner list: https://www.aicompetition-pz.com/4th/event_detail/105
