# GATES ACT Supplementary Methods and Results

Atharva Parande  
1 November 2026

## S1 Scope

This supplement records the frozen evidence boundary, model specification, acquisition comparators, statistical analyses, release logic, imaging stress tests, and reproducibility checks supporting the accompanying technical report. It distinguishes development evidence from external confirmation and separates measurement-count efficiency from physical laboratory savings.

## S2 Neural MEA datasets

### S2.1 Inclusion rule

A dataset is eligible for sequential replay only if it exposes well-level neural electrophysiology, repeated chronological measurements, dose identity, matched vehicle context, stable compound groups, and a defensible measurement-cost unit. Sources with terminal outcomes, incompatible endpoints, or missing acquisition order are not pooled.

| Source | Endpoint | Groups | Sequential role | Decision |
|---|---|---:|---|---|
| NTP EPA Phase 1 DNT MEA | Mean firing rate | 17 | held-out confirmation | Include |
| Brown neural ontogeny MEA | Mean firing rate | 6 | external compatibility | Include |
| EPA Tipping Point | Mean firing rate | 53 | shifted confirmation | Include |
| Vassallo acute MEA | Baseline and acute summaries | not pooled | order audit | Exclude from replay |
| OOC microscopy | image quality | 59 dates | independent stress test | Do not pool |
| Brain organoid microscopy | segmentation | 64 organoids | independent stress test | Do not pool |

### S2.2 Group isolation

The compound is the atomic split unit. A compound and all of its biological repeats, technical wells, doses, and timepoints are assigned to exactly one of training, calibration, development, or external evaluation. No well-level random split is used. Chronological replay exposes only the current and previous recording days.

### S2.3 Response construction

For each plate and scheduled day, the response is the median plate-matched log transformed treated signal minus the corresponding vehicle response. Replicates are aggregated only after matched normalization. A cell is affected when the absolute log1p contrast is at least 0.40.

## S3 Probabilistic model

The response surface uses a robust median prior and a shrinkage covariance estimated from training groups. A structural kernel provides smoothness across dose rank and recording day. A local biological-analog ensemble contributes a conservative covariance term from nearby training surfaces. These quantities are frozen before external confirmation.

For observed indices (O),

$$
\mu_{\mathrm{post}}=\mu+K_{:,O}(K_{O,O}+\sigma_n^2I)^{-1}(y_O-\mu_O),
$$

$$
K_{\mathrm{post}}=K-K_{:,O}(K_{O,O}+\sigma_n^2I)^{-1}K_{O,:}.
$$

Numerical solves use regularization and symmetry checks. If conditioning is unstable, the run fails closed rather than emitting a decision.

## S4 Acquisition policies

All policies share the same response model, initialization, eligible candidate set, batch size, vehicle requirement, and measurement budget.

### S4.1 Conditional maximum variance

The canonical rule selects the eligible candidate with largest posterior variance per cost. After each selection, the covariance is conditioned before the next member of the same batch is chosen.

### S4.2 Effect weighted acquisition

The rejected research rule multiplies covariance reduction by an estimated probability of biological effect and adds a direct-effect bonus. It is retained as an ablation because it slightly increases direct capture but does not improve pooled external recall or NRMSE.

### S4.3 Maximin

Maximin selects the candidate maximizing minimum structural distance from already observed cells. It tests geometric coverage without posterior uncertainty.

### S4.4 Fixed sentinels

Fixed sentinels use a predetermined sparse dose panel at each eligible day. They represent a reproducible non-adaptive laboratory design.

### S4.5 Random acquisition

Random acquisition samples from the same eligible set under the same budget. Frozen seeds make the comparison reproducible.

## S5 Outcome definitions

| Outcome | Definition | Direction |
|---|---|---|
| Protected effect recall | Fraction of affected hidden cells covered by the conformal protection rule | Higher is better |
| Effect precision | Fraction of protected cells that are affected | Higher is better |
| Hidden NRMSE | RMSE on hidden cells normalized by group response scale | Lower is better |
| Direct effect capture | Fraction of affected cells measured by acquisition | Higher is better |
| Recording sessions saved | Entire scheduled sessions safely avoided | Higher is better |
| Unsafe stop | Candidate stop followed by a missed protected effect or unacceptable reconstruction | Zero required for release evidence |

## S6 Pooled policy results

| Policy | Groups | Recall | Precision | NRMSE | Direct capture |
|---|---:|---:|---:|---:|---:|
| Conditional maximum variance | 76 | 0.97868 | 0.32040 | 0.33448 | 0.42679 |
| Effect weighted | 76 | 0.97469 | 0.31974 | 0.34198 | 0.44164 |
| Maximin | 76 | 0.97360 | 0.33555 | 0.35311 | 0.41430 |
| Random | 76 | 0.96202 | 0.35962 | 0.36150 | 0.39505 |
| Fixed sentinels | 76 | 0.95410 | 0.35441 | 0.34827 | 0.39088 |

## S7 Paired comparisons

Group-weighted differences compare conditional maximum variance with each baseline. Confidence intervals are grouped bootstrap intervals. P values are paired sign-flip tests.

| Comparator | Metric | Difference | 95% interval | p value |
|---|---|---:|---:|---:|
| Effect weighted | Recall | +0.00399 | -0.00600 to +0.01316 | 0.4272 |
| Effect weighted | NRMSE | -0.00750 | -0.01520 to -0.00066 | 0.0464 |
| Maximin | Recall | +0.00508 | -0.00442 to +0.01507 | 0.3458 |
| Maximin | NRMSE | -0.01862 | -0.03611 to -0.00471 | 0.0106 |
| Fixed sentinels | Recall | +0.02459 | +0.00940 to +0.04078 | 0.00187 |
| Fixed sentinels | NRMSE | -0.01379 | -0.02508 to -0.00334 | 0.0120 |
| Random | Recall | +0.01666 | +0.00477 to +0.02858 | 0.00318 |
| Random | NRMSE | -0.02702 | -0.04471 to -0.01289 | <0.0001 |

Positive recall differences and negative NRMSE differences favor conditional maximum variance. Recall differences against effect-weighted acquisition and maximin are unresolved.

## S8 Stopping release gate

For (x) unsafe outcomes among (n) independent validation stops, the one-sided 95% Clopper-Pearson upper risk bound is

$$
U(n,x,0.05)=F^{-1}_{\operatorname{Beta}(x+1,n-x)}(0.95).
$$

STOP authority requires all of the following:

1. the current group is inside the calibrated shift boundary;
2. reconstruction and protected-effect criteria are satisfied;
3. the upper unsafe-risk bound is no greater than 0.05; and
4. the evidence record and policy configuration match their frozen hashes.

External maximum-variance replay produced three candidate stops, two of which were unsafe. The upper bound is 0.983. The release action is therefore COMPLETE for in-boundary groups and ABSTAIN for shifted groups. No physical-session saving is claimed. If zero unsafe outcomes are observed, 59 independent validation stops are the minimum needed for the upper bound to fall below 0.05.

## S9 Imaging stress tests

### S9.1 Organ on chip image quality

The OOC archive contains 3,072 images from 59 acquisition dates. Dates, rather than images, define the split. Thirty-nine dates are used for training, ten for calibration, and the last ten for protected chronological testing. The protected test contains 1,373 images.

| Model or release quantity | Result |
|---|---:|
| Fine-tuned MobileNetV3 AUROC | 0.6471 |
| Fine-tuned MobileNetV3 average precision | 0.6726 |
| Frozen representation average precision | 0.595 |
| Bad-image recall at nominal threshold | 9.4% |
| Authorized automatic-pass coverage | 0% |

The model improves ranking performance but fails the release calibration test. Every image is routed to retake or review.

### S9.2 Cross laboratory brain organoid segmentation

The archive contains 1,407 images of 64 organoids. A deterministic clone-balanced split assigns 36 organoids to training, 12 to calibration, and 16 to protected testing. LR-ASPP is trained only on Lab A and evaluated on the same protected organoids in Lab A and Lab B.

| Protected result | Lab A | Lab B |
|---|---:|---:|
| LR-ASPP mean Dice | 0.96798 | 0.94160 |
| LR-ASPP mean area error | 4.08% | 9.87% |
| Images above 10% area error | 6.25% | 22.73% |
| Published SegFormer mean Dice | 0.960 | 0.960 |

LR-ASPP exceeds the published SegFormer in Lab A by 0.0079 Dice but is lower by 0.0181 in Lab B. An exact paired sign test on the 16 protected organoids gives 0 of 16 Lab-B wins and p=0.000031. The post-hoc acquisition fingerprint flags all Lab-B images, but because it was created after observing the failure it is secondary evidence and releases no images.

## S10 Source integrity

The following evidence classes are hash-pinned and checked by the focused verifier:

- public archive byte counts and SHA-256 digests;
- source-to-group mappings;
- frozen external acquisition traces;
- policy configuration and stopping lock;
- pooled policy summaries and paired comparisons;
- protected imaging predictions and split manifests; and
- disabled-stop deployment behavior.

Any changed source, missing policy row, altered group boundary, inconsistent metric, or unlocked stop produces a failing audit.

## S11 Reproduction

```bash
uv sync --frozen --extra dev
uv run python scripts/verify_gates_act_v3_sources.py
uv run python scripts/audit_gates_act_core.py
uv run python scripts/verify_gates_act_core.py
uv run python scripts/audit_gates_act_vision.py
uv run pytest
uv run python scripts/build_submission_report.py
```

Generated outputs are written to `output/pdf`, and report figures are written to `artifacts/act/submission_figures`.

## S12 Claim boundary

The evidence supports high-recall adaptive acquisition and lower hidden-response error at a fixed measurement budget. It does not support universal superiority, prospective wet-lab benefit, physical-session savings, autonomous early stopping, automatic OOC image acceptance, or brain-organoid deployment in an uncalibrated laboratory. Imaging datasets do not contain paired MEA outcomes and therefore do not validate an imaging-informed acquisition policy.

## S13 Data and literature references

1. NTP EPA Phase 1 DNT MEA curated release. https://doi.org/10.22427/NTP-DATA-500-018-001-000-5
2. Brown neural network ontogeny MEA data. https://doi.org/10.23719/1412682
3. EPA Tipping Point determinations. https://doi.org/10.23719/1407690
4. Vassallo multi-laboratory archive. https://doi.org/10.23719/1517583
5. Shafer et al. Network formation assay. https://doi.org/10.1093/toxsci/kfz052
6. OOC Image Dataset. https://doi.org/10.5281/zenodo.10203721
7. Cross-laboratory brain-organoid images. https://doi.org/10.5281/zenodo.10301912
