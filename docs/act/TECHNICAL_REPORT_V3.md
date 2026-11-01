# GATES ACT Adaptive Neural Assay Measurement Under Explicit Risk Limits

**Technical report and improvement audit**  
Atharva Parande  
1 November 2026

## Executive summary

GATES-ACT is a laboratory-constrained adaptive measurement system for longitudinal neural microelectrode-array assays. It predicts unmeasured dose-by-time responses, selects the next physically eligible batch, protects biologically meaningful effects, and refuses unsafe automation. The final system is deliberately simpler than the preceding research candidate: conditional maximum posterior variance replaces the effect-weighted acquisition score because it produced the best pooled external recall and the lowest hidden-response error.

Across 76 independent compound groups from three public neural-MEA datasets, the canonical policy achieved **97.87% protected effect recall**, **0.334 hidden-response NRMSE**, and **42.68% direct effect capture**. Relative to fixed sentinels, recall increased by 2.46 percentage points (95% grouped-bootstrap interval 0.94 to 4.08; paired sign-flip p=0.00187). Relative to random acquisition, recall increased by 1.67 points (0.48 to 2.86; p=0.00318). Hidden-response NRMSE was significantly lower than all four comparators. Recall differences against maximin and the rejected effect-weighted policy were not resolved, so no universal superiority claim is made.

The most consequential safety result is negative. Three external maximum-variance candidate stops included two unsafe outcomes, giving a one-sided 95% Clopper-Pearson upper risk bound of 98.3%, far above the 5% release limit. Early stopping is therefore disabled. The released workflow returns COMPLETE or ABSTAIN and claims zero avoided physical recording sessions. At least 59 independent zero-failure validation stops would be needed before a 5% upper risk limit could be supported.

![GATES-ACT system architecture](../../artifacts/act/submission_figures/01_system_architecture.png)

## What improved

The improvement is not a cosmetic version increment. Evidence expansion changed the selected method, the supported claim, and the deployable action.

| Dimension | Earlier state | Final state | Verified improvement |
|---|---|---|---|
| Evidence boundary | Six organoid replicates, then one neural-MEA development study | 76 external compound groups from three public MEA sources | Larger and biologically independent evaluation boundary |
| Acquisition rule | Effect-weighted biological utility | Conditional maximum posterior variance | Lower pooled NRMSE and higher pooled recall |
| Leakage control | Retrospective grid replay | Compound-group isolation and chronological eligibility | Future days and related wells cannot cross boundaries |
| Cost claim | Grid-cell reduction | Whole-session savings only | Measurement-count efficiency is no longer mislabeled as laboratory savings |
| Stopping | Candidate retrospective STOP | STOP disabled after external failures | Unsafe automation is prevented by an executable release gate |
| Imaging | Potential multimodal extension | Independent fail-closed stress tests | No unsupported paired-image/MEA claim |
| Reproducibility | Experimental artifacts | Hash-pinned sources, frozen traces, executable audits | Headline results can be recomputed and fail closed on changed evidence |

![Evidence evolution](../../artifacts/act/submission_figures/04_evidence_evolution.png)

## Scientific question

For compound group (g), dose rank (d), and recording day (t), let (y_g(d,t)) denote the plate-matched vehicle-normalized neural response. At each scheduled day, the system may select only currently eligible dose wells and must include the required vehicle anchor. Future days are unavailable. The system must estimate the unmeasured response surface while protecting affected cells defined by

$$
\left|\log(1+y_{\mathrm{treatment}})-\log(1+y_{\mathrm{vehicle}})\right| \ge 0.40.
$$

The primary estimands are protected effect recall, hidden-cell NRMSE, direct capture of affected cells, and unsafe-stop risk. Avoided physical recording sessions are a separate operational estimand. A well already allocated to a plate is never counted as saved.

## Evidence boundary and leakage control

| Public source | Independent groups | Role |
|---|---:|---|
| NTP EPA Phase 1 DNT MEA network formation | 17 | held-out confirmation |
| Brown neural network ontogeny MEA | 6 | external compatibility test |
| EPA Tipping Point determinations | 53 | largest shifted confirmation |
| **Total** | **76** | pooled group-weighted evidence |

Compound identity is the isolation boundary. Biological repeats, wells, dose levels, and timepoints from one compound remain in one split. Vehicle normalization is plate matched. Replicates are aggregated only after normalization. Training groups estimate the prior, calibration groups define conformal and shift quantities, and external groups remain untouched by fitting.

The Vassallo multi-laboratory acute-MEA archive was downloaded and hash-pinned but excluded from sequential replay because its processed release does not retain a validated phase-resolved dose sequence. Including it would require inventing an acquisition order. This exclusion reduces sample size but preserves the estimand.

## Model and acquisition rule

The response model combines a robust population median, Ledoit-Wolf shrinkage covariance, and a structural dose-time kernel. For observed cells (O), posterior mean and covariance are

$$
\mu_{\mathrm{post}}=\mu+K_{:,O}(K_{O,O}+\sigma_n^2I)^{-1}(y_O-\mu_O),
$$

$$
K_{\mathrm{post}}=K-K_{:,O}(K_{O,O}+\sigma_n^2I)^{-1}K_{O,:}.
$$

The batch policy greedily selects the eligible candidate with greatest posterior variance per unit cost,

$$
s(c)=\frac{K_{\mathrm{post}}(c,c)}{\operatorname{cost}(c)},
$$

then conditions covariance before choosing the next member. This within-batch update prevents redundant selection. Equal costs reduce the rule to conditional maximum variance.

The rejected alternative multiplied uncertainty reduction by an estimated effect probability and added a direct-effect bonus. It captured slightly more affected cells directly, but its probabilities were too imprecise under group shift. On the largest confirmation set it reached 96.48% recall versus 97.90% for maximum variance. The added term was removed rather than defended on complexity alone.

## Comparative results

![Policy comparison](../../artifacts/act/submission_figures/02_policy_performance.png)

| Policy | Protected recall | Effect precision | Hidden NRMSE | Direct capture |
|---|---:|---:|---:|---:|
| **Conditional maximum variance** | **97.87%** | 32.04% | **0.334** | 42.68% |
| Effect-weighted acquisition | 97.47% | 31.97% | 0.342 | **44.16%** |
| Maximin | 97.36% | 33.56% | 0.353 | 41.43% |
| Random | 96.20% | **35.96%** | 0.362 | 39.50% |
| Fixed sentinels | 95.41% | 35.44% | 0.348 | 39.09% |

Maximum variance is the canonical choice because it simultaneously has the highest pooled recall and lowest pooled NRMSE. Its NRMSE advantage is significant against effect-weighted acquisition (-0.0075; 95% interval -0.0152 to -0.00066; p=0.0464), maximin (-0.0186; p=0.0106), fixed sentinels (-0.0138; p=0.0120), and random acquisition (-0.0270; p<0.0001).

![Paired recall improvement](../../artifacts/act/submission_figures/03_paired_improvement.png)

The recall advantage over fixed sentinels and random acquisition remains positive under group-level paired resampling. The intervals against maximin and effect-weighted acquisition cross zero. The report therefore distinguishes “best pooled choice” from “significantly better on every metric.”

On the 53-group EPA Tipping Point stress test, maximum variance achieved 98.01% protected recall, 0.406 NRMSE, and 43.28% direct effect capture across decision rounds 2 and 3. This is the principal evidence that the method survives source shift.

## Uncertainty and refusal

Conformal upper bands flag any cell whose uncertainty interval can cross the biological threshold. This protects false negatives at the cost of low positive precision. A Mahalanobis-style group-shift score can trigger ABSTAIN. Neither device proves that a proposed early stop is safe.

Stopping authority is controlled by an exact finite-sample gate. If (x) of (n) independent validation stops are unsafe, the one-sided upper confidence bound is

$$
U(n,x,\alpha)=F^{-1}_{\operatorname{Beta}(x+1,n-x)}(1-\alpha).
$$

STOP is authorized only if (U\le0.05) and the current group is inside the calibrated shift boundary. Otherwise a candidate stop becomes COMPLETE, while a shifted group becomes ABSTAIN.

![Stopping safety evidence](../../artifacts/act/submission_figures/05_stopping_safety.png)

This refusal is a product improvement. It prevents a retrospective efficiency result from becoming an unsupported laboratory-savings claim. The current deployment status is shadow mode: complete plates continue while recommendations and outcomes are logged prospectively.

## Independent imaging stress tests

Two imaging studies test the refusal architecture, not the MEA acquisition claim.

### Organ on chip image quality

The public archive contains 3,072 expert-labelled images across six cell types and 59 acquisition dates. Date is the grouping boundary. The last ten dates form a protected chronological test of 1,373 images. Fine-tuned MobileNetV3 achieved AUROC 0.647 and average precision 0.673, improving average precision over the frozen representation from 0.595. Calibration shifted sharply: bad-image recall at the nominal threshold was only 9.4%. No automatic-pass region met the 5% risk limit, so authorized automatic coverage is 0%.

### Cross laboratory brain organoid segmentation

The second archive contains 1,407 images of 64 organoids across two laboratories. A Lab-A-only LR-ASPP model achieved 0.968 mean Dice in Lab A and 0.942 in Lab B. On the same 16 protected organoids, it beat the published multi-lab SegFormer in Lab A by 0.0079 Dice but lost in Lab B by 0.0181 Dice (0 of 16 wins; exact paired p=0.000031). Pixel and test-time-augmentation uncertainty accepted 99.4% of Lab-B images and failed as a shift safeguard. The released system therefore adopts the stronger published multi-lab measurement model and withholds automated release in an uncalibrated laboratory.

Neither image archive contains paired MEA responses. Imaging remains an independent quality and shift audit and does not alter the active-acquisition estimate.

## Implementation and reproducibility

The repository separates immutable evidence, acquisition logic, audits, protocols, figures, and report builders. Public archives are pinned by file size and SHA-256. Frozen external traces are hashed inside the canonical evidence record. Verification recomputes source boundaries, pooled metrics, paired comparisons, exact risk limits, disabled-stop behavior, protected imaging metrics, and data hashes.

Focused reproduction:

```bash
uv sync --frozen --extra dev
uv run python scripts/verify_gates_act_v3_sources.py
uv run python scripts/audit_gates_act_core.py
uv run python scripts/verify_gates_act_core.py
uv run python scripts/audit_gates_act_vision.py
uv run pytest
uv run python scripts/build_submission_report.py
```

## Limitations

- The study is retrospective and does not yet include a prospective wet-lab intervention.
- The final policy protects recall but has low positive effect precision under source shift.
- No physical-session saving is claimed because STOP is disabled.
- Exact stopping control assumes that validation stops are independent and exchangeable with deployment.
- OOC image-quality automation has zero authorized coverage under chronological shift.
- Brain-organoid automation is not authorized for an uncalibrated laboratory.
- Public finalist implementations are incomplete, so no unsupported competitor ranking is inferred.

## Conclusion

GATES-ACT improves adaptive neural-assay measurement in three defensible ways. It expands evaluation from a small retrospective replay to 76 external compound groups; it replaces an elaborate effect-weighted rule with the simpler policy that performs best; and it converts stopping from an optimistic heuristic into an exact release decision that currently refuses automation. The resulting system is less promotional and more useful: high-recall acquisition is supported, uncertainty is visible, savings are not overstated, and unsafe early stopping is impossible in the released workflow.

## References

1. NTP EPA Phase 1 DNT MEA curated release. https://doi.org/10.22427/NTP-DATA-500-018-001-000-5
2. Brown et al. neural-network ontogeny MEA data. https://doi.org/10.23719/1412682
3. EPA Tipping Point determinations. https://doi.org/10.23719/1407690
4. Shafer et al. Network formation assay. *Toxicological Sciences*. https://doi.org/10.1093/toxsci/kfz052
5. Vassallo et al. Acute neurotoxicity testing. *NeuroToxicology*. https://doi.org/10.1016/j.neuro.2016.03.019
6. OOC Image Dataset. https://doi.org/10.5281/zenodo.10203721
7. Cross-laboratory brain-organoid images. https://doi.org/10.5281/zenodo.10301912
8. Giraldo Gutierrez et al. Multi-output Gaussian processes. *npj Precision Oncology* 2024. https://doi.org/10.1038/s41698-024-00685-1
9. D'Souza et al. Active kernelized matrix completion. *BMC Bioinformatics* 2015. https://doi.org/10.1186/s12859-015-0713-8
10. Kontoudis and Stilwell. Expected integrated variance reduction. ACC 2023. https://doi.org/10.23919/ACC55779.2023.10155994
