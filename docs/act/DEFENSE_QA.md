# GATES-ACT difficult judge Q&A

## Why is the final policy “just” maximum variance?

Because it was the strongest measured policy. Across 76 external compound groups it had the highest
pooled protected recall and lowest NRMSE. The objective is scientific performance, not novelty for
its own sake. The contribution that survived is the laboratory-constrained batch implementation,
conservative effect protection, group-shift refusal, and exact stopping release gate.

## Did you tune the final rule on the external datasets?

No new acquisition score was tuned after external evaluation. We adopted the already-frozen
maximum-variance baseline unchanged. Its model, batch size, chronology, and vehicle constraint were
identical to the original comparison. External evidence selects the strongest defensible alternative
but does not fit new parameters.

## Is maximum variance significantly better than every baseline?

No. It significantly improves recall over fixed sentinels and random selection, and improves NRMSE
over all four comparators. Recall differences versus maximin and effect-weighted acquisition are not
significant. The claim is that it is the best supported canonical choice, not universally dominant.

## What happened to the effect-weighted acquisition?

It over-exploited effect probabilities that had low external precision. On the largest shifted
confirmation it reached 96.48% recall versus 97.90% for maximum variance. Its pooled direct capture
was slightly higher, but pooled recall and NRMSE were worse. The extra term was removed from the
released policy and retained as an ablation.

## Why is effect precision only 32%?

The conformal rule deliberately protects against false negatives by marking a cell as potentially
affected when its uncertainty band crosses the biological threshold. Under group shift this is
conservative and creates false positives. High recall does not imply accurate positive classification.

## Can GATES-ACT stop experiments early?

Not in the released system. Three external maximum-variance candidate stops included two unsafe
outcomes. The exact one-sided 95% unsafe-risk upper bound is 98.3%. STOP becomes COMPLETE, and
shifted groups receive ABSTAIN.

## Why require 59 zero-failure validation stops?

For zero failures, the one-sided 95% Clopper-Pearson upper bound is `1 - 0.05^(1/n)`. The smallest
`n` making that value at most 0.05 is 59. This is a minimum release condition, not a guarantee under
arbitrary distribution shift.

## What cost savings do you claim?

Zero physical-session savings. The acquisition policy improves information obtained at a fixed
measurement budget, but allocated wells and scheduled sessions are not counted as saved. Savings
will be claimed only after stopping is prospectively authorized and actually avoids a session.

## Why exclude Vassallo from sequential replay?

The Vassallo archive was downloaded and audited, but its public processed files do not preserve the
phase-resolved dose response required for sequential replay. The raw timestamps lack a validated
phase-to-dose mapping. Pooling it would change the estimand or require an invented acquisition order.

## Did the imaging program strengthen the primary acquisition claim?

No. The OOC and brain-organoid archives contain no paired MEA outcomes, so they cannot validate an
imaging-informed acquisition policy. They strengthen the safety case by exposing two failures:
chronological calibration collapse in OOC quality classification, and cross-laboratory degradation
in a Lab-A-only segmenter. The final system keeps those modules independent and fail-closed.

## What did the OOC model achieve?

Fine-tuned MobileNetV3 was the strongest locked candidate on the 1,373-image chronological test:
AUROC 0.647 and average precision 0.673, versus 0.650 average precision for the strongest classical
model. Yet its nominal-threshold bad-image recall was only 9.4%. The independent release dates
authorized zero automatic-pass coverage at the 5% risk limit, so every image remains review-only.

## Why mention an imaging model that cannot automate PASS?

Because the negative result changes the product. It prevents a plausible but unsafe image gate from
being marketed as a cost-saving feature. The reproducible benchmark and abstention path remain useful
for future multi-site calibration, while the released system makes no review-savings claim.

## Why adopt the published SegFormer?

The Lab-A-only LR-ASPP reached 0.968 Dice in Lab A, but dropped to 0.942 in Lab B and lost to the
published multi-lab SegFormer by 0.018 Dice across the same 16 protected organoids (0/16 wins,
exact paired p=0.000031). SegFormer is therefore the stronger measurement model. An uncalibrated
laboratory still receives ABSTAIN because the published evaluation is not a deployment guarantee.

## Did your uncertainty score detect the Lab-B failure?

No. Pixel and test-time-augmentation uncertainty accepted 99.4% of Lab-B images while 22.7% had
area error above 10%. A post-hoc acquisition fingerprint flagged all Lab-B images, but that analysis
followed the observed failure. It is disclosed as secondary evidence and releases zero Lab-B images.

## What is independently verified?

The core verifier checks source hashes, the 76-group boundary, complete policy coverage, pooled
metrics, the exact binomial risk bound, and the disabled-stop action. A separate vision audit
recomputes protected AUROC, average precision, Dice, area error, specimen/date separation, and both
dataset hashes. Both fail closed on changed evidence.

## What would change your decision?

A preregistered prospective shadow study using complete plates. If at least 59 independent eligible
candidate stops occur with zero unsafe outcomes and the shift boundary remains valid, STOP can be
reconsidered. Any failure raises the evidence requirement.
