# GATES-ACT five-minute defense

## Slide 1 - The decision

GATES-ACT predicts unmeasured neural-assay responses and chooses the next feasible measurement.
The final acquisition policy is conditional maximum variance because it produced the highest pooled
effect recall and lowest reconstruction error. We removed the effect-weighted score when it failed
to beat this simpler baseline.

## Slide 2 - The evidence boundary

The external evaluation contains 76 independent compound groups from three public neural-MEA
datasets. Splits are by compound identity, so wells, repeats, doses, and timepoints from one
biological group never cross train, calibration, and evaluation boundaries.

## Slide 3 - The acquisition rule

At each scheduled recording day, GATES-ACT selects the required vehicle and then greedily chooses
the dose with the largest conditional posterior variance per unit cost. Covariance is updated inside
the batch to prevent redundant measurements. Future days are never visible.

## Slide 4 - The main result

Across 76 external groups, maximum variance achieved 97.87% protected effect recall and 0.334 hidden
NRMSE. It beat fixed sentinels by 2.46 recall points and random acquisition by 1.67 points. Recall
differences versus maximin and effect-weighted acquisition were unresolved, but NRMSE was lower than
every comparator.

## Slide 5 - The stopping result

Early stopping is disabled. Three external candidate stops included two unsafe outcomes. The exact
one-sided 95% upper bound on unsafe-stop risk is 98.3%, not 5%. Even with zero failures, 59
independent validation stops would be required before the 5% limit could be supported.

## Slide 6 - The imaging stress test

The 3,072-image OOC stress test exposed severe chronological calibration shift. Fine-tuned
MobileNetV3 reached 0.673 average precision, but no automatic-pass threshold met the 5% risk limit.
In 1,407 cross-laboratory brain-organoid images, the Lab-A-only model lost to the published
multi-lab SegFormer in Lab B. GATES-ACT therefore keeps imaging separate and fail-closed.

## Slide 7 - The contribution

The contribution is a high-recall, leakage-free adaptive measurement system with an executable risk
boundary that refuses unsafe automation. The live workflow issues COMPLETE or ABSTAIN and reports
zero physical-session savings until prospective shadow validation earns STOP authority. Imaging
adds an independently audited failure detector, not an invented multimodal claim.

## Close

GATES-ACT improved effect protection, adopted the strongest baseline when it won, and converted an
unsafe stopping heuristic into a finite-sample release gate. The next study is prospective shadow
deployment on complete plates, with recommendations logged but never used to terminate acquisition.
