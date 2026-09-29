# Feasibility defense — superseded

Use the current [25-question defense](final/DEFENSE.md).

## Why is this more than an endpoint classifier

The classifier is a replaceable component. The contribution is the sequential decision contract,
separate calibration role, shift-aware refusal, group-level evaluation, and explicit risk–cost
evidence.

## Why not random frame splits

Adjacent frames from one well are highly dependent. A random frame split would let the model see the
same experimental unit during training and testing and would exaggerate generalization.

## Is the target risk guaranteed

No. The current calibration set is too small for a strong low-risk finite-sample statement. The
software computes the confidence bound, records whether it passes, and otherwise labels the result
empirical only.

## Why logistic regression

With seven training units per cyclic fold, a simple regularized model is easier to audit and less
likely to overfit than a large temporal network. Architecture complexity is deferred until the
evaluation population grows.

## What would change the conclusion

Failure to reproduce an early-signal curve, unstable stopping across held-out experiments, or a
large second dataset showing no safe savings would invalidate the current project direction.
