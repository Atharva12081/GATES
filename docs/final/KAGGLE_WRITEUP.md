# GATES: Knowing When to Stop Observing

Longitudinal biological experiments are often observed on a fixed schedule even when their outcome
becomes predictable earlier. GATES asks a different question from ordinary prediction: when is the
available evidence reliable enough to stop observing?

Across 988 retinal organoids from 11 independent experiments, evaluated with entire experiments
held out, Full GATES achieved 5.69% observed erroneous-early-stop rate while saving 12.26% of
scheduled observations at 44.43% early-decision coverage. Removing the refusal layer increased
error to 8.60%, while our CBES-style baseline reached 15.51%.

The result is empirical rather than a universal guarantee: several held-out experiments remained
difficult, Lens transfer failed, and an experiment-level refusal extension was unstable across
operating points.

## The decision, not just the prediction

At each observation time, GATES returns:

- **CONTINUE** when the evidence is insufficient;
- **STOP** when a separately calibrated confidence rule passes; or
- **ABSTAIN** when the prefix is outside the training-derived feature-distance boundary.

The independent unit is an organoid well, and the generalization boundary is a complete experiment.
All prefixes from the held-out experiment are excluded from fitting and calibration.

## Main evidence

At the frozen 5% operating point, Full GATES made 439 early decisions and 25 errors: 5.69% EESR
(exact 95% CI 3.72–8.29%), 44.43% coverage, and 12.26% savings. Unit-level refusal prevented 16
errors relative to GATES without refusal while costing 1.38 percentage points of savings.

The experiment-level extension was encouraging only at one point. At 5%, it refused E007/E008 and
prevented 17 errors. At 10%, it accepted every unsafe experiment and refused every safe one. We
therefore report it as an optional research extension, not part of the core architecture.

## Honest failure cases

E007 produced 15 errors among 49 early stops. Available numerical evidence suggests possible
conditional/concept shift, but is insufficient to identify a mechanism. E012 produced six
high-confidence false positives at 48 h, consistent with calibration failure. Lens transfer reached
17.82% EESR, and several stress tests exceeded the nominal target.

## Reproducibility

The repository includes the locked environment, source and dataset hashes, frozen protocols,
per-unit decisions, exact confidence intervals, experiment bootstrap intervals, a master evidence
table, figure source mappings, and a demo that reads evidence without retraining.

GATES does not claim certified error control or deployment readiness. It demonstrates an auditable
way to trade observation cost against premature-decision risk—and to keep observing when the model
does not deserve trust.
