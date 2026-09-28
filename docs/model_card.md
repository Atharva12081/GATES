# GATES Model Card

## Intended use

Retrospective research on whether longitudinal biological endpoints can be called before the final
scheduled observation. The model is designed to expose uncertainty, shift, and reason codes.

## Inputs

Prefix summaries of aggregate PI fluorescence and image-derived organoid morphology. Only values at
or before the current decision time are used.

## Output

Probability of low endpoint viability, predicted class, confidence, feature-distance score, and one
of STOP, CONTINUE, or ABSTAIN.

## Prohibited interpretation

Do not treat STOP as authorization to alter a real experiment, a medical recommendation, or a
validated biological conclusion. Do not apply the fitted artifacts to other assays or laboratories.

## Evaluation population

Twenty-one endpoint-labelled OrganoID gemcitabine dosage–replicate units evaluated through cyclic
replicate holdouts.

## Known failure modes

Small calibration samples, unseen acquisition conditions, missing or corrupted time points,
endpoint drift, dose-associated shortcuts, and overconfident extrapolation beyond the observed
population.

