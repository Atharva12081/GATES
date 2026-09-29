# GATES demo video — 4:45 shot-by-shot script

**Presenter:** Atharva Parande

Target runtime: **4 minutes 45 seconds**. Record at 1080p with the browser at 100% zoom.

## 0:00–0:15 — Hook

An experiment may already contain enough information to predict its endpoint. But prediction
confidence alone does not tell us whether it is safe to stop measuring.

## 0:15–0:35 — Problem

Longitudinal biological experiments follow fixed schedules. Each additional time point requires
acquisition, storage, and analysis, even when some trajectories have already become decisive.

## 0:35–0:55 — What GATES does

GATES turns prediction into a sequential decision. It says CONTINUE while evidence is weak, STOP
when a group-calibrated rule passes, and ABSTAIN when the trajectory is unfamiliar. It never trains
inside the demo; every result comes from held-out evidence. GATES does not only ask what will
happen. It asks whether we know enough to stop measuring.

## 0:55–1:30 — Successful example

Load the frozen representative correct-stop case. Advance time. Show probability and OOD ratio,
then the STOP decision, final endpoint, decision time, and saved fraction. Open its exact evidence
row.

Start at 12 h and advance through 48 h. Pause on experiment ID, observations consumed, endpoint
prediction, OOD ratio, and STOP. Ground truth remains hidden until the 72 h endpoint.

## 1:30–1:55 — Hard/refused example

Load the representative ABSTAIN case. Show that the OOD boundary is crossed and the unit continues
to the 72 h protocol. Emphasize that refusal trades savings for fewer premature errors.

## 1:55–2:25 — Evaluation design

Show the hierarchy: 114,510 published morphometric observations nested within 988 medaka retinal
organoids, nested within 11 independent experiments. Entire experiments are held out from fitting,
calibration, and refusal-threshold estimation.

## 2:25–2:55 — Main result

Show the risk–savings frontier and the frozen 5% row: 988 organoids, 11 held-out experiments, 439
early stops, 25 errors, 5.69% EESR, 44.43% coverage, and 12.26% savings. Show the exact 95% interval,
3.72–8.29%, and the wider experiment-bootstrap interval.

## 2:55–3:20 — Why refusal matters

Show the ablation: without refusal, 8.60% EESR and 13.65% estimated savings; with refusal, 5.69%
and 12.26%. Stopping early is easy. Knowing when not to stop is the harder problem.

## 3:20–3:45 — E007 failure

Show E007 at 30.61% EESR among early stops. Its erroneous stops did not cross the frozen OOD
threshold. State: aggregate performance can hide severe experiment-specific failure, and the
mechanism is undetermined.

## 3:45–4:10 — Baselines and negative results

Briefly show CBES-style at 15.51%, the safest tested fixed-time method at 13.82%, Lens transfer at
17.82%, and irregular sampling at 10.61%. Do not imply that methods at different savings are
directly equivalent.

## 4:10–4:30 — Reproducibility

Run `just verify-final`, then show the manifest, transparent erratum, and figure source map. State
that the audit-only correction produced zero changed scientific results after full recomputation.

## 4:30–4:45 — Limitation and close

GATES is retrospective evidence on medaka retinal organoids, not a deployment guarantee. The next
step is prospective validation in independent laboratories and direct organ-on-chip systems. GATES
turns prediction into a decision about whether another measurement is still necessary.
