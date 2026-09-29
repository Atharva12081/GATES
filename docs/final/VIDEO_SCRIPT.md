# GATES demo video — 4:45 shot-by-shot script

Target runtime: **4 minutes 45 seconds**. Record at 1080p with the browser at 100% zoom.

## 0:00–0:18 — Problem

Longitudinal experiments follow a fixed observation schedule, even when some outcomes become clear
early. Stopping early can save observation effort, but a wrong early decision can invalidate the
experiment.

## 0:18–0:38 — What GATES does

GATES turns prediction into a sequential decision. It says CONTINUE while evidence is weak, STOP
when a group-calibrated rule passes, and ABSTAIN when the trajectory is unfamiliar. It never trains
inside the demo; every result comes from held-out evidence.

## 0:38–1:18 — Successful example

Load the frozen representative correct-stop case. Advance time. Show probability and OOD ratio,
then the STOP decision, final endpoint, decision time, and saved fraction. Open its exact evidence
row.

Start at 12 h and advance through 48 h. Pause on experiment ID, observations consumed, endpoint
prediction, OOD ratio, and STOP. Ground truth remains hidden until the 72 h endpoint.

## 1:18–1:48 — Hard/refused example

Load the representative ABSTAIN case. Show that the OOD boundary is crossed and the unit continues
to the 72 h protocol. Emphasize that refusal trades savings for fewer premature errors.

## 1:48–2:25 — Main result

Show the risk–savings frontier and the frozen 5% row: 988 organoids, 11 held-out experiments, 439
early stops, 25 errors, 5.69% EESR, 44.43% coverage, and 12.26% savings. Show the exact 95% interval,
3.72–8.29%, and the wider experiment-bootstrap interval.

## 2:25–2:55 — Baselines

Show Full GATES beside fixed-time, naive confidence, non-group calibration, CBES-style, and GATES
without refusal. Removing refusal raises EESR from 5.69% to 8.60%. CBES-style reaches 15.51%.

## 2:55–3:28 — Experiment variation and E007

Show per-experiment EESR, then E007. It has 15 errors among 49 stops, and its erroneous stops do not
cross the frozen OOD boundary. State the evidence-based diagnosis: possible conditional/concept
shift; mechanism undetermined.

## 3:28–3:55 — Negative results

Show Lens at 17.82% EESR, irregular sampling at 10.61%, and the experiment gate’s collapse at the
10% operating point. Say clearly: the extension is not a general safety mechanism.

## 3:55–4:25 — Reproducibility

Run `just verify-final`, then show the manifest, transparent erratum, and figure source map. State
that the audit-only correction produced zero changed scientific results after full recomputation.

## 4:25–4:45 — Impact

GATES does not promise universal risk control. It demonstrates that experiment-held-out selective
stopping can save observations, reduce premature errors relative to tested alternatives, and fail
honestly when evidence is weak. That is the decision system we can defend today.
