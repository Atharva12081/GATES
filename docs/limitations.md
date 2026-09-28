# Limitations

1. **Small sample size.** There are only 21 endpoint-labelled units and three replicate groups. This
   is sufficient for a pipeline feasibility check, not a deployment-grade safety claim.
2. **Calibration power.** Seven units are available in each calibration role. A conservative exact
   upper confidence bound generally cannot certify a low target risk from so few observations. The
   system records this and labels unsupported thresholds `empirical_only`.
3. **Shared dose conditions.** The three replicates include the same seven doses. Holding out a
   replicate prevents frame/unit leakage but is weaker than holding out an independent experiment,
   donor, plate, laboratory, or acquisition campaign.
4. **Endpoint threshold.** Viability below 0.5 is a coarse, predefined demonstration endpoint. A
   domain expert must validate an operational threshold for a real assay.
5. **Shift detector.** Standardized nearest-neighbor refusal is a simple diagnostic, not proof of out-of-domain
   detection. Its behavior needs stress testing on truly shifted experiments.
6. **No prospective validation.** All decisions are retrospective simulations over public data.
7. **Licensing uncertainty.** The upstream repository lacks a detected license; public reuse terms
   must be confirmed before submission.
8. **No wet-lab or clinical instruction.** Outputs are research triage signals only.
