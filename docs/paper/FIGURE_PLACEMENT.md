# GATES paper figure placement

This record identifies every figure used in the final paper, its source asset, and the reason for its placement. The approved GATES overview image is reproduced unchanged; all quantitative plots come from the frozen project results.

| Figure | Source asset | Placement and purpose |
|---|---|---|
| GATES overview | `docs/paper/figures/gates_overview.png` | Near the introduction, spanning both columns, to give judges a single visual account of the inputs, staged decision process, and three possible outcomes before the technical detail. |
| Risk--savings frontier | `artifacts/final/figures/01_rpe_risk_savings_frontier.png` | Main results, beside the frontier discussion, to show the operating-point trade-off between erroneous early-stop risk and observation savings. |
| Risk--coverage frontier | `artifacts/final/figures/02_rpe_risk_coverage_frontier.png` | Main results, paired with the risk--savings plot, to show how the same operating points trade risk against decision coverage. |
| Refusal ablation | `artifacts/final/figures/06_ood_refusal_ablation.png` | Refusal-ablation section, where the effect of retaining versus removing the abstention mechanism is quantified. |
| Experiment heterogeneity | `artifacts/final/figures/03_per_experiment_eesr.png` | Heterogeneity section, before the E007 case discussion, to make experiment-to-experiment variation visible. |
| E007 failure case | `artifacts/final/figures/12_e007_failure_case.png` | Heterogeneity section, paired with the per-experiment plot, to localize the dominant failure pattern without asserting an unverified biological cause. |
| Representative correct stop | `artifacts/final/figures/09_representative_successful_stop.png` | Main-results trajectory panel, alongside the other representative trajectories, to illustrate a correct early-stop outcome. |
| Representative refusal | `artifacts/final/figures/10_representative_abstention.png` | Main-results trajectory panel to illustrate a case in which the system appropriately declines to stop early. |
| Representative erroneous stop | `artifacts/final/figures/11_representative_erroneous_stop.png` | Main-results trajectory panel to make the residual failure mode concrete and visible. |
| Per-experiment observation savings | `artifacts/final/figures/04_per_experiment_observation_savings.png` | Appendix, supporting the full per-experiment audit with the corresponding savings distribution. |
| Baseline comparison at 5% gate | `artifacts/final/figures/05_baseline_comparison_005.png` | Appendix, beside the baseline table, to provide a visual comparison at the documented 5% operating gate. |
| Per-experiment 5% gate | `artifacts/final/figures/07_experiment_gate_005.png` | Appendix robustness material, showing the experiment-level behavior at the lower exploratory gate. |
| Per-experiment 10% gate | `artifacts/final/figures/08_experiment_gate_010.png` | Appendix robustness material, paired with the 5% view, to expose the higher-risk failure behavior at the 10% gate. |

No figure embeds an absolute workstation path, and no asset introduces results beyond the frozen evidence used by the report.
