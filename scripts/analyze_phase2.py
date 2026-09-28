from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "fixed_time": "Fixed time",
    "naive_confidence": "Naive confidence",
    "cbes_style": "CBES-style",
    "gates_without_ood": "GATES without OOD",
    "gates_full": "Full GATES",
}


def metric(frame: pd.DataFrame) -> tuple[float, float, float]:
    stopped = frame[frame.decision == "STOP"]
    errors = (stopped.prediction != stopped.endpoint).sum()
    return (
        errors / len(stopped) if len(stopped) else np.nan,
        frame.saved_fraction.mean(),
        len(stopped) / len(frame),
    )


def bootstrap_groups(frame: pd.DataFrame, rng: np.random.Generator, n: int = 2000) -> dict:
    grouped = []
    for _, part in frame.groupby("group_id"):
        stopped = part.decision == "STOP"
        grouped.append(
            (
                len(part),
                int(stopped.sum()),
                int((stopped & (part.prediction != part.endpoint)).sum()),
                float(part.saved_fraction.sum()),
            )
        )
    summary = np.asarray(grouped)
    weights = rng.multinomial(len(summary), np.full(len(summary), 1 / len(summary)), size=n)
    totals = weights @ summary
    bootstrap_eesr = np.full(n, np.nan)
    np.divide(totals[:, 2], totals[:, 1], out=bootstrap_eesr, where=totals[:, 1] > 0)
    array = np.column_stack(
        (
            bootstrap_eesr,
            totals[:, 3] / totals[:, 0],
            totals[:, 1] / totals[:, 0],
        )
    )
    result = {}
    for index, name in enumerate(("EESR", "observation_savings", "coverage")):
        finite = array[np.isfinite(array[:, index]), index]
        result[f"{name}_group_bootstrap_lower"] = (
            np.quantile(finite, 0.025) if len(finite) else np.nan
        )
        result[f"{name}_group_bootstrap_upper"] = (
            np.quantile(finite, 0.975) if len(finite) else np.nan
        )
    return result


def select(decisions: pd.DataFrame, method: str, parameter: float) -> pd.DataFrame:
    return decisions[(decisions.method == method) & np.isclose(decisions.parameter, parameter)]


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    output = args.input
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    decisions = pd.read_csv(output / "unit_decisions.csv")
    master = pd.read_csv(output / "master_results.csv")
    predictions = pd.read_csv(output / "predictions_by_time.csv")
    rng = np.random.default_rng(args.seed)

    rows = []
    for (method, parameter), frame in decisions.groupby(["method", "parameter"]):
        eesr, savings, coverage = metric(frame)
        rows.append(
            {
                "method": method,
                "parameter": parameter,
                "independent_units": len(frame),
                "independent_groups": frame.group_id.nunique(),
                "stopped_early": int((frame.decision == "STOP").sum()),
                "incorrect_early": int(
                    ((frame.decision == "STOP") & (frame.prediction != frame.endpoint)).sum()
                ),
                "EESR": eesr,
                "observation_savings": savings,
                "coverage": coverage,
                **bootstrap_groups(frame, rng),
            }
        )
    ci = pd.DataFrame(rows)
    ci.to_csv(output / "aggregate_group_bootstrap_ci.csv", index=False)

    ablations = []
    for risk in sorted(
        decisions.loc[decisions.method == "gates_full", "parameter"].dropna().unique()
    ):
        without = select(decisions, "gates_without_ood", risk).set_index("unit_id")
        full = select(decisions, "gates_full", risk).set_index("unit_id")
        without_error = (without.decision == "STOP") & (without.prediction != without.endpoint)
        full_error = (full.decision == "STOP") & (full.prediction != full.endpoint)
        ablations.append(
            {
                "risk_target": risk,
                "errors_prevented": int((without_error & ~full_error).sum()),
                "additional_abstentions": int(
                    ((full.decision == "ABSTAIN") & (without.decision != "ABSTAIN")).sum()
                ),
                "observations_savings_sacrificed": without.saved_fraction.mean()
                - full.saved_fraction.mean(),
                "EESR_without_OOD": metric(without.reset_index())[0],
                "EESR_full": metric(full.reset_index())[0],
                "savings_without_OOD": without.saved_fraction.mean(),
                "savings_full": full.saved_fraction.mean(),
            }
        )
    pd.DataFrame(ablations).to_csv(output / "ood_ablation.csv", index=False)

    curve_methods = list(METHOD_LABELS)
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, method in zip(colors, curve_methods, strict=False):
        part = ci[ci.method == method].sort_values("observation_savings")
        ax.plot(
            part.observation_savings * 100,
            part.EESR * 100,
            "o-",
            label=METHOD_LABELS[method],
            color=color,
        )
        ax.fill_between(
            part.observation_savings * 100,
            part.EESR_group_bootstrap_lower * 100,
            part.EESR_group_bootstrap_upper * 100,
            alpha=0.12,
            color=color,
        )
    ax.axhline(5, color="black", linestyle="--", linewidth=1, label="Predefined 5% target")
    ax.set(
        xlabel="Observation savings (%)",
        ylabel="Held-out EESR (%)",
        title="Risk versus observation savings",
    )
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, figures / "risk_vs_observation_savings.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for color, method in zip(colors, curve_methods, strict=False):
        part = ci[ci.method == method].sort_values("coverage")
        ax.plot(
            part.coverage * 100, part.EESR * 100, "o-", label=METHOD_LABELS[method], color=color
        )
    ax.axhline(5, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Coverage (%)", ylabel="Held-out EESR (%)", title="Risk versus early-stop coverage"
    )
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, figures / "risk_vs_coverage.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in ("cbes_style", "gates_without_ood", "gates_full"):
        part = select(decisions, method, 0.05)
        ax.hist(
            part.terminal_time, bins=np.arange(6, 79, 12), alpha=0.45, label=METHOD_LABELS[method]
        )
    ax.set(
        xlabel="Terminal decision time (h)",
        ylabel="Independent organoids",
        title="Decision-time distribution at 5% target",
    )
    ax.legend()
    save_figure(fig, figures / "decision_time_distribution.png")

    fixed = ci[ci.method == "fixed_time"].sort_values("parameter")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(fixed.parameter, fixed.EESR * 100, "o-", color="tab:red", label="EESR")
    ax1.set(xlabel="Observation time (h)", ylabel="EESR (%)")
    ax2 = ax1.twinx()
    ax2.plot(
        fixed.parameter, fixed.observation_savings * 100, "s-", color="tab:blue", label="Savings"
    )
    ax2.set_ylabel("Observation savings (%)")
    ax1.set_title("Fixed-time performance versus observation time")
    save_figure(fig, figures / "performance_vs_observation_time.png")

    calibration = predictions[predictions.decision_time == 24].copy()
    calibration["bin"] = pd.qcut(calibration.probability, 10, duplicates="drop")
    calibration_curve = calibration.groupby("bin", observed=True).agg(
        predicted=("probability", "mean"), observed=("endpoint", "mean"), n=("endpoint", "size")
    )
    calibration_curve.to_csv(output / "calibration_curve_24h.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(calibration_curve.predicted, calibration_curve.observed, "o-")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Held-out calibration at 24 h",
    )
    save_figure(fig, figures / "calibration_curve.png")

    per_experiment = master[
        (master.method == "gates_full") & (master.risk_target == 0.05)
    ].sort_values("held_out_group")
    per_experiment.to_csv(output / "per_experiment_gates_005.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(per_experiment.held_out_group, per_experiment.EESR * 100)
    axes[0].axhline(5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("EESR (%)")
    axes[1].bar(per_experiment.held_out_group, per_experiment.observation_savings * 100)
    axes[1].set(ylabel="Savings (%)", xlabel="Held-out experiment")
    fig.suptitle("Full GATES at predefined 5% target by experiment")
    save_figure(fig, figures / "per_held_out_experiment.png")

    no_ood = select(decisions, "gates_without_ood", 0.05)
    stopped = no_ood[no_ood.decision == "STOP"].copy()
    stopped["outcome"] = np.where(stopped.prediction == stopped.endpoint, "correct", "error")
    fig, ax = plt.subplots(figsize=(6, 5))
    values = [
        stopped.loc[stopped.outcome == label, "ood_score"].dropna()
        for label in ("correct", "error")
    ]
    ax.boxplot(values, tick_labels=["Correct early stop", "Erroneous early stop"], showfliers=False)
    ax.set(ylabel="OOD score", title="OOD score versus early-stop failure")
    save_figure(fig, figures / "ood_score_vs_failure.png")

    full = select(decisions, "gates_full", 0.05)
    joined = no_ood.set_index("unit_id").join(
        full.set_index("unit_id"), lsuffix="_without", rsuffix="_full"
    )
    candidates = []
    prevented = joined[
        (joined.prediction_without != joined.endpoint_without)
        & (joined.decision_without == "STOP")
        & (joined.decision_full != "STOP")
    ]
    if len(prevented):
        candidates.append((prevented.index[0], "OOD-prevented error"))
    correct = joined[
        (joined.prediction_full == joined.endpoint_full) & (joined.decision_full == "STOP")
    ]
    if len(correct):
        candidates.append((correct.index[0], "Correct early stop"))
    residual = joined[
        (joined.prediction_full != joined.endpoint_full) & (joined.decision_full == "STOP")
    ]
    if len(residual):
        candidates.append((residual.index[0], "Residual error"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for unit_id, label in candidates:
        trajectory = predictions[predictions.unit_id == unit_id].sort_values("decision_time")
        ax.plot(trajectory.decision_time, trajectory.probability, "o-", label=f"{label}: {unit_id}")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Observation time (h)",
        ylabel="Endpoint probability",
        title="Representative held-out trajectories",
    )
    ax.legend(fontsize=8)
    save_figure(fig, figures / "representative_trajectories.png")


if __name__ == "__main__":
    main()
