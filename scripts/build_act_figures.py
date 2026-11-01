from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "act"
OUTPUT = EVIDENCE / "figures"
COLORS = {
    "act": "#16697A",
    "max_variance": "#489FB5",
    "maximin": "#82C0CC",
    "random": "#E76F51",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    traces = pd.read_csv(EVIDENCE / "acquisition_traces.csv")
    decisions = pd.read_csv(EVIDENCE / "gates_act_decisions.csv")
    comparisons = pd.read_csv(EVIDENCE / "paired_comparisons.csv")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})

    summary = traces.groupby(["strategy", "budget"])["nrmse"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for strategy, group in summary.groupby("strategy"):
        ax.plot(
            group["budget"],
            group["mean"],
            color=COLORS[strategy],
            linewidth=2.8 if strategy == "act" else 1.7,
            label=strategy.replace("_", " ").title(),
        )
        ax.fill_between(
            group["budget"],
            group["mean"] - group["std"],
            group["mean"] + group["std"],
            color=COLORS[strategy],
            alpha=0.08,
        )
    ax.set(
        xlabel="Measurements consumed from the 133-cell dose-time grid",
        ylabel="Held-out normalized RMSE",
        title="Integrated variance reduction improves reconstruction efficiency",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_acquisition_efficiency.png", dpi=220)
    plt.close(fig)

    act_decisions = decisions.loc[decisions["strategy"] == "act"].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    bars = ax.bar(
        act_decisions["test_replicate"].astype(str),
        act_decisions["measurements_consumed"],
        color=["#16697A" if value == "STOP" else "#E76F51" for value in act_decisions["decision"]],
    )
    for bar, (_, row) in zip(bars, act_decisions.iterrows(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            (
                f"{row['decision']}\nNRMSE {row['terminal_nrmse']:.3f}"
                if row["decision"] == "STOP"
                else f"{row['decision']}\ncomplete grid"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(133, color="#36454F", linewidth=1, linestyle="--", label="Complete grid")
    ax.set(
        xlabel="Held-out replicate",
        ylabel="Measurements consumed",
        ylim=(0, 150),
        title="GATES stops four groups and abstains on two shifted groups",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_calibrated_decisions.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(comparisons))
    means = comparisons["act_improvement_mean_nrmse"].to_numpy()
    low = means - comparisons["ci_low"].to_numpy()
    high = comparisons["ci_high"].to_numpy() - means
    ax.errorbar(means, y, xerr=np.vstack([low, high]), fmt="o", color="#16697A", capsize=5)
    ax.axvline(0, color="#36454F", linewidth=1)
    ax.set_yticks(y, comparisons["baseline"].str.replace("_", " ").str.title())
    ax.set(
        xlabel="Reduction in mean NRMSE over the acquisition trajectory",
        title="Paired group differences favor GATES-ACT",
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_paired_comparisons.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
