from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "act_v2"
OUTPUT = EVIDENCE / "figures"
COLORS = {
    "act": "#0B7285",
    "max_variance": "#5C7CFA",
    "maximin": "#2F9E44",
    "random": "#E8590C",
    "fixed_sentinel": "#868E96",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    traces = pd.read_csv(EVIDENCE / "neural_acquisition_traces.csv")
    decisions = pd.read_csv(EVIDENCE / "neural_decisions.csv")
    ablations = pd.read_csv(EVIDENCE / "neural_ablations.csv")
    test = traces.loc[traces["split"] == "test"]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})

    summary = test.groupby(["policy", "round"])["nrmse"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for policy, group in summary.groupby("policy"):
        ax.plot(
            group["round"],
            group["mean"],
            marker="o",
            color=COLORS[policy],
            linewidth=2.8 if policy == "act" else 1.5,
            label=policy.replace("_", " ").title(),
        )
        ax.fill_between(
            group["round"],
            group["mean"] - group["sem"],
            group["mean"] + group["sem"],
            color=COLORS[policy],
            alpha=0.10,
        )
    ax.set_xticks(range(1, 5), ["DIV5", "DIV7", "DIV9", "DIV12"])
    ax.set(
        ylabel="Held-out NRMSE (mean ± SE)",
        title="Chronological acquisition on 26 independent test compound groups",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_chronological_acquisition.png", dpi=240)
    plt.close(fig)

    policy_summary = (
        decisions.groupby("policy")
        .agg(
            sessions=("recording_sessions", "mean"),
            unsafe=(
                "stop_safe",
                lambda value: int(
                    ((~value) & (decisions.loc[value.index, "decision"] == "STOP")).sum()
                ),
            ),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bars = ax.bar(
        policy_summary["policy"].str.replace("_", " ").str.title(),
        4 - policy_summary["sessions"],
        color=[COLORS[value] for value in policy_summary["policy"]],
    )
    for bar, (_, row) in zip(bars, policy_summary.iterrows(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() - 0.012, 0.01),
            f"{100 * bar.get_height() / 4:.1f}%\n{row['unsafe']} unsafe",
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    ax.set(
        ylabel="Mean recording sessions avoided (of 4)",
        title="Actual batch-level savings and stop safety",
    )
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_cost_and_safety.png", dpi=240)
    plt.close(fig)

    primary_order = [
        "primary",
        "single_div_hit",
        "no_abstention",
        "effect_threshold_030",
        "effect_threshold_050",
        "strict_nrmse_015",
        "lenient_nrmse_025",
    ]
    plot = ablations.set_index("ablation").loc[primary_order].reset_index()
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    colors = ["#0B7285" if unsafe == 0 else "#C92A2A" for unsafe in plot["unsafe_stops"]]
    ax.barh(np.arange(len(plot)), 100 * plot["session_reduction"], color=colors)
    ax.set_yticks(np.arange(len(plot)), plot["ablation"].str.replace("_", " "))
    ax.invert_yaxis()
    for index, row in plot.iterrows():
        ax.text(
            100 * row["session_reduction"] + 0.3,
            index,
            f"{int(row['stops'])} stops / {int(row['unsafe_stops'])} unsafe",
            va="center",
            fontsize=9,
        )
    ax.set(
        xlabel="Mean recording-session reduction (%)",
        title="Ablations expose the safety–efficiency trade-off",
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_ablation_safety.png", dpi=240)
    plt.close(fig)

    act = decisions.loc[decisions["policy"] == "act"]
    counts = act["decision"].value_counts().reindex(["STOP", "ABSTAIN", "COMPLETE"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.bar(counts.index, counts.values, color=["#2F9E44", "#E8590C", "#868E96"])
    for index, value in enumerate(counts.values):
        ax.text(index, value + 0.35, str(value), ha="center", fontsize=11)
    ax.set(
        ylabel="Held-out compound runs",
        title="Primary GATES-ACT decisions: no unsafe stops, no missed compound effects",
    )
    ax.set_ylim(0, max(counts.values) + 4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "04_primary_decisions.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
