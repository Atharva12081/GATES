from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "act_v3"
OUT = ART / "figures"
COLORS = {
    "risk_sensitive": "#168C82",
    "max_variance": "#5668D8",
    "maximin": "#4F9D69",
    "fixed_sentinel": "#7B8790",
    "random": "#D77A2B",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#334155",
            "text.color": "#17212B",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    traces = pd.read_csv(ART / "tipping_fresh_traces.csv")
    decisions = pd.read_csv(ART / "tipping_fresh_decisions.csv")
    comparisons = pd.read_csv(ART / "tipping_fresh_comparisons.csv")
    audit = json.loads((ART / "independent_audit.json").read_text())

    order = ["risk_sensitive", "max_variance", "maximin", "fixed_sentinel", "random"]
    recall = (
        traces.loc[traces["round"].isin([2, 3])]
        .groupby("policy")["effect_recall"]
        .mean()
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(
        [x.replace("_", "\n") for x in order],
        recall * 100,
        color=[COLORS[x] for x in order],
        width=0.68,
    )
    ax.set_ylim(90, 100)
    ax.set_ylabel("Protected effect recall (%)")
    ax.set_title("Largest untouched confirmation: 53 compound groups")
    ax.bar_label(bars, labels=[f"{x:.2f}%" for x in recall * 100], padding=4, fontsize=10)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "v3_effect_recall.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    cmp = comparisons.set_index("baseline").loc[
        ["max_variance", "maximin", "fixed_sentinel", "random"]
    ]
    values = cmp["effect_recall_advantage"] * 100
    low = (cmp["effect_recall_advantage"] - cmp["effect_recall_ci_low"]) * 100
    high = (cmp["effect_recall_ci_high"] - cmp["effect_recall_advantage"]) * 100
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = range(len(cmp))
    ax.errorbar(values, y, xerr=[low, high], fmt="o", color="#168C82", capsize=5, lw=2)
    ax.axvline(0, color="#17212B", linewidth=1)
    ax.set_yticks(list(y), [x.replace("_", " ").title() for x in cmp.index])
    ax.invert_yaxis()
    ax.set_xlabel("GATES-ACT effect-recall advantage (percentage points)")
    ax.set_title("Paired compound-group differences with 95% bootstrap intervals")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "v3_baseline_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for policy, group in decisions.groupby("policy"):
        stops = group.loc[group["decision"].eq("STOP")]
        summary_rows.append(
            {
                "policy": policy,
                "stops": len(stops),
                "unsafe": int((~stops["stop_safe"]).sum()),
            }
        )
    stop = pd.DataFrame(summary_rows).set_index("policy").reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = range(len(order))
    ax.bar(x, stop["stops"], color=[COLORS[p] for p in order], label="Proposed stops")
    ax.bar(x, stop["unsafe"], color="#C94242", label="Unsafe stops")
    ax.set_xticks(list(x), [p.replace("_", "\n") for p in order])
    ax.set_ylabel("Fresh confirmation runs")
    ax.set_title("The external stopping rule fails the release gate")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    fig.text(
        0.5,
        0.01,
        f"Release status: {audit['release_status']} · v3 early stopping disabled",
        ha="center",
        color="#C94242",
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT / "v3_stopping_safety.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
