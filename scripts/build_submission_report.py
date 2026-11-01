from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "act"
FIGURES = ROOT / "artifacts" / "act" / "submission_figures"
OUTPUT = ROOT / "output" / "pdf"
CSS = ROOT / "docs" / "final" / "report.css"
PAPER_CSS = DOCS / "research_paper.css"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

NAVY = "#17212b"
TEAL = "#0f8f83"
BLUE = "#3f6fa8"
ORANGE = "#d97a28"
RED = "#c84444"
GREEN = "#4d9a68"
GREY = "#66717e"
PALE = "#eef3f5"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelcolor": NAVY,
            "axes.edgecolor": NAVY,
            "xtick.color": "#435066",
            "ytick.color": "#435066",
            "text.color": NAVY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def system_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 4.3))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    boxes = [
        (0.15, 1.45, 2.3, 1.55, "1  Eligible data", "Current-day doses\nplus vehicle anchor", BLUE),
        (2.9, 1.45, 2.3, 1.55, "2  Predict", "Robust prior\nand posterior covariance", TEAL),
        (5.65, 1.45, 2.3, 1.55, "3  Acquire", "Conditional variance\nper measurement cost", ORANGE),
        (
            8.4,
            1.45,
            2.3,
            1.55,
            "4  Protect",
            "Conformal effect band\nand shift-based refusal",
            GREEN,
        ),
        (11.15, 1.45, 2.3, 1.55, "5  Act", "COMPLETE or ABSTAIN\nSTOP remains locked", RED),
    ]
    for x, y, w, h, title, body, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.8,
                edgecolor=color,
                facecolor="white",
            )
        )
        ax.text(x + 0.2, y + 1.13, title, fontsize=11.3, weight="bold", color=color)
        ax.text(x + 0.2, y + 0.69, body, fontsize=9.1, va="top", linespacing=1.36)
    for current, following in zip(boxes[:-1], boxes[1:], strict=True):
        start_x = current[0] + current[2] + 0.05
        end_x = following[0] - 0.05
        ax.add_patch(
            FancyArrowPatch(
                (start_x, 2.22),
                (end_x, 2.22),
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.6,
                color=NAVY,
            )
        )
    feedback_path = MplPath(
        [(12.3, 1.38), (11.1, 0.55), (2.4, 0.55), (1.3, 1.38)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        FancyArrowPatch(
            path=feedback_path,
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.8,
            color=GREY,
        )
    )
    ax.text(
        6.8,
        0.22,
        "Prospective shadow mode: log recommendations while complete plates continue",
        ha="center",
        fontsize=9.8,
        color=GREY,
    )
    ax.set_title("GATES-ACT fail-closed adaptive measurement workflow", pad=8, fontsize=15)
    save(fig, "01_system_architecture.png")


def performance_figure() -> None:
    data = pd.read_csv(ROOT / "artifacts" / "act" / "canonical" / "policy_summary.csv")
    order = ["max_variance", "risk_sensitive", "maximin", "random", "fixed_sentinel"]
    labels = ["Max variance", "Effect weighted", "Maximin", "Random", "Fixed sentinels"]
    data = data.set_index("policy").loc[order]
    colors = [TEAL, ORANGE, BLUE, "#8d99ae", "#b6bec8"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4))
    metrics = [
        ("effect_recall", "Protected effect recall", 100, "%", (94, 99.2)),
        ("nrmse", "Hidden-response NRMSE", 1, "", (0.32, 0.37)),
        ("direct_effect_capture", "Direct effect capture", 100, "%", (37, 46)),
    ]
    for ax, (column, title, scale, suffix, limits) in zip(axes, metrics, strict=True):
        values = data[column].to_numpy() * scale
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
        ax.set_xlim(*limits)
        ax.set_title(title)
        ax.grid(axis="x", color="#dfe6eb", lw=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values[::-1], strict=True):
            fmt = f"{value:.2f}{suffix}" if scale == 100 else f"{value:.3f}"
            ax.text(
                value + (limits[1] - limits[0]) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                fmt,
                va="center",
                fontsize=8.5,
            )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].text(0.335, 4.45, "lower is better", fontsize=8.5, color=GREY)
    fig.suptitle(
        "The simpler conditional maximum-variance policy is the strongest pooled choice",
        y=1.03,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, "02_policy_performance.png")


def forest_figure() -> None:
    data = pd.read_csv(ROOT / "artifacts" / "act" / "canonical" / "paired_comparisons.csv")
    data = data[data.metric == "effect_recall"].copy()
    order = ["fixed_sentinel", "random", "maximin", "risk_sensitive"]
    labels = ["Fixed sentinels", "Random", "Maximin", "Effect weighted"]
    data = data.set_index("baseline").loc[order]
    estimate = data.difference.to_numpy() * 100
    low = data.ci_low.to_numpy() * 100
    high = data.ci_high.to_numpy() * 100
    pvals = data.paired_sign_flip_p.to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    y = np.arange(len(order))[::-1]
    sig = low > 0
    for yi, est, lo, hi, good in zip(y, estimate, low, high, sig, strict=True):
        color = TEAL if good else BLUE
        ax.plot([lo, hi], [yi, yi], color=color, lw=3)
        ax.plot(est, yi, "o", color=color, ms=8)
    ax.axvline(0, color=NAVY, lw=1.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Max-variance recall advantage (percentage points)")
    ax.set_title("Paired group-level improvement with 95% bootstrap intervals")
    ax.grid(axis="x", color="#dfe6eb", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yi, hi, p in zip(y, high, pvals, strict=True):
        ax.text(
            hi + 0.12,
            yi,
            f"p={p:.4f}" if p < 0.01 else f"p={p:.3f}",
            va="center",
            fontsize=9,
            color=GREY,
        )
    fig.tight_layout()
    save(fig, "03_paired_improvement.png")


def evidence_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.3))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    stages = [
        (
            0.4,
            "v1",
            "6 organoid\nreplicates",
            "Retrospective grid\nreduction only",
            "Interpretation\nwithdrawn",
            GREY,
        ),
        (
            3.2,
            "v2",
            "26 held-out\ncompound groups",
            "Chronology +\nphysical cost",
            "Promising internal\nevidence",
            BLUE,
        ),
        (
            6.0,
            "v3",
            "76 external\ncompound groups",
            "Three public\nMEA sources",
            "Canonical policy\nselected",
            TEAL,
        ),
        (
            8.8,
            "Release",
            "3 candidate\nstops",
            "2 unsafe\noutcomes",
            "STOP disabled\nshadow mode",
            RED,
        ),
    ]
    for x, label, n, method, result, color in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.55),
                2.25,
                2.85,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="white",
                edgecolor=color,
                lw=2,
            )
        )
        ax.text(x + 0.18, 3.08, label, fontsize=17, weight="bold", color=color)
        ax.text(x + 0.18, 2.38, n, fontsize=11, weight="bold")
        ax.text(x + 0.18, 1.68, method, fontsize=9.7, color=GREY)
        ax.text(x + 0.18, 0.85, result, fontsize=10.2, weight="bold", color=color)
    for x in [2.7, 5.5, 8.3]:
        ax.add_patch(
            FancyArrowPatch(
                (x, 1.97), (x + 0.38, 1.97), arrowstyle="-|>", mutation_scale=14, lw=1.8, color=NAVY
            )
        )
    ax.set_title("Evidence expansion changed both the model choice and the release decision", pad=6)
    save(fig, "04_evidence_evolution.png")


def stopping_figure() -> None:
    n = np.arange(1, 121)
    zero_fail_upper = 1 - np.power(0.05, 1 / n)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(n, zero_fail_upper * 100, color=TEAL, lw=2.8)
    ax.axhline(5, color=RED, lw=1.5, ls="--", label="5% release limit")
    ax.axvline(59, color=ORANGE, lw=1.5, ls=":")
    ax.scatter([59], [(1 - 0.05 ** (1 / 59)) * 100], color=ORANGE, s=55, zorder=3)
    ax.annotate(
        "59 zero-failure stops required",
        xy=(59, 5),
        xytext=(72, 18),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        color=ORANGE,
        fontsize=9.5,
    )
    ax.set_xlabel("Independent validation stops with zero failures")
    ax.set_ylabel("One-sided 95% upper risk bound")
    ax.set_ylim(0, 100)
    ax.grid(color="#dfe6eb", lw=0.8)
    ax.set_title("Evidence needed to authorize STOP")
    ax2 = axes[1]
    current_upper = beta.ppf(0.95, 3, 1) * 100
    ax2.bar(
        ["Observed unsafe\nrate", "95% upper\nrisk bound", "Release\nlimit"],
        [66.7, current_upper, 5],
        color=[ORANGE, RED, TEAL],
        width=0.62,
    )
    for i, value in enumerate([66.7, current_upper, 5]):
        ax2.text(i, value + 2, f"{value:.1f}%", ha="center", weight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Unsafe-stop risk")
    ax2.set_title("Current external evidence: 2 unsafe of 3")
    ax2.grid(axis="y", color="#dfe6eb", lw=0.8)
    ax2.set_axisbelow(True)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "The safety improvement is a refusal: early stopping is disabled",
        y=1.03,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, "05_stopping_safety.png")


def source_landscape_figure() -> None:
    data = pd.read_csv(ROOT / "artifacts" / "act" / "canonical" / "group_metrics.csv")
    policy_order = [
        "max_variance",
        "risk_sensitive",
        "maximin",
        "random",
        "fixed_sentinel",
    ]
    policy_labels = ["Max variance", "Effect weighted", "Maximin", "Random", "Fixed"]
    source_order = ["phase1", "brown", "tipping"]
    source_labels = ["Phase 1 (n=17)", "Brown (n=6)", "Tipping (n=53)"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    panels = [
        ("effect_recall", "Mean protected effect recall", 100, "%.1f%%", "YlGnBu", 90, 100),
        ("nrmse", "Mean hidden-response NRMSE", 1, "%.3f", "YlOrBr", 0.12, 0.48),
    ]
    for ax, (metric, title, scale, fmt, cmap, low, high) in zip(axes, panels, strict=True):
        matrix = (
            data.pivot_table(index="dataset", columns="policy", values=metric, aggfunc="mean")
            .loc[source_order, policy_order]
            .to_numpy()
            * scale
        )
        image = ax.imshow(matrix, cmap=cmap, vmin=low, vmax=high, aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                normalized = (matrix[row, col] - low) / (high - low)
                ax.text(
                    col,
                    row,
                    fmt % matrix[row, col],
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if normalized > 0.62 else NAVY,
                    weight="bold" if col == 0 else "normal",
                )
        ax.set_xticks(range(len(policy_labels)), policy_labels, rotation=32, ha="right")
        ax.set_yticks(range(len(source_labels)), source_labels)
        ax.set_title(title)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02)
    fig.suptitle(
        "Source-stratified performance shows where the pooled result comes from",
        y=1.03,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, "06_source_landscape.png")


def distribution_figure() -> None:
    data = pd.read_csv(ROOT / "artifacts" / "act" / "canonical" / "group_metrics.csv")
    order = ["max_variance", "risk_sensitive", "maximin", "random", "fixed_sentinel"]
    labels = ["Max variance", "Effect weighted", "Maximin", "Random", "Fixed sentinels"]
    colors = [TEAL, ORANGE, BLUE, "#8d99ae", "#b6bec8"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    for ax, metric, title, scale in [
        (axes[0], "effect_recall", "Group-level protected recall", 100),
        (axes[1], "nrmse", "Group-level hidden-response NRMSE", 1),
    ]:
        groups = [data.loc[data.policy == policy, metric].to_numpy() * scale for policy in order]
        box = ax.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
            medianprops={"color": NAVY, "linewidth": 1.7},
            whiskerprops={"color": GREY},
            capprops={"color": GREY},
        )
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        rng = np.random.default_rng(20261101)
        for index, (values, color) in enumerate(zip(groups, colors, strict=True), start=1):
            jitter = rng.normal(index, 0.055, len(values))
            ax.scatter(jitter, values, s=8, alpha=0.22, color=color, edgecolors="none")
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="#dfe6eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall (%)")
    axes[1].set_ylabel("NRMSE (lower is better)")
    fig.suptitle(
        "Aggregate means do not conceal the full 76-group distribution",
        y=1.03,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, "07_group_distributions.png")


def acquisition_trajectory_figure() -> None:
    phase1 = pd.read_csv(ROOT / "artifacts" / "act_v3" / "fresh_acquisition_traces.csv")
    phase1 = phase1.loc[phase1.v3_split == "test"].copy()
    phase1["dataset"] = "Phase 1"
    brown = pd.read_csv(ROOT / "artifacts" / "act_v3" / "brown_fresh_traces.csv")
    brown["dataset"] = "Brown"
    tipping = pd.read_csv(ROOT / "artifacts" / "act_v3" / "tipping_fresh_traces.csv")
    tipping["dataset"] = "Tipping"
    data = pd.concat([phase1, brown, tipping], ignore_index=True, sort=False)
    policies = ["max_variance", "risk_sensitive", "maximin", "random", "fixed_sentinel"]
    labels = ["Max variance", "Effect weighted", "Maximin", "Random", "Fixed"]
    colors = [TEAL, ORANGE, BLUE, "#8d99ae", "#b6bec8"]
    styles = ["-", "--", "-.", ":", "--"]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.1), sharex=True)
    for column, dataset in enumerate(["Phase 1", "Brown", "Tipping"]):
        subset = data.loc[data.dataset == dataset]
        for policy, label, color, line_style in zip(policies, labels, colors, styles, strict=True):
            policy_data = subset.loc[subset.policy == policy]
            mean = policy_data.groupby("round").agg(
                nrmse=("nrmse", "mean"), effect_recall=("effect_recall", "mean")
            )
            axes[0, column].plot(
                mean.index,
                mean.nrmse,
                marker="o",
                ms=4,
                lw=2,
                ls=line_style,
                color=color,
                label=label,
            )
            axes[1, column].plot(
                mean.index,
                mean.effect_recall * 100,
                marker="o",
                ms=4,
                lw=2,
                ls=line_style,
                color=color,
            )
        axes[0, column].set_title(dataset)
        axes[0, column].grid(color="#dfe6eb", lw=0.8)
        axes[1, column].grid(color="#dfe6eb", lw=0.8)
        axes[1, column].set_xticks([1, 2, 3, 4])
        axes[1, column].set_xlabel("Chronological decision round")
        for row in range(2):
            axes[row, column].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Mean hidden NRMSE")
    axes[1, 0].set_ylabel("Mean protected recall (%)")
    axes[0, 2].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Acquisition learning curves preserve chronology across all external sources",
        y=1.01,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, "08_acquisition_trajectories.png")


def pareto_figure() -> None:
    data = pd.read_csv(ROOT / "artifacts" / "act" / "canonical" / "policy_summary.csv")
    labels = {
        "max_variance": "Max variance",
        "risk_sensitive": "Effect weighted",
        "maximin": "Maximin",
        "random": "Random",
        "fixed_sentinel": "Fixed sentinels",
    }
    colors = {
        "max_variance": TEAL,
        "risk_sensitive": ORANGE,
        "maximin": BLUE,
        "random": "#8d99ae",
        "fixed_sentinel": "#b6bec8",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for row in data.itertuples(index=False):
        size = 90 + 900 * row.direct_effect_capture
        ax.scatter(
            row.nrmse,
            row.effect_recall * 100,
            s=size,
            color=colors[row.policy],
            alpha=0.85,
            edgecolor="white",
            linewidth=1.4,
            zorder=3,
        )
        offsets = {
            "max_variance": (-35, 12),
            "risk_sensitive": (8, -18),
            "maximin": (8, 8),
            "random": (8, -5),
            "fixed_sentinel": (-70, -18),
        }
        ax.annotate(
            labels[row.policy],
            (row.nrmse, row.effect_recall * 100),
            xytext=offsets[row.policy],
            textcoords="offset points",
            fontsize=9,
            weight="bold" if row.policy == "max_variance" else "normal",
        )
    ax.annotate(
        "preferred direction",
        xy=(0.326, 98.6),
        xytext=(0.351, 95.7),
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.7},
        color=TEAL,
        fontsize=10,
    )
    ax.set_xlabel("Hidden-response NRMSE (lower is better)")
    ax.set_ylabel("Protected effect recall (%)")
    ax.set_xlim(0.325, 0.368)
    ax.set_ylim(94.8, 99.1)
    ax.grid(color="#dfe6eb", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Policy efficiency frontier\nMarker area encodes direct effect capture")
    fig.tight_layout()
    save(fig, "09_policy_pareto.png")


def ablation_figure() -> None:
    data = pd.read_csv(ROOT / "artifacts" / "act_v2" / "neural_ablations.csv")
    display = {
        "primary": "Primary",
        "single_div_hit": "Single-DIV hit",
        "no_abstention": "No abstention",
        "allow_10pct_calibration_error": "10% calibration error",
        "strict_nrmse_015": "Strict NRMSE",
        "lenient_nrmse_025": "Lenient NRMSE",
        "effect_threshold_030": "Effect threshold 0.30",
        "effect_threshold_050": "Effect threshold 0.50",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    ax = axes[0]
    for row in data.itertuples(index=False):
        unsafe = row.unsafe_stops > 0
        ax.scatter(
            row.session_reduction * 100,
            row.stops,
            s=110 + row.abstentions * 35,
            facecolor=RED if unsafe else TEAL,
            edgecolor=NAVY,
            linewidth=1,
            marker="X" if unsafe else "o",
            alpha=0.88,
        )
        if unsafe or row.ablation in {"primary", "effect_threshold_050"}:
            offsets = {
                "primary": (6, -11),
                "single_div_hit": (5, 5),
                "no_abstention": (5, 5),
                "effect_threshold_030": (5, 5),
                "effect_threshold_050": (5, -10),
            }
            ax.annotate(
                display[row.ablation],
                (row.session_reduction * 100, row.stops),
                xytext=offsets[row.ablation],
                textcoords="offset points",
                fontsize=7.7,
            )
    ax.axvline(0, color=NAVY, lw=0.8)
    ax.set_xlabel("Retrospective session reduction (%)")
    ax.set_ylabel("Candidate stops")
    ax.set_title("Development ablations\nX marks an unsafe stop")
    ax.grid(color="#dfe6eb", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    sorted_data = data.sort_values("session_reduction")
    colors = [RED if value > 0 else TEAL for value in sorted_data.unsafe_stops]
    axes[1].barh(
        [display[value] for value in sorted_data.ablation],
        sorted_data.session_reduction * 100,
        color=colors,
        alpha=0.86,
    )
    for index, row in enumerate(sorted_data.itertuples(index=False)):
        axes[1].text(
            row.session_reduction * 100 + 0.35,
            index,
            f"{row.session_reduction:.1%} | unsafe={row.unsafe_stops}",
            va="center",
            fontsize=8,
        )
    axes[1].set_xlabel("Retrospective session reduction (%)")
    axes[1].set_title("Savings disappear when safety is enforced")
    axes[1].grid(axis="x", color="#dfe6eb", lw=0.8)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    fig.suptitle(
        "Ablation analysis exposes the safety-efficiency tradeoff",
        y=1.03,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, "10_ablation_safety.png")


def vision_diagnostics_figure() -> None:
    ooc = pd.read_csv(ROOT / "artifacts" / "act" / "canonical" / "ooc_vision_predictions.csv")
    brain = pd.read_csv(ROOT / "artifacts" / "act" / "canonical" / "brain_vision_predictions.csv")
    brain = brain.loc[(brain.split == "test") & (brain.model == "lraspp_lab_a_only")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3))
    axes[0].hist(
        [ooc.loc[ooc.bad == 0, "probability_bad"], ooc.loc[ooc.bad == 1, "probability_bad"]],
        bins=np.linspace(0, 1, 21),
        color=[BLUE, ORANGE],
        alpha=0.72,
        label=["Expert good", "Expert bad"],
    )
    axes[0].set_xlabel("Predicted probability of bad image")
    axes[0].set_ylabel("Protected test images")
    axes[0].set_title("OOC chronological test")
    axes[0].legend(frameon=False, fontsize=8)
    paired = brain.groupby(["org_id", "laboratory"], as_index=False).dice.mean()
    paired = paired.pivot(index="org_id", columns="laboratory", values="dice").dropna()
    for _, row in paired.iterrows():
        axes[1].plot([0, 1], [row.LabA, row.LabB], color="#aeb7c2", lw=0.9, alpha=0.8)
    axes[1].scatter(np.zeros(len(paired)), paired.LabA, color=TEAL, s=28, zorder=3)
    axes[1].scatter(np.ones(len(paired)), paired.LabB, color=RED, s=28, zorder=3)
    axes[1].plot([0, 1], [paired.LabA.mean(), paired.LabB.mean()], color=NAVY, lw=3)
    axes[1].set_xticks([0, 1], ["Lab A", "Lab B"])
    axes[1].set_ylabel("Mean Dice per protected organoid")
    axes[1].set_title("Cross-laboratory paired shift")
    area_a = brain.loc[brain.laboratory == "LabA", "area_relative_error"] * 100
    area_b = brain.loc[brain.laboratory == "LabB", "area_relative_error"] * 100
    box = axes[2].boxplot(
        [area_a, area_b],
        tick_labels=["Lab A", "Lab B"],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": NAVY, "linewidth": 1.6},
    )
    for patch, color in zip(box["boxes"], [TEAL, RED], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[2].axhline(10, color=ORANGE, ls="--", lw=1.4, label="10% error")
    axes[2].set_ylabel("Area relative error (%)")
    axes[2].set_title("Segmentation error shift")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#dfe6eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Independent imaging tests reveal calibration and laboratory shift",
        y=1.03,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, "11_vision_diagnostics.png")


def error_structure_figure() -> None:
    data = pd.read_csv(ROOT / "artifacts" / "act" / "canonical" / "group_metrics.csv")
    data = data.loc[data.policy == "max_variance"].copy()
    colors = {"phase1": BLUE, "brown": ORANGE, "tipping": TEAL}
    labels = {"phase1": "Phase 1", "brown": "Brown", "tipping": "Tipping"}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    for dataset, subset in data.groupby("dataset"):
        axes[0].scatter(
            subset.nrmse,
            subset.effect_recall * 100,
            color=colors[dataset],
            label=f"{labels[dataset]} (n={len(subset)})",
            s=34,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[1].scatter(
            subset.effect_precision * 100,
            subset.effect_recall * 100,
            color=colors[dataset],
            s=34,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
        )
    axes[0].set_xlabel("Hidden-response NRMSE")
    axes[0].set_ylabel("Protected recall (%)")
    axes[0].set_title("Reconstruction error versus protection")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("Effect precision (%)")
    axes[1].set_ylabel("Protected recall (%)")
    axes[1].set_title("Conservative protection tradeoff")
    for ax in axes:
        ax.grid(color="#dfe6eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Group-level error structure makes heterogeneity visible",
        y=1.03,
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, "12_error_structure.png")


def add_page_numbers(source: Path, destination: Path, label: str) -> None:
    reader = PdfReader(source)
    writer = PdfWriter()
    total = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        buffer = io.BytesIO()
        overlay = canvas.Canvas(buffer, pagesize=(width, height))
        overlay.setFont("Helvetica-Bold", 7.5)
        overlay.setFillColorRGB(0.30, 0.36, 0.40)
        overlay.drawString(48, 24, label)
        overlay.drawString(width - 104, 24, f"Page {index} of {total}")
        overlay.save()
        buffer.seek(0)
        page.merge_page(PdfReader(buffer).pages[0])
        writer.add_page(page)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        writer.write(handle)


def render_markdown(
    source: Path,
    destination: Path,
    label: str,
    toc: bool = True,
    css: Path = CSS,
) -> None:
    if not CHROME.exists() or shutil.which("pandoc") is None:
        raise FileNotFoundError("Google Chrome and pandoc are required")
    with tempfile.TemporaryDirectory(prefix="gates-submission-") as temporary:
        temporary_path = Path(temporary)
        html = temporary_path / "report.html"
        raw_pdf = temporary_path / "report.pdf"
        command = [
            "pandoc",
            str(source),
            "--standalone",
            "--embed-resources",
            "--math-method=mathml",
            "--css",
            str(css),
            "--resource-path",
            f"{source.parent}:{ROOT}",
            "--output",
            str(html),
        ]
        if toc:
            command[2:2] = ["--toc", "--toc-depth=2"]
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(
            [
                str(CHROME),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={raw_pdf}",
                html.as_uri(),
            ],
            check=True,
        )
        add_page_numbers(raw_pdf, destination, label)


def main() -> None:
    style()
    system_diagram()
    performance_figure()
    forest_figure()
    evidence_figure()
    stopping_figure()
    source_landscape_figure()
    distribution_figure()
    acquisition_trajectory_figure()
    pareto_figure()
    ablation_figure()
    vision_diagnostics_figure()
    error_structure_figure()
    render_markdown(
        DOCS / "TECHNICAL_REPORT_V3.md",
        OUTPUT / "GATES_ACT_Submission_Technical_Report.pdf",
        "GATES-ACT | technical report",
        css=PAPER_CSS,
    )
    render_markdown(
        DOCS / "SUPPLEMENTARY_MATERIAL.md",
        OUTPUT / "GATES_ACT_Supplementary_Material.pdf",
        "GATES-ACT | supplementary material",
    )


if __name__ == "__main__":
    main()
