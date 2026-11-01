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
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "act"
FIGURES = ROOT / "artifacts" / "act" / "submission_figures"
OUTPUT = ROOT / "output" / "pdf"
CSS = ROOT / "docs" / "final" / "report.css"
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
    fig, ax = plt.subplots(figsize=(12.4, 4.0))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.2, 2.0, 1.5, "1  Eligible data", "Current-day doses\n+ vehicle anchor", BLUE),
        (2.7, 1.2, 2.0, 1.5, "2  Predict", "Robust prior\n+ posterior covariance", TEAL),
        (5.2, 1.2, 2.0, 1.5, "3  Acquire", "Conditional maximum\nvariance per unit cost", ORANGE),
        (7.7, 1.2, 2.0, 1.5, "4  Protect", "Conformal effect band\n+ group-shift refusal", GREEN),
        (10.2, 1.2, 2.0, 1.5, "5  Act", "COMPLETE or ABSTAIN\nSTOP remains locked", RED),
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
        ax.text(x + 0.16, y + 1.08, title, fontsize=11.5, weight="bold", color=color)
        ax.text(x + 0.16, y + 0.68, body, fontsize=9.5, va="top", linespacing=1.35)
    for x in [2.25, 4.75, 7.25, 9.75]:
        ax.add_patch(
            FancyArrowPatch(
                (x, 1.95), (x + 0.4, 1.95), arrowstyle="-|>", mutation_scale=13, lw=1.6, color=NAVY
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (11.2, 1.15),
            (1.2, 0.78),
            connectionstyle="arc3,rad=-0.12",
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.8,
            color=GREY,
        )
    )
    ax.text(
        6.2,
        0.18,
        "Prospective shadow mode: recommendations are logged while complete plates continue",
        ha="center",
        fontsize=10.5,
        color=GREY,
    )
    ax.set_title(
        "GATES-ACT converts adaptive measurement into a fail-closed laboratory workflow", pad=10
    )
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


def add_page_numbers(source: Path, destination: Path, label: str) -> None:
    reader = PdfReader(source)
    writer = PdfWriter()
    total = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        buffer = io.BytesIO()
        overlay = canvas.Canvas(buffer, pagesize=(width, height))
        overlay.setFont("Helvetica", 7.5)
        overlay.setFillColorRGB(0.30, 0.36, 0.40)
        overlay.drawString(48, 24, label)
        overlay.drawRightString(width - 48, 24, f"{index} / {total}")
        overlay.save()
        buffer.seek(0)
        page.merge_page(PdfReader(buffer).pages[0])
        writer.add_page(page)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        writer.write(handle)


def render_markdown(source: Path, destination: Path, label: str, toc: bool = True) -> None:
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
            str(CSS),
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
    render_markdown(
        DOCS / "TECHNICAL_REPORT_V3.md",
        OUTPUT / "GATES_ACT_Submission_Technical_Report.pdf",
        "GATES-ACT | technical report",
    )
    render_markdown(
        DOCS / "SUPPLEMENTARY_MATERIAL.md",
        OUTPUT / "GATES_ACT_Supplementary_Material.pdf",
        "GATES-ACT | supplementary material",
    )


if __name__ == "__main__":
    main()
