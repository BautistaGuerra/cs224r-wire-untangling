"""Generate the milestone success-rate comparison figure."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def plot_success_comparison(output_stem: str, background: str, track: str, grid: str) -> None:
    rows = [
        ("Fixed-order demos", "Observation only", 1, "0-1", "obs"),
        ("Fixed-order demos", "Oracle context", 94, "94", "context"),
        ("Balanced-order demos", "Observation only", 68, "68", "obs"),
        ("Balanced-order demos", "Oracle context", 96, "96", "context"),
    ]
    y_positions = [3.15, 2.72, 1.42, 0.99]
    group_y = {"Fixed-order demos": 3.48, "Balanced-order demos": 1.75}

    # Technical-report style: editable text in vector outputs, simple sans-serif font.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )

    fig, ax = plt.subplots(figsize=(4.15, 2.35), dpi=300)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    colors = {
        "obs": "#e59381",
        "context": "#80bfe5",
    }
    text = "#242424"
    group_text = "#46423d"
    muted = "#6d6a64"

    ax.set_xlim(0, 104)
    ax.set_ylim(0.62, 3.78)

    for x_tick in [0, 50, 100]:
        ax.axvline(x_tick, color=grid, linewidth=0.6, zorder=0)

    bar_h = 0.22
    for (group, label, value, value_label, key), y in zip(rows, y_positions):
        ax.add_patch(
            FancyBboxPatch(
                (0, y - bar_h / 2),
                100,
                bar_h,
                boxstyle=f"round,pad=0,rounding_size={bar_h / 2}",
                linewidth=0,
                facecolor=track,
                zorder=1,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0, y - bar_h / 2),
                value,
                bar_h,
                boxstyle=f"round,pad=0,rounding_size={bar_h / 2}",
                linewidth=0,
                facecolor=colors[key],
                zorder=2,
            )
        )
        ax.text(
            -2.4,
            y,
            label,
            ha="right",
            va="center",
            fontsize=7.2,
            color=text,
            clip_on=False,
        )
        ax.text(
            min(value + 2.2, 101.5),
            y,
            f"{value_label}%",
            ha="left",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=colors[key],
        )

    for group, y in group_y.items():
        ax.text(
            0,
            y,
            group,
            ha="left",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=group_text,
        )

    ax.set_xlabel("Closed-loop success (%)", fontsize=7.2, color=muted, labelpad=5)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100"], fontsize=6.8, color=muted)
    ax.tick_params(axis="x", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.19, top=0.9)

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            Path("docs/assets") / f"{output_stem}.{suffix}",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def main() -> None:
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_success_comparison(
        output_stem="n2_bc_success_comparison",
        background="#fbfaf7",
        track="#ebe6dc",
        grid="#ded8ce",
    )
    plot_success_comparison(
        output_stem="n2_bc_success_comparison-2",
        background="#ffffff",
        track="#f0f0f0",
        grid="#e0e0e0",
    )


if __name__ == "__main__":
    main()
