"""Build a PDF/PNG rollout sequence chart with vector labels."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True, help="Directory containing frame images.")
    parser.add_argument("--output-stem", required=True, help="Output path without extension.")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern for frame images.")
    parser.add_argument("--limit", type=int, default=8, help="Number of frames to include.")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns.")
    parser.add_argument("--aspect", default="16:9", help="Per-frame aspect ratio, e.g. 16:9.")
    parser.add_argument("--background", default="#ffffff", help="Canvas background color.")
    return parser.parse_args()


def parse_aspect(aspect: str) -> float:
    width, height = aspect.split(":")
    return float(width) / float(height)


def center_crop(img: Image.Image, target_aspect: float) -> Image.Image:
    width, height = img.size
    aspect = width / height
    if aspect > target_aspect:
        new_width = int(height * target_aspect)
        left = (width - new_width) // 2
        return img.crop((left, 0, left + new_width, height))
    new_height = int(width / target_aspect)
    top = (height - new_height) // 2
    return img.crop((0, top, width, top + new_height))


def main() -> None:
    args = parse_args()
    frame_paths = sorted(Path(args.frames_dir).glob(args.pattern))[: args.limit]
    if not frame_paths:
        raise FileNotFoundError(f"No frames matched {args.pattern!r} in {args.frames_dir}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    aspect = parse_aspect(args.aspect)
    cols = args.cols
    rows = math.ceil(len(frame_paths) / cols)

    fig_w = 7.4
    frame_w = 1.74
    frame_h = frame_w / aspect
    gap = 0.07
    margin_x = 0.05
    margin_y = 0.05
    fig_h = rows * frame_h + (rows - 1) * gap + 2 * margin_y

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor=args.background)

    for i, path in enumerate(frame_paths):
        row = i // cols
        col = i % cols
        x = margin_x + col * (frame_w + gap)
        y = margin_y + (rows - 1 - row) * (frame_h + gap)
        ax = fig.add_axes([x / fig_w, y / fig_h, frame_w / fig_w, frame_h / fig_h])

        img = Image.open(path).convert("RGB")
        img = center_crop(img, aspect)
        ax.imshow(img)
        ax.set_axis_off()

        badge_x = 0.035
        badge_y = 0.835
        badge_w = 0.10
        badge_h = 0.12
        badge = FancyBboxPatch(
            (badge_x, badge_y),
            badge_w,
            badge_h,
            boxstyle="round,pad=0,rounding_size=0.026",
            transform=ax.transAxes,
            facecolor="#f9f9f9",
            edgecolor="#a1a1a1",
            linewidth=1.25,
            zorder=3,
        )
        ax.add_patch(badge)
        ax.text(
            badge_x + badge_w / 2,
            badge_y + badge_h / 2 - 0.006,
            str(i + 1),
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#646464",
            fontsize=6.7,
            fontweight="bold",
            zorder=4,
        )

    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            output_stem.with_suffix(f".{ext}"),
            bbox_inches="tight",
            pad_inches=0,
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)
    print(f"Wrote {output_stem}.pdf/.svg/.png from {len(frame_paths)} frames")


if __name__ == "__main__":
    main()
