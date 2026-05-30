"""Build a rollout sequence chart from a folder of frame images."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True, help="Directory containing frame images.")
    parser.add_argument("--output", required=True, help="Output PNG/JPG/PDF path.")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern for frame images.")
    parser.add_argument("--limit", type=int, default=8, help="Number of frames to include.")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns.")
    parser.add_argument("--frame-width", type=int, default=420, help="Per-frame output width.")
    parser.add_argument("--aspect", default="16:9", help="Per-frame aspect ratio, e.g. 16:9.")
    parser.add_argument("--gap", type=int, default=14, help="Gap between frames in pixels.")
    parser.add_argument("--margin", type=int, default=18, help="Outer margin in pixels.")
    parser.add_argument("--background", default="#ffffff", help="Canvas background color.")
    parser.add_argument("--label-prefix", default="", help="Optional prefix before frame number.")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw frame labels.")
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


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, label: str) -> None:
    label_font = font(22)
    bbox = draw.textbbox((0, 0), label, font=label_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    pad_x = 12
    pad_y = 8
    rect = (x, y, x + width + 2 * pad_x, y + height + 2 * pad_y)
    draw.rounded_rectangle(
        rect,
        radius=10,
        fill="#f9f9f9",
        outline="#a1a1a1",
        width=2,
    )
    text_x = x + (rect[2] - rect[0] - width) / 2 - bbox[0]
    text_y = y + (rect[3] - rect[1] - height) / 2 - bbox[1]
    draw.text((text_x, text_y), label, fill="#646464", font=label_font)


def main() -> None:
    args = parse_args()
    frame_paths = sorted(Path(args.frames_dir).glob(args.pattern))[: args.limit]
    if not frame_paths:
        raise FileNotFoundError(f"No frames matched {args.pattern!r} in {args.frames_dir}")

    aspect = parse_aspect(args.aspect)
    frame_w = args.frame_width
    frame_h = int(round(frame_w / aspect))
    cols = args.cols
    rows = math.ceil(len(frame_paths) / cols)
    canvas_w = args.margin * 2 + cols * frame_w + (cols - 1) * args.gap
    canvas_h = args.margin * 2 + rows * frame_h + (rows - 1) * args.gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), args.background)
    draw = ImageDraw.Draw(canvas, "RGBA")

    for i, path in enumerate(frame_paths):
        row = i // cols
        col = i % cols
        x = args.margin + col * (frame_w + args.gap)
        y = args.margin + row * (frame_h + args.gap)

        img = Image.open(path).convert("RGB")
        img = center_crop(img, aspect).resize((frame_w, frame_h), Image.Resampling.LANCZOS)
        canvas.paste(img, (x, y))
        if not args.no_labels:
            draw_label(draw, x + 10, y + 10, f"{args.label_prefix}{i + 1}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Wrote {output} from {len(frame_paths)} frames")


if __name__ == "__main__":
    main()
