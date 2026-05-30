"""Extract evenly spaced PNG frames from a recorded rollout video."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted PNGs.")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of frames to extract.")
    parser.add_argument("--start-frac", type=float, default=0.05, help="Fraction of video to start at.")
    parser.add_argument("--end-frac", type=float, default=0.95, help="Fraction of video to end at.")
    parser.add_argument(
        "--indices",
        default=None,
        help="Optional comma-separated frame indices. Overrides --num-frames.",
    )
    parser.add_argument("--prefix", default="frame", help="Output filename prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(video_path)
    try:
        n_frames = reader.count_frames()
    except Exception:
        n_frames = reader.get_length()
    if n_frames == float("inf"):
        frames = list(reader)
        n_frames = len(frames)
    else:
        frames = None

    if args.indices:
        indices = [int(i.strip()) for i in args.indices.split(",") if i.strip()]
    else:
        start = int(max(0, min(1, args.start_frac)) * (n_frames - 1))
        end = int(max(0, min(1, args.end_frac)) * (n_frames - 1))
        if end <= start:
            raise ValueError("--end-frac must be greater than --start-frac")
        indices = np.linspace(start, end, args.num_frames, dtype=int).tolist()

    written = []
    for out_i, frame_i in enumerate(indices, start=1):
        frame = frames[frame_i] if frames is not None else reader.get_data(frame_i)
        out_path = output_dir / f"{args.prefix}_{out_i:03d}_f{frame_i:05d}.png"
        imageio.imwrite(out_path, frame)
        written.append(out_path)

    reader.close()
    print(f"Extracted {len(written)} frames from {video_path} to {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
