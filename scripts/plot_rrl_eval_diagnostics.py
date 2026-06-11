"""Plot residual-RL evaluation diagnostics from per-timestep CSV output.

The input CSV is produced by scripts.play_env / modal_eval_policy.py when
evaluation is run with --save-step-diagnostics. The plots intentionally focus
on directly recorded quantities:

1. Critic Q trajectory for the final residual action and base action.
2. Critic Q advantage over the base action.
3. Residual magnitude over time.

For each selected episode, vertical markers indicate the first timestep where
each stick satisfies the environment's placed predicate.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _float_or_nan(value: str | None) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def _int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _read_rows(path: str) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, fieldnames


def _group_by_episode(rows: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode"])].append(row)
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: int(row["timestep"]))
    return dict(sorted(grouped.items()))


def _stick_ids(fieldnames: list[str]) -> list[int]:
    ids = []
    for name in fieldnames:
        if name.startswith("stick") and name.endswith("_placed"):
            middle = name[len("stick") : -len("_placed")]
            if middle.isdigit():
                ids.append(int(middle))
    return sorted(ids)


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([_float_or_nan(row.get(key)) for row in rows], dtype=float)


def _first_placed_timestep(rows: list[dict], stick_id: int) -> int | None:
    key = f"stick{stick_id}_placed"
    for row in rows:
        if _int_or_none(row.get(key)) == 1:
            return int(row["timestep"])
    return None


def _mark_placements(ax, rows: list[dict], stick_ids: list[int]) -> None:
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for idx, stick_id in enumerate(stick_ids):
        timestep = _first_placed_timestep(rows, stick_id)
        if timestep is None:
            continue
        color = colors[idx % len(colors)]
        ax.axvline(timestep, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            timestep,
            0.98,
            f"stick {stick_id} placed",
            color=color,
            rotation=90,
            va="top",
            ha="right",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
        )


def _plot_episode(rows: list[dict], stick_ids: list[int], out_path: str, title: str) -> None:
    t = _series(rows, "timestep")
    q_final = _series(rows, "q_final_mean")
    q_base = _series(rows, "q_base_mean")
    q_adv = _series(rows, "q_advantage_mean")
    residual_l1 = _series(rows, "residual_l1")
    residual_l2 = _series(rows, "residual_l2")
    reward = _series(rows, "reward")

    missing = [
        name
        for name, arr in [
            ("q_final_mean", q_final),
            ("q_base_mean", q_base),
            ("q_advantage_mean", q_adv),
            ("residual_l1", residual_l1),
            ("residual_l2", residual_l2),
        ]
        if np.all(np.isnan(arr))
    ]
    if missing:
        raise ValueError(
            "CSV is missing RRL diagnostics needed for plotting: "
            + ", ".join(missing)
            + ". Re-run eval with an RRL checkpoint and --save-step-diagnostics."
        )

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(title)

    axes[0].plot(t, q_final, label="Q(final action)", color="#1f77b4", linewidth=1.6)
    axes[0].plot(t, q_base, label="Q(base action)", color="#7f7f7f", linewidth=1.3)
    _mark_placements(axes[0], rows, stick_ids)
    axes[0].set_ylabel("critic Q")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, q_adv, label="Q(final) - Q(base)", color="#d62728", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linewidth=0.9, alpha=0.6)
    _mark_placements(axes[1], rows, stick_ids)
    axes[1].set_ylabel("Q advantage")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, residual_l1, label="residual L1", color="#17becf", linewidth=1.5)
    axes[2].plot(t, residual_l2, label="residual L2", color="#9467bd", linewidth=1.2)
    reward_scaled = reward / max(np.nanmax(np.abs(reward)), 1e-6)
    axes[2].plot(t, reward_scaled, label="reward (scaled)", color="#2ca02c", linewidth=1.0, alpha=0.65)
    _mark_placements(axes[2], rows, stick_ids)
    axes[2].set_ylabel("magnitude")
    axes[2].set_xlabel("environment timestep")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _episode_score(rows: list[dict], mode: str) -> float:
    if mode == "first":
        return -int(rows[0]["episode"])
    if mode == "success":
        return float(int(rows[-1].get("success", "0")))
    if mode == "failure":
        return 1.0 - float(int(rows[-1].get("success", "0")))
    if mode == "reward":
        return float(_float_or_nan(rows[-1].get("cumulative_reward")))
    raise ValueError(f"Unknown episode selection mode: {mode}")


def _select_episodes(grouped: dict[int, list[dict]], mode: str, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if mode == "first":
        return list(grouped)[:count]
    scored = sorted(
        ((_episode_score(rows, mode), episode) for episode, rows in grouped.items()),
        reverse=True,
    )
    return [episode for _, episode in scored[:count]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-csv", required=True, help="Per-timestep eval diagnostics CSV")
    parser.add_argument("--out-dir", required=True, help="Directory for PNG plots")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Specific episode ids to plot. Default uses --select.")
    parser.add_argument("--select", choices=["first", "success", "failure", "reward"], default="success",
                        help="How to auto-select episodes when --episodes is omitted.")
    parser.add_argument("--count", type=int, default=3, help="Number of auto-selected episodes to plot")
    parser.add_argument("--title-prefix", default="RRL eval diagnostics")
    args = parser.parse_args()

    rows, fieldnames = _read_rows(args.steps_csv)
    grouped = _group_by_episode(rows)
    stick_ids = _stick_ids(fieldnames)
    if not stick_ids:
        raise ValueError("No stick*_placed columns found in diagnostics CSV.")

    selected = args.episodes or _select_episodes(grouped, args.select, args.count)
    missing = [episode for episode in selected if episode not in grouped]
    if missing:
        raise ValueError(f"Episodes not found in CSV: {missing}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loaded {len(rows)} steps from {len(grouped)} episodes")
    print(f"Plotting episodes: {selected}")
    for episode in selected:
        episode_rows = grouped[episode]
        order = episode_rows[0].get("order_str", "")
        success = int(float(episode_rows[-1].get("success", "0")))
        reward = _float_or_nan(episode_rows[-1].get("cumulative_reward"))
        title = f"{args.title_prefix} | episode {episode} | order {order} | success={success} | reward={reward:.1f}"
        out_path = os.path.join(args.out_dir, f"episode_{episode:03d}_q_residual.png")
        _plot_episode(episode_rows, stick_ids, out_path, title)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
