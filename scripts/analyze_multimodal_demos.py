"""Analyze paired-order multimodal demo datasets.

The paired N=2 ablation stores each accepted reset seed twice: once with
expert order [0, 1] and once with [1, 0]. This script verifies that paired
demos really start from the same observation and quantifies how different the
early expert actions are.
"""

import argparse
from dataclasses import dataclass

import h5py
import numpy as np


@dataclass(frozen=True)
class PairSummary:
    pair_id: int
    initial_obs_max_abs_diff: float
    initial_action_l2: float
    prefix_action_l2: float
    initial_action_mean_norm_ratio: float
    initial_action_mean_y_abs: float
    branch_initial_action_y_abs_mean: float
    orders: tuple[tuple[int, ...], ...]


def _demo_sort_key(name: str) -> int:
    return int(name.split("_")[-1])


def _order_attr(demo) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(demo.attrs["stick_order"]).tolist())


def _format_order(order: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(int(i)) for i in order) + "]"


def load_pair_summaries(
    demos_path: str,
    prefix_horizon: int = 16,
) -> list[PairSummary]:
    """Return per-pair multimodality diagnostics from an HDF5 dataset."""
    pair_entries: dict[int, list[tuple[int, str]]] = {}
    with h5py.File(demos_path, "r") as f:
        if f.attrs.get("multimodal_collection", "none") != "paired_order":
            raise ValueError(
                f"{demos_path} is not marked as a paired-order multimodal dataset"
            )
        data = f["data"]
        for name in sorted(data.keys(), key=_demo_sort_key):
            demo = data[name]
            if "multimodal_pair_id" not in demo.attrs:
                raise ValueError(f"data/{name} is missing multimodal_pair_id")
            pair_id = int(demo.attrs["multimodal_pair_id"])
            branch = int(demo.attrs.get("multimodal_branch", len(pair_entries.get(pair_id, []))))
            pair_entries.setdefault(pair_id, []).append((branch, name))

        summaries = []
        for pair_id, entries in sorted(pair_entries.items()):
            entries = sorted(entries)
            if len(entries) != 2:
                raise ValueError(f"pair {pair_id} has {len(entries)} demos; expected 2")
            demos = [data[name] for _, name in entries]
            obs0 = [demo["obs"][0] for demo in demos]
            act0 = [demo["actions"][0] for demo in demos]
            horizon = min(prefix_horizon, *(len(demo["actions"]) for demo in demos))
            chunks = [demo["actions"][:horizon].reshape(-1) for demo in demos]

            obs_diff = float(np.max(np.abs(obs0[0] - obs0[1])))
            action_delta = act0[0] - act0[1]
            chunk_delta = chunks[0] - chunks[1]
            action_norms = [np.linalg.norm(a) for a in act0]
            denom = float(np.mean(action_norms))
            mean_norm_ratio = 0.0 if denom == 0.0 else float(np.linalg.norm(np.mean(act0, axis=0)) / denom)

            summaries.append(
                PairSummary(
                    pair_id=pair_id,
                    initial_obs_max_abs_diff=obs_diff,
                    initial_action_l2=float(np.linalg.norm(action_delta)),
                    prefix_action_l2=float(np.linalg.norm(chunk_delta)),
                    initial_action_mean_norm_ratio=mean_norm_ratio,
                    initial_action_mean_y_abs=float(abs(np.mean([a[1] for a in act0]))),
                    branch_initial_action_y_abs_mean=float(np.mean([abs(a[1]) for a in act0])),
                    orders=tuple(_order_attr(demo) for demo in demos),
                )
            )
    return summaries


def summarize_pairs(
    summaries: list[PairSummary],
    obs_tolerance: float = 1e-6,
    action_diff_threshold: float = 0.25,
) -> dict:
    if not summaries:
        raise ValueError("No paired demos found")

    obs_diffs = np.array([s.initial_obs_max_abs_diff for s in summaries], dtype=np.float64)
    action_l2 = np.array([s.initial_action_l2 for s in summaries], dtype=np.float64)
    prefix_l2 = np.array([s.prefix_action_l2 for s in summaries], dtype=np.float64)
    mean_norm_ratio = np.array([s.initial_action_mean_norm_ratio for s in summaries], dtype=np.float64)
    mean_y_abs = np.array([s.initial_action_mean_y_abs for s in summaries], dtype=np.float64)
    branch_y_abs = np.array([s.branch_initial_action_y_abs_mean for s in summaries], dtype=np.float64)
    y_cancellation_ratio = np.divide(
        mean_y_abs,
        np.maximum(branch_y_abs, 1e-12),
    )

    order_counts: dict[tuple[int, ...], int] = {}
    for summary in summaries:
        for order in summary.orders:
            order_counts[order] = order_counts.get(order, 0) + 1

    return {
        "num_pairs": len(summaries),
        "num_demos": 2 * len(summaries),
        "identical_initial_obs_pairs": int(np.sum(obs_diffs <= obs_tolerance)),
        "action_multimodal_pairs": int(np.sum(action_l2 >= action_diff_threshold)),
        "max_initial_obs_diff": float(np.max(obs_diffs)),
        "mean_initial_action_l2": float(np.mean(action_l2)),
        "median_initial_action_l2": float(np.median(action_l2)),
        "mean_prefix_action_l2": float(np.mean(prefix_l2)),
        "median_prefix_action_l2": float(np.median(prefix_l2)),
        "mean_action_mean_norm_ratio": float(np.mean(mean_norm_ratio)),
        "mean_y_cancellation_ratio": float(np.mean(y_cancellation_ratio)),
        "order_counts": order_counts,
    }


def print_summary(summary: dict) -> None:
    print("Paired-order multimodal demo summary")
    print(f"  pairs: {summary['num_pairs']} ({summary['num_demos']} demos)")
    print(
        "  identical initial observations: "
        f"{summary['identical_initial_obs_pairs']}/{summary['num_pairs']}"
    )
    print(
        "  action-multimodal pairs: "
        f"{summary['action_multimodal_pairs']}/{summary['num_pairs']}"
    )
    print(f"  max initial obs |delta|: {summary['max_initial_obs_diff']:.3e}")
    print(f"  initial action L2 mean/median: "
          f"{summary['mean_initial_action_l2']:.3f} / {summary['median_initial_action_l2']:.3f}")
    print(f"  prefix action L2 mean/median: "
          f"{summary['mean_prefix_action_l2']:.3f} / {summary['median_prefix_action_l2']:.3f}")
    print(
        "  ||mean initial action|| / mean(||branch action||): "
        f"{summary['mean_action_mean_norm_ratio']:.3f}"
    )
    print(
        "  |mean initial y action| / mean(|branch y action|): "
        f"{summary['mean_y_cancellation_ratio']:.3f}"
    )
    print("  order counts:")
    for order, count in sorted(summary["order_counts"].items()):
        print(f"    {_format_order(order)}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-path", required=True)
    parser.add_argument("--prefix-horizon", type=int, default=16)
    parser.add_argument("--obs-tolerance", type=float, default=1e-6)
    parser.add_argument("--action-diff-threshold", type=float, default=0.25)
    args = parser.parse_args()

    summaries = load_pair_summaries(args.demos_path, prefix_horizon=args.prefix_horizon)
    summary = summarize_pairs(
        summaries,
        obs_tolerance=args.obs_tolerance,
        action_diff_threshold=args.action_diff_threshold,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
