"""
Phase-conditional analysis of the scripted expert policy.

Reads a demo HDF5 produced by collect_demos.py and runs five analyses, each
printed as a table and optionally saved as a PNG plot.

Schema (matches scripts/collect_demos.py):
    data/demo_<i>/{obs, actions, rewards, dones, next_obs, phase, is_success}
    where `phase` is int8 (0..7) and `is_success` is per-step bool.

Analyses:
  1. Phase duration distribution     — mean/std/median/p95 step counts per phase
  2. Phase-conditional failures       — only meaningful if HDF5 contains failed
                                          demos (current collect_demos.py drops
                                          them; degrades gracefully)
  3. Action statistics by phase       — per-axis mean |action| and saturation
  4. Reward decomposition by phase    — total + per-step reward per phase
  5. Initial-condition correlations   — initial stick yaw misalignment / XY
                                          distance to goal vs phase durations.
                                          Uses the obs_index_map saved on the
                                          HDF5 root by recent collect_demos.py.
                                          For older datasets without that
                                          attribute, the env is rebuilt from
                                          env_config (skip with --no-rebuild-env).

Usage:
    python scripts/analyze_expert.py --demos data/demos.hdf5
    python scripts/analyze_expert.py --demos data/demos.hdf5 --save-plots analysis/expert/
    python scripts/analyze_expert.py --demos data/demos.hdf5 --phase APPROACH
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Optional

import h5py
import matplotlib
import numpy as np
from robosuite.wrappers import GymWrapper

# Use the non-interactive Agg backend so plots can be saved headlessly.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after .use())

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies import build_obs_index_map


PHASE_ORDER = [
    "APPROACH", "DESCEND", "GRASP", "LIFT",
    "TRANSPORT", "PLACE", "RELEASE", "RETREAT",
]

ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
SATURATION_THRESHOLD = 0.95


def _decode_phase(arr: np.ndarray) -> np.ndarray:
    """Map per-step phase to string names. Accepts either int8 (Phase IntEnum
    value) or fixed-length-string (legacy schema)."""
    if arr.dtype.kind in ("i", "u"):
        return np.array([PHASE_ORDER[int(i)] for i in arr])
    return np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in arr])


def _quat_to_yaw(q_xyzw):
    x, y, z, w = q_xyzw
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap_pi_over_2(angle: float) -> float:
    """Wrap to [-pi/2, pi/2] (stick has 180° symmetry around z)."""
    a = (angle + np.pi) % (2 * np.pi) - np.pi
    if a > np.pi / 2:
        a -= np.pi
    elif a < -np.pi / 2:
        a += np.pi
    return a


def _build_obs_map_from_env_config(env_config: dict) -> dict:
    """Rebuild env from env_config and return its obs index map. Skipped via
    --no-rebuild-env when only running analyses that don't need it
    (durations, action stats, reward decomposition)."""

    raw_env = StickReorderEnv(
        robots=env_config.get("robot", "Panda"),
        num_sticks=env_config.get("num_sticks", 1),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=env_config.get("control_freq", 20),
        horizon=env_config.get("horizon", 500),
    )
    gym_env = GymWrapper(raw_env)
    obs_map = build_obs_index_map(gym_env)
    gym_env.close()
    return obs_map


def load_demos(path: str, rebuild_env: bool = True):
    """Load all demos. Returns (demos, obs_map_or_None, meta).

    If rebuild_env=True and the file doesn't carry an `obs_index_map` attr,
    the env is reconstructed from `env_config` to recover one. Set False to
    skip — initial-condition analysis will be unavailable.
    """
    with h5py.File(path, "r") as f:
        # Support both schemas: my old one stored obs_index_map directly;
        # the canonical schema requires rebuilding from env_config.
        obs_map = None
        if "obs_index_map" in f.attrs:
            raw = json.loads(f.attrs["obs_index_map"])
            obs_map = {k: slice(v[0], v[1]) for k, v in raw.items()}
        env_config = json.loads(f.attrs.get("env_config", "{}"))

        demos = []
        for key in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
            grp = f["data"][key]
            phase_arr = _decode_phase(grp["phase"][:])
            # Determine per-episode success: prefer per-step is_success,
            # else fall back to the per-demo `success` attribute (legacy).
            if "is_success" in grp:
                final_success = bool(grp["is_success"][-1])
            elif "success" in grp.attrs:
                final_success = bool(grp.attrs["success"])
            else:
                final_success = True  # main only saves successes
            demos.append({
                "success": final_success,
                "obs": grp["obs"][:],
                "actions": grp["actions"][:],
                "rewards": grp["rewards"][:],
                "phase": phase_arr,
            })

        # Compute meta (the canonical schema only stores num_demos, not the
        # success/failure split — derive it from per-demo success flags).
        n_succ = sum(1 for d in demos if d["success"])
        n_fail = len(demos) - n_succ
        meta = {
            "num_demos": int(f.attrs.get("num_demos", len(demos))),
            "num_successes": n_succ,
            "num_failures": n_fail,
            "obs_dim": int(f.attrs.get("obs_dim", demos[0]["obs"].shape[1])),
            "env_config": env_config,
        }

    if obs_map is None and rebuild_env:
        # Let exceptions propagate — the original traceback is far more
        # informative than a swallowed warning. Pass --no-rebuild-env to
        # skip this path entirely if you don't need the obs_index_map.
        obs_map = _build_obs_map_from_env_config(env_config)
    return demos, obs_map, meta


def _phase_durations(demos):
    """Per-episode dict[phase] -> step count. Returns list of dicts."""
    out = []
    for d in demos:
        counts = defaultdict(int)
        for p in d["phase"]:
            counts[p] += 1
        out.append(dict(counts))
    return out


def _print_table(rows, headers):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


def _maybe_save_bar(values, labels, title, ylabel, save_dir, fname):
    if save_dir is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(save_dir, fname)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → saved {out}")


def analyze_phase_durations(demos, save_dir, phase_filter):
    print("\n=== 1. Phase duration distribution ===")
    durations = _phase_durations(demos)
    rows = []
    means = []
    labels = []
    for phase in PHASE_ORDER:
        if phase_filter and phase != phase_filter:
            continue
        vals = np.array([d.get(phase, 0) for d in durations], dtype=np.float64)
        nonzero = vals[vals > 0]
        if len(nonzero) == 0:
            rows.append([phase, "0", "—", "—", "—", "—"])
            continue
        rows.append([
            phase,
            f"{int((vals > 0).sum())}",
            f"{nonzero.mean():.1f}",
            f"{nonzero.std():.1f}",
            f"{np.median(nonzero):.0f}",
            f"{np.percentile(nonzero, 95):.0f}",
        ])
        means.append(nonzero.mean())
        labels.append(phase)
    _print_table(rows, ["phase", "n_episodes", "mean", "std", "median", "p95"])
    _maybe_save_bar(means, labels, "Phase duration (mean steps)",
                    "steps", save_dir, "phase_durations.png")


def analyze_failures(demos, save_dir, phase_filter):
    print("\n=== 2. Phase-conditional failure attribution ===")
    failed = [d for d in demos if not d["success"]]
    if not failed:
        print("  No failed episodes in dataset (collect_demos.py drops failures by default). "
              "Skipping. For failure attribution, run scripts/diagnose_expert.py.")
        return
    print(f"  {len(failed)} failed episodes.")
    counts = defaultdict(int)
    for d in failed:
        # Phase active at the last action of the failed episode
        last_phase = d["phase"][-1] if len(d["phase"]) else "UNKNOWN"
        counts[last_phase] += 1
    rows = []
    values = []
    labels = []
    for phase in PHASE_ORDER:
        if phase_filter and phase != phase_filter:
            continue
        c = counts.get(phase, 0)
        rows.append([phase, c, f"{100 * c / len(failed):.1f}%"])
        values.append(c)
        labels.append(phase)
    _print_table(rows, ["phase", "failures", "pct"])
    _maybe_save_bar(values, labels, "Failure attribution by phase",
                    "count", save_dir, "phase_failures.png")


def analyze_action_stats(demos, save_dir, phase_filter):
    print("\n=== 3. Action statistics by phase ===")
    # Aggregate per-phase actions across all episodes
    per_phase = defaultdict(list)
    for d in demos:
        for action, phase in zip(d["actions"], d["phase"]):
            per_phase[phase].append(action)
    rows = []
    for phase in PHASE_ORDER:
        if phase_filter and phase != phase_filter:
            continue
        actions = per_phase.get(phase)
        if not actions:
            rows.append([phase, "0"] + ["—"] * (2 * len(ACTION_LABELS)))
            continue
        actions = np.array(actions)
        means = np.abs(actions).mean(axis=0)
        sat = (np.abs(actions) > SATURATION_THRESHOLD).mean(axis=0)
        row = [phase, f"{len(actions)}"]
        for i in range(len(ACTION_LABELS)):
            row.append(f"{means[i]:.2f}")
            row.append(f"{100 * sat[i]:.0f}%")
        rows.append(row)
    headers = ["phase", "n_steps"]
    for lbl in ACTION_LABELS:
        headers += [f"|{lbl}|", f"{lbl}_sat"]
    _print_table(rows, headers)


def analyze_rewards(demos, save_dir, phase_filter):
    print("\n=== 4. Reward decomposition by phase ===")
    # Per-episode dict[phase] -> total reward in that phase
    per_ep = []
    for d in demos:
        sums = defaultdict(float)
        steps = defaultdict(int)
        for r, p in zip(d["rewards"], d["phase"]):
            sums[p] += float(r)
            steps[p] += 1
        per_ep.append((dict(sums), dict(steps)))
    rows = []
    for phase in PHASE_ORDER:
        if phase_filter and phase != phase_filter:
            continue
        totals = np.array([ep[0].get(phase, 0.0) for ep in per_ep])
        steps = np.array([ep[1].get(phase, 0) for ep in per_ep])
        nonzero_mask = steps > 0
        if not nonzero_mask.any():
            rows.append([phase, "0", "—", "—", "—"])
            continue
        per_step = totals[nonzero_mask] / steps[nonzero_mask]
        rows.append([
            phase,
            f"{int(nonzero_mask.sum())}",
            f"{totals[nonzero_mask].mean():.3f}",
            f"{totals[nonzero_mask].std():.3f}",
            f"{per_step.mean():.4f}",
        ])
    _print_table(rows, ["phase", "n_episodes", "total_mean", "total_std", "per_step_mean"])


def analyze_initial_conditions(demos, obs_map, save_dir, phase_filter):
    print("\n=== 5. Initial-condition correlations ===")
    if obs_map is None:
        print("  obs_index_map unavailable (rebuild env_config or pass --no-rebuild-env was used); skipping.")
        return
    required = {"robot0_eef_quat", "stick0_pos", "stick0_quat", "goal0_pos"}
    missing = required - obs_map.keys()
    if missing:
        print(f"  Missing observables {missing}; skipping.")
        return

    yaw_misalign = []
    xy_dist = []
    durations_by_phase = defaultdict(list)

    for d in demos:
        first = d["obs"][0]
        eef_quat = first[obs_map["robot0_eef_quat"]]
        stick_pos = first[obs_map["stick0_pos"]]
        stick_quat = first[obs_map["stick0_quat"]]
        goal_pos = first[obs_map["goal0_pos"]]

        eef_yaw = _quat_to_yaw(eef_quat)
        stick_yaw = _quat_to_yaw(stick_quat)
        misalign = abs(_wrap_pi_over_2(stick_yaw - eef_yaw))
        dist = float(np.linalg.norm(stick_pos[:2] - goal_pos[:2]))

        yaw_misalign.append(misalign)
        xy_dist.append(dist)

        counts = defaultdict(int)
        for p in d["phase"]:
            counts[p] += 1
        for phase in PHASE_ORDER:
            durations_by_phase[phase].append(counts.get(phase, 0))

    yaw_misalign = np.array(yaw_misalign)
    xy_dist = np.array(xy_dist)

    print(f"  yaw misalignment: mean={yaw_misalign.mean():.3f} rad, max={yaw_misalign.max():.3f}")
    print(f"  xy distance:      mean={xy_dist.mean():.3f} m,   max={xy_dist.max():.3f}")

    rows = []
    for phase in PHASE_ORDER:
        if phase_filter and phase != phase_filter:
            continue
        durs = np.array(durations_by_phase[phase])
        if durs.std() == 0:
            rows.append([phase, "—", "—"])
            continue
        # Pearson correlation
        r_yaw = float(np.corrcoef(yaw_misalign, durs)[0, 1]) if yaw_misalign.std() else float("nan")
        r_dist = float(np.corrcoef(xy_dist, durs)[0, 1]) if xy_dist.std() else float("nan")
        rows.append([phase, f"{r_yaw:+.2f}", f"{r_dist:+.2f}"])
    _print_table(rows, ["phase", "r(yaw_misalign, dur)", "r(xy_dist, dur)"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", required=True)
    parser.add_argument("--save-plots", default=None,
                        help="Directory to write PNGs into. Default: no plots.")
    parser.add_argument("--phase", default=None,
                        help="Filter all analyses to a single phase (e.g. APPROACH).")
    parser.add_argument("--no-rebuild-env", action="store_true",
                        help="Skip rebuilding the env from env_config (used for "
                             "the initial-condition correlation analysis).")
    args = parser.parse_args()

    if args.phase and args.phase not in PHASE_ORDER:
        raise SystemExit(f"--phase must be one of {PHASE_ORDER}")

    save_dir: Optional[str] = args.save_plots
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    demos, obs_map, meta = load_demos(args.demos, rebuild_env=not args.no_rebuild_env)
    print(f"Loaded {meta['num_demos']} demos "
          f"({meta['num_successes']} success, {meta['num_failures']} fail) "
          f"from {args.demos}")

    analyze_phase_durations(demos, save_dir, args.phase)
    analyze_failures(demos, save_dir, args.phase)
    analyze_action_stats(demos, save_dir, args.phase)
    analyze_rewards(demos, save_dir, args.phase)
    analyze_initial_conditions(demos, obs_map, save_dir, args.phase)


if __name__ == "__main__":
    main()
