"""
Collect expert demonstrations for behavior cloning.

Runs the scripted PickPlaceExpertPolicy, stores only successful episodes in HDF5.
Per-step phase labels and success flags are recorded so downstream analysis
(e.g. phase-conditional BC, error attribution) can run without re-rollout.

HDF5 layout:
    data/
        demo_0/
            obs        (T, obs_dim)   float32
            actions    (T, 7)         float32
            rewards    (T,)           float32
            dones      (T,)           bool
            next_obs   (T, obs_dim)   float32
            phase      (T,)           int8       0..7  (Phase IntEnum)
            active_stick (T,)         int8       active stick index
            is_success (T,)           bool       env._check_success per step
            attrs:
                seed        int
                stick_order int8[]    expert order for this episode
        demo_1/ ...
    attrs:
        num_demos        int    — number of successful episodes
        obs_dim          int    — observation dimensionality
        total_samples    int    — total transitions across all demos
        env_config       str    — JSON of env kwargs
        env_config_hash  str    — SHA256-12 of env+robosuite version (reproducibility)
        robosuite_version str   — robosuite package version used
        oracle_version   str    — version tag of the expert policy

Usage:
    # Collect 200 successful demos
    python scripts/collect_demos.py --num-demos 200 --output data/demos.hdf5

    # Smoke test: 50 rollouts, no save, fail if success rate < 95%
    python scripts/collect_demos.py --smoke --seed 42

    # With live rendering (debug, slow)
    python scripts/collect_demos.py --num-demos 10 --output /tmp/test.hdf5 --render
"""

import argparse
import hashlib
import json
import os
import sys

import h5py
import numpy as np
import robosuite
import yaml
from robosuite.wrappers import GymWrapper

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map
from wire_untangling.policies.pick_place_expert import Phase
from wire_untangling.utils.seeding import demo_seed, resolve_seed, seed_env
from wire_untangling.utils.stick_order import StickOrderScheduler


ORACLE_VERSION_N1 = "1.1-n1-orientation"
ORACLE_VERSION_N2_SIDE = "1.2-n2-side-goals"
ORACLE_VERSION_N2_RANDOM_ORDER = "1.3-n2-random-order"
ORACLE_VERSION_N2_PAIRED_ORDER = "1.4-n2-paired-order"

DEMO_DATA_KEYS = (
    "obs",
    "actions",
    "rewards",
    "dones",
    "next_obs",
    "phase",
    "active_stick",
    "is_success",
)


def oracle_version_for(env_cfg: dict, expert_cfg: dict | None = None) -> str:
    if env_cfg.get("placement_mode") == "two_stick_side":
        order_mode = (expert_cfg or {}).get("order_mode")
        if order_mode == "paired_balanced":
            return ORACLE_VERSION_N2_PAIRED_ORDER
        if order_mode == "balanced":
            return ORACLE_VERSION_N2_RANDOM_ORDER
        return ORACLE_VERSION_N2_SIDE
    return ORACLE_VERSION_N1


def env_config_hash(env_kwargs: dict, robosuite_version: str) -> str:
    """SHA256-12 over the env config + robosuite version (controller is the
    robosuite default OSC_POSE for this codebase, so it is implicit)."""
    payload = {
        "env_kwargs": env_kwargs,
        "robosuite_version": robosuite_version,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def make_env(env_cfg: dict, render: bool = False):
    kw = dict(
        robots=env_cfg.get("robot", "Panda"),
        num_sticks=env_cfg.get("num_sticks", 1),
        stick_length=env_cfg.get("stick_length", 0.20),
        stick_radius=env_cfg.get("stick_radius", 0.0075),
        goal_spacing=env_cfg.get("goal_spacing", 0.06),
        success_threshold=env_cfg.get("success_threshold", 0.03),
        orientation_threshold=env_cfg.get("orientation_threshold", np.deg2rad(10.0)),
        lambda_rot=env_cfg.get("lambda_rot", 0.1),
        goal_yaw=env_cfg.get("goal_yaw", 0.0),
        reward_shaping=env_cfg.get("reward_shaping", True),
        # When False, episodes continue past success — useful for collecting
        # full trajectories that include RELEASE/RETREAT phases. The
        # success_hold_steps mechanism in run_episode still caps total length
        # so demos don't run the full horizon idling.
        terminate_on_success=env_cfg.get("terminate_on_success", True),
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=env_cfg.get("horizon", 500),
    )
    optional_env_keys = (
        "placement_mode",
        "init_x_range",
        "init_y_range",
        "side_init_x_range",
        "side_init_y_ranges",
        "side_init_yaw_range",
        "side_goal_x",
        "side_goal_y_ranges",
        "stick_color_indices",
    )
    for key in optional_env_keys:
        if key in env_cfg:
            kw[key] = env_cfg[key]
    return StickReorderEnv(**kw)


def run_episode(
    gym_env,
    expert,
    render: bool = False,
    seed: int | None = None,
    success_hold_steps: int = 5,
    stick_order=None,
):
    """Run one expert rollout. Returns per-step lists and final success.

    If ``seed`` is provided, the env's RNG is rebound before reset so the
    initial stick placement is deterministic from the seed.
    """
    if seed is not None:
        seed_env(gym_env.env, seed)
    obs, _ = gym_env.reset()
    expert.reset(stick_order=stick_order)
    episode_order = tuple(expert.stick_order)

    ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []
    ep_next_obs, ep_phase, ep_active_stick, ep_success = [], [], [], []
    done = False
    info: dict = {}
    consecutive_success = 0

    while not done:
        # Capture phase BEFORE the action — labels the action with the phase
        # that produced it, matching the convention of action-time labels.
        phase_before = int(expert.phase)
        active_stick_before = int(expert.active_stick)
        action, _ = expert.predict(obs)
        next_obs, reward, terminated, truncated, info = gym_env.step(action)
        step_success = bool(info.get("is_success", False))
        if step_success:
            consecutive_success += 1
        else:
            consecutive_success = 0
        done = terminated or truncated or consecutive_success >= success_hold_steps

        ep_obs.append(obs)
        ep_actions.append(action)
        ep_rewards.append(reward)
        ep_dones.append(done)
        ep_next_obs.append(next_obs)
        ep_phase.append(phase_before)
        ep_active_stick.append(active_stick_before)
        ep_success.append(step_success)

        obs = next_obs

        if render:
            gym_env.env.render()

    final_success = bool(info.get("is_success", False))
    return {
        "obs": np.array(ep_obs, dtype=np.float32),
        "actions": np.array(ep_actions, dtype=np.float32),
        "rewards": np.array(ep_rewards, dtype=np.float32),
        "dones": np.array(ep_dones, dtype=bool),
        "next_obs": np.array(ep_next_obs, dtype=np.float32),
        "phase": np.array(ep_phase, dtype=np.int8),
        "active_stick": np.array(ep_active_stick, dtype=np.int8),
        "is_success": np.array(ep_success, dtype=bool),
        "stick_order": np.array(episode_order, dtype=np.int8),
        "final_success": final_success,
        "final_phase": int(expert.phase),
        "final_active_stick": int(expert.active_stick),
    }


def smoke_test(
    env_cfg: dict,
    expert_cfg: dict,
    n_rollouts: int,
    threshold: float,
    top_seed: int,
):
    """Run n_rollouts headless and report success rate. Exit 1 if below threshold.

    Each rollout uses ``demo_seed(top_seed, i)`` so the smoke test is itself
    bytewise reproducible.
    """
    raw_env = make_env(env_cfg, render=False)
    gym_env = GymWrapper(raw_env)
    obs_map = build_obs_index_map(gym_env)
    order_schedule = StickOrderScheduler(expert_cfg, raw_env.num_sticks)
    expert = PickPlaceExpertPolicy(
        obs_map,
        goal_yaw=env_cfg.get("goal_yaw", 0.0),
        stick_order=expert_cfg.get("stick_order"),
    )

    successes = 0
    successes_by_order: dict[tuple[int, ...], int] = {}
    attempts_by_order: dict[tuple[int, ...], int] = {}
    failure_phases: dict[int, int] = {}

    if order_schedule.uses_paired_seeds:
        order_schedule.require_exact_balance(n_rollouts)
        rollout_specs = []
        n_pairs = n_rollouts // len(order_schedule.order_choices)
        for pair_id in range(n_pairs):
            seed = demo_seed(top_seed, pair_id)
            for branch, stick_order in enumerate(order_schedule.order_choices):
                rollout_specs.append((pair_id, branch, seed, stick_order))
    else:
        rollout_specs = [
            (None, order_schedule.branch_for(i), demo_seed(top_seed, i), order_schedule.order_for(i))
            for i in range(n_rollouts)
        ]

    for i, (pair_id, branch, seed, stick_order) in enumerate(rollout_specs):
        attempts_by_order[stick_order] = attempts_by_order.get(stick_order, 0) + 1
        ep = run_episode(gym_env, expert, seed=seed, stick_order=stick_order)
        if ep["final_success"]:
            successes += 1
            successes_by_order[stick_order] = successes_by_order.get(stick_order, 0) + 1
        else:
            failure_phases[ep["final_phase"]] = failure_phases.get(ep["final_phase"], 0) + 1
        pair_text = "" if pair_id is None else f" pair={pair_id} branch={branch}"
        print(f"  rollout {i + 1}/{n_rollouts}: seed={seed} "
              f"{pair_text} "
              f"order={StickOrderScheduler.format_order(stick_order)} "
              f"success={ep['final_success']} "
              f"final_active_stick={ep['final_active_stick']} "
              f"final_phase={Phase(ep['final_phase']).name}")

    gym_env.close()
    rate = successes / n_rollouts
    print(f"\nSmoke test: {successes}/{n_rollouts} succeeded ({rate:.1%})")
    print("Per-order success:")
    for order in order_schedule.order_choices:
        order_successes = successes_by_order.get(order, 0)
        order_attempts = attempts_by_order.get(order, 0)
        order_rate = order_successes / order_attempts if order_attempts else 0.0
        print(
            f"  {StickOrderScheduler.format_order(order)}: "
            f"{order_successes}/{order_attempts} ({order_rate:.1%})"
        )
    if failure_phases:
        print("Per-phase failure counts (phase at terminal step):")
        for p, c in sorted(failure_phases.items()):
            print(f"  {Phase(p).name}: {c}")

    if rate < threshold:
        print(f"FAIL: success rate {rate:.1%} < {threshold:.0%} threshold")
        sys.exit(1)
    print("PASS")


def _write_demo_group(grp, seed: int, demo: dict) -> None:
    for key in DEMO_DATA_KEYS:
        grp.create_dataset(key, data=demo[key], compression="gzip")
    grp.attrs["seed"] = int(seed)
    grp.attrs["stick_order"] = np.asarray(demo["stick_order"], dtype=np.int8)
    if "multimodal_pair_id" in demo:
        grp.attrs["multimodal_pair_id"] = int(demo["multimodal_pair_id"])
    if "multimodal_branch" in demo:
        grp.attrs["multimodal_branch"] = int(demo["multimodal_branch"])
    if "multimodal_pair_seed" in demo:
        grp.attrs["multimodal_pair_seed"] = int(demo["multimodal_pair_seed"])
    if "multimodal_pair_attempt" in demo:
        grp.attrs["multimodal_pair_attempt"] = int(demo["multimodal_pair_attempt"])


def collect(
    config: dict,
    num_demos: int,
    output_path: str,
    top_seed: int,
    render: bool = False,
    max_attempts_factor: int = 3,
    success_hold_steps: int = 5,
    save_failures: bool = False,
):
    env_cfg = dict(config.get("env", {}))
    expert_cfg = dict(config.get("expert", {}))
    # Demo collection uses its own consecutive-success hold so labels include
    # a short stable terminal segment instead of ending on the first success.
    env_cfg["terminate_on_success"] = False

    raw_env = make_env(env_cfg, render=render)
    gym_env = GymWrapper(raw_env)
    obs_map = build_obs_index_map(gym_env)
    order_schedule = StickOrderScheduler(expert_cfg, raw_env.num_sticks)
    order_schedule.require_exact_balance(num_demos)
    expert = PickPlaceExpertPolicy(
        obs_map,
        goal_yaw=env_cfg.get("goal_yaw", 0.0),
        # TODO(alexta): in the main the line was: stick_order=expert_cfg.get("stick_order"),
        # Check if everything is correct.
        stick_order=order_schedule.order_for(0),
    )
    obs_dim = gym_env.observation_space.shape[0]

    cfg_hash = env_config_hash(env_cfg, robosuite.__version__)
    oracle_version = oracle_version_for(env_cfg, expert_cfg)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # successful_demos holds (seed, ep_dict). For standard collection, we use
    # the attempt index as the reset seed. For paired collection, each pair
    # attempt uses one reset seed for every order branch, and the pair is
    # accepted only when all branches succeed.
    successful_demos: list[tuple[int, dict]] = []
    failed_demos: list[tuple[int, dict]] = []
    attempts = 0
    max_attempts = num_demos * max_attempts_factor

    if order_schedule.uses_paired_seeds:
        target_pairs = num_demos // len(order_schedule.order_choices)
        successful_pairs = 0
        while successful_pairs < target_pairs and attempts < max_attempts:
            seed = demo_seed(top_seed, attempts)
            pair_demos: list[tuple[int, int, tuple[int, ...], dict]] = []
            attempts += 1

            for branch, stick_order in enumerate(order_schedule.order_choices):
                ep = run_episode(
                    gym_env,
                    expert,
                    render=render,
                    seed=seed,
                    success_hold_steps=success_hold_steps,
                    stick_order=stick_order,
                )
                ep["multimodal_branch"] = branch
                ep["multimodal_pair_seed"] = seed
                ep["multimodal_pair_attempt"] = attempts - 1
                pair_demos.append((seed, branch, stick_order, ep))

            if all(ep["final_success"] for _, _, _, ep in pair_demos):
                for seed_i, branch, stick_order, ep in pair_demos:
                    ep["multimodal_pair_id"] = successful_pairs
                    successful_demos.append((seed_i, ep))
                    print(
                        f"  Demo {len(successful_demos)}/{num_demos} collected "
                        f"(pair {successful_pairs}, branch {branch}, "
                        f"attempt {attempts}, seed={seed_i}, "
                        f"order={StickOrderScheduler.format_order(stick_order)}, "
                        f"{len(ep['obs'])} steps)"
                    )
                successful_pairs += 1
            else:
                if save_failures:
                    failed_demos.extend((seed_i, ep) for seed_i, _, _, ep in pair_demos)
                failed = [
                    (
                        branch,
                        StickOrderScheduler.format_order(stick_order),
                        Phase(ep["final_phase"]).name,
                    )
                    for _, branch, stick_order, ep in pair_demos
                    if not ep["final_success"]
                ]
                print(
                    f"  Pair attempt {attempts} failed "
                    f"(seed={seed}, failed_branches={failed}) — skipping entire pair"
                )
    else:
        while len(successful_demos) < num_demos and attempts < max_attempts:
            seed = demo_seed(top_seed, attempts)
            stick_order = order_schedule.order_for(len(successful_demos))
            ep = run_episode(gym_env, expert, render=render, seed=seed,
                             success_hold_steps=success_hold_steps,
                             stick_order=stick_order)
            attempts += 1

            if ep["final_success"]:
                successful_demos.append((seed, ep))
                print(f"  Demo {len(successful_demos)}/{num_demos} collected "
                      f"(attempt {attempts}, seed={seed}, "
                      f"order={StickOrderScheduler.format_order(stick_order)}, "
                      f"{len(ep['obs'])} steps)")
            else:
                if save_failures:
                    failed_demos.append((seed, ep))
                print(f"  Attempt {attempts} failed "
                      f"(seed={seed}, order={StickOrderScheduler.format_order(stick_order)}, "
                      f"final_active_stick={ep['final_active_stick']}, "
                      f"final_phase={Phase(ep['final_phase']).name}, "
                      f"{len(ep['obs'])} steps) — skipping")

    gym_env.close()

    if not successful_demos:
        print("No successful demos collected!")
        return
    if len(successful_demos) < num_demos and order_schedule.mode in ("balanced", "paired_balanced"):
        print(
            f"Only collected {len(successful_demos)}/{num_demos} successful demos; "
            f"not writing an incomplete {order_schedule.mode} dataset."
        )
        return

    total_samples = sum(d["obs"].shape[0] for _, d in successful_demos)
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        for i, (seed, demo) in enumerate(successful_demos):
            grp = data_grp.create_group(f"demo_{i}")
            _write_demo_group(grp, seed, demo)

        if save_failures and failed_demos:
            failures_grp = f.create_group("failures")
            for i, (seed, demo) in enumerate(failed_demos):
                grp = failures_grp.create_group(f"failure_{i}")
                _write_demo_group(grp, seed, demo)
                grp.attrs["final_phase"] = int(demo["final_phase"])
                grp.attrs["final_active_stick"] = int(demo["final_active_stick"])
                grp.attrs["final_success"] = bool(demo["final_success"])

        f.attrs["num_demos"] = len(successful_demos)
        f.attrs["num_failures"] = len(failed_demos)
        f.attrs["obs_dim"] = obs_dim
        f.attrs["total_samples"] = total_samples
        f.attrs["env_config"] = json.dumps(env_cfg)
        f.attrs["expert_config"] = json.dumps(expert_cfg)
        f.attrs["env_config_hash"] = cfg_hash
        f.attrs["robosuite_version"] = robosuite.__version__
        f.attrs["oracle_version"] = oracle_version
        f.attrs["top_seed"] = int(top_seed)
        f.attrs["order_mode"] = order_schedule.mode
        if order_schedule.uses_paired_seeds:
            f.attrs["multimodal_collection"] = "paired_order"
            f.attrs["num_pairs"] = len(successful_demos) // len(order_schedule.order_choices)
            f.attrs["paired_order_choices"] = json.dumps([list(o) for o in order_schedule.order_choices])
        else:
            f.attrs["multimodal_collection"] = "none"
        # Save the observable→slice mapping so analysis tools can locate
        # named obs fields without re-instantiating the env. Slices serialise
        # as [start, stop] pairs.
        f.attrs["obs_index_map"] = json.dumps(
            {name: [s.start, s.stop] for name, s in obs_map.items()}
        )

    print(f"\nSaved {len(successful_demos)} demos ({total_samples} total transitions) "
          f"to {output_path}")
    print(f"  env_config_hash={cfg_hash}  oracle_version={oracle_version}  top_seed={top_seed}")
    print("  order_counts:")
    for order in order_schedule.order_choices:
        count = sum(tuple(d["stick_order"].tolist()) == order for _, d in successful_demos)
        print(f"    {StickOrderScheduler.format_order(order)}: {count}")
    if order_schedule.uses_paired_seeds:
        print(f"  paired_order_pairs={len(successful_demos) // len(order_schedule.order_choices)}")
    accepted_attempts = (
        len(successful_demos) // len(order_schedule.order_choices)
        if order_schedule.uses_paired_seeds
        else len(successful_demos)
    )
    failed_attempts = attempts - accepted_attempts
    if failed_attempts > 0:
        if order_schedule.uses_paired_seeds:
            print(f"  ({failed_attempts} failed pair attempts discarded)")
        else:
            print(f"  ({failed_attempts} failed attempts discarded)")
    if save_failures:
        print(f"  saved_failures={len(failed_demos)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stick_reorder.yaml")
    parser.add_argument("--num-demos", type=int, default=200)
    parser.add_argument("--output", default="data/demos.hdf5")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Run N headless rollouts, report success rate, no save.")
    parser.add_argument("--smoke-n", type=int, default=50)
    parser.add_argument("--smoke-threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None,
                        help="Top-level seed. Attempt i uses seed * 1_000_003 + i. "
                             "Random if not specified.")
    parser.add_argument("--num-sticks", type=int, default=None,
                        help="Override env.num_sticks from config.")
    parser.add_argument("--save-failures", action="store_true",
                        help="Store failed attempts under /failures for diagnostics. "
                             "BC training still reads only /data.")
    parser.add_argument("--no-terminate-on-success", action="store_true",
                        help="Let episodes continue past success so demos cover "
                             "the full RELEASE phase. Auto-bumps "
                             "--success-hold-steps to 15 unless overridden.")
    parser.add_argument("--success-hold-steps", type=int, default=None,
                        help="End an episode after this many consecutive success "
                             "steps. Default: 5 (terminate quickly), or 15 when "
                             "--no-terminate-on-success is set — just past the "
                             "10-step RELEASE phase, before idle RETREAT.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides on env_cfg
    if args.num_sticks is not None:
        config.setdefault("env", {})["num_sticks"] = args.num_sticks
    if args.no_terminate_on_success:
        config.setdefault("env", {})["terminate_on_success"] = False

    success_hold_steps = args.success_hold_steps
    if success_hold_steps is None:
        # 15 is just past the 10-step RELEASE phase, so demos cover the full
        # gripper-opening segment plus a few tail steps without entering the
        # state-independent RETREAT idle (which dilutes per-phase statistics).
        success_hold_steps = 15 if args.no_terminate_on_success else 5

    args.seed = resolve_seed(args.seed)

    if args.smoke:
        env_cfg = dict(config.get("env", {}))
        expert_cfg = dict(config.get("expert", {}))
        smoke_test(
            env_cfg,
            expert_cfg,
            n_rollouts=args.smoke_n,
            threshold=args.smoke_threshold,
            top_seed=args.seed,
        )
    else:
        collect(
            config,
            num_demos=args.num_demos,
            output_path=args.output,
            top_seed=args.seed,
            render=args.render,
            success_hold_steps=success_hold_steps,
            save_failures=args.save_failures,
        )
