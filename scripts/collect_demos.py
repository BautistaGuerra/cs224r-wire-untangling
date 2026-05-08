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
            is_success (T,)           bool       env._check_success per step
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
import yaml


ORACLE_VERSION = "1.1-n1-orientation"


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
    from wire_untangling.envs import StickReorderEnv

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
        terminate_on_success=env_cfg.get("terminate_on_success", True),
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=env_cfg.get("horizon", 500),
    )
    if "init_x_range" in env_cfg:
        kw["init_x_range"] = tuple(env_cfg["init_x_range"])
    if "init_y_range" in env_cfg:
        kw["init_y_range"] = tuple(env_cfg["init_y_range"])
    return StickReorderEnv(**kw)


def run_episode(
    gym_env,
    expert,
    render: bool = False,
    seed: int | None = None,
    success_hold_steps: int = 5,
):
    """Run one expert rollout. Returns per-step lists and final success.

    If ``seed`` is provided, the env's RNG is rebound before reset so the
    initial stick placement is deterministic from the seed.
    """
    if seed is not None:
        from wire_untangling.utils.seeding import seed_env
        seed_env(gym_env.env, seed)
    obs, _ = gym_env.reset()
    expert.reset()

    ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []
    ep_next_obs, ep_phase, ep_success = [], [], []
    done = False
    info: dict = {}
    consecutive_success = 0

    while not done:
        # Capture phase BEFORE the action — labels the action with the phase
        # that produced it, matching the convention of action-time labels.
        phase_before = int(expert.phase)
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
        "is_success": np.array(ep_success, dtype=bool),
        "final_success": final_success,
        "final_phase": int(expert.phase),
    }


def smoke_test(env_cfg: dict, n_rollouts: int, threshold: float, top_seed: int):
    """Run n_rollouts headless and report success rate. Exit 1 if below threshold.

    Each rollout uses ``demo_seed(top_seed, i)`` so the smoke test is itself
    bytewise reproducible.
    """
    from robosuite.wrappers import GymWrapper

    from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map
    from wire_untangling.policies.pick_place_expert import Phase
    from wire_untangling.utils.seeding import demo_seed

    raw_env = make_env(env_cfg, render=False)
    gym_env = GymWrapper(raw_env)
    obs_map = build_obs_index_map(gym_env)
    expert = PickPlaceExpertPolicy(obs_map, goal_yaw=env_cfg.get("goal_yaw", 0.0))

    successes = 0
    failure_phases: dict[int, int] = {}

    for i in range(n_rollouts):
        seed = demo_seed(top_seed, i)
        ep = run_episode(gym_env, expert, seed=seed)
        if ep["final_success"]:
            successes += 1
        else:
            failure_phases[ep["final_phase"]] = failure_phases.get(ep["final_phase"], 0) + 1
        print(f"  rollout {i + 1}/{n_rollouts}: seed={seed} "
              f"success={ep['final_success']} "
              f"final_phase={Phase(ep['final_phase']).name}")

    gym_env.close()
    rate = successes / n_rollouts
    print(f"\nSmoke test: {successes}/{n_rollouts} succeeded ({rate:.1%})")
    if failure_phases:
        print("Per-phase failure counts (phase at terminal step):")
        for p, c in sorted(failure_phases.items()):
            print(f"  {Phase(p).name}: {c}")

    if rate < threshold:
        print(f"FAIL: success rate {rate:.1%} < {threshold:.0%} threshold")
        sys.exit(1)
    print("PASS")


def collect(
    config: dict,
    num_demos: int,
    output_path: str,
    top_seed: int,
    render: bool = False,
    max_attempts_factor: int = 3,
):
    import robosuite
    from robosuite.wrappers import GymWrapper

    from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map
    from wire_untangling.utils.seeding import demo_seed

    env_cfg = dict(config.get("env", {}))
    env_cfg["num_sticks"] = 1  # single-stick BC for now
    # Demo collection uses its own consecutive-success hold so labels include
    # a short stable terminal segment instead of ending on the first success.
    env_cfg["terminate_on_success"] = False

    raw_env = make_env(env_cfg, render=render)
    gym_env = GymWrapper(raw_env)
    obs_map = build_obs_index_map(gym_env)
    expert = PickPlaceExpertPolicy(obs_map, goal_yaw=env_cfg.get("goal_yaw", 0.0))
    obs_dim = gym_env.observation_space.shape[0]

    cfg_hash = env_config_hash(env_cfg, robosuite.__version__)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # successful_demos holds (seed, ep_dict). We use the attempt index, not the
    # success index, so re-running collection with the same top_seed retries
    # the same failed seeds in the same order — bytewise reproducible.
    successful_demos: list[tuple[int, dict]] = []
    attempts = 0
    max_attempts = num_demos * max_attempts_factor

    while len(successful_demos) < num_demos and attempts < max_attempts:
        seed = demo_seed(top_seed, attempts)
        ep = run_episode(gym_env, expert, render=render, seed=seed)
        attempts += 1

        if ep["final_success"]:
            successful_demos.append((seed, ep))
            print(f"  Demo {len(successful_demos)}/{num_demos} collected "
                  f"(attempt {attempts}, seed={seed}, {len(ep['obs'])} steps)")
        else:
            from wire_untangling.policies.pick_place_expert import Phase
            print(f"  Attempt {attempts} failed "
                  f"(seed={seed}, final_phase={Phase(ep['final_phase']).name}, "
                  f"{len(ep['obs'])} steps) — skipping")

    gym_env.close()

    if not successful_demos:
        print("No successful demos collected!")
        return

    total_samples = sum(d["obs"].shape[0] for _, d in successful_demos)
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        for i, (seed, demo) in enumerate(successful_demos):
            grp = data_grp.create_group(f"demo_{i}")
            for key in ("obs", "actions", "rewards", "dones",
                        "next_obs", "phase", "is_success"):
                grp.create_dataset(key, data=demo[key], compression="gzip")
            grp.attrs["seed"] = int(seed)

        f.attrs["num_demos"] = len(successful_demos)
        f.attrs["obs_dim"] = obs_dim
        f.attrs["total_samples"] = total_samples
        f.attrs["env_config"] = json.dumps(env_cfg)
        f.attrs["env_config_hash"] = cfg_hash
        f.attrs["robosuite_version"] = robosuite.__version__
        f.attrs["oracle_version"] = ORACLE_VERSION
        f.attrs["top_seed"] = int(top_seed)

    print(f"\nSaved {len(successful_demos)} demos ({total_samples} total transitions) "
          f"to {output_path}")
    print(f"  env_config_hash={cfg_hash}  oracle_version={ORACLE_VERSION}  top_seed={top_seed}")
    if attempts > len(successful_demos):
        print(f"  ({attempts - len(successful_demos)} failed attempts discarded)")


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
    parser.add_argument("--seed", type=int, default=42,
                        help="Top-level seed. Attempt i uses seed * 1_000_003 + i.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke:
        env_cfg = dict(config.get("env", {}))
        env_cfg["num_sticks"] = 1
        smoke_test(
            env_cfg,
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
        )
