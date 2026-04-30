"""
Collect expert demonstrations for behavior cloning.

Runs the scripted PickPlaceExpertPolicy, stores only successful episodes in HDF5.

HDF5 layout:
    data/
        demo_0/
            obs        (T, obs_dim)   float32
            actions    (T, 7)         float32
            rewards    (T,)           float32
            dones      (T,)           bool
            next_obs   (T, obs_dim)   float32
        demo_1/ ...
    attrs: num_demos, env_config, obs_dim, total_samples

Usage:
    python scripts/collect_demos.py --num-demos 200 --output data/demos.hdf5
    python scripts/collect_demos.py --num-demos 10 --output /tmp/test.hdf5 --render
"""

import argparse
import json
import os

import h5py
import numpy as np
import yaml


def collect(
    config: dict,
    num_demos: int,
    output_path: str,
    render: bool = False,
    max_attempts_factor: int = 3,
):
    from robosuite.wrappers import GymWrapper

    from wire_untangling.envs import StickReorderEnv
    from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map

    env_cfg = config.get("env", {})
    env_cfg["num_sticks"] = 1

    raw_env = StickReorderEnv(
        robots=env_cfg.get("robot", "Panda"),
        num_sticks=1,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=500,
    )
    gym_env = GymWrapper(raw_env)

    obs_map = build_obs_index_map(gym_env)
    expert = PickPlaceExpertPolicy(obs_map)
    obs_dim = gym_env.observation_space.shape[0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    successful_demos = []
    attempts = 0
    max_attempts = num_demos * max_attempts_factor

    while len(successful_demos) < num_demos and attempts < max_attempts:
        attempts += 1
        obs, _ = gym_env.reset()
        expert.reset()

        ep_obs, ep_actions, ep_rewards, ep_dones, ep_next_obs = [], [], [], [], []
        done = False

        while not done:
            action, _ = expert.predict(obs)
            next_obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated

            ep_obs.append(obs)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            ep_next_obs.append(next_obs)

            obs = next_obs

            if render:
                gym_env.env.render()

        success = info.get("is_success", False)
        if success:
            successful_demos.append({
                "obs": np.array(ep_obs, dtype=np.float32),
                "actions": np.array(ep_actions, dtype=np.float32),
                "rewards": np.array(ep_rewards, dtype=np.float32),
                "dones": np.array(ep_dones, dtype=bool),
                "next_obs": np.array(ep_next_obs, dtype=np.float32),
            })
            print(f"  Demo {len(successful_demos)}/{num_demos} collected "
                  f"(attempt {attempts}, {len(ep_obs)} steps)")
        else:
            print(f"  Attempt {attempts} failed (phase={expert._phase.name}, "
                  f"{len(ep_obs)} steps) — skipping")

    gym_env.close()

    if not successful_demos:
        print("No successful demos collected!")
        return

    # Write HDF5
    total_samples = sum(d["obs"].shape[0] for d in successful_demos)
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        for i, demo in enumerate(successful_demos):
            grp = data_grp.create_group(f"demo_{i}")
            for key, arr in demo.items():
                grp.create_dataset(key, data=arr, compression="gzip")

        f.attrs["num_demos"] = len(successful_demos)
        f.attrs["obs_dim"] = obs_dim
        f.attrs["total_samples"] = total_samples
        f.attrs["env_config"] = json.dumps(env_cfg)

    print(f"\nSaved {len(successful_demos)} demos ({total_samples} total transitions) "
          f"to {output_path}")
    if attempts > len(successful_demos):
        print(f"  ({attempts - len(successful_demos)} failed attempts discarded)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stick_reorder.yaml")
    parser.add_argument("--num-demos", type=int, default=200)
    parser.add_argument("--output", default="data/demos.hdf5")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    collect(config, num_demos=args.num_demos, output_path=args.output, render=args.render)
