"""Demo-seeded SAC baseline for the stick-reorder task.

This trains a standard full-action Stable-Baselines3 SAC policy. It does not
call DPFM during training or inference. When demo seeding is enabled, expert
HDF5 transitions are inserted into the SAC replay buffer before online learning
starts.

TODO(alexta): this is a very simplistic baseline SAC method that does not really used and is
only for establishing a baselione. Consider removing in the future versions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from gymnasium import spaces
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from wire_untangling.envs import StickReorderEnv
from wire_untangling.utils.seeding import resolve_seed, resolve_device


def load_config(
    env_config: str = "configs/stick_reorder.yaml",
    sac_config: str = "configs/sac_baseline.yaml",
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for path in (env_config, sac_config):
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)
    return cfg


def make_gym_env(env_cfg: dict[str, Any]):
    """Create the same state-based robosuite env surface used by playback."""
    kwargs = dict(
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
        success_bonus=env_cfg.get("success_bonus", 1.0),
        terminate_on_success=env_cfg.get("terminate_on_success", True),
        has_renderer=False,
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
            kwargs[key] = env_cfg[key]
    return GymWrapper(StickReorderEnv(**kwargs))


def _as_comparable(value):
    if isinstance(value, np.ndarray):
        return _as_comparable(value.tolist())
    if isinstance(value, tuple):
        return [_as_comparable(v) for v in value]
    if isinstance(value, list):
        return [_as_comparable(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_comparable(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _warn_demo_config_mismatch(demos_path: str, env_cfg: dict[str, Any]) -> None:
    import h5py

    with h5py.File(demos_path, "r") as f:
        raw_demo_env = f.attrs.get("env_config")
        demo_hash = f.attrs.get("env_config_hash", "<missing>")
        top_seed = f.attrs.get("top_seed", "<missing>")
        oracle_version = f.attrs.get("oracle_version", "<missing>")

    print(
        "Demo provenance: "
        f"env_config_hash={demo_hash}, top_seed={top_seed}, oracle_version={oracle_version}"
    )
    if raw_demo_env is None:
        print("WARNING: demo file has no env_config attr; cannot compare provenance.")
        return

    try:
        demo_env = json.loads(raw_demo_env)
    except json.JSONDecodeError:
        print("WARNING: demo env_config attr is not valid JSON; cannot compare provenance.")
        return

    mismatches = []
    for key, demo_value in demo_env.items():
        if key not in env_cfg:
            continue
        train_value = env_cfg[key]
        if _as_comparable(demo_value) != _as_comparable(train_value):
            mismatches.append((key, demo_value, train_value))

    if not mismatches:
        return

    print("WARNING: demo env_config differs from the training env config:")
    for key, demo_value, train_value in mismatches:
        print(f"  {key}: demo={demo_value!r} train={train_value!r}")


def _episode_sort_key(key: str):
    try:
        return int(key.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return key


def _check_demo_shapes(model: SAC, obs: np.ndarray, actions: np.ndarray, demos_path: str) -> None:
    obs_space = model.observation_space
    action_space = model.action_space
    if not isinstance(obs_space, spaces.Box) or not isinstance(action_space, spaces.Box):
        raise TypeError("SAC demo seeding expects Box observation and action spaces.")
    if tuple(obs.shape[1:]) != tuple(obs_space.shape):
        raise ValueError(
            f"{demos_path}: obs shape {obs.shape[1:]} does not match env "
            f"observation shape {obs_space.shape}"
        )
    if tuple(actions.shape[1:]) != tuple(action_space.shape):
        raise ValueError(
            f"{demos_path}: action shape {actions.shape[1:]} does not match env "
            f"action shape {action_space.shape}"
        )


def seed_sac_replay_buffer(
    model: SAC,
    demos_path: str,
    env_cfg: dict[str, Any] | None = None,
    max_transitions: int | None = None,
    warn_on_config_mismatch: bool = True,
) -> int:
    """Insert expert HDF5 transitions into an SB3 SAC replay buffer."""
    import h5py

    if model.replay_buffer is None:
        raise RuntimeError("SAC model has no replay buffer; call with an initialized off-policy model.")
    if warn_on_config_mismatch and env_cfg is not None:
        _warn_demo_config_mismatch(demos_path, env_cfg)

    transitions = 0
    with h5py.File(demos_path, "r") as f:
        data_grp = f["data"]
        for ep_key in sorted(data_grp.keys(), key=_episode_sort_key):
            demo = data_grp[ep_key]
            obs_all = np.asarray(demo["obs"][:], dtype=np.float32)
            actions_all = np.asarray(demo["actions"][:], dtype=np.float32)
            rewards_all = np.asarray(demo["rewards"][:], dtype=np.float32)
            dones_all = np.asarray(demo["dones"][:], dtype=bool)
            if "next_obs" in demo:
                next_obs_all = np.asarray(demo["next_obs"][:], dtype=np.float32)
            else:
                next_obs_all = np.concatenate(
                    [obs_all[1:], np.zeros_like(obs_all[:1])],
                    axis=0,
                )

            _check_demo_shapes(model, obs_all, actions_all, demos_path)
            for t in range(obs_all.shape[0]):
                if max_transitions is not None and transitions >= max_transitions:
                    return transitions
                info = {"TimeLimit.truncated": False}
                if "is_success" in demo:
                    info["is_success"] = bool(demo["is_success"][t])
                model.replay_buffer.add(
                    obs_all[t][None, ...],
                    next_obs_all[t][None, ...],
                    actions_all[t][None, ...],
                    np.asarray([rewards_all[t]], dtype=np.float32),
                    np.asarray([dones_all[t]], dtype=bool),
                    [info],
                )
                transitions += 1

    return transitions


def _policy_kwargs(sac_cfg: dict[str, Any]) -> dict[str, Any]:
    hidden_dims = sac_cfg.get("policy_hidden_dims", [256, 256])
    return {"net_arch": [int(h) for h in hidden_dims]}


def train(
    config: dict[str, Any],
    seed: int | None = None,
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints/sac_demo_seeded_n1",
    device: str | None = None,
) -> None:
    env_cfg = dict(config.get("env", {}))
    sac_cfg = dict(config.get("sac_baseline", {}))
    demo_cfg = dict(sac_cfg.get("demo_seed", {}))
    if device is not None:
        sac_cfg["device"] = device
    seed = resolve_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "periodic"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "logs"), exist_ok=True)

    train_env = Monitor(make_gym_env(env_cfg))
    eval_env = Monitor(make_gym_env(env_cfg))

    demo_seed_enabled = bool(demo_cfg.get("enabled", True))
    learning_starts = int(
        sac_cfg.get(
            "learning_starts" if demo_seed_enabled else "no_demo_seed_learning_starts",
            0 if demo_seed_enabled else 10000,
        )
    )

    run = None
    tensorboard_log = None
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(checkpoint_dir, "best"),
            log_path=os.path.join(checkpoint_dir, "logs"),
            eval_freq=int(sac_cfg.get("eval_freq", 10000)),
            n_eval_episodes=int(sac_cfg.get("n_eval_episodes", 10)),
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=int(sac_cfg.get("checkpoint_freq", 50000)),
            save_path=os.path.join(checkpoint_dir, "periodic"),
            name_prefix="sac",
        ),
    ]

    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project="cs224r-wire-untangling",
            config={
                **config,
                "seed": seed,
                "checkpoint_dir": checkpoint_dir,
                "demo_seed_enabled": demo_seed_enabled,
                "learning_starts": learning_starts,
            },
            tags=["sac-baseline", "demo-seeded" if demo_seed_enabled else "online-only"],
            sync_tensorboard=True,
        )
        tensorboard_log = os.path.join(checkpoint_dir, "runs", run.id)
        callbacks.append(WandbCallback(verbose=0))

    model = SAC(
        "MlpPolicy",
        train_env,
        seed=seed,
        learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
        batch_size=int(sac_cfg.get("batch_size", 256)),
        buffer_size=int(sac_cfg.get("buffer_size", 1_000_000)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        tau=float(sac_cfg.get("tau", 0.005)),
        train_freq=int(sac_cfg.get("train_freq", 1)),
        gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
        ent_coef=sac_cfg.get("ent_coef", "auto"),
        learning_starts=learning_starts,
        policy_kwargs=_policy_kwargs(sac_cfg),
        tensorboard_log=tensorboard_log,
        device=sac_cfg.get("device", "auto"),
        verbose=1,
    )

    seeded_transitions = 0
    if demo_seed_enabled:
        demos_path = str(demo_cfg.get("demos_path", "data/stick_n1_orientation_demos.hdf5"))
        if not os.path.exists(demos_path):
            raise FileNotFoundError(f"Demo seed file not found: {demos_path}")
        max_transitions = demo_cfg.get("max_transitions")
        if max_transitions is not None:
            max_transitions = int(max_transitions)
        seeded_transitions = seed_sac_replay_buffer(
            model,
            demos_path=demos_path,
            env_cfg=env_cfg,
            max_transitions=max_transitions,
            warn_on_config_mismatch=bool(demo_cfg.get("warn_on_config_mismatch", True)),
        )
        print(f"Seeded SAC replay buffer with {seeded_transitions} expert transitions.")
    else:
        print("Demo replay seeding disabled; running pure online SAC.")

    run_config_path = os.path.join(checkpoint_dir, "run_config.yaml")
    with open(run_config_path, "w") as f:
        yaml.safe_dump(
            {
                **config,
                "seed": seed,
                "checkpoint_dir": checkpoint_dir,
                "seeded_transitions": seeded_transitions,
            },
            f,
            sort_keys=False,
        )

    total_timesteps = int(sac_cfg.get("total_timesteps", 500000))
    print(
        "Starting SAC baseline training: "
        f"timesteps={total_timesteps}, seed={seed}, "
        f"num_sticks={env_cfg.get('num_sticks', 1)}, "
        f"reward_shaping={env_cfg.get('reward_shaping', True)}, "
        f"demo_seed={demo_seed_enabled}, learning_starts={learning_starts}"
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=int(sac_cfg.get("log_interval", 10)),
    )

    final_path = os.path.join(checkpoint_dir, "sac_final")
    model.save(final_path)
    print(f"Final SAC model saved to {final_path}.zip")

    if run is not None:
        run.finish()
    train_env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/stick_reorder.yaml")
    parser.add_argument("--sac-config", default="configs/sac_baseline.yaml")
    parser.add_argument("--demos-path", default=None, help="Override demo seed HDF5 path")
    parser.add_argument("--no-demo-seed", action="store_true", help="Disable expert replay seeding")
    parser.add_argument("--demo-max-transitions", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/sac_demo_seeded_n1")
    parser.add_argument("--num-sticks", type=int, default=1)
    parser.add_argument("--reward-shaping", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (e.g. cpu, cuda, cuda:0, cuda:1). Default: auto-detect")
    args = parser.parse_args()

    cfg = load_config(args.env_config, args.sac_config)
    env_cfg = cfg.setdefault("env", {})
    sac_cfg = cfg.setdefault("sac_baseline", {})
    demo_cfg = sac_cfg.setdefault("demo_seed", {})

    if args.num_sticks is not None:
        env_cfg["num_sticks"] = args.num_sticks
    if args.reward_shaping is not None:
        env_cfg["reward_shaping"] = args.reward_shaping
    if args.no_demo_seed:
        demo_cfg["enabled"] = False
    if args.demos_path is not None:
        demo_cfg["demos_path"] = args.demos_path
    if args.demo_max_transitions is not None:
        demo_cfg["max_transitions"] = args.demo_max_transitions
    if args.total_timesteps is not None:
        sac_cfg["total_timesteps"] = args.total_timesteps
    if args.learning_starts is not None:
        sac_cfg["learning_starts"] = args.learning_starts
        sac_cfg["no_demo_seed_learning_starts"] = args.learning_starts

    train(
        cfg,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()

