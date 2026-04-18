"""
SAC training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/stick_reorder.yaml
    python scripts/train.py --timesteps 500000 --seed 0
    python scripts/train.py --no-wandb   # skip WandB logging
"""

import argparse
import os
import random

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_gym_env(env_cfg: dict):
    from robosuite.wrappers import GymWrapper
    from wire_untangling.envs import StickReorderEnv

    env = StickReorderEnv(
        robots=env_cfg.get("robot", "Panda"),
        num_sticks=env_cfg.get("num_sticks", 3),
        stick_length=env_cfg.get("stick_length", 0.18),
        stick_radius=env_cfg.get("stick_radius", 0.012),
        goal_spacing=env_cfg.get("goal_spacing", 0.06),
        success_threshold=env_cfg.get("success_threshold", 0.03),
        reward_shaping=env_cfg.get("reward_shaping", True),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=500,
    )
    return GymWrapper(env)


def train(
    config: dict,
    total_timesteps: int,
    seed: int = 42,
    use_wandb: bool = True,
    checkpoint_dir: str = "checkpoints",
):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor

    # Seed everything for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = config.get("env", {})
    train_cfg = config.get("training", {})

    os.makedirs(f"{checkpoint_dir}/best", exist_ok=True)
    os.makedirs(f"{checkpoint_dir}/periodic", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_env = Monitor(make_gym_env(env_cfg))
    eval_env = Monitor(make_gym_env(env_cfg))   # separate env — never reuse train env for eval

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{checkpoint_dir}/best/",
        log_path="logs/",
        eval_freq=train_cfg.get("eval_freq", 10_000),
        n_eval_episodes=train_cfg.get("n_eval_episodes", 10),
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg.get("checkpoint_freq", 50_000),
        save_path=f"{checkpoint_dir}/periodic/",
        name_prefix="sac",
    )
    callbacks = [eval_callback, checkpoint_callback]

    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        algo = train_cfg.get("algorithm", "SAC")
        run = wandb.init(
            project="cs224r-wire-untangling",
            config={**config, "seed": seed, "algorithm": algo, "total_timesteps": total_timesteps},
            sync_tensorboard=True,
        )
        tensorboard_log = f"runs/{run.id}"
        callbacks.append(WandbCallback(verbose=0))
    else:
        run = None
        tensorboard_log = None

    # TODO: The SAC algorithm is only a baseline to test the whole repo functionality
    # we will replace it with our custom one
    model = SAC(
        "MlpPolicy",
        train_env,
        seed=seed,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        batch_size=train_cfg.get("batch_size", 256),
        buffer_size=train_cfg.get("buffer_size", 1_000_000),
        tensorboard_log=tensorboard_log,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Log best checkpoint as WandB artifact
    if run is not None:
        artifact = wandb.Artifact(f"{algo.lower()}-best-model", type="model")
        artifact.add_dir(f"{checkpoint_dir}/best/")
        run.log_artifact(artifact)
        run.finish()

    train_env.close()
    eval_env.close()
    print(f"Training complete. Best model saved to {checkpoint_dir}/best/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stick_reorder.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})

    total_timesteps = args.timesteps or train_cfg.get("total_timesteps", 1_000_000)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)

    train(
        config,
        total_timesteps=total_timesteps,
        seed=seed,
        use_wandb=not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
    )
