"""
Training entry point — SAC baseline with Stable-Baselines3.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/stick_reorder.yaml
    python scripts/train.py --timesteps 500000 --wandb
"""

import argparse
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_gym_env(env_cfg: dict):
    from wire_untangling.envs import StickReorderEnv
    from robosuite.wrappers import GymWrapper

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


def train(config: dict, total_timesteps: int, use_wandb: bool):
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    env_cfg = config.get("env", {})
    train_cfg = config.get("training", {})

    env = Monitor(make_gym_env(env_cfg))

    kwargs = dict(
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        batch_size=train_cfg.get("batch_size", 256),
        buffer_size=train_cfg.get("buffer_size", 1_000_000),
        verbose=1,
    )

    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(project="cs224r-wire-untangling", config=config, sync_tensorboard=True)
        kwargs["tensorboard_log"] = f"runs/{run.id}"
        callback = WandbCallback(verbose=2)
    else:
        callback = None

    model = SAC("MlpPolicy", env, **kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("checkpoints/sac_stick_reorder")
    print("Model saved to checkpoints/sac_stick_reorder")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stick_reorder.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    total_timesteps = args.timesteps or config.get("training", {}).get("total_timesteps", 1_000_000)

    train(config, total_timesteps=total_timesteps, use_wandb=args.wandb)
