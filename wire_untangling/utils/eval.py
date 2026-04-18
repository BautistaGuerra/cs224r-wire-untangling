"""
Algorithm-agnostic evaluation utilities.

Works with any policy that exposes:
    action, _state = policy.predict(obs, deterministic=True)

This includes SB3 models, and can be adapted for custom PyTorch policies
or VLA wrappers by adding a thin .predict() shim.
"""

import numpy as np


def evaluate(policy, env_cfg: dict, n_episodes: int = 50, seed: int = 42) -> dict:
    """
    Run n_episodes with a deterministic policy and return aggregate metrics.

    Args:
        policy: Any object with .predict(obs, deterministic=True) -> (action, state).
        env_cfg: Environment config dict (same schema as configs/stick_reorder.yaml env section).
        n_episodes: Number of evaluation episodes.
        seed: Random seed for the eval environment.

    Returns:
        dict with keys: success_rate, mean_reward, std_reward, mean_length, std_length.
    """
    from stable_baselines3.common.monitor import Monitor

    from scripts.train import make_gym_env

    env = Monitor(make_gym_env(env_cfg))

    successes, rewards, lengths = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        last_info = {}

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += r
            steps += 1
            last_info = info

        successes.append(bool(last_info.get("is_success", False)))
        rewards.append(ep_reward)
        lengths.append(steps)

    env.close()

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "n_episodes": n_episodes,
    }
