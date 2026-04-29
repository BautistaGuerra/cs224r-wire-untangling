"""
Sanity-check and visualization script: instantiate StickReorderEnv,
run random or trained-policy actions, and optionally render via the MuJoCo viewer.

Without --checkpoint: uses random actions (raw Robosuite env, no GymWrapper).
With --checkpoint: loads an SB3 model and runs its policy (requires GymWrapper
to produce the flat observation vector the policy expects).

Usage:
    # Headless random actions — just checks everything loads and steps correctly
    python scripts/play_env.py

    # Random actions with MuJoCo viewer (Linux: python, macOS: mjpython)
    python scripts/play_env.py --render
    python scripts/play_env.py --render --fps 20

    # Visualize a trained policy
    python scripts/play_env.py --render --checkpoint checkpoints/best/best_model.zip

    # Wrap as Gymnasium env and print observation/action spaces
    python scripts/play_env.py --gym
"""

import argparse
import time

import numpy as np


def make_env(render: bool = False):
    from wire_untangling.envs import StickReorderEnv

    return StickReorderEnv(
        robots="Panda",
        num_sticks=3,
        reward_shaping=True,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=500,
    )


def run_random(env, n_episodes: int = 2, render: bool = False, fps: int = 20):
    sleep_time = 1.0 / fps if render else 0.0

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        for i, body_id in enumerate(env.stick_body_ids):
            pos = env.sim.data.body_xpos[body_id]
            print(f"  stick{i} initial pos: {pos}")

        while not done:
            # Sample random action within the env's valid action range
            low, high = env.action_spec
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            if render:
                env.render()
                if sleep_time:
                    time.sleep(sleep_time)

        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={info.get('success', False)}")

    env.close()


def run_policy(env, checkpoint: str, n_episodes: int = 2, render: bool = False, fps: int = 20):
    """Run a trained SB3 policy in the environment.
    Uses GymWrapper to produce the flat obs vector the policy expects,
    while keeping the underlying Robosuite renderer active."""
    from robosuite.wrappers import GymWrapper
    from stable_baselines3 import SAC

    gym_env = GymWrapper(env)
    model = SAC.load(checkpoint, env=gym_env)
    sleep_time = 1.0 / fps if render else 0.0

    for ep in range(n_episodes):
        obs, _ = gym_env.reset()
        total_reward = 0.0
        done = False
        step = 0

        for i, body_id in enumerate(env.stick_body_ids):
            pos = env.sim.data.body_xpos[body_id]
            print(f"  stick{i} initial pos: {pos}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if render:
                env.render()
                if sleep_time:
                    time.sleep(sleep_time)

        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={info.get('is_success', False)}")

    gym_env.close()


def print_gym_spaces(env):
    """Wrap in GymWrapper to show what SB3 sees: flat observation and action spaces."""
    from robosuite.wrappers import GymWrapper

    gym_env = GymWrapper(env)
    print("Observation space:", gym_env.observation_space)
    print("Action space:     ", gym_env.action_space)
    gym_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer (use mjpython on macOS)")
    parser.add_argument("--fps", type=int, default=20, help="Target render FPS (default 20)")
    parser.add_argument("--gym", action="store_true", help="Print Gymnasium spaces")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SB3 .zip checkpoint for trained policy")
    args = parser.parse_args()

    env = make_env(render=args.render)

    if args.gym:
        print_gym_spaces(env)
    elif args.checkpoint:
        run_policy(env, args.checkpoint, n_episodes=args.episodes, render=args.render, fps=args.fps)
    else:
        run_random(env, n_episodes=args.episodes, render=args.render, fps=args.fps)
