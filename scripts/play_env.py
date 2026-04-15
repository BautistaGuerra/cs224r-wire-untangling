"""
Sanity-check script: instantiate StickReorderEnv, run random actions, and
optionally render.

Usage:
    # Headless (no window) — just checks everything loads correctly
    python scripts/play_env.py

    # With MuJoCo viewer (requires a display)
    python scripts/play_env.py --render

    # Wrap as Gymnasium env and print spaces
    python scripts/play_env.py --gym
"""

import argparse
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


def run_random(env, n_episodes: int = 2, render: bool = False):
    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done:
            low, high = env.action_spec
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            if render:
                env.render()

        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={info.get('success', False)}")

    env.close()


def print_gym_spaces(env):
    from robosuite.wrappers import GymWrapper

    gym_env = GymWrapper(env)
    print("Observation space:", gym_env.observation_space)
    print("Action space:     ", gym_env.action_space)
    gym_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument("--gym", action="store_true", help="Print Gymnasium spaces")
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()

    env = make_env(render=args.render)

    if args.gym:
        print_gym_spaces(env)
    else:
        run_random(env, n_episodes=args.episodes, render=args.render)
