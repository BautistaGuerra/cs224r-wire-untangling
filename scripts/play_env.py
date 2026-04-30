"""
Sanity-check and visualization script: instantiate StickReorderEnv,
run random, trained-policy, or scripted expert actions, and optionally render.

Usage:
    # Headless random actions — just checks everything loads and steps correctly
    python scripts/play_env.py

    # Random actions with MuJoCo viewer (Linux: python, macOS: mjpython)
    python scripts/play_env.py --render
    python scripts/play_env.py --render --fps 20

    # Visualize a trained policy
    python scripts/play_env.py --render --checkpoint checkpoints/best/best_model.zip

    # Visualize the scripted expert policy (single stick)
    python scripts/play_env.py --render --expert
    python scripts/play_env.py --expert --episodes 10   # headless success rate check

    # Wrap as Gymnasium env and print observation/action spaces
    python scripts/play_env.py --gym
"""

import argparse
import time

import numpy as np


def make_env(render: bool = False, num_sticks: int = 3):
    from wire_untangling.envs import StickReorderEnv

    return StickReorderEnv(
        robots="Panda",
        num_sticks=num_sticks,
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


def run_expert(env, n_episodes: int = 2, render: bool = False, fps: int = 20):
    """Run the scripted pick-and-place expert policy.
    Uses GymWrapper for flat observations + underlying Robosuite renderer."""
    from robosuite.wrappers import GymWrapper

    from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map

    gym_env = GymWrapper(env)
    obs_map = build_obs_index_map(gym_env)
    expert = PickPlaceExpertPolicy(obs_map)
    sleep_time = 1.0 / fps if render else 0.0

    successes = 0
    for ep in range(n_episodes):
        obs, _ = gym_env.reset()
        expert.reset()
        total_reward = 0.0
        done = False
        step = 0

        for i, body_id in enumerate(env.stick_body_ids):
            pos = env.sim.data.body_xpos[body_id]
            print(f"  stick{i} initial pos: {pos}")

        while not done:
            action, _ = expert.predict(obs)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if render:
                env.render()
                if sleep_time:
                    time.sleep(sleep_time)

        success = info.get("is_success", False)
        successes += int(success)
        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={success}  phase={expert._phase.name}")

    print(f"\nSuccess rate: {successes}/{n_episodes} ({successes/n_episodes:.0%})")
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
    parser.add_argument("--expert", action="store_true", help="Run scripted pick-and-place expert (single stick)")
    parser.add_argument("--num-sticks", type=int, default=None, help="Override number of sticks")
    args = parser.parse_args()

    # Expert mode defaults to 1 stick
    num_sticks = args.num_sticks if args.num_sticks is not None else (1 if args.expert else 3)
    env = make_env(render=args.render, num_sticks=num_sticks)

    if args.gym:
        print_gym_spaces(env)
    elif args.expert:
        run_expert(env, n_episodes=args.episodes, render=args.render, fps=args.fps)
    elif args.checkpoint:
        run_policy(env, args.checkpoint, n_episodes=args.episodes, render=args.render, fps=args.fps)
    else:
        run_random(env, n_episodes=args.episodes, render=args.render, fps=args.fps)
