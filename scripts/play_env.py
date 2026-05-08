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

    # Record video to disk (no GUI needed)
    python scripts/play_env.py --record videos/expert_demo.mp4 --expert
    python scripts/play_env.py --record videos/policy.mp4 --checkpoint checkpoints/best/best_model.zip

    # Wrap as Gymnasium env and print observation/action spaces
    python scripts/play_env.py --gym
"""

import argparse
import os
import time

import imageio
import numpy as np
from stable_baselines3 import SAC
import torch
from wire_untangling.policies.flow_matching_policy import FlowMatchingPolicy


def _make_writer(path: str, fps: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8)


def _grab_frame(env):
    return env.sim.render(width=1280, height=720, camera_name="agentview")[::-1]


def make_env(render: bool = False, record: bool = False, num_sticks: int = 3):
    from wire_untangling.envs import StickReorderEnv

    return StickReorderEnv(
        robots="Panda",
        num_sticks=num_sticks,
        reward_shaping=True,
        has_renderer=render,
        has_offscreen_renderer=record,
        use_camera_obs=False,
        control_freq=20,
        horizon=500,
        camera_names="agentview",
        camera_heights=720,
        camera_widths=1280,
    )


def run_random(env, n_episodes: int = 2, render: bool = False, fps: int = 20, record_path: str = None):
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

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
            if writer:
                writer.append_data(_grab_frame(env))

        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={info.get('success', False)}")

    if writer:
        writer.close()
        print(f"Video saved to {record_path}")
    env.close()


class ModelPolicy(object):
    def __init__(self, model_path:str, gym_env):
        pass

    def predict(self, obs:torch.Tensor) -> torch.Tensor:
        pass


class SACModelPolicy(ModelPolicy):
    def __init__(self, model_path: str, gym_env):
        super().__init__(model_path, gym_env)
        self.model = SAC.load(model_path, env=gym_env)
        self.gym_env = gym_env

    def predict(self, obs:torch.Tensor) ->torch.Tensor:
        action, _ = self.model.predict(obs, deterministic=True)
        return action



class DPFMModelPolicy(ModelPolicy):
    def __init__(self, model_path: str, gym_env):
        self.gym_env = gym_env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.action_dim = int(checkpoint["action_dim"])

        self.model = FlowMatchingPolicy(
            state_dim=int(checkpoint["state_dim"]),
            action_dim=self.action_dim,
            pred_horizon=int(checkpoint["pred_horizon"]),
            num_steps=int(checkpoint["num_steps"]),
            device=self.device,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_chunk = self.model.schedule.sample(self.model.model, state)
        first_action = action_chunk[0, :self.action_dim]
        return first_action.cpu().numpy()


def run_policy(env, policy:ModelPolicy, n_episodes: int = 2, render: bool = False, fps: int = 20, record_path: str = None):
    """Run a trained SB3 policy in the environment.
    Uses GymWrapper to produce the flat obs vector the policy expects,
    while keeping the underlying Robosuite renderer active."""
    from robosuite.wrappers import GymWrapper

    gym_env = GymWrapper(env)
    # model = SAC.load(checkpoint, env=gym_env)
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

    for ep in range(n_episodes):
        obs, _ = gym_env.reset()
        total_reward = 0.0
        done = False
        step = 0

        for i, body_id in enumerate(env.stick_body_ids):
            pos = env.sim.data.body_xpos[body_id]
            print(f"  stick{i} initial pos: {pos}")

        while not done:
            # action, _ = model.predict(obs, deterministic=True)
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if render:
                env.render()
                if sleep_time:
                    time.sleep(sleep_time)
            if writer:
                writer.append_data(_grab_frame(env))

        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={info.get('is_success', False)}")

    if writer:
        writer.close()
        print(f"Video saved to {record_path}")
    gym_env.close()


def run_expert(env, n_episodes: int = 2, render: bool = False, fps: int = 20, record_path: str = None):
    """Run the scripted pick-and-place expert policy.
    Uses GymWrapper for flat observations + underlying Robosuite renderer."""
    from robosuite.wrappers import GymWrapper

    from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map

    gym_env = GymWrapper(env)
    obs_map = build_obs_index_map(gym_env)
    expert = PickPlaceExpertPolicy(obs_map)
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

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
            if writer:
                writer.append_data(_grab_frame(env))

        success = info.get("is_success", False)
        successes += int(success)
        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={success}  phase={expert._phase.name}")

    print(f"\nSuccess rate: {successes}/{n_episodes} ({successes/n_episodes:.0%})")
    if writer:
        writer.close()
        print(f"Video saved to {record_path}")
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
    parser.add_argument("--record", type=str, default=None, metavar="PATH", help="Save video to .mp4 file (offscreen, no GUI needed)")
    parser.add_argument("--fps", type=int, default=20, help="Target render FPS (default 20)")
    parser.add_argument("--gym", action="store_true", help="Print Gymnasium spaces")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--sac_checkpoint", type=str, default=None, help="Path to SB3 .zip checkpoint for trained policy")
    parser.add_argument("--dpfm_checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint for trained DPFM policy")
    parser.add_argument("--expert", action="store_true", help="Run scripted pick-and-place expert (single stick)")
    parser.add_argument("--num-sticks", type=int, default=None, help="Override number of sticks")
    args = parser.parse_args()

    # Expert and DPFM modes default to 1 stick
    # num_sticks = args.num_sticks if args.num_sticks is not None else (1 if (args.expert or args.dpfm_checkpoint) else 3)
    num_sticks = args.num_sticks if args.num_sticks is not None else (1 if args.expert else 3)
    env = make_env(render=args.render, record=bool(args.record), num_sticks=num_sticks)

    if args.gym:
        print_gym_spaces(env)
    elif args.expert:
        run_expert(env, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
    elif args.sac_checkpoint:
        policy = SACModelPolicy(args.sac_checkpoint, env)
        run_policy(env, policy, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
    elif args.dpfm_checkpoint:
        policy = DPFMModelPolicy(args.dpfm_checkpoint, env)
        run_policy(env, policy, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
    else:
        run_random(env, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
