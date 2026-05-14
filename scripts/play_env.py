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
import torch
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map
from wire_untangling.policies.mlp_bc import MLPBCPolicy
from wire_untangling.policies.flow_matching_policy import FlowMatchingPolicy
from wire_untangling.utils.normalizer import Normalizer


def _make_writer(path: str, fps: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8)


def _grab_frame(env):
    return env.sim.render(width=1280, height=720, camera_name="agentview")[::-1]


def make_env(render: bool = False, record: bool = False, num_sticks: int = 3):
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

    def reset(self):
        pass


class SACModelPolicy(ModelPolicy):
    def __init__(self, model_path: str, gym_env):
        super().__init__(model_path, gym_env)
        self.model = SAC.load(model_path)
        self.gym_env = gym_env

    def predict(self, obs:torch.Tensor) ->torch.Tensor:
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class MLPBCModelPolicy(ModelPolicy):
    def __init__(self, model_path: str, gym_env=None):
        super().__init__(model_path, gym_env)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.action_dim = int(ckpt["action_dim"])
        self.model = MLPBCPolicy(
            state_dim=int(ckpt["state_dim"]),
            action_dim=self.action_dim,
            hidden_dims=tuple(int(h) for h in ckpt["hidden_dims"]),
            dropout=float(ckpt.get("dropout", 0.0)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.obs_norm = Normalizer(loc=ckpt["state_mean"], scale=ckpt["state_std"])

    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normed = self.obs_norm.normalize_torch(
                torch.tensor(obs, dtype=torch.float32, device=self.device),
            ).unsqueeze(0)
            action = self.model(normed)[0]
        return action.cpu().numpy()

    def reset(self):
        pass


class DPFMModelPolicy(ModelPolicy):
    def __init__(self, model_path: str, gym_env, execute_steps: int | None = None, stochastic: bool = False):
        super().__init__(model_path, gym_env)
        self.gym_env = gym_env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.action_dim = int(checkpoint["action_dim"])
        self.pred_horizon = int(checkpoint["pred_horizon"])
        self.execute_steps = int(
            execute_steps if execute_steps is not None
            else checkpoint.get("execute_steps", max(1, self.pred_horizon // 2))
        )
        self.execute_steps = max(1, min(self.execute_steps, self.pred_horizon))
        self.stochastic = stochastic

        if "obs_norm" in checkpoint:
            self.obs_norm = Normalizer.from_state_dict(checkpoint["obs_norm"])
            self.action_norm = Normalizer.from_state_dict(checkpoint["action_norm"])

        self.model = FlowMatchingPolicy(
            state_dim=int(checkpoint["state_dim"]),
            action_dim=self.action_dim,
            pred_horizon=self.pred_horizon,
            num_steps=int(checkpoint["num_steps"]),
            device=self.device,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self._chunk = None
        self._chunk_idx = 0

    def reset(self):
        self._chunk = None
        self._chunk_idx = 0

    def _sample_chunk(self, obs: np.ndarray) -> np.ndarray:
        obs = self.obs_norm.normalize(obs)
        with torch.no_grad():
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            initial_noise = None
            if not self.stochastic:
                initial_noise = torch.zeros(
                    1,
                    self.pred_horizon * self.action_dim,
                    dtype=torch.float32,
                    device=self.device,
                )
            flat_chunk = self.model.schedule.sample(
                self.model.model,
                state,
                initial_noise=initial_noise,
            )
        chunk = flat_chunk[0].reshape(self.pred_horizon, self.action_dim).cpu().numpy()
        return self.action_norm.denormalize(chunk)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self._chunk is None or self._chunk_idx >= min(self.execute_steps, self.pred_horizon):
            self._chunk = self._sample_chunk(obs)
            self._chunk_idx = 0
        action = self._chunk[self._chunk_idx]
        self._chunk_idx += 1
        return action


def run_policy(env, policy:ModelPolicy, n_episodes: int = 2, render: bool = False, fps: int = 20, record_path: str = None):
    """Run a trained policy in the environment and report success rate.
    Uses GymWrapper to produce the flat obs vector the policy expects,
    while keeping the underlying Robosuite renderer active."""
    gym_env = GymWrapper(env)
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

    successes = 0
    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = gym_env.reset()
        policy.reset()
        total_reward = 0.0
        done = False
        step = 0

        for i, body_id in enumerate(env.stick_body_ids):
            pos = env.sim.data.body_xpos[body_id]
            print(f"  stick{i} initial pos: {pos}")

        while not done:
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

        success = info.get("is_success", False)
        successes += int(success)
        total_rewards.append(total_reward)
        print(f"Episode {ep + 1}: steps={step}  total_reward={total_reward:.3f}  success={success}")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nSuccess rate: {successes}/{n_episodes} ({successes/n_episodes:.0%})")
    print(f"Reward: {mean_reward:.3f} ± {std_reward:.3f}")

    if writer:
        writer.close()
        print(f"Video saved to {record_path}")
    gym_env.close()


def run_expert(env, n_episodes: int = 2, render: bool = False, fps: int = 20, record_path: str = None):
    """Run the scripted pick-and-place expert policy.
    Uses GymWrapper for flat observations + underlying Robosuite renderer."""
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
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for trained MLP-BC policy")
    parser.add_argument("--sac_checkpoint", type=str, default=None, help="Path to SB3 .zip checkpoint for trained policy")
    parser.add_argument("--dpfm_checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint for trained DPFM policy")
    parser.add_argument("--dpfm-execute-steps", type=int, default=None,
                        help="Override DPFM chunk actions executed before re-planning")
    parser.add_argument("--dpfm-stochastic", action="store_true",
                        help="Use random Flow Matching initial noise instead of deterministic zero-noise sampling")
    parser.add_argument("--expert", action="store_true", help="Run scripted pick-and-place expert (single stick)")
    parser.add_argument("--num-sticks", type=int, default=None, help="Override number of sticks")
    args = parser.parse_args()

    # Expert and DPFM modes currently target the one-stick BC setup.
    num_sticks = args.num_sticks if args.num_sticks is not None else (1 if (args.expert or args.dpfm_checkpoint or args.bc_checkpoint) else 3)
    env = make_env(render=args.render, record=bool(args.record), num_sticks=num_sticks)

    if args.gym:
        print_gym_spaces(env)
    elif args.expert:
        run_expert(env, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
    elif args.bc_checkpoint:
        policy = MLPBCModelPolicy(args.bc_checkpoint, env)
        run_policy(env, policy, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
    elif args.sac_checkpoint:
        policy = SACModelPolicy(args.sac_checkpoint, env)
        run_policy(env, policy, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
    elif args.dpfm_checkpoint:
        policy = DPFMModelPolicy(
            args.dpfm_checkpoint,
            env,
            execute_steps=args.dpfm_execute_steps,
            stochastic=args.dpfm_stochastic,
        )
        run_policy(env, policy, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
    else:
        run_random(env, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
