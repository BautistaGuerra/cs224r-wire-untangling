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
import yaml
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies import ContextPredictor, PickPlaceExpertPolicy, build_obs_index_map
from wire_untangling.policies.mlp_bc import MLPBCPolicy
from wire_untangling.policies.flow_matching_policy import FlowMatchingPolicy
from wire_untangling.utils.normalizer import Normalizer
from wire_untangling.utils.stick_order import StickOrderScheduler


def _make_writer(path: str, fps: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8)


def _grab_frame(env):
    return env.sim.render(width=1280, height=720, camera_name="agentview")[::-1]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(
    render: bool = False,
    record: bool = False,
    num_sticks: int | None = None,
    env_cfg: dict | None = None,
):
    env_cfg = dict(env_cfg or {})
    if num_sticks is not None:
        env_cfg["num_sticks"] = num_sticks

    kwargs = dict(
        robots=env_cfg.get("robot", "Panda"),
        num_sticks=env_cfg.get("num_sticks", 3),
        stick_length=env_cfg.get("stick_length", 0.20),
        stick_radius=env_cfg.get("stick_radius", 0.0075),
        goal_spacing=env_cfg.get("goal_spacing", 0.06),
        success_threshold=env_cfg.get("success_threshold", 0.03),
        orientation_threshold=env_cfg.get("orientation_threshold", np.deg2rad(10.0)),
        lambda_rot=env_cfg.get("lambda_rot", 0.1),
        goal_yaw=env_cfg.get("goal_yaw", 0.0),
        reward_shaping=env_cfg.get("reward_shaping", True),
        terminate_on_success=env_cfg.get("terminate_on_success", True),
        has_renderer=render,
        has_offscreen_renderer=record,
        use_camera_obs=False,
        control_freq=20,
        horizon=env_cfg.get("horizon", 500),
        camera_names="agentview",
        camera_heights=720,
        camera_widths=1280,
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

    return StickReorderEnv(
        **kwargs,
    )


def make_phase_active_features(
    phase: int,
    active_stick: int,
    num_phases: int,
    num_sticks: int,
) -> np.ndarray:
    features = np.zeros(num_phases + num_sticks, dtype=np.float32)
    features[int(phase)] = 1.0
    features[num_phases + int(active_stick)] = 1.0
    return features


def hard_context_from_logits(
    phase_logits: torch.Tensor,
    active_logits: torch.Tensor,
    num_phases: int,
    num_sticks: int,
) -> tuple[int, int, np.ndarray]:
    """Convert predictor logits into the hard one-hot BC context features."""
    if not isinstance(phase_logits, torch.Tensor):
        phase_logits = torch.as_tensor(phase_logits)
    if not isinstance(active_logits, torch.Tensor):
        active_logits = torch.as_tensor(active_logits)
    phase_flat = phase_logits.detach().reshape(-1)
    active_flat = active_logits.detach().reshape(-1)
    if phase_flat.numel() != num_phases:
        raise ValueError(
            f"Expected {num_phases} phase logits, got {phase_flat.numel()}"
        )
    if active_flat.numel() != num_sticks:
        raise ValueError(
            f"Expected {num_sticks} active-stick logits, got {active_flat.numel()}"
        )

    phase = int(torch.argmax(phase_flat).item())
    active_stick = int(torch.argmax(active_flat).item())
    features = make_phase_active_features(
        phase,
        active_stick,
        num_phases=num_phases,
        num_sticks=num_sticks,
    )
    return phase, active_stick, features


def load_context_predictor_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ContextPredictor, Normalizer, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "model_state_dict",
        "num_phases",
        "num_sticks",
        "hidden_dims",
        "obs_mean",
        "obs_std",
    }
    if "obs_dim" not in ckpt and "raw_obs_dim" not in ckpt:
        required.add("obs_dim")
    missing = required - set(ckpt.keys())
    if missing:
        raise ValueError(
            f"Context predictor checkpoint missing keys: {sorted(missing)}"
        )

    obs_dim = int(ckpt.get("obs_dim", ckpt.get("raw_obs_dim")))
    model = ContextPredictor(
        obs_dim=obs_dim,
        num_phases=int(ckpt["num_phases"]),
        num_sticks=int(ckpt["num_sticks"]),
        hidden_dims=tuple(int(h) for h in ckpt["hidden_dims"]),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    obs_norm = Normalizer(loc=ckpt["obs_mean"], scale=ckpt["obs_std"])
    meta = {
        "obs_dim": obs_dim,
        "num_phases": int(ckpt["num_phases"]),
        "num_sticks": int(ckpt["num_sticks"]),
        "hidden_dims": tuple(int(h) for h in ckpt["hidden_dims"]),
        "dropout": float(ckpt.get("dropout", 0.0)),
    }
    return model, obs_norm, meta


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

    def reset(self, stick_order=None):
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
    def __init__(
        self,
        model_path: str,
        gym_env=None,
        context_predictor_checkpoint: str | None = None,
        compare_oracle_context: bool = False,
    ):
        super().__init__(model_path, gym_env)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.action_dim = int(ckpt["action_dim"])
        self.state_dim = int(ckpt["state_dim"])
        self.conditioning = ckpt.get("conditioning", "obs")
        self.raw_obs_dim = int(ckpt.get("raw_obs_dim", self.state_dim))
        self.num_phases = int(ckpt.get("num_phases", 8))
        self.num_sticks = int(ckpt.get("num_sticks", 1))
        self.goal_yaw = float(getattr(gym_env, "goal_yaw", 0.0))
        self._phase_tracker = None
        self._context_predictor = None
        self._context_obs_norm = None
        self._context_meta = None
        self._compare_oracle_context = bool(compare_oracle_context)
        self._context_diag = {
            "steps": 0,
            "phase_disagreements": 0,
            "active_stick_disagreements": 0,
            "joint_disagreements": 0,
        }
        if self._compare_oracle_context and context_predictor_checkpoint is None:
            raise ValueError("--compare-oracle-context requires a context predictor checkpoint")
        if context_predictor_checkpoint is not None and self.conditioning != "phase-active":
            raise ValueError(
                "A context predictor can only be used with phase-active MLP-BC; "
                f"checkpoint conditioning is {self.conditioning!r}"
            )
        if self.conditioning == "phase-active":
            expected_state_dim = self.raw_obs_dim + self.num_phases + self.num_sticks
            if self.state_dim != expected_state_dim:
                raise ValueError(
                    f"phase-active MLP-BC state_dim={self.state_dim} does not match "
                    f"raw_obs_dim + num_phases + num_sticks = {expected_state_dim}"
                )

        self.model = MLPBCPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=tuple(int(h) for h in ckpt["hidden_dims"]),
            dropout=float(ckpt.get("dropout", 0.0)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.obs_norm = Normalizer(loc=ckpt["state_mean"], scale=ckpt["state_std"])
        if context_predictor_checkpoint is not None:
            (
                self._context_predictor,
                self._context_obs_norm,
                self._context_meta,
            ) = load_context_predictor_checkpoint(context_predictor_checkpoint, self.device)
            self._validate_context_predictor_metadata()

    def _validate_context_predictor_metadata(self) -> None:
        if self._context_meta is None:
            return
        if int(self._context_meta["obs_dim"]) != self.raw_obs_dim:
            raise ValueError(
                f"Context predictor obs_dim={self._context_meta['obs_dim']} "
                f"does not match BC raw_obs_dim={self.raw_obs_dim}"
            )
        if int(self._context_meta["num_phases"]) != self.num_phases:
            raise ValueError(
                f"Context predictor num_phases={self._context_meta['num_phases']} "
                f"does not match BC num_phases={self.num_phases}"
            )
        if int(self._context_meta["num_sticks"]) != self.num_sticks:
            raise ValueError(
                f"Context predictor num_sticks={self._context_meta['num_sticks']} "
                f"does not match BC num_sticks={self.num_sticks}"
            )

    def set_gym_env(self, gym_env, expert_cfg: dict | None = None):
        if self.conditioning != "phase-active":
            return
        env_num_sticks = int(getattr(gym_env.env, "num_sticks", self.num_sticks))
        if env_num_sticks != self.num_sticks:
            raise ValueError(
                f"Checkpoint expects num_sticks={self.num_sticks}, "
                f"but env has num_sticks={env_num_sticks}"
            )
        if self._context_meta is not None and env_num_sticks != int(self._context_meta["num_sticks"]):
            raise ValueError(
                f"Context predictor expects num_sticks={self._context_meta['num_sticks']}, "
                f"but env has num_sticks={env_num_sticks}"
            )
        if self._context_predictor is not None and not self._compare_oracle_context:
            return
        self._order_schedule = StickOrderScheduler(expert_cfg, self.num_sticks)
        obs_map = build_obs_index_map(gym_env)
        self._phase_tracker = PickPlaceExpertPolicy(
            obs_map,
            goal_yaw=self.goal_yaw,
            stick_order=self._order_schedule.order_for(0),
        )

    def _predict_learned_context(self, obs: np.ndarray) -> tuple[int, int, np.ndarray]:
        if self._context_predictor is None or self._context_obs_norm is None:
            raise RuntimeError("No context predictor is loaded")
        if obs.shape[0] != self.raw_obs_dim:
            raise ValueError(
                f"Observation dim {obs.shape[0]} does not match BC raw_obs_dim={self.raw_obs_dim}"
            )
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            normed = self._context_obs_norm.normalize_torch(obs_tensor).unsqueeze(0)
            phase_logits, active_logits = self._context_predictor(normed)
        return hard_context_from_logits(
            phase_logits[0],
            active_logits[0],
            num_phases=self.num_phases,
            num_sticks=self.num_sticks,
        )

    def _compare_with_oracle_context(
        self,
        obs: np.ndarray,
        learned_phase: int,
        learned_active_stick: int,
    ) -> None:
        if self._phase_tracker is None:
            raise RuntimeError(
                "--compare-oracle-context requires a GymWrapper-backed phase tracker; "
                "run it via run_policy/play_env so set_gym_env() is called."
            )
        oracle_phase = int(self._phase_tracker.phase)
        oracle_active_stick = int(self._phase_tracker.active_stick)
        phase_disagree = int(learned_phase != oracle_phase)
        active_disagree = int(learned_active_stick != oracle_active_stick)
        self._context_diag["steps"] += 1
        self._context_diag["phase_disagreements"] += phase_disagree
        self._context_diag["active_stick_disagreements"] += active_disagree
        self._context_diag["joint_disagreements"] += int(
            bool(phase_disagree or active_disagree)
        )
        # Keep the oracle tracker synchronized for the next diagnostic step.
        self._phase_tracker.predict(obs)

    def context_diagnostics(self) -> dict[str, float | int]:
        steps = int(self._context_diag["steps"])
        if steps == 0:
            return dict(self._context_diag)
        out = dict(self._context_diag)
        out["phase_disagreement_rate"] = self._context_diag["phase_disagreements"] / steps
        out["active_stick_disagreement_rate"] = (
            self._context_diag["active_stick_disagreements"] / steps
        )
        out["joint_disagreement_rate"] = self._context_diag["joint_disagreements"] / steps
        return out

    def _build_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).flatten()
        if obs.shape[0] != self.raw_obs_dim:
            raise ValueError(
                f"Observation dim {obs.shape[0]} does not match BC raw_obs_dim={self.raw_obs_dim}"
            )
        if self.conditioning == "obs":
            return obs
        if self.conditioning != "phase-active":
            raise ValueError(f"Unsupported MLP-BC conditioning: {self.conditioning!r}")
        if self._context_predictor is not None:
            phase, active_stick, features = self._predict_learned_context(obs)
            if self._compare_oracle_context:
                self._compare_with_oracle_context(obs, phase, active_stick)
            return np.concatenate([obs, features], axis=0)
        if self._phase_tracker is None:
            raise RuntimeError(
                "phase-active MLP-BC requires a GymWrapper-backed phase tracker; "
                "run it via run_policy/play_env so set_gym_env() is called."
            )

        phase = int(self._phase_tracker.phase)
        active_stick = int(self._phase_tracker.active_stick)
        # Advance the tracker using current obs, but discard its scripted action.
        # The MLP still controls the robot.
        self._phase_tracker.predict(obs)
        features = make_phase_active_features(
            phase,
            active_stick,
            num_phases=self.num_phases,
            num_sticks=self.num_sticks,
        )
        return np.concatenate([obs, features], axis=0)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_np = self._build_state(obs)
            normed = self.obs_norm.normalize_torch(
                torch.tensor(state_np, dtype=torch.float32, device=self.device),
            ).unsqueeze(0)
            action = self.model(normed)[0]
        return action.cpu().numpy()

    def reset(self, stick_order=None):
        if self._phase_tracker is not None:
            self._phase_tracker.reset(stick_order=stick_order)


class DPFMModelPolicy(ModelPolicy):
    def __init__(self, model_path: str, gym_env, execute_steps: int | None = None, stochastic: bool = False):
        super().__init__(model_path, gym_env)
        self.gym_env = gym_env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.action_dim = int(checkpoint["action_dim"])
        self.state_dim = int(checkpoint["state_dim"])
        self.conditioning = checkpoint.get("conditioning", "obs")
        self.raw_obs_dim = int(checkpoint.get("raw_obs_dim", self.state_dim))
        self.num_phases = int(checkpoint.get("num_phases", 8))
        self.num_sticks = int(checkpoint.get("num_sticks", 1))
        self.goal_yaw = float(getattr(gym_env, "goal_yaw", 0.0))
        self.pred_horizon = int(checkpoint["pred_horizon"])
        self.execute_steps = int(
            execute_steps if execute_steps is not None
            else checkpoint.get("execute_steps", max(1, self.pred_horizon // 2))
        )
        self.execute_steps = max(1, min(self.execute_steps, self.pred_horizon))
        self.stochastic = stochastic
        self._phase_tracker = None

        if "obs_norm" in checkpoint:
            self.obs_norm = Normalizer.from_state_dict(checkpoint["obs_norm"])
            self.action_norm = Normalizer.from_state_dict(checkpoint["action_norm"])

        self.model = FlowMatchingPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            pred_horizon=self.pred_horizon,
            num_steps=int(checkpoint["num_steps"]),
            device=self.device,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self._chunk = None
        self._chunk_idx = 0

    def set_gym_env(self, gym_env, expert_cfg: dict | None = None):
        if self.conditioning != "phase-active":
            return
        env_num_sticks = int(getattr(gym_env.env, "num_sticks", self.num_sticks))
        if env_num_sticks != self.num_sticks:
            raise ValueError(
                f"Checkpoint expects num_sticks={self.num_sticks}, "
                f"but env has num_sticks={env_num_sticks}"
            )
        self._order_schedule = StickOrderScheduler(expert_cfg, self.num_sticks)
        obs_map = build_obs_index_map(gym_env)
        self._phase_tracker = PickPlaceExpertPolicy(
            obs_map,
            goal_yaw=self.goal_yaw,
            stick_order=self._order_schedule.order_for(0),
        )

    def reset(self, stick_order=None):
        self._chunk = None
        self._chunk_idx = 0
        if self._phase_tracker is not None:
            self._phase_tracker.reset(stick_order=stick_order)

    def _build_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self.conditioning == "obs":
            return obs
        if self.conditioning != "phase-active":
            raise ValueError(f"Unsupported DPFM conditioning: {self.conditioning!r}")
        if self._phase_tracker is None:
            raise RuntimeError(
                "phase-active DPFM requires a GymWrapper-backed phase tracker; "
                "run it via run_policy/play_env so set_gym_env() is called."
            )

        phase = int(self._phase_tracker.phase)
        active_stick = int(self._phase_tracker.active_stick)
        # Advance the oracle tracker with the current observation; DPFM still
        # supplies the control actions.
        self._phase_tracker.predict(obs)
        features = make_phase_active_features(
            phase,
            active_stick,
            num_phases=self.num_phases,
            num_sticks=self.num_sticks,
        )
        return np.concatenate([obs, features], axis=0)

    def _sample_chunk(self, obs: np.ndarray) -> np.ndarray:
        obs = self.obs_norm.normalize(self._build_state(obs))
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


def run_policy(
    env,
    policy: ModelPolicy,
    n_episodes: int = 2,
    render: bool = False,
    fps: int = 20,
    record_path: str = None,
    expert_cfg: dict | None = None,
):
    """Run a trained policy in the environment and report success rate.
    Uses GymWrapper to produce the flat obs vector the policy expects,
    while keeping the underlying Robosuite renderer active."""
    gym_env = GymWrapper(env)
    expert_cfg = dict(expert_cfg or {})
    order_schedule = StickOrderScheduler(expert_cfg, env.num_sticks)
    if hasattr(policy, "set_gym_env"):
        policy.set_gym_env(gym_env, expert_cfg=expert_cfg)
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

    successes = 0
    successes_by_order: dict[tuple[int, ...], int] = {}
    attempts_by_order: dict[tuple[int, ...], int] = {}
    total_rewards = []
    for ep in range(n_episodes):
        stick_order = order_schedule.order_for(ep)
        attempts_by_order[stick_order] = attempts_by_order.get(stick_order, 0) + 1
        obs, _ = gym_env.reset()
        policy.reset(stick_order=stick_order)
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
        if success:
            successes_by_order[stick_order] = successes_by_order.get(stick_order, 0) + 1
        total_rewards.append(total_reward)
        print(
            f"Episode {ep + 1}: order={StickOrderScheduler.format_order(stick_order)} "
            f"steps={step}  total_reward={total_reward:.3f}  success={success}"
        )

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nSuccess rate: {successes}/{n_episodes} ({successes/n_episodes:.0%})")
    print("Per-order success:")
    for order in order_schedule.order_choices:
        order_successes = successes_by_order.get(order, 0)
        order_attempts = attempts_by_order.get(order, 0)
        order_rate = order_successes / order_attempts if order_attempts else 0.0
        print(
            f"  {StickOrderScheduler.format_order(order)}: "
            f"{order_successes}/{order_attempts} ({order_rate:.0%})"
        )
    print(f"Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    if hasattr(policy, "context_diagnostics"):
        diag = policy.context_diagnostics()
        if int(diag.get("steps", 0)):
            print(
                "Learned-vs-oracle context disagreement: "
                f"phase={diag['phase_disagreements']}/{diag['steps']} "
                f"({diag['phase_disagreement_rate']:.1%}), "
                f"active={diag['active_stick_disagreements']}/{diag['steps']} "
                f"({diag['active_stick_disagreement_rate']:.1%}), "
                f"joint={diag['joint_disagreements']}/{diag['steps']} "
                f"({diag['joint_disagreement_rate']:.1%})"
            )

    if writer:
        writer.close()
        print(f"Video saved to {record_path}")
    gym_env.close()


def run_expert(
    env,
    n_episodes: int = 2,
    render: bool = False,
    fps: int = 20,
    record_path: str = None,
    expert_cfg: dict | None = None,
):
    """Run the scripted pick-and-place expert policy.
    Uses GymWrapper for flat observations + underlying Robosuite renderer."""
    gym_env = GymWrapper(env)
    obs_map = build_obs_index_map(gym_env)
    expert_cfg = dict(expert_cfg or {})
    order_schedule = StickOrderScheduler(expert_cfg, env.num_sticks)
    expert = PickPlaceExpertPolicy(
        obs_map,
        goal_yaw=getattr(env, "goal_yaw", 0.0),
        stick_order=order_schedule.order_for(0),
    )
    sleep_time = 1.0 / fps if render else 0.0
    writer = _make_writer(record_path, fps) if record_path else None

    successes = 0
    successes_by_order: dict[tuple[int, ...], int] = {}
    attempts_by_order: dict[tuple[int, ...], int] = {}
    for ep in range(n_episodes):
        stick_order = order_schedule.order_for(ep)
        attempts_by_order[stick_order] = attempts_by_order.get(stick_order, 0) + 1
        obs, _ = gym_env.reset()
        expert.reset(stick_order=stick_order)
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
        if success:
            successes_by_order[stick_order] = successes_by_order.get(stick_order, 0) + 1
        print(
            f"Episode {ep + 1}: order={StickOrderScheduler.format_order(stick_order)} "
            f"steps={step}  total_reward={total_reward:.3f}  "
            f"success={success}  phase={expert._phase.name}"
        )

    print(f"\nSuccess rate: {successes}/{n_episodes} ({successes/n_episodes:.0%})")
    print("Per-order success:")
    for order in order_schedule.order_choices:
        order_successes = successes_by_order.get(order, 0)
        order_attempts = attempts_by_order.get(order, 0)
        order_rate = order_successes / order_attempts if order_attempts else 0.0
        print(
            f"  {StickOrderScheduler.format_order(order)}: "
            f"{order_successes}/{order_attempts} ({order_rate:.0%})"
        )
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
    parser.add_argument("--config", default=None,
                        help="Optional YAML config for env/expert settings, e.g. configs/stick_reorder_n2.yaml")
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for trained MLP-BC policy")
    parser.add_argument("--context-predictor-checkpoint", type=str, default=None,
                        help="Optional learned context predictor for phase-active MLP-BC")
    parser.add_argument("--compare-oracle-context", action="store_true",
                        help="Track oracle-vs-learned context disagreement while MLP-BC consumes learned context")
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
    if args.context_predictor_checkpoint and not args.bc_checkpoint:
        parser.error("--context-predictor-checkpoint is valid only with --bc_checkpoint")
    if args.compare_oracle_context and not args.context_predictor_checkpoint:
        parser.error("--compare-oracle-context requires --context-predictor-checkpoint")

    cfg = load_config(args.config) if args.config else {}
    env_cfg = cfg.get("env", {})
    expert_cfg = cfg.get("expert", {})

    if args.num_sticks is not None:
        num_sticks = args.num_sticks
    elif args.config:
        num_sticks = None
    else:
        # Preserve the old one-stick default for BC / expert checkpoint smoke tests.
        num_sticks = 1 if (args.expert or args.dpfm_checkpoint or args.bc_checkpoint) else 3
    env = make_env(
        render=args.render,
        record=bool(args.record),
        num_sticks=num_sticks,
        env_cfg=env_cfg,
    )

    if args.gym:
        print_gym_spaces(env)
    elif args.expert:
        run_expert(
            env,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
        )
    elif args.bc_checkpoint:
        policy = MLPBCModelPolicy(
            args.bc_checkpoint,
            env,
            context_predictor_checkpoint=args.context_predictor_checkpoint,
            compare_oracle_context=args.compare_oracle_context,
        )
        run_policy(
            env,
            policy,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
        )
    elif args.sac_checkpoint:
        policy = SACModelPolicy(args.sac_checkpoint, env)
        run_policy(
            env,
            policy,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
        )
    elif args.dpfm_checkpoint:
        policy = DPFMModelPolicy(
            args.dpfm_checkpoint,
            env,
            execute_steps=args.dpfm_execute_steps,
            stochastic=args.dpfm_stochastic,
        )
        run_policy(
            env,
            policy,
            n_episodes=args.episodes,
            render=args.render,
            fps=args.fps,
            record_path=args.record,
            expert_cfg=expert_cfg,
        )
    else:
        run_random(env, n_episodes=args.episodes, render=args.render, fps=args.fps, record_path=args.record)
