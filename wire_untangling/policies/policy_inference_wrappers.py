"""The module contains high-level implementation of policies that contain the reference to the active environment."""
from typing import Tuple

import numpy as np
import torch
from stable_baselines3 import SAC
from wire_untangling.policies.mlp_bc import MLPBCPolicy
from wire_untangling.policies.flow_matching_policy import FlowMatchingPolicy
from wire_untangling.policies.context_predictor import ContextPredictor
from wire_untangling.policies.rl.agent import TD3Agent
from wire_untangling.utils.normalizer import Normalizer, MinMaxNormalizer, IdentityNormalizer, load_normalizer
from wire_untangling.utils.seeding import resolve_device
from wire_untangling.utils.stick_order import StickOrderScheduler
from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map


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
    """Convert predictor logits into hard one-hot BC context features."""
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


class ModelPolicy(object):
    def __init__(self, model_path:str, gym_env):
        self.obs_norm = None
        self.action_norm = None
        pass

    def predict(self, obs:np.ndarray) -> np.ndarray:
        pass

    def predict_norm(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def reset(self, stick_order=None):
        pass


class SACModelPolicy(ModelPolicy):
    def __init__(self, model_path: str, gym_env):
        super().__init__(model_path, gym_env)
        self.model = SAC.load(model_path)
        self.gym_env = gym_env

    def predict(self, obs: np.ndarray) ->np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class MLPBCModelPolicy(ModelPolicy):
    def __init__(
        self,
        model_path: str,
        gym_env=None,
        device: str | None = None,
        context_predictor_checkpoint: str | None = None,
        compare_oracle_context: bool = False,
    ):
        super().__init__(model_path, gym_env)
        self.device = resolve_device(device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
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

        # self.obs_norm = Normalizer(loc=ckpt["state_mean"], scale=ckpt["state_std"])
        assert "obs_norm" in ckpt
        self.obs_norm = Normalizer.from_state_dict(ckpt["obs_norm"])
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
        # env_num_sticks = int(getattr(gym_env.env, "num_sticks", self.num_sticks))
        # if env_num_sticks != self.num_sticks:
        #     raise ValueError(
        #         f"Checkpoint expects num_sticks={self.num_sticks}, "
        #         f"but env has num_sticks={env_num_sticks}"
        #     )
        # order_schedule = StickOrderScheduler(expert_cfg, self.num_sticks)
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
        order_schedule = StickOrderScheduler(expert_cfg, self.num_sticks)
        obs_map = build_obs_index_map(gym_env)
        self._phase_tracker = PickPlaceExpertPolicy(
            obs_map,
            goal_yaw=self.goal_yaw,
            stick_order=order_schedule.order_for(0),
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
    def __init__(
        self,
        model_path: str,
        gym_env,
        execute_steps: int | None = None,
        stochastic: bool = True,
        replan_on_context_change: bool = False,
        device: str | None = None,
    ):
        super().__init__(model_path, gym_env)
        self.gym_env = gym_env
        self.device = resolve_device(device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.action_dim = int(checkpoint["action_dim"])
        self.state_dim = int(checkpoint["state_dim"])
        self.conditioning = checkpoint.get("conditioning", "obs")
        self.raw_obs_dim = int(checkpoint.get("raw_obs_dim", self.state_dim))
        self.num_phases = int(checkpoint.get("num_phases", 8))
        self.num_sticks = int(checkpoint.get("num_sticks", 1))
        self.goal_yaw = float(getattr(gym_env, "goal_yaw", 0.0))
        self._phase_tracker = None
        self.pred_horizon = int(checkpoint["pred_horizon"])
        self.num_integration_steps = int(checkpoint["num_integration_steps"])
        self.execute_steps = int(
            execute_steps if execute_steps is not None
            else checkpoint.get("execute_steps", FlowMatchingPolicy.default_execute_steps(self.pred_horizon))
        )
        self.execute_steps = max(1, min(self.execute_steps, self.pred_horizon))
        self.stochastic = stochastic
        self.replan_on_context_change = replan_on_context_change

        print('')
        print('Initializing the DPFM policy')
        print('')
        print(f'Integration steps: {self.num_integration_steps}')
        print(f'Prediction horizon: {self.pred_horizon}')
        print(f'Execution steps: {self.execute_steps}')
        print(f'Stochastic: {self.stochastic}')
        print(f'Replan on context change: {self.replan_on_context_change}')

        assert "obs_norm" in checkpoint
        print('Creating an observation normalizer')
        self.obs_norm = Normalizer.from_state_dict(checkpoint["obs_norm"])
        assert "action_norm" in checkpoint
        print('Creating an action normalizer')
        self.action_norm = load_normalizer(checkpoint["action_norm"])

        self.model = FlowMatchingPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            pred_horizon=self.pred_horizon,
            num_integration_steps=self.num_integration_steps,
            device=self.device,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        # Unnormalized (i.e. in the original action space) chunk
        self._chunk = None
        # normalized chunk for residual RL
        self._nchunk = None
        self._chunk_idx = 0
        self._chunk_context = None
        self._last_built_state_context = None
        self._last_policy_state = None
        self._last_normalized_policy_state = None

    def set_gym_env(self, gym_env, expert_cfg: dict | None = None):
        if self.conditioning != "phase-active":
            return
        env_num_sticks = int(getattr(gym_env.env, "num_sticks", self.num_sticks))
        if env_num_sticks != self.num_sticks:
            raise ValueError(
                f"Checkpoint expects num_sticks={self.num_sticks}, "
                f"but env has num_sticks={env_num_sticks}"
            )
        order_schedule = StickOrderScheduler(expert_cfg, self.num_sticks)
        obs_map = build_obs_index_map(gym_env)
        self._phase_tracker = PickPlaceExpertPolicy(
            obs_map,
            goal_yaw=self.goal_yaw,
            stick_order=order_schedule.order_for(0),
        )

    def reset(self, stick_order=None):
        self._chunk = None
        self._nchunk = None
        self._chunk_idx = 0
        self._chunk_context = None
        self._last_built_state_context = None
        self._last_policy_state = None
        self._last_normalized_policy_state = None
        if self._phase_tracker is not None:
            self._phase_tracker.reset(stick_order=stick_order)

    def _current_phase_active_context(self) -> tuple[int, int]:
        if self._phase_tracker is None:
            raise RuntimeError(
                "phase-active DPFM requires a GymWrapper-backed phase tracker; "
                "run it via run_policy/play_env so set_gym_env() is called."
            )
        return int(self._phase_tracker.phase), int(self._phase_tracker.active_stick)

    def _build_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if getattr(self, "conditioning", "obs") == "obs":
            self._last_built_state_context = None
            return obs
        if self.conditioning != "phase-active":
            raise ValueError(f"Unsupported DPFM conditioning: {self.conditioning!r}")

        phase, active_stick = self._current_phase_active_context()
        self._last_built_state_context = (phase, active_stick)
        self._phase_tracker.predict(obs)
        features = make_phase_active_features(
            phase,
            active_stick,
            num_phases=self.num_phases,
            num_sticks=self.num_sticks,
        )
        return np.concatenate([obs, features], axis=0)

    def _record_policy_state(self, state_np: np.ndarray) -> np.ndarray:
        self._last_policy_state = np.asarray(state_np, dtype=np.float32)
        self._last_normalized_policy_state = self.obs_norm.normalize(self._last_policy_state)
        return self._last_normalized_policy_state

    def get_last_normalized_policy_state(self) -> np.ndarray:
        if self._last_normalized_policy_state is None:
            raise RuntimeError("No DPFM policy state has been built yet; call predict_norm first.")
        return np.asarray(self._last_normalized_policy_state, dtype=np.float32).copy()

    def _advance_context_tracker(self, obs: np.ndarray) -> None:
        if getattr(self, "conditioning", "obs") == "phase-active":
            if self._phase_tracker is None:
                raise RuntimeError(
                    "phase-active DPFM requires a GymWrapper-backed phase tracker; "
                    "run it via run_policy/play_env so set_gym_env() is called."
                )
            self._phase_tracker.predict(obs)

    def _cached_context_changed(self) -> bool:
        if not getattr(self, "replan_on_context_change", False):
            return False
        if getattr(self, "conditioning", "obs") != "phase-active":
            return False
        if self._chunk is None or self._chunk_context is None:
            return False
        return self._current_phase_active_context() != self._chunk_context

    def _sample_chunk(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        state_np = self._build_state(obs)
        obs = self._record_policy_state(state_np)
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
        # Return both unnormalized (original action space) and normalized action chunks
        return self.action_norm.denormalize(chunk), chunk

    def predict_norm(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict both unnormalized, as well as normalized action. Normalized is used for residual RL."""
        needs_replan = self._chunk is None or self._chunk_idx >= min(self.execute_steps, self.pred_horizon)
        if not needs_replan and self._cached_context_changed():
            needs_replan = True
        if needs_replan:
            self._chunk, self._nchunk = self._sample_chunk(obs)
            self._chunk_idx = 0
            self._chunk_context = getattr(self, "_last_built_state_context", None)
        else:
            if hasattr(self, "obs_norm"):
                state_np = self._build_state(obs)
                self._record_policy_state(state_np)
            else:
                self._advance_context_tracker(obs)
        action = self._chunk[self._chunk_idx]
        naction = self._nchunk[self._chunk_idx]
        self._chunk_idx += 1
        return action, naction

    def predict_norm_with_state(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        action, naction = self.predict_norm(obs)
        return action, naction, self.get_last_normalized_policy_state()

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict unnormalized action value, i.e. in the original policy scope. Used for rollouts."""
        action, _ = self.predict_norm(obs)
        return action


class ResidualRLPolicy(ModelPolicy):
    """A residual RL policy. Incorporates the base behavior cloning policy."""

    def __init__(self, rl_model_path: str, base_model_path: str, base_policy: ModelPolicy, gym_env, rrl_cfg=None, device: str | None = None):
        super().__init__(rl_model_path, gym_env)
        print('')
        print('Initializing the ResidualRL policy')
        print('')
        self.gym_env = gym_env
        self.device = resolve_device(device)
        self.base_policy = base_policy

        base_checkpoint = torch.load(base_model_path, map_location=self.device, weights_only=True)
        self.action_dim = int(base_checkpoint["action_dim"])
        self.state_dim = int(base_checkpoint["state_dim"])

        rl_checkpoint = torch.load(rl_model_path, map_location=self.device, weights_only=True)
        if "rrl_config" in rl_checkpoint:
            from scripts.train_residual_rl import DictConfig
            saved_raw = dict(rl_checkpoint["rrl_config"])
            if saved_raw["state_dim"] != self.state_dim:
                raise ValueError(f"State dimensions for the base policy ({self.state_dim}) do not match state dimensions for the residual policy ({saved_raw['state_dim']})")
            if saved_raw["action_dim"] != self.action_dim:
                raise ValueError(f"Action dimensions for the base policy ({self.action_dim}) do not match action dimensions for the residual policy ({saved_raw['action_dim']})")
            saved_raw["device"] = str(self.device)
            rrl_cfg = DictConfig(saved_raw)
        elif rrl_cfg is not None:
            rrl_cfg.state_dim = self.state_dim
            rrl_cfg.action_dim = self.action_dim
            rrl_cfg.device = str(self.device)
        else:
            raise ValueError(
                "RRL checkpoint does not contain 'rrl_config' and no rrl_cfg was provided. "
                "Re-train or pass --rrl-config with matching hyperparameters."
            )

        # Use exactly the same normalizers as the base policy
        self.obs_norm = self.base_policy.obs_norm
        self.action_norm = self.base_policy.action_norm

        self.rrl_model = TD3Agent(rrl_cfg)
        self.rrl_model.load_state_dict(rl_checkpoint["model_state_dict"])
        self.rrl_model.eval()


    def reset(self, stick_order=None):
        self.base_policy.reset(stick_order=stick_order)

    def predict_rrl(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts the action using residual RL.
        Args:
            obs: UNNORMALIZED observations
        Returns:
            final_action: final action, in the (denormalized) action space
            residual_action: residual action, as predicted by RRL
            base_naction: a base normalized action from the BC policy. we return it so we could store
                normalized observations in the buffer and pass these to residual RL during the training.
        """
        final_action, residual_action, base_naction, _ = self._predict_rrl(obs, include_diagnostics=False)
        return final_action, residual_action, base_naction

    def _predict_rrl(
        self,
        obs: np.ndarray,
        include_diagnostics: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        # Get the base action - unscaled / unnormalized back to the input [-1, 1] space
        _, base_naction, nobs = self.base_policy.predict_norm_with_state(obs)
        with torch.no_grad():
            nobs_t = torch.tensor(nobs, dtype=torch.float32, device=self.device)
            base_naction_t = torch.tensor(base_naction, dtype=torch.float32, device=self.device)
            # Sample action from the residual RL policy that is within [-action_scale, action_scale]
            # Note: in the original residual RL we do not apply any scaling here. It will not be sampled from the RRL
            # policy; no noise will be added - just mean value of the predicted action is reported.
            # Final scaling will be applied by the training / evaluation script
            residual_action_t = self.rrl_model.act(
                nobs_t,
                base_naction_t,
                # Do not add any noise to the action.
                eval_mode=True,
            )
            residual_action_t = residual_action_t.to(self.device)
            residual_action = residual_action_t.cpu().numpy()

            diagnostics = {}
            if include_diagnostics:
                final_naction_t = TD3Agent.get_combined_action_torch(base_naction_t, residual_action_t)
                q_final_all = self.rrl_model.critic(nobs_t.unsqueeze(0), final_naction_t.unsqueeze(0)).squeeze(-1)
                q_base_all = self.rrl_model.critic(nobs_t.unsqueeze(0), base_naction_t.unsqueeze(0)).squeeze(-1)
                context = getattr(self.base_policy, "_last_built_state_context", None)
                diagnostics = {
                    "q_final_mean": float(q_final_all.mean().item()),
                    "q_final_min": float(q_final_all.min().item()),
                    "q_base_mean": float(q_base_all.mean().item()),
                    "q_base_min": float(q_base_all.min().item()),
                    "q_advantage_mean": float((q_final_all.mean() - q_base_all.mean()).item()),
                    "q_advantage_min": float((q_final_all.min() - q_base_all.min()).item()),
                    "residual_l1": float(torch.mean(torch.abs(residual_action_t)).item()),
                    "residual_l2": float(torch.mean(torch.square(residual_action_t)).item()),
                    "base_action_l2": float(torch.mean(torch.square(base_naction_t)).item()),
                    "final_action_l2": float(torch.mean(torch.square(final_naction_t)).item()),
                }
                if context is not None:
                    diagnostics["phase"] = int(context[0])
                    diagnostics["active_stick"] = int(context[1])

        final_action, residual_action, base_naction = ResidualRLPolicy.combine_actions(
            self.action_norm,
            base_naction,
            residual_action,
        )
        return final_action, residual_action, base_naction, diagnostics

    def predict_with_diagnostics(self, obs: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Predict an action and return RRL critic/residual diagnostics for evaluation."""
        final_action, _, _, diagnostics = self._predict_rrl(obs, include_diagnostics=True)
        return final_action, diagnostics

    @staticmethod
    def combine_actions(action_norm, base_naction: np.ndarray, residual_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Here, the prediction is in the normalized space
        # TODO(alexta): verify if adding actions to the always-zero action dimensions screws training
        final_naction = TD3Agent.get_combined_action_numpy(base_naction, residual_action)
        final_action = action_norm.denormalize(final_naction)
        return final_action, residual_action, base_naction

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict unnormalized action value, i.e. in the original policy scope. Used for rollouts."""
        action, _, _ = self.predict_rrl(obs)
        return action
