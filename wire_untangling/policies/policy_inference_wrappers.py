"""The module contains high-level implementation of policies that contain the reference to the active environment."""
from typing import Tuple

import numpy as np
import torch
from stable_baselines3 import SAC
from wire_untangling.policies.mlp_bc import MLPBCPolicy
from wire_untangling.policies.flow_matching_policy import FlowMatchingPolicy
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
    def __init__(self, model_path: str, gym_env=None, device: str | None = None):
        super().__init__(model_path, gym_env)
        self.device = resolve_device(device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.action_dim = int(ckpt["action_dim"])
        self.conditioning = ckpt.get("conditioning", "obs")
        self.raw_obs_dim = int(ckpt.get("raw_obs_dim", ckpt["state_dim"]))
        self.num_phases = int(ckpt.get("num_phases", 8))
        self.num_sticks = int(ckpt.get("num_sticks", 1))
        self.goal_yaw = float(getattr(gym_env, "goal_yaw", 0.0))
        self._phase_tracker = None
        self.model = MLPBCPolicy(
            state_dim=int(ckpt["state_dim"]),
            action_dim=self.action_dim,
            hidden_dims=tuple(int(h) for h in ckpt["hidden_dims"]),
            dropout=float(ckpt.get("dropout", 0.0)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # self.obs_norm = Normalizer(loc=ckpt["state_mean"], scale=ckpt["state_std"])
        assert "obs_norm" in ckpt
        self.obs_norm = Normalizer.from_state_dict(ckpt["obs_norm"])

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

    def _build_state(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self.conditioning == "obs":
            return obs
        if self.conditioning != "phase-active":
            raise ValueError(f"Unsupported MLP-BC conditioning: {self.conditioning!r}")
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
