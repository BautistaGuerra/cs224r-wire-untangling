import h5py
import numpy as np
import torch

import scripts.rrl_env_creation as rrl_env_creation
import scripts.train_residual_rl as train_residual_rl
from wire_untangling.policies.policy_inference_wrappers import DPFMModelPolicy, ResidualRLPolicy
from wire_untangling.utils.normalizer import IdentityNormalizer
from wire_untangling.utils.stick_order import StickOrderScheduler


def test_rrl_env_creation_forwards_n2_side_keys(monkeypatch):
    captured = {}

    class FakeStickReorderEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(rrl_env_creation, "StickReorderEnv", FakeStickReorderEnv)

    env = rrl_env_creation.make_rrl_gym_env(
        {
            "num_sticks": 2,
            "placement_mode": "two_stick_side",
            "side_init_x_range": [-0.11, 0.11],
            "side_init_y_ranges": [[-0.24, -0.12], [0.12, 0.24]],
            "side_init_yaw_range": [-0.5, 0.5],
            "side_goal_x": 0.0,
            "side_goal_y_ranges": [[-0.08, -0.02], [0.02, 0.08]],
            "stick_color_indices": [1, 0],
            "horizon": 1000,
        }
    )

    assert isinstance(env, FakeStickReorderEnv)
    assert captured["num_sticks"] == 2
    assert captured["placement_mode"] == "two_stick_side"
    assert captured["side_init_y_ranges"] == [[-0.24, -0.12], [0.12, 0.24]]
    assert captured["side_goal_y_ranges"] == [[-0.08, -0.02], [0.02, 0.08]]
    assert captured["stick_color_indices"] == [1, 0]
    assert captured["horizon"] == 1000


def test_offline_buffer_resets_base_policy_with_demo_stick_order(tmp_path, monkeypatch):
    path = tmp_path / "demos.hdf5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("data/demo_0")
        grp.create_dataset("obs", data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        grp.create_dataset("actions", data=np.array([[0.1], [0.2]], dtype=np.float32))
        grp.create_dataset("rewards", data=np.array([0.0, 1.0], dtype=np.float32))
        grp.create_dataset("dones", data=np.array([False, True], dtype=bool))
        grp.attrs["stick_order"] = np.array([1, 0], dtype=np.int8)

    class FakeBasePolicy:
        def __init__(self):
            self.resets = []
            self.calls = 0

        def reset(self, stick_order=None):
            self.resets.append(stick_order)

        def predict_norm_with_state(self, obs):
            self.calls += 1
            base_action = np.array([float(self.calls)], dtype=np.float32)
            normalized_state = np.asarray(obs, dtype=np.float32) + 10.0
            return base_action, base_action, normalized_state

    transitions = []
    base_policy = FakeBasePolicy()
    schedule = StickOrderScheduler(
        {"order_mode": "balanced", "order_choices": [[0, 1], [1, 0]]},
        num_sticks=2,
    )

    def fake_add_transition(**kwargs):
        transitions.append(kwargs)

    monkeypatch.setattr(train_residual_rl, "add_transition_to_buffer", fake_add_transition)

    train_residual_rl.populate_offline_buffer(
        demos_path=str(path),
        offline_rb=object(),
        base_policy=base_policy,
        obs_norm=None,
        action_norm=IdentityNormalizer(ndims=1),
        gamma=0.97,
        order_schedule=schedule,
    )

    assert base_policy.resets == [(1, 0)]
    assert len(transitions) == 2
    np.testing.assert_allclose(transitions[0]["obs"], [11.0, 12.0])
    np.testing.assert_allclose(transitions[0]["next_obs"], [13.0, 14.0])
    np.testing.assert_allclose(transitions[0]["action_base"], [1.0])


def test_dpfm_predict_norm_with_state_tracks_cached_action_state():
    class Tracker:
        phase = 0
        active_stick = 1

        def predict(self, obs):
            self.phase += 1
            return np.zeros(7, dtype=np.float32), {}

    policy = DPFMModelPolicy.__new__(DPFMModelPolicy)
    policy.conditioning = "phase-active"
    policy.num_phases = 8
    policy.num_sticks = 2
    policy.action_dim = 1
    policy.pred_horizon = 3
    policy.execute_steps = 2
    policy.replan_on_context_change = False
    policy.obs_norm = IdentityNormalizer(ndims=11)
    policy._phase_tracker = Tracker()
    policy._chunk = None
    policy._nchunk = None
    policy._chunk_idx = 0
    policy._chunk_context = None
    policy._last_built_state_context = None
    policy._last_policy_state = None
    policy._last_normalized_policy_state = None

    def sample_chunk(obs):
        state = policy._build_state(obs)
        policy._record_policy_state(state)
        chunk = np.array([[10.0], [11.0], [12.0]], dtype=np.float32)
        return chunk, chunk

    policy._sample_chunk = sample_chunk

    _, _, state0 = policy.predict_norm_with_state(np.array([0.5], dtype=np.float32))
    _, _, state1 = policy.predict_norm_with_state(np.array([0.6], dtype=np.float32))

    assert state0[1 + 0] == 1.0
    assert state0[1 + 8 + 1] == 1.0
    assert state1[1 + 1] == 1.0
    assert state1[1 + 8 + 1] == 1.0


def test_residual_policy_uses_base_policy_normalized_state():
    class FakeBasePolicy:
        obs_norm = IdentityNormalizer(ndims=2)
        action_norm = IdentityNormalizer(ndims=1)

        def predict_norm_with_state(self, obs):
            return (
                np.array([0.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
                np.array([7.0, 8.0], dtype=np.float32),
            )

    class FakeRRLModel:
        def __init__(self):
            self.obs = None
            self.base_action = None

        def act(self, obs, base_action, eval_mode=True):
            self.obs = obs.detach().cpu().numpy()
            self.base_action = base_action.detach().cpu().numpy()
            return torch.zeros_like(base_action)

    policy = ResidualRLPolicy.__new__(ResidualRLPolicy)
    policy.device = torch.device("cpu")
    policy.base_policy = FakeBasePolicy()
    policy.obs_norm = policy.base_policy.obs_norm
    policy.action_norm = policy.base_policy.action_norm
    policy.rrl_model = FakeRRLModel()

    final_action, residual_action, base_action = policy.predict_rrl(
        np.array([100.0, 200.0], dtype=np.float32)
    )

    np.testing.assert_allclose(policy.rrl_model.obs, [7.0, 8.0])
    np.testing.assert_allclose(policy.rrl_model.base_action, [2.0])
    np.testing.assert_allclose(final_action, [2.0])
    np.testing.assert_allclose(residual_action, [0.0])
    np.testing.assert_allclose(base_action, [2.0])


def test_find_latest_training_checkpoint_ignores_lower_and_temp_files(tmp_path):
    for name in [
        "td3_step10000.pt",
        "td3_step60000.pt",
        "td3_step360000.pt",
        "td3_step370000.pt.tmp",
        "td3_final.pt",
    ]:
        (tmp_path / name).write_text("x")

    latest = train_residual_rl.find_latest_training_checkpoint(str(tmp_path))

    assert latest == str(tmp_path / "td3_step360000.pt")
