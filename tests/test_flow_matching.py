import types

import h5py
import numpy as np
import torch

from scripts.train_flow_matching import load_data
from wire_untangling.utils.normalizer import Normalizer
from scripts.play_env import DPFMModelPolicy
from wire_untangling.policies.flow_matching_policy import FlowMatchingSchedule


def test_dataset_standardization_roundtrip():
    mean = np.array([0.0, 2.0, 1.0], dtype=np.float32)
    std = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    raw = np.array(
        [
            [-1.0, 0.0, 4.0],
            [0.5, -2.0, 1.0],
        ],
        dtype=np.float32,
    )

    norm = Normalizer(loc=mean, scale=std)
    normalized = norm.normalize(raw)
    np.testing.assert_allclose(norm.denormalize(normalized), raw)


def test_flow_matching_sample_is_unclamped_by_default():
    class ConstantVelocity(torch.nn.Module):
        def forward(self, action, state, timestep):
            return torch.full_like(action, -10.0)

    schedule = FlowMatchingSchedule(action_dim=5, device="cpu", num_steps=2)
    state = torch.zeros(3, 4)
    sample = schedule.sample(ConstantVelocity(), state)

    assert sample.shape == (3, 5)
    assert torch.any(sample < -1.0)


def test_flow_matching_sample_accepts_deterministic_initial_noise():
    class ConstantVelocity(torch.nn.Module):
        def forward(self, action, state, timestep):
            return torch.ones_like(action)

    schedule = FlowMatchingSchedule(action_dim=4, device="cpu", num_steps=2)
    state = torch.zeros(2, 3)
    initial_noise = torch.zeros(2, 4)
    sample_a = schedule.sample(ConstantVelocity(), state, initial_noise=initial_noise)
    sample_b = schedule.sample(ConstantVelocity(), state, initial_noise=initial_noise)

    np.testing.assert_allclose(sample_a.numpy(), sample_b.numpy())
    np.testing.assert_allclose(sample_a.numpy(), np.ones((2, 4), dtype=np.float32))


def test_load_data_normalizes_action_chunks(tmp_path):
    path = tmp_path / "demos.hdf5"
    obs = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    actions = np.array(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [0.5, 3.0],
            [-0.5, 4.0],
        ],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as f:
        grp = f.create_group("data/demo_0")
        grp.create_dataset("obs", data=obs)
        grp.create_dataset("actions", data=actions)

    loader, state_dim, action_dim, obs_norm, action_norm = load_data(
        str(path),
        chunk_size=3,
        batch_size=8,
        shuffle=False,
    )
    states, chunks = next(iter(loader))

    assert state_dim == 2
    assert action_dim == 2
    assert states.shape[1] == 2
    assert chunks.shape[1] == 6
    assert isinstance(obs_norm, Normalizer)
    assert isinstance(action_norm, Normalizer)
    assert abs(float(states.mean())) < 2.0
    assert abs(float(chunks.mean())) < 2.0


def test_dpfm_policy_executes_chunk_before_requerying():
    policy = DPFMModelPolicy.__new__(DPFMModelPolicy)
    policy.action_dim = 2
    policy.pred_horizon = 4
    policy.execute_steps = 3
    policy.action_low = np.array([-1.0, -1.0], dtype=np.float32)
    policy.action_high = np.array([1.0, 1.0], dtype=np.float32)
    policy._chunk = None
    policy._nchunk = None
    policy._chunk_idx = 0
    calls = {"n": 0}

    def sample_chunk(self, obs):
        calls["n"] += 1
        offset = 10 * calls["n"]
        chunk = np.array(
            [[offset + i, offset + i + 0.5] for i in range(self.pred_horizon)],
            dtype=np.float32,
        )
        return chunk, chunk

    policy._sample_chunk = types.MethodType(sample_chunk, policy)

    obs = np.zeros(2, dtype=np.float32)
    a0 = policy.predict(obs)
    a1 = policy.predict(obs)
    a2 = policy.predict(obs)
    a3 = policy.predict(obs)

    assert calls["n"] == 2
    np.testing.assert_array_equal(a0, np.array([10.0, 10.5], dtype=np.float32))
    np.testing.assert_array_equal(a1, np.array([11.0, 11.5], dtype=np.float32))
    np.testing.assert_array_equal(a2, np.array([12.0, 12.5], dtype=np.float32))
    np.testing.assert_array_equal(a3, np.array([20.0, 20.5], dtype=np.float32))
