"""Smoke tests for the MLP-BC policy and its training script.

These don't run the full robosuite env — they verify:
  - MLPBCPolicy has the right shape, output range, and gradient flow
  - mse_loss gives the expected scalar
  - train_bc.train() runs end-to-end on a synthetic HDF5 and produces a
    loadable checkpoint with the expected keys (model_state_dict + the
    state normalisation stats + arch metadata)
"""

import os
import sys

import h5py
import numpy as np
import pytest
import torch

# Make scripts/ importable so we can pull train symbols from train_bc.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from train_bc import CONDITIONING_PHASE_ACTIVE, load_data, train  # noqa: E402

from wire_untangling.policies.mlp_bc import MLPBCPolicy, mse_loss
from wire_untangling.utils.normalizer import Normalizer


# ── Policy unit tests ──────────────────────────────────────────────────

def test_mlp_bc_policy_shape_and_range():
    policy = MLPBCPolicy(state_dim=60, action_dim=7, hidden_dims=(64, 64))
    s = torch.randn(8, 60)
    a = policy(s)
    assert a.shape == (8, 7), f"Expected (8, 7), got {a.shape}"
    # tanh saturates near ±1; magnitudes must always be < 1
    assert torch.all(a.abs() < 1.0), "Tanh should keep |a| strictly below 1"


def test_mlp_bc_gradient_flow():
    """Loss has a non-zero gradient — confirms backprop reaches every layer."""
    policy = MLPBCPolicy(state_dim=10, action_dim=3, hidden_dims=(16, 16))
    s = torch.randn(4, 10)
    a = torch.randn(4, 3).clamp(-1, 1)
    loss = mse_loss(policy, s, a)
    loss.backward()
    norms = [p.grad.norm().item() for p in policy.parameters() if p.grad is not None]
    assert all(n > 0.0 for n in norms), "Some params had zero gradient"


def test_mlp_bc_dropout_in_eval_is_deterministic():
    """With dropout > 0, eval() should give bit-exact repeated predictions."""
    policy = MLPBCPolicy(state_dim=10, action_dim=3, hidden_dims=(16,), dropout=0.5)
    policy.eval()
    s = torch.randn(2, 10)
    with torch.no_grad():
        a1 = policy(s)
        a2 = policy(s)
    torch.testing.assert_close(a1, a2)


# ── End-to-end training test on synthetic HDF5 ─────────────────────────

@pytest.fixture
def synthetic_demos(tmp_path):
    """A tiny synthetic HDF5 in the same schema as collect_demos.py output."""
    out = tmp_path / "synth.hdf5"
    rng = np.random.default_rng(0)
    state_dim, action_dim = 8, 7
    with h5py.File(out, "w") as f:
        # Two demos, 50 steps each, with arbitrary but valid shapes.
        for i in range(2):
            T = 50
            grp = f.create_group(f"data/demo_{i}")
            grp.create_dataset("obs", data=rng.standard_normal((T, state_dim)).astype(np.float32))
            grp.create_dataset("actions",
                               data=rng.uniform(-1, 1, size=(T, action_dim)).astype(np.float32))
            grp.create_dataset("rewards", data=rng.standard_normal(T).astype(np.float32))
            grp.create_dataset("dones", data=np.zeros(T, dtype=bool))
            grp.create_dataset("next_obs",
                               data=rng.standard_normal((T, state_dim)).astype(np.float32))
            grp.create_dataset("phase", data=rng.integers(0, 8, size=T, dtype=np.int8))
            grp.create_dataset("active_stick",
                               data=np.array([0] * (T // 2) + [1] * (T - T // 2), dtype=np.int8))
            grp.create_dataset("is_success", data=np.zeros(T, dtype=bool))
        f.attrs["num_demos"] = 2
        f.attrs["obs_dim"] = state_dim
        f.attrs["env_config"] = '{"num_sticks": 2}'
        f.attrs["env_config_hash"] = "synth0000"
        f.attrs["top_seed"] = 42
        f.attrs["oracle_version"] = "test"
    return str(out)


def test_train_bc_end_to_end(synthetic_demos, tmp_path):
    """train() should produce a loadable checkpoint with the expected keys."""
    ckpt_dir = tmp_path / "ckpt"
    train(
        demos_path=synthetic_demos,
        epochs=2,
        batch_size=32,
        lr=1e-3,
        hidden_dims=(32, 32),
        dropout=0.0,
        seed=0,
        use_wandb=False,
        checkpoint_dir=str(ckpt_dir),
    )
    save_path = ckpt_dir / "mlp_bc_policy.pt"
    assert save_path.exists(), "Checkpoint file was not written"

    ckpt = torch.load(str(save_path), map_location="cpu", weights_only=True)
    expected_keys = {"model_state_dict", "state_dim", "action_dim",
                     "hidden_dims", "dropout", "obs_norm"}
    assert expected_keys.issubset(ckpt.keys()), \
        f"Missing keys: {expected_keys - set(ckpt.keys())}"

    assert int(ckpt["state_dim"]) == 8
    assert int(ckpt["action_dim"]) == 7
    assert isinstance(ckpt["obs_norm"], dict)
    obs_norm = Normalizer.from_state_dict(ckpt["obs_norm"])
    assert obs_norm.loc.shape == (8,)
    assert obs_norm.scale.shape == (8,)
    # Re-instantiate and load — should not raise
    policy = MLPBCPolicy(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dims=tuple(int(h) for h in ckpt["hidden_dims"]),
        dropout=float(ckpt["dropout"]),
    )
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()


def test_load_data_phase_active_appends_one_hot_features(synthetic_demos):
    loader, state_dim, action_dim, obs_norm, meta = load_data(
        synthetic_demos,
        batch_size=16,
        shuffle=False,
        conditioning=CONDITIONING_PHASE_ACTIVE,
    )

    assert state_dim == 8 + 8 + 2
    assert action_dim == 7
    assert meta["raw_obs_dim"] == 8
    assert meta["num_phases"] == 8
    assert meta["num_sticks"] == 2
    assert obs_norm.loc.shape == (18,)
    assert obs_norm.scale.shape == (18,)

    states, actions = next(iter(loader))
    assert states.shape[1] == 18
    assert actions.shape[1] == 7


def test_train_bc_phase_active_checkpoint_metadata(synthetic_demos, tmp_path):
    ckpt_dir = tmp_path / "phase_ckpt"
    train(
        demos_path=synthetic_demos,
        epochs=2,
        batch_size=32,
        lr=1e-3,
        hidden_dims=(32, 32),
        dropout=0.0,
        seed=0,
        use_wandb=False,
        checkpoint_dir=str(ckpt_dir),
        conditioning=CONDITIONING_PHASE_ACTIVE,
    )

    ckpt = torch.load(str(ckpt_dir / "mlp_bc_policy.pt"), map_location="cpu", weights_only=True)
    assert ckpt["conditioning"] == CONDITIONING_PHASE_ACTIVE
    assert int(ckpt["raw_obs_dim"]) == 8
    assert int(ckpt["state_dim"]) == 18
    assert int(ckpt["num_phases"]) == 8
    assert int(ckpt["num_sticks"]) == 2
    obs_norm = Normalizer.from_state_dict(ckpt["obs_norm"])
    assert obs_norm.loc.shape == (18,)
    assert obs_norm.scale.shape == (18,)
