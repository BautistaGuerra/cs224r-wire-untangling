"""
Tests for the demo HDF5 schema and env_config_hash helper in scripts/collect_demos.py.

These tests don't run a full collection (that's a separate manual step);
they verify the hashing function and round-trip a synthetic demo through
HDF5 with all of the new fields (phase, is_success).
"""

import os
import sys

import h5py
import numpy as np
import pytest


# Make scripts/ importable so we can pull env_config_hash from collect_demos.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))


def test_env_config_hash_deterministic():
    from collect_demos import env_config_hash

    cfg = {"num_sticks": 1, "stick_length": 0.18, "goal_yaw": 0.0}
    h1 = env_config_hash(cfg, "1.4.0")
    h2 = env_config_hash(cfg, "1.4.0")
    assert h1 == h2, "Same input should produce the same hash"
    assert len(h1) == 12, "Hash should be 12 hex chars"


def test_env_config_hash_distinguishes_inputs():
    from collect_demos import env_config_hash

    cfg_a = {"num_sticks": 1, "stick_length": 0.18}
    cfg_b = {"num_sticks": 1, "stick_length": 0.20}
    assert env_config_hash(cfg_a, "1.4.0") != env_config_hash(cfg_b, "1.4.0")
    assert env_config_hash(cfg_a, "1.4.0") != env_config_hash(cfg_a, "1.5.0")


def test_hdf5_roundtrip_with_phase_and_success(tmp_path):
    """Write a synthetic demo with all new fields and read it back."""
    out = tmp_path / "tiny.hdf5"
    T, obs_dim = 10, 60
    rng = np.random.default_rng(0)

    demo = {
        "obs":        rng.standard_normal((T, obs_dim)).astype(np.float32),
        "actions":    rng.standard_normal((T, 7)).astype(np.float32),
        "rewards":    rng.standard_normal(T).astype(np.float32),
        "dones":      np.array([False] * (T - 1) + [True], dtype=bool),
        "next_obs":   rng.standard_normal((T, obs_dim)).astype(np.float32),
        "phase":      rng.integers(0, 8, size=T, dtype=np.int8),
        "is_success": np.array([False] * (T - 1) + [True], dtype=bool),
    }

    with h5py.File(out, "w") as f:
        grp = f.create_group("data/demo_0")
        for k, v in demo.items():
            grp.create_dataset(k, data=v, compression="gzip")
        f.attrs["num_demos"] = 1
        f.attrs["obs_dim"] = obs_dim
        f.attrs["env_config_hash"] = "abcdef012345"
        f.attrs["oracle_version"] = "test"

    with h5py.File(out, "r") as f:
        assert f.attrs["num_demos"] == 1
        assert f.attrs["env_config_hash"] == "abcdef012345"
        for k, v in demo.items():
            np.testing.assert_array_equal(f[f"data/demo_0/{k}"][:], v)
        # Phase must be 8-way (0..7)
        phases = f["data/demo_0/phase"][:]
        assert phases.dtype == np.int8
        assert phases.min() >= 0 and phases.max() <= 7
