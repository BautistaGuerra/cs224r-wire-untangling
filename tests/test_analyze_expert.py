"""Tests for scripts/analyze_expert.py — synthetic HDF5, no env required."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Make scripts/ importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_expert  # noqa: E402


PHASES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE", "RETREAT"]


def _make_synthetic_hdf5(path, episodes):
    """episodes: list of dicts with keys
        success, phases (list[str]), rewards (list[float]),
        actions (np.ndarray T x 7), obs (np.ndarray T x obs_dim)
    """
    obs_dim = episodes[0]["obs"].shape[1]
    # Mock obs_index_map: 7 eef_quat at [0:4], stick_pos at [4:7], stick_quat at [7:11], goal_pos at [11:14]
    obs_map = {
        "robot0_eef_quat": [0, 4],
        "stick0_pos": [4, 7],
        "stick0_quat": [7, 11],
        "goal0_pos": [11, 14],
    }
    n_succ = sum(1 for e in episodes if e["success"])
    n_fail = len(episodes) - n_succ
    total = sum(len(e["phases"]) for e in episodes)

    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, ep in enumerate(episodes):
            T = len(ep["phases"])
            grp = data.create_group(f"demo_{i}")
            grp.attrs["success"] = ep["success"]
            grp.create_dataset("obs", data=ep["obs"].astype(np.float32))
            grp.create_dataset("actions", data=ep["actions"].astype(np.float32))
            grp.create_dataset("rewards", data=np.array(ep["rewards"], dtype=np.float32))
            grp.create_dataset("dones", data=np.zeros(T, dtype=bool))
            grp.create_dataset("next_obs", data=ep["obs"].astype(np.float32))
            grp.create_dataset(
                "phase",
                data=np.array(ep["phases"], dtype="S16"),
            )
        f.attrs["schema_version"] = 2
        f.attrs["num_demos"] = len(episodes)
        f.attrs["num_successes"] = n_succ
        f.attrs["num_failures"] = n_fail
        f.attrs["obs_dim"] = obs_dim
        f.attrs["total_samples"] = total
        f.attrs["env_config"] = "{}"
        f.attrs["obs_index_map"] = json.dumps(obs_map)


@pytest.fixture
def synthetic_demos(tmp_path):
    """Two episodes:
       ep0: success — APPROACH(3), DESCEND(2), GRASP(1)            T=6
       ep1: failure — APPROACH(2), DESCEND(2), GRASP(2)            T=6, terminated in GRASP
    """
    obs_dim = 14

    def mk_obs(stick_pos, goal_pos, stick_yaw_quat=(0, 0, 0, 1), eef_quat=(0, 0, 0, 1)):
        return np.array([list(eef_quat) + list(stick_pos) + list(stick_yaw_quat) + list(goal_pos)] * 6)

    ep0 = {
        "success": True,
        "phases": ["APPROACH", "APPROACH", "APPROACH", "DESCEND", "DESCEND", "GRASP"],
        "rewards": [0.0, 0.1, 0.2, 0.3, 0.4, 1.0],
        "actions": np.tile([0.5, 0, 0, 0, 0, 0, -1.0], (6, 1)),
        "obs": mk_obs(stick_pos=(0.1, 0.0, 0.825), goal_pos=(0.3, 0.0, 0.825)),
    }
    ep1 = {
        "success": False,
        "phases": ["APPROACH", "APPROACH", "DESCEND", "DESCEND", "GRASP", "GRASP"],
        "rewards": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "actions": np.tile([0.99, 0, 0, 0, 0, 0, 1.0], (6, 1)),  # saturated dx
        "obs": mk_obs(stick_pos=(0.0, 0.0, 0.825), goal_pos=(0.5, 0.0, 0.825)),
    }
    path = tmp_path / "demos.hdf5"
    _make_synthetic_hdf5(path, [ep0, ep1])
    return str(path)


def test_load_demos(synthetic_demos):
    demos, obs_map, meta = analyze_expert.load_demos(synthetic_demos)
    assert meta["num_demos"] == 2
    assert meta["num_successes"] == 1
    assert meta["num_failures"] == 1
    assert len(demos) == 2
    # phase decoded to unicode
    assert demos[0]["phase"][0] == "APPROACH"
    assert isinstance(obs_map["stick0_pos"], slice)


def test_phase_durations(synthetic_demos):
    demos, _, _ = analyze_expert.load_demos(synthetic_demos)
    durs = analyze_expert._phase_durations(demos)
    assert durs[0]["APPROACH"] == 3
    assert durs[0]["DESCEND"] == 2
    assert durs[0]["GRASP"] == 1
    assert durs[1]["APPROACH"] == 2
    assert durs[1]["DESCEND"] == 2
    assert durs[1]["GRASP"] == 2


def test_failure_phase_attribution(synthetic_demos, capsys):
    demos, _, _ = analyze_expert.load_demos(synthetic_demos)
    analyze_expert.analyze_failures(demos, save_dir=None, phase_filter=None)
    out = capsys.readouterr().out
    # ep1 failed in GRASP
    assert "GRASP" in out
    # parse the GRASP row: should show 1 failure
    grasp_lines = [l for l in out.splitlines() if l.strip().startswith("GRASP")]
    assert grasp_lines and "1" in grasp_lines[0]


def test_action_saturation(synthetic_demos, capsys):
    demos, _, _ = analyze_expert.load_demos(synthetic_demos)
    analyze_expert.analyze_action_stats(demos, save_dir=None, phase_filter=None)
    out = capsys.readouterr().out
    # ep1 has dx=0.99 (saturated). DESCEND in ep1 has 2 saturated; ep0 DESCEND has 0 saturated.
    # So DESCEND total: 2 of (2+2)=4 steps saturated → 50%
    descend_line = [l for l in out.splitlines() if l.strip().startswith("DESCEND")][0]
    assert "50%" in descend_line


def test_reward_decomposition(synthetic_demos, capsys):
    demos, _, _ = analyze_expert.load_demos(synthetic_demos)
    analyze_expert.analyze_rewards(demos, save_dir=None, phase_filter=None)
    out = capsys.readouterr().out
    # APPROACH: ep0 sums 0+0.1+0.2 = 0.3, ep1 sums 0+0=0 → mean total 0.15
    approach_line = [l for l in out.splitlines() if l.strip().startswith("APPROACH")][0]
    assert "0.150" in approach_line


def test_canonical_int8_schema(tmp_path):
    """Analyzer reads main's canonical schema: int8 phase, no obs_index_map attr,
    no per-demo `success` attribute (every saved demo is a success)."""
    path = tmp_path / "canonical.hdf5"
    obs_dim = 14
    T = 4
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        grp = data.create_group("demo_0")
        grp.create_dataset("obs", data=np.zeros((T, obs_dim), dtype=np.float32))
        grp.create_dataset("actions", data=np.zeros((T, 7), dtype=np.float32))
        grp.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))
        grp.create_dataset("dones", data=np.zeros(T, dtype=bool))
        grp.create_dataset("next_obs", data=np.zeros((T, obs_dim), dtype=np.float32))
        # int8 phase: APPROACH(0), APPROACH(0), DESCEND(1), GRASP(2)
        grp.create_dataset("phase", data=np.array([0, 0, 1, 2], dtype=np.int8))
        # Per-step is_success (always True at end for a saved demo)
        grp.create_dataset("is_success", data=np.array([False, False, False, True]))
        f.attrs["num_demos"] = 1
        f.attrs["obs_dim"] = obs_dim
        f.attrs["env_config"] = "{}"

    demos, obs_map, meta = analyze_expert.load_demos(str(path), rebuild_env=False)
    assert meta["num_demos"] == 1
    assert meta["num_successes"] == 1
    assert obs_map is None  # no obs_index_map attr, rebuild_env=False
    # int8 → string conversion via Phase enum order
    assert list(demos[0]["phase"]) == ["APPROACH", "APPROACH", "DESCEND", "GRASP"]
