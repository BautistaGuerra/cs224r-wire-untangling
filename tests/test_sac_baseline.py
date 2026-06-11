import numpy as np
import h5py
from gymnasium import spaces

from scripts.train_sac_baseline import seed_sac_replay_buffer


class FakeReplayBuffer:
    def __init__(self):
        self.items = []

    def add(self, obs, next_obs, action, reward, done, infos):
        self.items.append((obs, next_obs, action, reward, done, infos))


class FakeModel:
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.replay_buffer = FakeReplayBuffer()


def _write_demo(path):
    with h5py.File(path, "w") as f:
        f.attrs["env_config"] = '{"num_sticks": 1, "reward_shaping": true}'
        f.attrs["env_config_hash"] = "abc123"
        f.attrs["top_seed"] = 42
        f.attrs["oracle_version"] = "test"
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset(
            "obs",
            data=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )
        demo.create_dataset(
            "next_obs",
            data=np.asarray([[4, 5, 6], [7, 8, 9]], dtype=np.float32),
        )
        demo.create_dataset(
            "actions",
            data=np.asarray([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32),
        )
        demo.create_dataset("rewards", data=np.asarray([0.5, 1.0], dtype=np.float32))
        demo.create_dataset("dones", data=np.asarray([False, True], dtype=bool))
        demo.create_dataset("is_success", data=np.asarray([False, True], dtype=bool))


def test_seed_sac_replay_buffer_loads_hdf5_transitions(tmp_path):
    demos_path = tmp_path / "demos.hdf5"
    _write_demo(demos_path)
    model = FakeModel()

    count = seed_sac_replay_buffer(
        model,
        str(demos_path),
        env_cfg={"num_sticks": 1, "reward_shaping": True},
    )

    assert count == 2
    assert len(model.replay_buffer.items) == 2
    obs, next_obs, action, reward, done, infos = model.replay_buffer.items[0]
    np.testing.assert_allclose(obs, [[1, 2, 3]])
    np.testing.assert_allclose(next_obs, [[4, 5, 6]])
    np.testing.assert_allclose(action, [[0.1, -0.1]])
    np.testing.assert_allclose(reward, [0.5])
    np.testing.assert_array_equal(done, [False])
    assert infos == [{"TimeLimit.truncated": False, "is_success": False}]


def test_seed_sac_replay_buffer_respects_max_transitions(tmp_path):
    demos_path = tmp_path / "demos.hdf5"
    _write_demo(demos_path)
    model = FakeModel()

    count = seed_sac_replay_buffer(
        model,
        str(demos_path),
        env_cfg={"num_sticks": 1, "reward_shaping": True},
        max_transitions=1,
    )

    assert count == 1
    assert len(model.replay_buffer.items) == 1

