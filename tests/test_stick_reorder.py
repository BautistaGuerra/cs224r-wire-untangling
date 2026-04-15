"""
Basic integration tests for StickReorderEnv.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def env():
    from wire_untangling.envs import StickReorderEnv

    e = StickReorderEnv(
        robots="Panda",
        num_sticks=2,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=50,
    )
    yield e
    e.close()


def test_reset_returns_obs(env):
    obs = env.reset()
    assert isinstance(obs, dict)
    assert "robot0_eef_pos" in obs
    assert obs["robot0_eef_pos"].shape == (3,)


def test_obs_contains_sticks(env):
    obs = env.reset()
    for i in range(env.num_sticks):
        assert f"stick{i}_pos" in obs, f"stick{i}_pos missing from observations"
        assert f"stick{i}_quat" in obs
        assert f"goal{i}_pos" in obs
        assert obs[f"stick{i}_pos"].shape == (3,)
        assert obs[f"goal{i}_pos"].shape == (3,)


def test_step_returns_valid_outputs(env):
    env.reset()
    low, high = env.action_spec
    action = np.zeros_like(low)
    obs, reward, done, info = env.step(action)

    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_reward_is_finite(env):
    env.reset()
    low, high = env.action_spec
    for _ in range(5):
        action = np.random.uniform(low, high)
        _, reward, _, _ = env.step(action)
        assert np.isfinite(reward), f"Non-finite reward: {reward}"


def test_goal_positions_shape(env):
    assert env._goal_positions.shape == (env.num_sticks, 3)


def test_gym_wrapper(env):
    from robosuite.wrappers import GymWrapper

    gym_env = GymWrapper(env)
    obs, _ = gym_env.reset()
    assert obs is not None
    gym_env.close()
