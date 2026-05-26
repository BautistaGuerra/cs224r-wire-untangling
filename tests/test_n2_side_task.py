import numpy as np
import pytest
from robosuite.wrappers import GymWrapper

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map
from wire_untangling.policies.pick_place_expert import Phase
from wire_untangling.utils.transform import yaw_from_quat_wxyz


@pytest.fixture(scope="module")
def side_env():
    env = StickReorderEnv(
        robots="Panda",
        num_sticks=2,
        placement_mode="two_stick_side",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
    )
    yield env
    env.close()


def test_n2_side_reset_ranges(side_env):
    side_env.reset()

    stick0 = side_env.sim.data.body_xpos[side_env.stick_body_ids[0]]
    stick1 = side_env.sim.data.body_xpos[side_env.stick_body_ids[1]]
    yaw0 = yaw_from_quat_wxyz(side_env.sim.data.body_xquat[side_env.stick_body_ids[0]])
    yaw1 = yaw_from_quat_wxyz(side_env.sim.data.body_xquat[side_env.stick_body_ids[1]])

    assert -0.11 <= stick0[0] <= 0.11
    assert -0.11 <= stick1[0] <= 0.11
    assert -0.24 <= stick0[1] <= -0.12
    assert 0.12 <= stick1[1] <= 0.24
    assert abs(yaw0) <= np.deg2rad(30.0) + 1e-6
    assert abs(yaw1) <= np.deg2rad(30.0) + 1e-6


def test_n2_side_goals_randomize_on_y_only(side_env):
    goals = []
    for _ in range(5):
        side_env.reset()
        goals.append(side_env._goal_positions.copy())
        np.testing.assert_allclose(side_env._goal_positions[:, 0], [0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            side_env._goal_positions[:, 2],
            [side_env.table_offset[2] + side_env.stick_radius] * 2,
            atol=1e-9,
        )
        assert -0.08 <= side_env._goal_positions[0, 1] <= -0.02
        assert 0.02 <= side_env._goal_positions[1, 1] <= 0.08

    y_samples = np.array([g[:, 1] for g in goals])
    assert np.unique(np.round(y_samples, decimals=5), axis=0).shape[0] > 1


def test_n2_side_resets_are_not_already_solved(side_env):
    for _ in range(20):
        side_env.reset()
        assert not side_env._check_success()


def test_n2_expert_switches_active_stick_at_retreat(side_env):
    gym_env = GymWrapper(side_env)
    obs_map = build_obs_index_map(gym_env)
    expert = PickPlaceExpertPolicy(obs_map, stick_order=(0, 1))

    obs, _ = gym_env.reset()
    expert.reset()
    assert expert.active_stick == 0

    obs = obs.copy()
    obs[obs_map["robot0_eef_pos"]] = np.array([
        side_env._goal_positions[0, 0],
        side_env._goal_positions[0, 1],
        expert.lift_height,
    ])
    expert._phase = Phase.RETREAT

    expert.predict(obs)

    assert expert.active_stick == 1
    assert expert.phase == Phase.APPROACH


def test_n2_expert_reset_accepts_episode_stick_order(side_env):
    gym_env = GymWrapper(side_env)
    obs_map = build_obs_index_map(gym_env)
    expert = PickPlaceExpertPolicy(obs_map, stick_order=(0, 1))

    expert.reset(stick_order=(1, 0))

    assert expert.active_stick == 1
    assert expert.phase == Phase.APPROACH
