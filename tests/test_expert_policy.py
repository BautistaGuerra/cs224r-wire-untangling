"""
Unit tests for PickPlaceExpertPolicy.

These don't aim to verify the full success rate (that's measured separately
via play_env.py / smoke mode); they check the FSM contract: phase property
exists, action shape is right, episodes can advance through phases, and
goal_yaw is plumbed through to the rotation logic.
"""

import numpy as np
import pytest
from robosuite.wrappers import GymWrapper

from wire_untangling.envs import StickReorderEnv
from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map
from wire_untangling.policies.pick_place_expert import Phase, _wrap_to_half_pi


@pytest.fixture(scope="module")
def gym_env_n1():
    raw = StickReorderEnv(
        robots="Panda",
        num_sticks=1,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
    )
    g = GymWrapper(raw)
    yield g
    g.close()


def test_expert_action_shape_and_phase(gym_env_n1):
    obs_map = build_obs_index_map(gym_env_n1)
    expert = PickPlaceExpertPolicy(obs_map)

    obs, _ = gym_env_n1.reset()
    expert.reset()
    assert expert.phase == Phase.APPROACH, "Expert should start in APPROACH after reset"

    action, _ = expert.predict(obs)
    assert action.shape == (7,), f"Expected (7,) action, got {action.shape}"
    assert np.all(np.abs(action) <= 1.0), "Action must be in [-1, 1]"


def test_expert_phase_advances(gym_env_n1):
    """Across one rollout, the expert should advance past APPROACH."""
    obs_map = build_obs_index_map(gym_env_n1)
    expert = PickPlaceExpertPolicy(obs_map)

    obs, _ = gym_env_n1.reset()
    expert.reset()

    seen = {expert.phase}
    for _ in range(150):
        action, _ = expert.predict(obs)
        obs, _, term, trunc, _ = gym_env_n1.step(action)
        seen.add(expert.phase)
        if term or trunc:
            break

    assert Phase.APPROACH in seen
    # At minimum the expert should reach DESCEND or further within 150 steps
    later_phases = {p for p in seen if int(p) >= int(Phase.DESCEND)}
    assert later_phases, f"Expert never advanced past APPROACH; seen={seen}"


def test_goal_yaw_plumbed(gym_env_n1):
    """goal_yaw param is used by the place-yaw computation (smoke check that
    nothing crashes when goal_yaw is non-default)."""
    obs_map = build_obs_index_map(gym_env_n1)
    expert = PickPlaceExpertPolicy(obs_map, goal_yaw=np.pi / 4)

    obs, _ = gym_env_n1.reset()
    expert.reset()
    action, _ = expert.predict(obs)
    assert action.shape == (7,)
    assert np.all(np.isfinite(action))


def test_wrap_to_half_pi():
    tol = 1e-9
    assert abs(_wrap_to_half_pi(0.0)) < tol
    assert abs(_wrap_to_half_pi(np.pi / 4) - np.pi / 4) < tol
    # +π/2 stays at +π/2 (boundary case can be ±π/2; we accept either sign)
    assert abs(abs(_wrap_to_half_pi(np.pi / 2)) - np.pi / 2) < tol
    # 3π/4 should fold to -π/4 (closer rotation under 180° symmetry)
    assert abs(_wrap_to_half_pi(3 * np.pi / 4) - (-np.pi / 4)) < tol
    # -3π/4 folds to +π/4
    assert abs(_wrap_to_half_pi(-3 * np.pi / 4) - (np.pi / 4)) < tol
