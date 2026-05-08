"""
Env-readiness battery for StickReorderEnv with orientation-aware success.

These tests are the precondition for trusting the demo-collection pipeline:
they verify the action/obs spaces, observable wiring, success/reward semantics
(including the new yaw term), and the yaw-error helper used by both the env
and the expert policy.

Run with:  pytest tests/test_env_readiness.py -v
"""

import numpy as np
import pytest
from robosuite.wrappers import GymWrapper

from wire_untangling.envs import StickReorderEnv
from wire_untangling.utils.transform import yaw_error_mod_pi


# ── Module-scoped env fixture (single stick — current scope of BC work) ──

@pytest.fixture(scope="module")
def env():
    e = StickReorderEnv(
        robots="Panda",
        num_sticks=1,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
    )
    yield e
    e.close()


def _set_stick_pose(env, stick_idx: int, pos, yaw: float):
    """Helper: teleport stick to pos with given yaw via joint qpos."""
    qw = np.cos(yaw / 2.0)
    qz = np.sin(yaw / 2.0)
    quat_wxyz = np.array([qw, 0.0, 0.0, qz])
    env.sim.data.set_joint_qpos(
        env.stick_objects[stick_idx].joints[0],
        np.concatenate([np.asarray(pos, dtype=float), quat_wxyz]),
    )
    env.sim.forward()


# ── Tests ──────────────────────────────────────────────────────────────

def test_action_spec_shape_and_bounds(env):
    low, high = env.action_spec
    assert low.shape == (7,), f"Expected 7-d action, got {low.shape}"
    assert high.shape == (7,)
    assert np.all(np.isfinite(low))
    assert np.all(np.isfinite(high))
    assert np.all(low < high)


def test_gym_wrapper_obs_dim(env):
    """For N=1: 50-d proprio + 10-d object (stick_pos 3 + stick_quat 4 + goal_pos 3) = 60."""
    gym_env = GymWrapper(env)
    obs, _ = gym_env.reset()
    assert obs.shape == (60,), f"Expected (60,) flat obs for N=1, got {obs.shape}"
    assert obs.ndim == 1
    gym_env.close()


def test_observable_wiring(env):
    env.reset()
    low, _ = env.action_spec
    obs, _, _, _ = env.step(np.zeros_like(low))

    # Stick obs reflects sim state
    body_pos = env.sim.data.body_xpos[env.stick_body_ids[0]]
    np.testing.assert_array_equal(obs["stick0_pos"], body_pos)

    # Goal pos is constant across steps
    g0 = obs["goal0_pos"].copy()
    obs2, _, _, _ = env.step(np.zeros_like(low))
    np.testing.assert_array_equal(obs2["goal0_pos"], g0)


def test_check_success_with_orientation(env):
    env.reset()

    # All sticks at goal with goal_yaw → success
    for i in range(env.num_sticks):
        _set_stick_pose(env, i, env._goal_positions[i], env.goal_yaw)
    assert env._check_success(), "Should succeed at goal pose with goal_yaw"

    # Far away → fail
    _set_stick_pose(env, 0, env._goal_positions[0] + np.array([0.5, 0.0, 0.0]), env.goal_yaw)
    assert not env._check_success(), "Should fail when stick is far"

    # At goal but yaw 20° off → fail (threshold is 10°)
    _set_stick_pose(env, 0, env._goal_positions[0], env.goal_yaw + np.deg2rad(20))
    assert not env._check_success(), "Should fail when yaw error exceeds threshold"

    # At goal with yaw 5° off → success
    _set_stick_pose(env, 0, env._goal_positions[0], env.goal_yaw + np.deg2rad(5))
    assert env._check_success(), "Should succeed inside yaw threshold"

    # Mod-π symmetry: yaw = goal_yaw + π should also count
    _set_stick_pose(env, 0, env._goal_positions[0], env.goal_yaw + np.pi)
    assert env._check_success(), "Should succeed under 180° symmetry"


def test_reward_components(env):
    env.reset()

    # At goal with correct yaw: reward ≈ +1 (sparse) and 0 dense
    for i in range(env.num_sticks):
        _set_stick_pose(env, i, env._goal_positions[i], env.goal_yaw)
    r_at_goal = env.reward()
    assert r_at_goal >= 0.9, f"At-goal reward should be near 1.0, got {r_at_goal}"

    # Far away with bad yaw: strongly negative
    for i in range(env.num_sticks):
        _set_stick_pose(env, i, env._goal_positions[i] + np.array([0.5, 0.0, 0.0]), np.pi / 2)
    r_far = env.reward()
    assert r_far < -0.4, f"Far-from-goal reward should be negative, got {r_far}"

    # At goal but with wrong yaw: dense should subtract lambda_rot * yaw_err
    for i in range(env.num_sticks):
        _set_stick_pose(env, i, env._goal_positions[i], env.goal_yaw + np.deg2rad(45))
    r_bad_yaw = env.reward()
    assert r_bad_yaw < r_at_goal, "Wrong-yaw reward must be strictly worse than at-goal"


def test_placement_non_trivial(env):
    """Resets must not start with the stick already at its goal (would trivialize the task)."""
    n_resets = 30
    trivial = 0
    for _ in range(n_resets):
        env.reset()
        if env._check_success():
            trivial += 1
    assert trivial == 0, f"{trivial}/{n_resets} resets started at goal"


def test_post_action_emits_is_success(env):
    env.reset()
    low, _ = env.action_spec
    _, _, _, info = env.step(np.zeros_like(low))
    assert "is_success" in info
    assert isinstance(info["is_success"], (bool, np.bool_))


def test_terminate_on_success_toggle(env):
    """Env-level early termination should be configurable.

    Demo collection disables this and uses its own consecutive-success hold.
    """
    original = env.terminate_on_success
    low, _ = env.action_spec
    try:
        env.terminate_on_success = False
        env.reset()
        _set_stick_pose(env, 0, env._goal_positions[0], env.goal_yaw)
        _, _, done, info = env.step(np.zeros_like(low))
        assert info["is_success"]
        assert not done

        env.terminate_on_success = True
        env.reset()
        _set_stick_pose(env, 0, env._goal_positions[0], env.goal_yaw)
        _, _, done, info = env.step(np.zeros_like(low))
        assert info["is_success"]
        assert done
    finally:
        env.terminate_on_success = original


def test_yaw_error_mod_pi_helper():
    tol = 1e-6
    assert abs(yaw_error_mod_pi(0.0, 0.0)) < tol,                       "yaw=0 → 0"
    assert abs(yaw_error_mod_pi(np.pi, 0.0)) < tol,                     "yaw=π → 0 (symmetric)"
    assert abs(yaw_error_mod_pi(np.pi / 2, 0.0) - np.pi / 2) < tol,     "yaw=π/2 → π/2"
    assert abs(yaw_error_mod_pi(np.pi / 4, 0.0) - np.pi / 4) < tol,     "yaw=π/4 → π/4"
    assert abs(yaw_error_mod_pi(3 * np.pi / 4, 0.0) - np.pi / 4) < tol, "yaw=3π/4 → π/4"
    # Negative yaws fold the same way
    assert abs(yaw_error_mod_pi(-np.pi / 4, 0.0) - np.pi / 4) < tol,    "yaw=-π/4 → π/4"
