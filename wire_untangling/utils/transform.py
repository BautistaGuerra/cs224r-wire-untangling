"""
Quaternion / yaw helpers shared by the env and the expert policy.

The stick is rotationally symmetric about its long axis, so yaw_error_mod_pi
treats yaw and yaw+pi as equivalent and returns a value in [0, pi/2].
"""

import numpy as np


def yaw_from_quat_wxyz(q) -> float:
    """Extract yaw from a quaternion in (w, x, y, z) order (MuJoCo convention)."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_from_quat_xyzw(q) -> float:
    """Extract yaw from a quaternion in (x, y, z, w) order (Robosuite obs convention)."""
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_error_mod_pi(yaw: float, goal_yaw: float = 0.0) -> float:
    """Absolute yaw error reduced to [0, pi/2] by 180-degree stick symmetry."""
    diff = (yaw - goal_yaw) % np.pi
    if diff > np.pi / 2:
        diff = np.pi - diff
    return float(diff)


def signed_yaw_error_mod_pi(yaw: float, goal_yaw: float = 0.0) -> float:
    """Signed yaw error in [-pi/2, pi/2], for driving a controller toward goal_yaw."""
    err = (yaw - goal_yaw + np.pi / 2) % np.pi - np.pi / 2
    return float(err)
