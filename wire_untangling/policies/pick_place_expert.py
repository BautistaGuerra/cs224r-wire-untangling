"""
Scripted pick-and-place expert policy for single-stick reordering.

Uses proportional control over OSC_POSE deltas (7D: 3 pos + 3 ori + 1 gripper).
The OSC controller handles IK internally — this policy only outputs Cartesian
delta targets and gripper commands.

Gripper convention (PandaGripper): -1 = open, +1 = closed.

The Panda gripper fingers close along the EEF y-axis. To grasp a stick reliably,
the gripper yaw must be aligned with the stick's heading so the fingers close
perpendicular to the stick's long axis.

Exposes .predict(obs, deterministic=True) -> (action, None) for SB3 compatibility.
"""

from collections import OrderedDict
from enum import Enum, auto

import numpy as np


class Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    PLACE = auto()
    RELEASE = auto()
    RETREAT = auto()


GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


def build_obs_index_map(gym_env) -> dict[str, slice]:
    """Build a mapping from observable names to their slices in the flat obs vector.

    Iterates through the env's observables grouped by modality, matching
    GymWrapper's concatenation order (e.g. ["object-state", "robot0_proprio-state"]).
    This avoids hardcoded indices that break if Robosuite changes active sensors.

    Args:
        gym_env: A GymWrapper-wrapped Robosuite env (must have been reset at least once).

    Returns:
        Dict mapping observable names (e.g. "stick0_pos", "robot0_eef_pos") to slices.
    """
    env = gym_env.env

    modality_groups: dict[str, list[tuple[str, int]]] = OrderedDict()
    for name, obs_obj in env._observables.items():
        if not (obs_obj.is_enabled() and obs_obj.is_active()):
            continue
        group_key = f"{obs_obj.modality}-state"
        size = int(np.prod(obs_obj.obs.shape))
        if group_key not in modality_groups:
            modality_groups[group_key] = []
        modality_groups[group_key].append((name, size))

    index_map = {}
    offset = 0
    for key in gym_env.keys:
        if key not in modality_groups:
            continue
        for name, size in modality_groups[key]:
            index_map[name] = slice(offset, offset + size)
            offset += size

    return index_map


def _quat_to_yaw(q_xyzw):
    """Extract yaw (rotation around z) from quaternion in xyzw format."""
    x, y, z, w = q_xyzw
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PickPlaceExpertPolicy:
    """Scripted waypoint-based pick-and-place policy for a single stick.

    8-phase state machine:
        APPROACH  → move above the stick at lift_height, rotate gripper to align
        DESCEND   → lower to grasp height, maintain yaw alignment
        GRASP     → close gripper for grasp_steps
        LIFT      → lift to lift_height
        TRANSPORT → move horizontally to above goal
        PLACE     → lower to goal height
        RELEASE   → open gripper for release_steps
        RETREAT   → lift away, then idle

    During APPROACH and DESCEND, the gripper rotates around z to align its
    finger-closing axis perpendicular to the stick's long axis.

    Args:
        obs_index_map: Dict mapping observable names to slices in flat obs vector.
                       Obtain via build_obs_index_map(gym_env).
        lift_height: Z height for safe transport above the table (world frame).
        eef_z_offset: Vertical offset from stick center to EEF target during grasp.
        pos_threshold: XY distance threshold for phase transitions (metres).
        z_threshold: Z distance threshold for phase transitions (metres).
        yaw_threshold: Yaw alignment threshold for APPROACH→DESCEND transition (rad).
        grasp_steps: Steps to hold gripper closed before lifting.
        release_steps: Steps to hold gripper open after placing.
        gain: Proportional gain for position deltas.
        yaw_gain: Proportional gain for yaw correction.
    """

    def __init__(
        self,
        obs_index_map: dict[str, slice],
        lift_height: float = 0.95,
        eef_z_offset: float = 0.0,
        pos_threshold: float = 0.02,
        z_threshold: float = 0.02,
        yaw_threshold: float = 0.15,
        grasp_steps: int = 25,
        release_steps: int = 10,
        gain: float = 10.0,
        yaw_gain: float = 3.0,
    ):
        self.obs_map = obs_index_map
        self.lift_height = lift_height
        self.eef_z_offset = eef_z_offset
        self.pos_threshold = pos_threshold
        self.z_threshold = z_threshold
        self.yaw_threshold = yaw_threshold
        self.grasp_steps = grasp_steps
        self.release_steps = release_steps
        self.gain = gain
        self.yaw_gain = yaw_gain

        self._phase = Phase.APPROACH
        self._phase_step = 0

    def reset(self):
        """Reset internal state at the start of each episode."""
        self._phase = Phase.APPROACH
        self._phase_step = 0

    def predict(self, obs, deterministic=True):
        """Compute action from flat observation vector.

        Args:
            obs: Flat numpy array from GymWrapper.
            deterministic: Ignored (policy is always deterministic).

        Returns:
            (action, None): 7D action array and None (SB3 state compatibility).
        """
        obs = np.asarray(obs).flatten()
        eef_pos = obs[self.obs_map["robot0_eef_pos"]]
        eef_quat = obs[self.obs_map["robot0_eef_quat"]]
        stick_pos = obs[self.obs_map["stick0_pos"]]
        stick_quat = obs[self.obs_map["stick0_quat"]]
        goal_pos = obs[self.obs_map["goal0_pos"]]

        grasp_z = stick_pos[2] + self.eef_z_offset
        place_z = goal_pos[2] + self.eef_z_offset

        # Compute yaw alignment: target EEF yaw = stick yaw (mod pi)
        # so gripper fingers close perpendicular to the stick's long axis
        stick_yaw = _quat_to_yaw(stick_quat)
        eef_yaw = _quat_to_yaw(eef_quat)
        yaw_error = _wrap_angle(stick_yaw - eef_yaw)
        # Stick has 180° symmetry — take the shorter rotation
        if yaw_error > np.pi / 2:
            yaw_error -= np.pi
        elif yaw_error < -np.pi / 2:
            yaw_error += np.pi

        action = self._step_phase(eef_pos, stick_pos, goal_pos, grasp_z, place_z, yaw_error)
        return action, None

    def _step_phase(self, eef_pos, stick_pos, goal_pos, grasp_z, place_z, yaw_error):
        """Execute current phase and handle transitions."""
        phase = self._phase

        if phase == Phase.APPROACH:
            target = np.array([stick_pos[0], stick_pos[1], self.lift_height])
            action = self._move_to(eef_pos, target, GRIPPER_OPEN)
            # Rotate gripper to align with stick while approaching
            action[5] = np.clip(self.yaw_gain * yaw_error, -1.0, 1.0)
            xy_dist = np.linalg.norm(eef_pos[:2] - stick_pos[:2])
            z_dist = abs(eef_pos[2] - self.lift_height)
            yaw_aligned = abs(yaw_error) < self.yaw_threshold
            if xy_dist < self.pos_threshold and z_dist < self.z_threshold and yaw_aligned:
                self._advance(Phase.DESCEND)

        elif phase == Phase.DESCEND:
            target = np.array([stick_pos[0], stick_pos[1], grasp_z])
            action = self._move_to(eef_pos, target, GRIPPER_OPEN)
            # Maintain yaw alignment during descent
            action[5] = np.clip(self.yaw_gain * yaw_error, -1.0, 1.0)
            if abs(eef_pos[2] - grasp_z) < self.z_threshold:
                self._advance(Phase.GRASP)

        elif phase == Phase.GRASP:
            target = np.array([stick_pos[0], stick_pos[1], grasp_z])
            action = self._move_to(eef_pos, target, GRIPPER_CLOSE)
            if self._phase_step >= self.grasp_steps:
                self._advance(Phase.LIFT)

        elif phase == Phase.LIFT:
            target = np.array([eef_pos[0], eef_pos[1], self.lift_height])
            action = self._move_to(eef_pos, target, GRIPPER_CLOSE)
            if abs(eef_pos[2] - self.lift_height) < self.z_threshold:
                self._advance(Phase.TRANSPORT)

        elif phase == Phase.TRANSPORT:
            target = np.array([goal_pos[0], goal_pos[1], self.lift_height])
            action = self._move_to(eef_pos, target, GRIPPER_CLOSE)
            xy_dist = np.linalg.norm(eef_pos[:2] - goal_pos[:2])
            if xy_dist < self.pos_threshold:
                self._advance(Phase.PLACE)

        elif phase == Phase.PLACE:
            # Use goal z, not current stick z (stick is in the gripper)
            target = np.array([goal_pos[0], goal_pos[1], place_z])
            action = self._move_to(eef_pos, target, GRIPPER_CLOSE)
            if abs(eef_pos[2] - place_z) < self.z_threshold:
                self._advance(Phase.RELEASE)

        elif phase == Phase.RELEASE:
            target = np.array([goal_pos[0], goal_pos[1], place_z])
            action = self._move_to(eef_pos, target, GRIPPER_OPEN)
            if self._phase_step >= self.release_steps:
                self._advance(Phase.RETREAT)

        elif phase == Phase.RETREAT:
            target = np.array([goal_pos[0], goal_pos[1], self.lift_height])
            action = self._move_to(eef_pos, target, GRIPPER_OPEN)
            if abs(eef_pos[2] - self.lift_height) < self.z_threshold:
                action = np.zeros(7)
                action[6] = GRIPPER_OPEN

        self._phase_step += 1
        return action

    def _move_to(self, eef_pos, target_pos, gripper_action):
        """Proportional controller: compute 7D action toward target."""
        error = target_pos - eef_pos
        delta_pos = np.clip(self.gain * error, -1.0, 1.0)
        action = np.zeros(7)
        action[:3] = delta_pos
        action[6] = gripper_action
        return action

    def _advance(self, next_phase):
        self._phase = next_phase
        self._phase_step = 0
