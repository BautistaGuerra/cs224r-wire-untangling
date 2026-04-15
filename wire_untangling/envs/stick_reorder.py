"""
Stick reorder environment.

The robot must pick and place N rigid sticks so that each one reaches its
assigned goal position. Goal positions form a parallel row on the table,
evenly spaced along the y-axis.

Observation space (flat dict, via Robosuite observables):
    robot0_eef_pos          (3,)   end-effector position
    robot0_eef_quat         (4,)   end-effector orientation (xyzw)
    robot0_gripper_qpos     (2,)   gripper joint positions
    robot0_joint_pos        (7,)   arm joint positions
    robot0_joint_vel        (7,)   arm joint velocities
    stickN_pos              (3,)   per-stick position
    stickN_quat             (4,)   per-stick orientation (xyzw)
    goalN_pos               (3,)   per-stick goal position (constant)

Action space:
    Controlled by the robot controller (default: OSC_POSE, 6-DoF + gripper).

Reward:
    Dense (reward_shaping=True):  -sum of distances from sticks to goals
    Sparse bonus:                 +1.0 when all sticks within success_threshold
"""

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

from wire_untangling.models.objects import StickObject


class StickReorderEnv(SingleArmEnv):
    """
    Tabletop stick-reordering task.

    Args:
        robots: Robot specification passed to SingleArmEnv (e.g. "Panda").
        num_sticks: Number of sticks to reorder.
        stick_length: Full stick length in metres.
        stick_radius: Stick cross-section half-extent in metres.
        goal_spacing: Y-axis spacing between goal positions in metres.
        success_threshold: Per-stick distance tolerance for task success (m).
        reward_shaping: If True, add dense shaped reward on top of sparse bonus.
        table_full_size: (x, y, z) full size of the table surface.
        table_friction: MuJoCo friction parameters for the table.
        **kwargs: Forwarded to SingleArmEnv (horizon, control_freq, renderer, …).
    """

    def __init__(
        self,
        robots,
        num_sticks: int = 3,
        stick_length: float = 0.18,
        stick_radius: float = 0.012,
        goal_spacing: float = 0.06,
        success_threshold: float = 0.03,
        reward_shaping: bool = True,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 0.005, 0.0001),
        **kwargs,
    ):
        self.num_sticks = num_sticks
        self.stick_length = stick_length
        self.stick_radius = stick_radius
        self.goal_spacing = goal_spacing
        self.success_threshold = success_threshold
        self.reward_shaping = reward_shaping
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array([0.0, 0.0, 0.8])

        # Computed before super().__init__ so they're available during setup.
        self._goal_positions = self._compute_goal_positions()

        self.stick_objects: list[StickObject] = []
        self.stick_body_ids: list[int] = []

        super().__init__(robots=robots, **kwargs)

    # ------------------------------------------------------------------
    # Goal geometry
    # ------------------------------------------------------------------

    def _compute_goal_positions(self) -> np.ndarray:
        """Return (num_sticks, 3) array of goal xyz positions on the table."""
        total_span = (self.num_sticks - 1) * self.goal_spacing
        table_surface_z = self.table_offset[2] + self.table_full_size[2] / 2
        return np.array(
            [
                [
                    0.0,
                    -total_span / 2 + i * self.goal_spacing,
                    table_surface_z + self.stick_radius + 0.001,
                ]
                for i in range(self.num_sticks)
            ]
        )

    # ------------------------------------------------------------------
    # Robosuite lifecycle
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        self.mujoco_arena.set_origin([0, 0, 0])

        self.stick_objects = [
            StickObject(
                name=f"stick{i}",
                length=self.stick_length,
                radius=self.stick_radius,
                color_idx=i,
            )
            for i in range(self.num_sticks)
        ]

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.stick_objects,
            x_range=[-0.25, 0.25],
            y_range=[-0.25, 0.25],
            rotation=(-np.pi, np.pi),
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        )

        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.stick_objects,
        )

    def _setup_references(self):
        super()._setup_references()
        self.stick_body_ids = [
            self.sim.model.body_name2id(stick.root_body)
            for stick in self.stick_objects
        ]

    def _setup_observables(self):
        observables = super()._setup_observables()

        for i in range(self.num_sticks):
            # Default-argument capture avoids the loop-variable closure bug.
            @sensor(modality="object")
            def stick_pos(obs_cache, idx=i):
                return np.array(self.sim.data.body_xpos[self.stick_body_ids[idx]])

            @sensor(modality="object")
            def stick_quat(obs_cache, idx=i):
                return T.convert_quat(
                    self.sim.data.body_xquat[self.stick_body_ids[idx]], to="xyzw"
                )

            @sensor(modality="object")
            def goal_pos(obs_cache, idx=i):
                return self._goal_positions[idx].copy()

            observables[f"stick{i}_pos"] = Observable(
                name=f"stick{i}_pos",
                sensor=stick_pos,
                sampling_rate=self.control_freq,
            )
            observables[f"stick{i}_quat"] = Observable(
                name=f"stick{i}_quat",
                sensor=stick_quat,
                sampling_rate=self.control_freq,
            )
            observables[f"goal{i}_pos"] = Observable(
                name=f"goal{i}_pos",
                sensor=goal_pos,
                sampling_rate=self.control_freq,
            )

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self.placement_initializer.reset()
        object_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
            )

    # ------------------------------------------------------------------
    # Reward & termination
    # ------------------------------------------------------------------

    def reward(self, action=None) -> float:
        reward = 0.0

        if self.reward_shaping:
            for i, body_id in enumerate(self.stick_body_ids):
                pos = self.sim.data.body_xpos[body_id]
                dist = np.linalg.norm(pos - self._goal_positions[i])
                reward -= dist

        if self._check_success():
            reward += 1.0

        return reward

    def _check_success(self) -> bool:
        for i, body_id in enumerate(self.stick_body_ids):
            pos = self.sim.data.body_xpos[body_id]
            if np.linalg.norm(pos - self._goal_positions[i]) > self.success_threshold:
                return False
        return True
