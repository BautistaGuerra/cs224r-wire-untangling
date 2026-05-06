"""
Stick reorder environment (Robosuite / MuJoCo).

The robot must pick and place N rigid sticks so that each one reaches its
assigned goal position. Goal positions form a parallel row on the table,
evenly spaced along the y-axis.

Architecture:
    ManipulationEnv (Robosuite)
      └── ManipulationTask (self.model)
            ├── TableArena      — the table, positioned at table_offset in world frame
            ├── Panda robot     — positioned relative to the table via base_xpos_offset
            └── StickObject[]   — rigid bodies placed on the table surface

Robosuite lifecycle (called in this order by super().__init__):
    _load_model()          → Modeling API: build MuJoCo XML from arena + robot + objects
    (MuJoCo compiles XML into self.sim)
    _setup_references()    → Simulation API: cache body IDs from compiled sim
    _setup_observables()   → Simulation API: register sensor functions for observations
    _reset_internal()      → Simulation API: randomize object placements each episode

Observation space (flat vector, ~53 dims, concatenated by GymWrapper):
    Robot proprioception (23 dims):
        robot0_eef_pos          (3,)   end-effector position
        robot0_eef_quat         (4,)   end-effector orientation (xyzw)
        robot0_gripper_qpos     (2,)   gripper joint positions
        robot0_joint_pos        (7,)   arm joint positions
        robot0_joint_vel        (7,)   arm joint velocities
    Object state — ground truth from simulator (7 * num_sticks dims):
        stickN_pos              (3,)   per-stick center position
        stickN_quat             (4,)   per-stick orientation (xyzw)
    Goal positions — constant within episode (3 * num_sticks dims):
        goalN_pos               (3,)   per-stick goal position

    No images: has_offscreen_renderer=False, use_camera_obs=False.

Action space (7 dims, from OSC_POSE controller):
    End-effector delta pose (dx, dy, dz, droll, dpitch, dyaw) + gripper open/close.

Reward:
    Dense (reward_shaping=True):  -sum_i (dist_i + lambda_rot * yaw_err_i)
    Sparse bonus:                 +1.0 when every stick is within success_threshold
                                  AND yaw_err <= orientation_threshold.

    Yaw error is reduced mod π (the stick is rotationally symmetric about its
    long axis, so yaw and yaw+π are equivalent).
"""

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from wire_untangling.models.objects import StickObject
from wire_untangling.utils.transform import yaw_from_quat_wxyz, yaw_error_mod_pi


class StickReorderEnv(ManipulationEnv):
    """
    Tabletop stick-reordering task.

    Args:
        robots: Robot specification passed to ManipulationEnv (e.g. "Panda").
        num_sticks: Number of sticks to reorder.
        stick_length: Full stick length in metres.
        stick_radius: Stick cross-section half-extent in metres.
        goal_spacing: Y-axis spacing between goal positions in metres.
        success_threshold: Per-stick distance tolerance for task success (m).
        orientation_threshold: Per-stick yaw tolerance for task success (rad).
            Yaw is reduced mod π (stick has 180° symmetry).
        lambda_rot: Weight of the yaw-error term in the dense reward.
        goal_yaw: Target yaw for every stick at its goal (rad). Default 0
            aligns sticks with the world x-axis.
        reward_shaping: If True, add dense shaped reward on top of sparse bonus.
        table_full_size: (x, y, z) full size of the table surface.
        table_friction: MuJoCo friction parameters for the table.
        **kwargs: Forwarded to ManipulationEnv (horizon, control_freq, renderer, …).
    """

    def __init__(
        self,
        robots,
        num_sticks: int = 3,
        # 1.5 cm-thick × 20 cm-long sticks: closer to the wire-untangling target
        # than the previous 18 cm × 1.2 cm rod.
        stick_length: float = 0.20,
        stick_radius: float = 0.0075,
        goal_spacing: float = 0.06,
        success_threshold: float = 0.03,
        orientation_threshold: float = np.deg2rad(10.0),
        lambda_rot: float = 0.1,
        goal_yaw: float = 0.0,
        # Initial placement: a narrow band close to the robot but offset from
        # the goal at x=0. This avoids trivial N=1 resets that already satisfy
        # the 3 cm success threshold.
        init_x_range=(-0.12, -0.06),
        init_y_range=(-0.15, 0.15),
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
        self.orientation_threshold = orientation_threshold
        self.lambda_rot = lambda_rot
        self.goal_yaw = goal_yaw
        self.init_x_range = tuple(init_x_range)
        self.init_y_range = tuple(init_y_range)
        self.reward_shaping = reward_shaping
        self.reward_scale = 1.0   # required by GymWrapper
        self.use_object_obs = True  # always include stick positions in obs
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        # Table TOP in world frame: (0, 0, 0.8). (In this robosuite version
        # table_offset == arena.table_top_abs, despite what older code/comments
        # may suggest.)
        self.table_offset = np.array([0.0, 0.0, 0.8])

        # Computed before super().__init__ so they're available during setup.
        self._goal_positions = self._compute_goal_positions()

        self.stick_objects: list[StickObject] = []
        self.stick_body_ids: list[int] = []

        # Triggers the lifecycle: _load_model → _setup_references → _setup_observables
        super().__init__(robots=robots, **kwargs)


    # ── Goal geometry ──────────────────────────────────────────────────

    def _compute_goal_positions(self) -> np.ndarray:
        """Return (num_sticks, 3) array of goal xyz positions on the table.

        In this robosuite version `table_offset` is the table TOP (matches
        ``arena.table_top_abs``), not the center. A stick at rest therefore
        sits at z = table_offset[2] + stick_radius, which is what we use as
        the goal_z. The previous implementation added ``table_full_size[2]/2``
        on top, so goal_z floated ~2.5 cm above any reachable rest pose and
        every demo grazed the 3 cm success_threshold on z alone.
        """
        total_span = (self.num_sticks - 1) * self.goal_spacing
        table_top_z = self.table_offset[2]
        return np.array(
            [
                [
                    0.0,
                    -total_span / 2 + i * self.goal_spacing,
                    # Stick center rests one radius above the table top.
                    table_top_z + self.stick_radius,
                ]
                for i in range(self.num_sticks)
            ]
        )


    # ── Robosuite lifecycle: Modeling API (XML generation) ─────────────

    def _load_model(self):
        """Build the MuJoCo XML scene: arena, robot, and stick objects.
        Nothing is simulated yet — this is purely declarative."""
        super()._load_model()

        # Position robot base relative to table (Robosuite convention:
        # each robot model provides offsets for known arena types)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        self.mujoco_arena.set_origin([0, 0, 0])

        # Create stick objects (thin BoxObject wrappers that generate rigid body XML)
        self.stick_objects = [
            StickObject(
                name=f"stick{i}",
                length=self.stick_length,
                radius=self.stick_radius,
                color_idx=i,
            )
            for i in range(self.num_sticks)
        ]

        # Placement sampler for random initial positions on the table.
        # ensure_valid_placement=True prevents sticks from overlapping at spawn.
        # Ranges are narrowed via init_x_range / init_y_range to keep sticks
        # near the robot (avoids the wrist's long-way-around rotation when
        # transporting from the far edge of the workspace).
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.stick_objects,
            x_range=list(self.init_x_range),
            y_range=list(self.init_y_range),
            rotation=(-np.pi, np.pi),
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )

        # Assemble arena + robot + objects into a single MuJoCo XML model.
        # After this, super().__init__ compiles it into self.sim.
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.stick_objects,
        )

    # ── Robosuite lifecycle: Simulation API (compiled sim) ─────────────

    def _setup_references(self):
        """translates object names into MuJoCo array indices.
        These IDs are used to index into sim.data.body_xpos / body_xquat."""
        super()._setup_references()
        self.stick_body_ids = [
            self.sim.model.body_name2id(stick.root_body)
            for stick in self.stick_objects
        ]

    def _setup_observables(self):
        """Register per-stick position, orientation, and goal sensors.
        Each Observable wraps a @sensor function that reads from the live sim.
        These are concatenated into the flat obs vector by GymWrapper."""
        observables = super()._setup_observables()

        for i in range(self.num_sticks):
            # Default-argument capture (idx=i) avoids the loop-variable closure bug:
            # without it, all closures would share the final value of i.
            @sensor(modality="object")
            def stick_pos(obs_cache, idx=i):
                return np.array(self.sim.data.body_xpos[self.stick_body_ids[idx]])

            @sensor(modality="object")
            def stick_quat(obs_cache, idx=i):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[self.stick_body_ids[idx]]), to="xyzw"
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
        """Randomize stick positions on the table at the start of each episode.
        Writes 7D joint state (xyz + quaternion) directly into MuJoCo sim."""
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )


    # ── Reward and success ─────────────────────────────────────────────

    def _yaw_error(self, body_id: int) -> float:
        """Yaw error for a stick body, reduced to [0, π/2] by stick symmetry."""
        q = self.sim.data.body_xquat[body_id]  # wxyz (MuJoCo)
        yaw = yaw_from_quat_wxyz(q)
        return yaw_error_mod_pi(yaw, self.goal_yaw)

    def reward(self, action=None) -> float:
        """Compute reward for current state.
        Dense: -sum_i (dist_i + lambda_rot * yaw_err_i).
        Sparse: +1.0 bonus when all sticks are within success_threshold of goals
        AND yaw_err <= orientation_threshold."""
        reward = 0.0

        if self.reward_shaping:
            for i, body_id in enumerate(self.stick_body_ids):
                pos = self.sim.data.body_xpos[body_id]
                dist = np.linalg.norm(pos - self._goal_positions[i])
                yaw_err = self._yaw_error(body_id)
                reward -= dist + self.lambda_rot * yaw_err

        if self._check_success():
            reward += 1.0

        return reward

    def _post_action(self, action):
        """Inject is_success flag into the info dict after each step."""
        reward, done, info = super()._post_action(action)
        info["is_success"] = self._check_success()
        return reward, done, info

    def _check_success(self) -> bool:
        """All sticks must be within success_threshold of their goal positions
        AND within orientation_threshold of goal_yaw (mod π)."""
        for i, body_id in enumerate(self.stick_body_ids):
            pos = self.sim.data.body_xpos[body_id]
            if np.linalg.norm(pos - self._goal_positions[i]) > self.success_threshold:
                return False
            if self._yaw_error(body_id) > self.orientation_threshold:
                return False
        return True
