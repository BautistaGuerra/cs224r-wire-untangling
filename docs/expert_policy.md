# Expert Policy and Demonstration Collection

A scripted pick-and-place expert policy generates demonstrations for training a behavior cloning policy. The expert operates on a single stick (`num_sticks=1`) and achieves ~92% success rate across randomized initial configurations.

## Quick Start

```bash
# Visualize the expert in the MuJoCo viewer
python scripts/play_env.py --render --expert

# Headless success rate check
python scripts/play_env.py --expert --episodes 100

# Collect 200 successful demonstrations
python scripts/collect_demos.py --num-demos 200 --output data/demos.hdf5

# Collect with live rendering (slower, useful for debugging)
python scripts/collect_demos.py --num-demos 10 --output /tmp/test.hdf5 --render
```

## Demonstration Collection

### Output Format

Demonstrations are stored in HDF5. Only successful episodes (stick placed within `success_threshold` of the goal) are kept. Failed attempts are discarded automatically.

```
demos.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs        (T, 60)    float32   — flat observation vector
│   │   ├── actions    (T, 7)     float32   — 7D OSC_POSE + gripper actions
│   │   ├── rewards    (T,)       float32   — per-step reward
│   │   ├── dones      (T,)       bool      — episode termination flag
│   │   └── next_obs   (T, 60)    float32   — observation after action
│   ├── demo_1/ ...
├── attrs:
│   ├── num_demos      int        — number of successful episodes
│   ├── obs_dim        int        — observation dimensionality (60)
│   ├── total_samples  int        — total transitions across all demos
│   └── env_config     str (JSON) — environment parameters used
```

Each episode runs for the full horizon (T=500 steps at 20 Hz = 25 seconds of sim time).

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/stick_reorder.yaml` | Environment config file |
| `--num-demos` | 200 | Number of successful demos to collect |
| `--output` | `data/demos.hdf5` | Output HDF5 path |
| `--render` | off | Show MuJoCo viewer during collection |

The collector attempts up to `3x num_demos` episodes to reach the target count. At ~92% expert success rate, this headroom is sufficient.

### Loading Demonstrations

```python
import h5py
import numpy as np

with h5py.File("data/demos.hdf5", "r") as f:
    print(f"Demos: {f.attrs['num_demos']}, Samples: {f.attrs['total_samples']}")

    # Load a single episode
    obs = f["data/demo_0/obs"][:]          # (500, 60)
    actions = f["data/demo_0/actions"][:]  # (500, 7)

    # Load all episodes into flat arrays for BC training
    all_obs = np.concatenate([f[f"data/demo_{i}/obs"][:] for i in range(f.attrs["num_demos"])])
    all_act = np.concatenate([f[f"data/demo_{i}/actions"][:] for i in range(f.attrs["num_demos"])])
```

## Expert Policy Design

### Overview

The expert is a scripted state machine that outputs 7D actions for the Panda robot's OSC_POSE controller. The OSC controller handles inverse kinematics internally — the policy only specifies Cartesian delta targets and gripper commands.

**Source:** `wire_untangling/policies/pick_place_expert.py`

### Action Space

The policy outputs a 7D action vector each timestep:

| Dims | Range | Meaning |
|------|-------|---------|
| 0-2 | [-1, 1] | Delta end-effector position (dx, dy, dz), mapped to [-0.05, 0.05] m/step |
| 3-4 | [-1, 1] | Delta end-effector orientation (droll, dpitch), set to 0 |
| 5 | [-1, 1] | Delta yaw (dyaw), mapped to [-0.5, 0.5] rad/step. Used for gripper alignment |
| 6 | [-1, 1] | Gripper: -1 = open, +1 = closed |

### State Machine

The policy executes an 8-phase sequence:

```
APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE → RELEASE → RETREAT
```

| Phase | Target Position | Gripper | Transition Condition |
|-------|----------------|---------|---------------------|
| APPROACH | (stick_x, stick_y, 0.95) | open | XY near stick, Z near 0.95, yaw aligned |
| DESCEND | (stick_x, stick_y, stick_z) | open | Z near stick_z |
| GRASP | hold at stick | closed | After 25 steps (~1.25s) |
| LIFT | (eef_x, eef_y, 0.95) | closed | Z near 0.95 |
| TRANSPORT | (goal_x, goal_y, 0.95) | closed | XY near goal |
| PLACE | (goal_x, goal_y, goal_z) | closed | Z near goal_z |
| RELEASE | hold at goal | open | After 10 steps (~0.5s) |
| RETREAT | (goal_x, goal_y, 0.95) | open | Z near 0.95, then idle |

### Proportional Controller

Each phase computes the action using proportional control:

```
delta = clip(gain * (target - eef_pos), -1, 1)
```

With `gain=10.0`, this creates a natural deceleration profile: far from target, the action saturates at full speed (0.05 m/step); close to target, the action scales down proportionally (e.g., at 1 cm error: `10 * 0.01 = 0.1`, commanding 0.005 m/step).

Phase transitions trigger when the EEF is within `pos_threshold` (20 mm XY) or `z_threshold` (20 mm Z) of the target.

### Gripper Yaw Alignment

The Panda gripper has two parallel fingers that close along the EEF y-axis. A stick lying on the table has a random z-rotation. If the fingers are parallel to the stick's long axis, they cannot close around it.

During APPROACH and DESCEND, the policy rotates the gripper around z to align with the stick's heading:

1. Extract stick yaw from `stick0_quat` (quaternion → yaw angle)
2. Extract EEF yaw from `robot0_eef_quat`
3. Compute yaw error, wrapped to [-pi/2, pi/2] (stick has 180 deg symmetry)
4. Command `action[5] = clip(yaw_gain * yaw_error, -1, 1)`

This ensures the finger-closing axis is perpendicular to the stick regardless of its orientation. Without yaw alignment, the success rate drops from ~92% to ~40%.

### Observation Parsing

The policy reads from the flat observation vector produced by GymWrapper. Rather than hardcoding array indices (which break if Robosuite changes active sensors), `build_obs_index_map(gym_env)` dynamically computes a `dict[str, slice]` by inspecting the environment's registered observables and their modalities.

For `num_sticks=1`, the flat vector is 60-dimensional:

| Segment | Fields | Dims |
|---------|--------|------|
| object-state (10) | `stick0_pos`(3), `stick0_quat`(4), `goal0_pos`(3) | 0-9 |
| robot0_proprio-state (50) | joint_pos(7), joint_pos_cos(7), joint_pos_sin(7), joint_vel(7), joint_acc(7), eef_pos(3), eef_quat(4), eef_quat_site(4), gripper_qpos(2), gripper_qvel(2) | 10-59 |

The policy reads 5 fields: `robot0_eef_pos`, `robot0_eef_quat`, `stick0_pos`, `stick0_quat`, `goal0_pos`.

### API

The policy exposes the SB3-compatible `.predict()` interface:

```python
from wire_untangling.policies import PickPlaceExpertPolicy, build_obs_index_map

obs_map = build_obs_index_map(gym_env)
expert = PickPlaceExpertPolicy(obs_map)

# Each episode:
expert.reset()
obs, _ = gym_env.reset()
while not done:
    action, _ = expert.predict(obs)
    obs, reward, terminated, truncated, info = gym_env.step(action)
    done = terminated or truncated
```

### Tunable Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lift_height` | 0.95 | Transport altitude above table (m). Higher is safer but slower |
| `eef_z_offset` | 0.0 | Vertical offset for grasp target relative to stick center (m) |
| `pos_threshold` | 0.02 | XY convergence threshold for phase transitions (m) |
| `z_threshold` | 0.02 | Z convergence threshold for phase transitions (m) |
| `yaw_threshold` | 0.15 | Yaw alignment required before descending (rad, ~8.6 deg) |
| `grasp_steps` | 25 | Hold-closed duration before lifting (~1.25s at 20 Hz) |
| `release_steps` | 10 | Hold-open duration after placing (~0.5s at 20 Hz) |
| `gain` | 10.0 | Position proportional gain |
| `yaw_gain` | 3.0 | Yaw proportional gain |

### Limitations

- Single stick only (`num_sticks=1`). Multi-stick sequencing would require planning which stick to move first.
- ~8% failure rate on random configurations, primarily when sticks spawn at extreme table positions near the Panda's workspace boundary.
- Top-down grasps only. The gripper orientation (pointing down) is fixed except for z-rotation.
- No re-grasp or error recovery. If the stick slips during transport, the episode fails.
