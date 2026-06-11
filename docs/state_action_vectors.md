# State and Action Vector Composition

## Action Vector (7 dims, all in [-1, 1])

| Index | Name   | Description                          | Scaling by OSC controller     |
|-------|--------|--------------------------------------|-------------------------------|
| 0     | dx     | End-effector delta x position        | ±0.05 m/step                  |
| 1     | dy     | End-effector delta y position        | ±0.05 m/step                  |
| 2     | dz     | End-effector delta z position        | ±0.05 m/step                  |
| 3     | droll  | End-effector delta roll              | Typically set to 0 by expert  |
| 4     | dpitch | End-effector delta pitch             | Typically set to 0 by expert  |
| 5     | dyaw   | End-effector delta yaw               | ±0.5 rad/step                 |
| 6     | grip   | Gripper command (-1 open, +1 close)  | Binary open/close             |

All policy outputs use tanh activation to stay in [-1, 1]. The OSC_POSE controller handles denormalization and inverse kinematics.

## Observation Vector (flat, float32)

The observation is a concatenation of three groups. Total size depends on the number of sticks N.

### Robot Proprioception (50 dims)

| Dims | Name                   | Description                         |
|------|------------------------|-------------------------------------|
| 7    | robot0_joint_pos       | Arm joint angles (rad)              |
| 7    | robot0_joint_pos_cos   | cos(joint angles)                   |
| 7    | robot0_joint_pos_sin   | sin(joint angles)                   |
| 7    | robot0_joint_vel       | Joint velocities                    |
| 7    | robot0_joint_acc       | Joint accelerations                 |
| 3    | robot0_eef_pos         | End-effector xyz (meters)           |
| 4    | robot0_eef_quat        | End-effector quaternion (xyzw)      |
| 4    | robot0_eef_quat_site   | Alternative EEF quaternion          |
| 2    | robot0_gripper_qpos    | Gripper joint positions             |
| 2    | robot0_gripper_qvel    | Gripper joint velocities            |

### Object State (7 dims per stick)

| Dims | Name        | Description                           |
|------|-------------|---------------------------------------|
| 3    | stick_i_pos | Stick center position (meters)        |
| 4    | stick_i_quat| Stick orientation quaternion (xyzw)   |

### Goal Positions (3 dims per stick)

| Dims | Name       | Description                            |
|------|------------|----------------------------------------|
| 3    | goal_i_pos | Target position for stick i (constant) |

### Total Observation Size

| N (sticks) | Robot | Objects | Goals | Total |
|------------|-------|---------|-------|-------|
| 1          | 50    | 7       | 3     | 60    |
| 2          | 50    | 14      | 6     | 70    |
| 3          | 50    | 21      | 9     | 80    |

## Phase-Conditional Augmentation

When training with `--conditioning phase-active`, additional features are appended to the raw observation:

| Dims           | Name          | Description                                |
|----------------|---------------|--------------------------------------------|
| 8              | phase         | One-hot encoded phase (APPROACH..RETREAT)  |
| N              | active_stick  | One-hot encoded index of current stick     |

Augmented observation sizes: 69 (N=1), 80 (N=2), 91 (N=3).

## Normalization

Observations are z-score normalized per dimension before policy input:

```
normalized_obs = (obs - mean) / max(std, 1e-6)
```

Statistics (mean, std) are computed over the demonstration corpus and saved with the checkpoint. Actions are natively in [-1, 1] and are not normalized for behavior cloning.

## Expert Policy

The expert only reads 5 fields from the full observation (17 dims total):
`robot0_eef_pos`, `robot0_eef_quat`, `stick0_pos`, `stick0_quat`, `goal0_pos`.
It uses proportional control (position gain 10.0, yaw gain 3.0) clipped to [-1, 1].
