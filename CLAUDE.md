# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS224R (Stanford Deep RL) final project: training a Franka Panda robot arm to reorder rigid sticks (simplified wires) into goal positions using deep reinforcement learning, simulated in Robosuite/MuJoCo.

## Environment Setup

```bash
mamba activate cs224r-wire-untangling
# or: conda activate cs224r-wire-untangling
```

Environment is defined in `environment.yml` (Python 3.11, PyTorch + CUDA 12.9 for Blackwell GPUs). Install with `mamba env create -f environment.yml`. The project itself is installed as an editable package (`-e .`).

## Common Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_stick_reorder.py::test_reward_is_finite -v

# Sanity-check environment (headless)
python scripts/play_env.py

# Render with MuJoCo viewer (Linux: python, macOS: mjpython)
python scripts/play_env.py --render

# Visualize a trained policy
python scripts/play_env.py --render --checkpoint checkpoints/best/best_model.zip

# Train locally (SAC baseline, reads config from YAML)
python scripts/train.py --no-wandb
python scripts/train.py --timesteps 50000 --seed 0 --no-wandb

# Evaluate a checkpoint
python scripts/eval.py --checkpoint checkpoints/best/best_model.zip --episodes 50

# Scripted expert policy (single stick)
python scripts/play_env.py --render --expert        # visualize
python scripts/play_env.py --expert --episodes 100  # headless success rate

# Collect demonstrations for behavior cloning
python scripts/collect_demos.py --num-demos 200 --output data/demos.hdf5
python scripts/collect_demos.py --num-demos 10 --output /tmp/test.hdf5 --render

# Modal GPU training
modal run modal_train.py --total-timesteps 1000   # smoke test
modal run modal_train.py                           # full run
```

## Architecture

### Framework Stack

- **MuJoCo**: physics engine (contacts, rigid bodies, solver)
- **Robosuite**: robotics layer on top of MuJoCo (robot models, controllers, task structure, sensors)
- **Gymnasium**: standard RL interface (`reset`/`step`/`observation_space`/`action_space`)
- **GymWrapper**: bridges Robosuite's dict-based API to Gymnasium's flat-vector interface for SB3

### Robosuite Environment Lifecycle

`StickReorderEnv` extends `ManipulationEnv`. The lifecycle is triggered by `super().__init__()`:

1. **`_load_model()`** — Modeling API (XML generation): creates `TableArena`, `StickObject` instances, `Panda` robot model, assembles them into a `ManipulationTask`. Nothing is simulated yet.
2. *(MuJoCo compiles the XML into `self.sim`)*
3. **`_setup_references()`** — Simulation API: translates object names to MuJoCo body IDs (`body_name2id`) for indexing into `sim.data` arrays. Internal use only (reward, reset).
4. **`_setup_observables()`** — Simulation API: registers `@sensor`-decorated functions as `Observable` objects. These produce the agent-facing observation vector.
5. **`_reset_internal()`** — Called each `env.reset()`. Randomizes stick placements via `UniformRandomSampler`, writes 7D joint state (xyz + quaternion) into `sim.data`.

### Simulation State Access

`self.sim` is the compiled MuJoCo simulation:
- `self.sim.model` — static structure, used for lookups like `body_name2id()`
- `self.sim.data` — live state updated every `sim.step()`: `body_xpos` (positions), `body_xquat` (orientations), `set_joint_qpos` (write joint state)

### Observation / Action Spaces

Observations (~53-dim flat float32 vector): robot proprioception (eef pose, gripper, joint pos/vel — 23 dims) + ground-truth object state from simulator (stick pos + quat — 7 per stick) + goal positions (3 per stick). State-based only, no images.

Actions (7-dim): OSC_POSE controller outputs delta end-effector pose (dx, dy, dz, droll, dpitch, dyaw) + gripper open/close. The controller handles IK internally.

### Training Pipeline

`scripts/train.py` → `make_gym_env()` wraps `StickReorderEnv` in `GymWrapper` + `Monitor` → SB3's `SAC` auto-wraps in `DummyVecEnv(batch_size=1)` → `model.learn()` runs the collect-store-sample-update loop. SAC is off-policy (replay buffer), so single-env is fine.

### Configuration

`configs/stick_reorder.yaml` is the single source of truth for env params, training hyperparams, and Modal GPU settings. CLI args override config values.

### Expert Policy

`wire_untangling/policies/pick_place_expert.py` implements a scripted pick-and-place policy for a single stick. 8-phase state machine (APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → PLACE → RELEASE → RETREAT) using proportional control over OSC_POSE deltas. Aligns gripper yaw with stick heading before grasping so fingers close perpendicular to the stick's long axis. Exposes `.predict(obs)` for SB3 compatibility and `.reset()` between episodes.

`build_obs_index_map(gym_env)` dynamically maps observable names to flat-vector slices by inspecting the env's active observables, avoiding hardcoded indices.

`scripts/collect_demos.py` runs the expert, saves successful episodes to HDF5 with `(obs, actions, rewards, dones, next_obs)` per timestep.

### Coordinate System

`table_offset = [0, 0, 0.8]` is the table *center* in world frame. Table surface is at `0.8 + 0.05/2 = 0.825m`. Goal positions and stick placements are computed relative to `table_offset`. Robot base is positioned relative to the table via `base_xpos_offset["table"]`.
