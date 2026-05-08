# CS224R Final Project — Robot Wire Untangling

Stanford CS224R: Deep Reinforcement Learning (Spring 2026)

## Project Overview

Training a robot arm to untangle wires using deep RL. Wires are modelled as
rigid sticks, reducing the problem to contact-rich pick-and-place manipulation.

**Simulation engine:** Robosuite (MuJoCo backend)  
**Robot:** Franka Panda  
**Algorithm:** SAC (baseline)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
```

## Usage

```bash
# Sanity-check the environment (headless)
python scripts/play_env.py

# With MuJoCo viewer (macOS: use mjpython instead of python)
mjpython scripts/play_env.py --render

# Print Gymnasium observation/action spaces
python scripts/play_env.py --gym

# Train locally (SAC, reads total_timesteps from config)
python scripts/train.py

# Train without WandB logging
python scripts/train.py --no-wandb

# Train with custom config / overrides
python scripts/train.py --config configs/stick_reorder.yaml --timesteps 500000 --seed 1

# Evaluate a saved checkpoint
python scripts/eval.py --checkpoint checkpoints/best/best_model.zip
python scripts/eval.py --checkpoint checkpoints/best/best_model.zip --episodes 100 --seed 0

# Run tests
pytest tests/ -v
```

### Modal GPU training

```bash
pip install modal
modal setup   # authenticate once

# Create the WandB secret (one-time)
modal secret create wandb WANDB_API_KEY=<your_token>

# Smoke test (1000 steps)
modal run modal_train.py --total-timesteps 1000

# Full run (reads config, GPU type, and timeout from configs/stick_reorder.yaml)
modal run modal_train.py
```

GPU type and timeout are set in `configs/stick_reorder.yaml` under `modal.gpu` / `modal.timeout` — no code changes needed to switch hardware.

## Structure

```
cs224r-wire-untangling/
├── wire_untangling/
│   ├── envs/
│   │   └── stick_reorder.py     # StickReorderEnv
│   ├── models/objects/
│   │   └── stick_object.py      # StickObject (thin BoxObject wrapper)
│   └── utils/
│       └── eval.py              # algorithm-agnostic evaluate()
├── scripts/
│   ├── play_env.py              # render & sanity-check
│   ├── train.py                 # training entry point
│   └── eval.py                  # evaluation CLI
├── configs/
│   └── stick_reorder.yaml       # env, training, and modal params
├── modal_train.py               # Modal GPU deployment
└── tests/
    └── test_stick_reorder.py
```

## Environment

`StickReorderEnv` places N sticks randomly on a table. The goal is to move
each stick to its assigned position in a parallel row arrangement.

Key parameters (see `configs/stick_reorder.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_sticks` | 3 | Number of sticks |
| `stick_length` | 0.18 m | Stick length |
| `goal_spacing` | 0.06 m | Y-spacing between goal positions |
| `success_threshold` | 0.03 m | Per-stick distance tolerance |
| `reward_shaping` | True | Dense reward (−Σdist) + sparse bonus |

## Expert Policy and Demonstrations

A scripted pick-and-place expert generates demonstrations for behavior cloning.
The expert uses proportional control over OSC_POSE deltas with automatic gripper
yaw alignment to grasp sticks regardless of their orientation (~92% success rate).

```bash
# Visualize the expert
python scripts/play_env.py --render --expert

# Collect 200 successful demonstrations
python scripts/collect_demos.py --num-demos 200 --output data/demos.hdf5
```

See [docs/expert_policy.md](docs/expert_policy.md) for the full design
documentation, HDF5 format specification, and tunable parameters.

### Canonical training set (team default)

Everyone training a BC variant should use the same demo file so policy
comparisons aren't confounded by data differences. The reproducible recipe:

```bash
python scripts/collect_demos.py --num-demos 1000 --output data/demos.hdf5 --seed 42
```

The seeding utility (`wire_untangling/utils/seeding.py`) makes this
**bytewise-reproducible**: same seed + same `env_config_hash` ⇒ identical
HDF5. Each demo file records its `top_seed`, `env_config_hash`,
`oracle_version`, and `robosuite_version` as root attrs so you can verify
provenance after the fact. `train_bc.py` prints these at startup and
supports `--require-config-hash <hash>` / `--require-top-seed <int>` flags
that hard-fail if the loaded HDF5 doesn't match — useful for CI or when
sharing checkpoints across machines.

To collect demos that include the full RELEASE phase (useful when training
sequence models that need terminal-segment coverage):

```bash
python scripts/collect_demos.py --num-demos 1000 --output data/demos.hdf5 \
    --seed 42 --no-terminate-on-success
```

### Train MLP-BC

```bash
python scripts/train_bc.py --demos-path data/demos.hdf5 \
    --checkpoint-dir checkpoints/mlp_bc \
    --require-config-hash <hash from collected file> --require-top-seed 42

# Evaluate (deterministic single-step inference)
python scripts/play_env.py --bc_checkpoint checkpoints/mlp_bc/mlp_bc_policy.pt --episodes 100
```

Defaults: 500 epochs, batch 256, lr 1e-3, hidden (256, 256, 256), tanh
output. ~5 min on Apple Silicon MPS for the canonical 1000-demo dataset.
On a 200-demo dataset the same recipe gives ~87% N=1 success rate; on
1000 demos, ~92%.

## Recording Videos

Save MP4 videos of any policy for presentations or debugging. Uses offscreen
rendering — no display or GUI required.

```bash
# Record the expert policy (1280x720 H.264)
python scripts/play_env.py --record videos/expert_demo.mp4 --expert --episodes 3

# Record a trained checkpoint
python scripts/play_env.py --record videos/policy.mp4 --checkpoint checkpoints/best/best_model.zip

# Watch live AND save to disk simultaneously
python scripts/play_env.py --render --record videos/demo.mp4 --expert

# Custom framerate
python scripts/play_env.py --record videos/slow.mp4 --expert --fps 10
```

Requires `imageio` and `imageio-ffmpeg` (included in `environment.yml`).
