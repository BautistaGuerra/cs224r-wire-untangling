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
