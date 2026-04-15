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

# With MuJoCo viewer
python scripts/play_env.py --render

# Print Gymnasium observation/action spaces
python scripts/play_env.py --gym

# Train (SAC, 1M steps)
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/stick_reorder.yaml --timesteps 500000

# Run tests
pytest tests/ -v
```

## Structure

```
cs224r-wire-untangling/
├── wire_untangling/
│   ├── envs/
│   │   └── stick_reorder.py     # StickReorderEnv
│   ├── models/objects/
│   │   └── stick_object.py      # StickObject (thin BoxObject)
│   └── utils/
├── scripts/
│   ├── play_env.py              # render & sanity-check
│   └── train.py                 # SAC training entry point
├── configs/
│   └── stick_reorder.yaml
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
