# CS224R Final Project — Robot Wire Untangling

Stanford CS224R: Deep Reinforcement Learning (Spring 2026)

## Project Overview

Training a robot arm to untangle wires using deep RL. In the first approximation, wires are modeled as rigid objects (solid sticks), reducing the problem to contact-rich pick-and-place / reordering manipulation.

**Simulation engine:** Robosuite (MuJoCo backend)

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Rigid wire approximation (sticks) — environment setup | In progress |
| 2 | RL training (PPO / SAC baseline) | Planned |
| 3 | Deformable wire simulation | Planned |

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Structure

```
source/
├── envs/           # Custom Robosuite environments
├── scripts/        # Training and evaluation scripts
├── configs/        # Experiment configs
├── tests/          # Environment sanity checks
└── requirements.txt
```

## Simulation Engine Choice

After evaluating ManiSkill 3, Robosuite, IsaacLab, MuJoCo+dm_control, Genesis, and PyBullet:

- **Robosuite** selected for Phase 1: MuJoCo physics accuracy, native macOS support, clean API for custom rigid-body tasks, strong RL baselines
- **ManiSkill 3** kept as alternative for GPU-parallel training on Modal if throughput becomes a bottleneck
