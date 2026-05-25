# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package for CS224R robot wire/stick untangling experiments in Robosuite/MuJoCo. Core package code lives in `wire_untangling/`: environments in `envs/`, policies in `policies/`, object models in `models/objects/`, and shared helpers in `utils/`. CLI entry points live in `scripts/`, with `modal_train.py` for Modal GPU runs. Configuration is stored in `configs/*.yaml`. Tests are in `tests/`, documentation in `docs/`, and images in `docs/assets/`. Treat `data/`, `checkpoints/`, `logs/`, `wandb/`, and local debug files as generated artifacts unless a task explicitly says otherwise.

## Build, Test, and Development Commands

Create a local editable install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
```

Alternatively, use `mamba env create -f environment.yml` and activate `cs224r-wire-untangling`.

Common commands:

```bash
python -m pytest tests/ -v
python -m pytest tests/test_stick_reorder.py::test_reward_is_finite -v
python scripts/play_env.py
mjpython scripts/play_env.py --render
python scripts/train.py --config configs/stick_reorder.yaml --no-wandb
python scripts/eval.py --checkpoint checkpoints/best/best_model.zip --episodes 50
python scripts/collect_demos.py --num-demos 200 --output data/demos.hdf5
modal run modal_train.py --total-timesteps 1000
```

Use `mjpython` for rendered MuJoCo viewer runs on macOS.

## Coding Style & Naming Conventions

Follow standard Python style with 4-space indentation, clear type/shape comments where useful, and small functions around environment, policy, and data responsibilities. Use `snake_case` for modules, functions, variables, YAML keys, and CLI flags; use `PascalCase` for classes such as `StickReorderEnv` and policy/model classes. Prefer config-driven changes in `configs/*.yaml` over hardcoded training constants. Keep comments focused on non-obvious Robosuite, MuJoCo, or observation-index behavior.

## Testing Guidelines

Tests use `pytest`; name files `test_*.py` and functions `test_*`. Add focused tests near the behavior being changed, especially for environment resets, reward/success logic, expert policy phases, dataset provenance, normalizers, and policy output shapes. Run the full suite before broad changes and a targeted test for narrow fixes.

## Commit & Pull Request Guidelines

Recent history mixes concise descriptive commits with Conventional Commit prefixes such as `feat:` and `docs:`. Prefer imperative, scoped messages, for example `feat: add n2 bc baseline` or `docs: update expert policy notes`. Pull requests should summarize the behavior change, list verification commands, note config/data/checkpoint assumptions, and include screenshots or videos only for rendered behavior or documentation asset changes.

## Security & Configuration Tips

Do not commit secrets, WandB tokens, large generated datasets, or private checkpoints. For Modal, create secrets with `modal secret create` and keep GPU type, timeout, and training settings in YAML configs.
