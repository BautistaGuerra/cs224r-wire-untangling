# Testing and End-to-End Evaluation

## Unit and Integration Tests

Run the full test suite with:

```bash
python -m pytest tests/ -v
```

The test files cover individual components without requiring long training runs:

| Test file | What it covers |
|-----------|---------------|
| `test_stick_reorder.py` | Environment basics: spaces, step/reset, reward shape |
| `test_env_readiness.py` | Observable wiring, success/reward semantics, yaw-error helper |
| `test_expert_policy.py` | Expert FSM contract: phase transitions, action shape, goal_yaw plumbing |
| `test_mlp_bc.py` | MLP-BC output shape/range, gradient flow, checkpoint round-trip on synthetic data |
| `test_flow_matching.py` | Dataset normalization round-trip, DPFM checkpoint loading |
| `test_normalizer.py` | Normalizer statistics, forward/inverse transforms, serialization, torch parity |
| `test_dataset.py` | HDF5 demo schema, env_config_hash helper |
| `test_analyze_expert.py` | Expert analysis script on synthetic HDF5 |
| `test_n2_side_task.py` | Two-stick environment and expert policy |

Most tests use synthetic data or short rollouts. Tests that create a Robosuite environment use `scope="module"` fixtures to avoid repeated initialization.

## End-to-End Training and Evaluation

`scripts/e2e_train_eval.py` runs the full pipeline on a single-stick environment: demo collection, training of both MLP-BC and DPFM policies, and evaluation with success rate reporting.

### Quick start

```bash
# Full pipeline: collect 200 demos, train both policies, evaluate 50 episodes each
python -m scripts.e2e_train_eval --skip-collect=false

# Reuse existing demos (default), just retrain and evaluate
python -m scripts.e2e_train_eval

# Fast smoke test with fewer demos and episodes
python -m scripts.e2e_train_eval --skip-collect=false --num-demos 50 --eval-episodes 10
```

### What the script does

1. **Collect demos** — Runs the scripted expert for `--num-demos` episodes, saves to `data/demos.hdf5`. Skipped by default (`--skip-collect` is true); pass `--skip-collect=false` to re-collect.
2. **Train MLP-BC** — 500 epochs, 3x256 MLP with tanh output, MSE loss. Checkpoint saved to `checkpoints/mlp_bc/mlp_bc_policy.pt`.
3. **Train DPFM** — 200 epochs, temporal 1D U-Net with action chunk horizon 10. Checkpoint saved to `checkpoints/flow_matching/flow_matching_policy.pt`.
4. **Evaluate MLP-BC** — Rolls out the trained policy for `--eval-episodes` episodes and reports success rate.
5. **Evaluate DPFM** — Same evaluation for the flow matching policy.

The script prints a comparison table at the end:

```
============================================================
  FINAL COMPARISON (50 episodes each)
============================================================
  Policy                      Success          Reward
  ───────────────────────── ────────── ───────────────
  MLP-BC                       92.0%          187.43
  DPFM (Flow Matching)         88.0%          179.21
============================================================
```

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-demos` | 200 | Expert episodes to collect |
| `--eval-episodes` | 50 | Episodes per policy evaluation |
| `--seed` | 42 | Random seed for reproducibility |
| `--skip-collect` | true | Skip demo collection, reuse existing HDF5 |
| `--demos-path` | `data/demos.hdf5` | Path to demo dataset |
| `--env-config` | `configs/stick_reorder.yaml` | Environment config file |

### Success criteria

An episode is successful when **all sticks** simultaneously satisfy:
- Position error to goal <= 0.03 m (`success_threshold`)
- Yaw error (mod pi) <= 10 deg (`orientation_threshold`)

This is reported via the `is_success` key in the environment's info dict at the final timestep of each episode.
