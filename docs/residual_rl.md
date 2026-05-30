# Residual RL (TD3) Policy

This document describes the residual reinforcement learning (RRL) policy: what it
is, the architectural decisions behind the current implementation, and how to run
it. A short section at the end gives orientation context for AI coding assistants.

## What it is

The residual RL policy improves on a **frozen** flow-matching (DPFM) base policy
rather than learning manipulation from scratch. At each step the base policy
proposes an action; a learned TD3 actor proposes a small **residual** that is
added on top. The combined action is what the environment executes:

```
action_executed = base_action + residual_action      (in normalized action space)
```

The base policy already solves the single-stick pick-and-place task ~80–87% of
the time (see `docs/dpfm_sweep.csv`). RRL exists to push past that ceiling by
correcting the base policy's mistakes online, while keeping the base behavior as
a strong, stable prior.

> **Scope:** RRL is currently **single-stick only** (`--num-sticks 1`).

## Architectural decisions

These are the deliberate choices in the current implementation, with the
reasoning behind each.

### 1. Residual on a frozen base policy

The DPFM checkpoint is loaded and never updated. Only the residual actor/critic
train. This keeps a reliable prior in place: if the residual collapses to zero,
behavior degrades gracefully to the base policy rather than to noise. The actor's
final layer is initialized with near-zero weights (`actor.py`), so at the start of
training the residual is ~0 and the agent behaves exactly like the base policy.

### 2. Everything happens in normalized action space, with no clipping

Both base and residual actions live in the normalized space defined by the base
policy's normalizers. The combined action is added there and only then
denormalized for execution. We deliberately **do not** clip the combined action
to `[-1, 1]` (see the commented-out `torch.clamp` calls in `agent.py`): the agent
always operates in a normalized space where the base policy's outputs are already
well-scaled, and clipping would distort the residual's learning signal.

### 3. Shared normalizers with the base policy

`ResidualRLPolicy` reuses the base `DPFMModelPolicy`'s `obs_norm` and
`action_norm` directly instead of storing its own copies. This guarantees the
residual is trained and evaluated in exactly the same coordinate frame the base
policy uses — any mismatch would make the additive residual meaningless. The
normalizer *type* (z-score / min-max / identity) is whatever the DPFM checkpoint
was trained with; it is auto-detected on load.

### 4. TD3 actor-critic with a critic ensemble

- **Actor** (`RRLActor`): input is `concat(obs, base_action)`; an MLP trunk
  (widths from `actor.hidden_dims`) with LayerNorm + ReLU produces a `Tanh`-bounded
  output, scaled by `actor.action_scale` to keep residuals small relative to the
  base action. The output mean parameterizes a `TruncatedNormal` whose std is a
  hyperparameter (not learned) — this makes the exploration/smoothing noise
  directly interpretable.
- **Critic** (`Critic`): an **ensemble** of `critic.num_critics` Q-networks.
  - For the **target** (critic update), Q is aggregated by taking the **min over a
    random subset** of `critic.num_sampled_critics` heads — clipped double-Q,
    which counteracts value overestimation.
  - For the **actor** update, Q is aggregated by **mean** over all heads. We use
    mean rather than min here on purpose: we don't want to *underestimate* value
    when optimizing the policy.

### 5. TD3 stabilizers: target smoothing, delayed actor updates, critic warmup

- **Target policy smoothing:** when computing the Bellman target, noise is added to
  the next action (`critic_smoothing_stddev`, clipped by
  `critic_smoothing_stddev_clip`) so the critic can't overfit sharp peaks in the
  value landscape.
- **Critic warmup:** the actor does not update until the critic has done
  `critic_warmup_steps` updates — training the policy against an untrained critic
  is destabilizing.
- **Delayed / configurable update ratio:** critic and actor update counts per
  gradient step are configurable (`critic_updates_per_gradient_step`,
  `actor_updates_per_gradient_step`).
- **Target networks** are updated with EMA (Polyak averaging, rate `tau`).

### 6. n-step returns via a torchrl replay buffer

Transitions are stored in a `TensorDictPrioritizedReplayBuffer` (currently
`alpha=0`, i.e. uniform sampling — the prioritization machinery is wired up but
disabled for now). n-step returns are computed by a `MultiStepTransform`
(`wire_untangling/utils/rb_transforms.py`), so the Bellman target uses
`gamma^n` discounting over `n_step` steps. The terminal-state correction is
handled by the `nonterminal` mask the transform produces, not by a separate
`done` flag.

`MultiStepTransform` is vendored from Amazon's **residual-offpolicy-rl (ResFiT)**
reference implementation (CC-BY-NC-4.0); several comments in the code cite line
numbers in that reference (`train_residual_td3.py:...`).

### 7. RLPD-style mixed offline/online sampling

An optional **offline buffer** is seeded from HDF5 demonstrations. For each demo
step it stores the **ground-truth demo action** as the executed action and the
**frozen base policy's action** as `action_base`. This teaches the critic the
value landscape around demonstrated behavior before online data accumulates.

During training each batch mixes online and offline transitions according to
`offline_fraction` (e.g. `0.5` = half the batch from demos). Set
`offline_fraction: 0.0` for online-only training.

### 8. Interpretable exploration noise schedule

Online exploration adds `TruncatedNormal` noise whose std decays linearly from
`exploration_stddev_max` to `exploration_stddev_min` over
`exploration_stddev_decay_steps`. Because std is an explicit hyperparameter (not a
learned log-std head), the exploration magnitude is easy to read and tune.

## Configuration

All RRL hyperparameters live in `configs/residual_td3.yaml`. Key knobs:

| Param | Meaning |
|-------|---------|
| `actor.hidden_dims` | Sequence of actor MLP layer widths, e.g. `[512, 512]` |
| `actor.action_scale` | Caps residual magnitude relative to the base action |
| `actor.lr` / `critic.lr` | Learning rates (actor LR is kept lower) |
| `critic.hidden_dims` | Sequence of critic MLP layer widths |
| `critic.num_critics` | Size of the Q-ensemble |
| `critic.num_sampled_critics` | Heads sampled for the clipped (min) target |
| `gamma` / `n_step` | Discount and n-step return horizon |
| `offline_fraction` | Fraction of each batch drawn from the demo buffer |
| `offline_demos_path` | HDF5 demos for the offline buffer (`null` to disable) |
| `learning_starts` | Random-action steps before learning (must be `> n_step`) |
| `critic_warmup_steps` | Critic-only updates before the actor starts |
| `tau` | EMA rate for target networks |

`state_dim` / `action_dim` are left `null` and inferred at runtime from the DPFM
checkpoint. CLI flags override config values (see below).

## How to run

### Train

```bash
# Basic: residual TD3 on top of a DPFM checkpoint
python -m scripts.train_residual_rl \
    --dpfm-checkpoint checkpoints/flow_matching/flow_matching_policy.pt \
    --num-sticks 1 \
    --td3-config configs/residual_td3.yaml

# Without WandB logging
python -m scripts.train_residual_rl \
    --dpfm-checkpoint checkpoints/flow_matching/flow_matching_policy.pt \
    --num-sticks 1 --no-wandb
```

Useful overrides: `--action-scale`, `--actor-lr`, `--critic-lr`,
`--actor-hidden-dims 512 512`, `--critic-hidden-dims 512 512`,
`--offline-fraction`, `--demos-path`.

### Sweep

`sweep_rrl.sh` runs a grid over `action_scale × offline_fraction × DPFM checkpoint`
(up to 4 processes in parallel), writing checkpoints under `checkpoints/sweep_rrl/`.

```bash
bash sweep_rrl.sh
```

### Visualize / evaluate

RRL inference needs **both** the RRL checkpoint and its DPFM base checkpoint:

```bash
# Render in the MuJoCo viewer
python -m scripts.play_env --render \
    --rrl-checkpoint checkpoints/td3/td3_final.pt \
    --dpfm_checkpoint checkpoints/flow_matching/flow_matching_policy.pt

# Headless success rate over 100 episodes
python -m scripts.play_env --config configs/stick_reorder.yaml \
    --rrl-checkpoint checkpoints/td3/td3_step500000.pt \
    --rrl-config configs/residual_td3.yaml \
    --dpfm_checkpoint checkpoints/flow_matching/flow_matching_policy.pt \
    --dpfm-stochastic --num-sticks 1 --episodes 100
```

Experiment results and run metadata are tracked in `docs/rrl_runs.csv` and on
WandB (entity `alexta-uw`, project `cs224r-wire-untangling`).

## Sweep results (Phase 5 — large networks, 512×512)

All runs share: `[512, 512]` actor/critic hidden dims, actor_lr=5e-6,
critic_lr=1e-4, 4 critics, n_step=3, gamma=0.97, offline_fraction=0.5,
base DPFM checkpoint `checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt`.

### Training commands

**Baseline — action_scale=0.2, dense reward, UTD=4** (silvery-gorge-77, [wandb](https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/ctes47gj)):
```bash
python -m scripts.train_residual_rl \
    --dpfm-checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
    --action-scale 0.2 --offline-fraction 0.5 \
    --checkpoint-dir checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512 \
    --num-sticks 1 --demos-path data/demos.hdf5
```

**action_scale=0.3** (valiant-sea-78, [wandb](https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/crezep81)):
```bash
python -m scripts.train_residual_rl \
    --dpfm-checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
    --action-scale 0.3 --offline-fraction 0.5 \
    --checkpoint-dir checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512_a0.3 \
    --num-sticks 1 --demos-path data/demos.hdf5
```

**UTD=1 (4 env steps per gradient update)** (glorious-disco-80, [wandb](https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/idw7odys)):
```bash
# Requires modified configs/residual_td3.yaml with:
#   gradient_update_per_env_steps: 4
#   total_timesteps: 1000000
#   learning_starts: 35000
#   critic_warmup_steps: 35000
#   critic_stddev_decay_steps: 1000000
#   exploration_stddev_decay_steps: 600000
python -m scripts.train_residual_rl \
    --dpfm-checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
    --action-scale 0.2 --offline-fraction 0.5 \
    --checkpoint-dir checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512_utd0.25 \
    --num-sticks 1 --demos-path data/demos.hdf5
```

**Sparse reward (no shaping)** (sage-bee-79, [wandb](https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/r0f00lyd)):
```bash
python -m scripts.train_residual_rl \
    --dpfm-checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
    --action-scale 0.2 --offline-fraction 0.5 \
    --checkpoint-dir checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512_noshape \
    --no-reward-shaping --num-sticks 1 --demos-path data/demos.hdf5
```

### Evaluation results

Each configuration was evaluated 3× with 100 episodes per run. Success rate
averaged over 300 episodes. Baseline: silvery-gorge-77 (dense, scale=0.2, UTD=4).

| Run | Reward Type | Action Scale | Critic UTD | Avg Success Rate | Fisher p vs baseline |
|-----|-------------|-------------|------------|-----------------|---------------------|
| silvery-gorge-77 (baseline) | dense | 0.2 | 4 | **99.0%** | — |
| valiant-sea-78 | dense | 0.3 | 4 | 97.0% | p=0.142 (not sig.) |
| glorious-disco-80 | dense | 0.2 | 1 | 97.3% | p=0.222 (not sig.) |
| sage-bee-79 | sparse | 0.2 | 4 | 95.0% | p=0.007 (sig. at α=0.05) |

Full metrics in `docs/rrl_sweep_summary.csv`.

## Context for AI assistants

If you are an AI assistant picking up this work, orient here first:

- **Key source files:**
  - `scripts/train_residual_rl.py` — training loop, replay buffers, offline buffer
    population, logging.
  - `wire_untangling/policies/rl/agent.py` — `TD3Agent`: critic/actor updates,
    action combination, target EMA.
  - `wire_untangling/policies/rl/actor.py` — `RRLActor` (residual policy network).
  - `wire_untangling/policies/rl/critic.py` — `Critic` (Q-ensemble).
  - `wire_untangling/policies/policy_inference_wrappers.py` — `ResidualRLPolicy`
    wraps a base `DPFMModelPolicy` + `TD3Agent` for inference.
  - `wire_untangling/utils/rb_transforms.py` — vendored ResFiT `MultiStepTransform`.
  - `configs/residual_td3.yaml` — all hyperparameters.

- **Non-obvious invariants worth preserving:**
  - The residual is added in **normalized** action space and **not clipped** — this
    is intentional, not an oversight.
  - `ResidualRLPolicy` must reuse the **base policy's** normalizers; do not give it
    independent ones.
  - The actor trunk must end with `Linear → Tanh`; the near-zero init in
    `RRLActor.__init__` grabs `self.policy[-2]` assuming Tanh is last.
  - Reward/discount come from the `MultiStepTransform` output under
    `("next", "reward")` and `gamma`, with a `nonterminal` mask — not a raw `done`.
  - `learning_starts` must be `> n_step`.

- **State of the work:** this pipeline is experimental and under active iteration.
  Some code carries author TODOs (e.g. duplicated combine-action helpers in
  `agent.py`). The design follows the ResFiT reference; comments citing
  `train_residual_td3.py:<lines>` point into that reference, not this repo.

- **Collaboration note:** when you find a bug, **diagnose and propose the fix, then
  wait for the maintainer's approval before editing** — they prefer to decide how
  fixes land in this experimental code.
```
