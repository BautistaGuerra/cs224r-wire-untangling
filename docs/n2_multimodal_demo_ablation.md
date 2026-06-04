# N=2 Paired-Order Multimodal Demo Ablation

This ablation creates obs-only behavioral-cloning data where the same physical
initial scene has two valid expert continuations. For each accepted reset seed,
the collector saves one demo with order `[0, 1]` and one demo with order
`[1, 0]`.

The final task success condition is unchanged: stick0 must reach goal0 and
stick1 must reach goal1. The order only changes which valid subtask is solved
first. This keeps the ablation focused on multimodal supervision rather than
on goal reassignment or phase-active oracle context.

## Why

The existing random-order N=2 dataset alternates orders, but different orders
usually appear under different reset seeds. That acts partly like data
augmentation, and obs-only MLP-BC remains strong.

The paired-order dataset instead creates the supervised-learning pattern that
deterministic MSE policies handle poorly:

```text
same obs_0 -> move toward stick0 first
same obs_0 -> move toward stick1 first
```

MLP-BC is trained to predict the conditional mean action. If the two first
motions point to different sides of the table, the mean can point between the
two sticks. A stochastic DPFM policy can model a distribution over action
chunks and sample one coherent branch.

## Config

The paired dataset uses `configs/stick_reorder_n2_paired_order.yaml`:

```yaml
expert:
  order_mode: paired_balanced
  order_choices:
    - [0, 1]
    - [1, 0]
```

During collection, the pair is accepted only if both branches succeed. HDF5
metadata includes:

- root attr `multimodal_collection = "paired_order"`
- root attr `num_pairs`
- per-demo attr `multimodal_pair_id`
- per-demo attr `multimodal_branch`
- per-demo attr `multimodal_pair_seed`
- per-demo attr `stick_order`

## Commands

Smoke test:

```bash
.venv/bin/python -m scripts.collect_demos \
  --config configs/stick_reorder_n2_paired_order.yaml \
  --smoke \
  --smoke-n 20 \
  --smoke-threshold 0.80 \
  --seed 42
```

Collect paired demos:

```bash
.venv/bin/python -m scripts.collect_demos \
  --config configs/stick_reorder_n2_paired_order.yaml \
  --num-demos 200 \
  --output data/stick_n2_paired_order_demos.hdf5 \
  --seed 42
```

Verify that paired demos share initial observations but have different early
actions:

```bash
.venv/bin/python -m scripts.analyze_multimodal_demos \
  --demos-path data/stick_n2_paired_order_demos.hdf5 \
  --prefix-horizon 16
```

Train obs-only MLP-BC:

```bash
WANDB_NAME=mlp_bc_n2_paired_order_obs \
.venv/bin/python -m scripts.train_bc \
  --demos-path data/stick_n2_paired_order_demos.hdf5 \
  --checkpoint-dir checkpoints/mlp_bc_n2_paired_order_obs \
  --conditioning obs \
  --seed 42
```

Train obs-only DPFM on Modal:

```bash
modal run --detach modal_train_flow_matching.py \
  --env-config configs/stick_reorder_n2_paired_order.yaml \
  --dpfm-config configs/flow_matching.yaml \
  --demos-path data/stick_n2_paired_order_demos.hdf5 \
  --checkpoint-dir checkpoints/dpfm_n2_paired_order_obs_h4_e1_i10 \
  --action-normalizer zscore \
  --conditioning obs \
  --action-chunk-horizon 4 \
  --execute-steps 1 \
  --integration-steps 10 \
  --seed 42 \
  --upload-demos \
  --wandb-name dpfm_n2_paired_order_obs_h4_e1_i10
```

Evaluate MLP-BC:

```bash
.venv/bin/python -m scripts.play_env \
  --config configs/stick_reorder_n2_paired_order.yaml \
  --bc_checkpoint checkpoints/mlp_bc_n2_paired_order_obs/mlp_bc_policy.pt \
  --episodes 100
```

Evaluate DPFM:

```bash
.venv/bin/python -m scripts.play_env \
  --config configs/stick_reorder_n2_paired_order.yaml \
  --dpfm_checkpoint checkpoints/dpfm_n2_paired_order_obs_h4_e1_i10/flow_matching_policy.pt \
  --dpfm-execute-steps 1 \
  --episodes 100
```

## Reporting

Report:

- paired dataset diagnostics from `analyze_multimodal_demos.py`
- MLP-BC closed-loop success
- stochastic DPFM closed-loop success
- optionally, deterministic DPFM success with `--dpfm-deterministic`

The key comparison is not only success rate. The data diagnostic should show
that the dataset contains identical observations with distinct early expert
actions, which is the regime where deterministic MSE behavior cloning is
expected to average modes.

## Results

Closed-loop evaluation on the N=2 paired-order obs-only task:

| Policy | Success |
|---|---:|
| MLP-BC obs-only | 34% |
| DPFM-BC obs-only `h4_e1_i10` | 66% |

The paired-order dataset exposes a gap that was hidden by the less-conflicting
random-order dataset. MLP-BC suffers when the same observation has two valid
expert actions, while stochastic DPFM maintains roughly the prior N=2 obs-only
performance. This supports using DPFM as the residual-RL base policy: it is the
closer match to realistic human demonstration data, where valid strategies are
often multimodal rather than a single scripted trajectory family.
