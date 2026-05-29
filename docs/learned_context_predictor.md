# Learned Context Predictor

This baseline replaces the oracle phase tracker used by phase-active MLP-BC
with a supervised predictor trained from demonstration labels.

## V1 Design

- Input: current raw GymWrapper observation only.
- Output heads: 8-way `Phase` classification and `num_sticks` active-stick
  classification.
- Architecture: shared MLP trunk with hidden dims `[256, 256, 256]` by default.
- Loss: unweighted `CE_phase + CE_active`.
- Rollout context: hard `argmax` predictions converted to the same one-hot
  phase-active feature vector used by MLP-BC training.
- Normalization: raw-observation train-split mean/std are saved in the context
  predictor checkpoint. These stats are separate from BC state normalization.

V1 intentionally does not use temporal history, order ID, sequence smoothing,
class weighting, soft probability context, or a `none/done` active-stick class.

## Training

Canonical N=2 random-order command:

```bash
python scripts/train_context_predictor.py \
  --demos-path data/stick_n2_random_order_demos.hdf5 \
  --checkpoint-dir checkpoints/context_predictor_n2_random_order \
  --no-wandb
```

The script uses sorted demo keys with a deterministic episode-level split,
shuffled by `--split-seed 42` and default fractions `80/10/10`.

It writes:

- `context_predictor_best.pt`: selected by validation joint accuracy.
- `context_predictor_last.pt`: final epoch.
- `metrics.json`: train/val/test metrics, split keys, history, and V1 design
  choices.

Reported metrics include phase accuracy, active-stick accuracy, joint accuracy
(`bc_input_agreement`), within-one-phase accuracy, per-phase precision/recall/F1,
and an 8x8 phase confusion matrix labeled with `Phase` enum names. Test metrics
are computed only after training from the best checkpoint.

## Rollout

Canonical learned-context rollout:

```bash
python scripts/play_env.py \
  --config configs/stick_reorder_n2_random_order.yaml \
  --bc_checkpoint checkpoints/mlp_bc_n2_random_order_phase_active/mlp_bc_policy.pt \
  --context-predictor-checkpoint checkpoints/context_predictor_n2_random_order/context_predictor_best.pt \
  --episodes 100
```

If `--context-predictor-checkpoint` is omitted, phase-active MLP-BC keeps using
the existing oracle phase tracker path.

For diagnostics, compare learned context against the oracle tracker while still
feeding learned context to BC:

```bash
python scripts/play_env.py \
  --config configs/stick_reorder_n2_random_order.yaml \
  --bc_checkpoint checkpoints/mlp_bc_n2_random_order_phase_active/mlp_bc_policy.pt \
  --context-predictor-checkpoint checkpoints/context_predictor_n2_random_order/context_predictor_best.pt \
  --compare-oracle-context \
  --episodes 100
```

Rollout hard-fails when predictor metadata does not match the BC checkpoint:
raw observation dimension, `num_phases`, or `num_sticks`. It also hard-fails if
the environment `num_sticks` differs, or if a context predictor is supplied for
an obs-only BC checkpoint.

## Known Limitations

`GRASP` and `RELEASE` include timer-driven behavior, so a stateless single-frame
classifier can be ambiguous even when the raw observation is correct. Errors in
phase or active-stick prediction also directly change the BC input vector, so
offline joint accuracy is the most relevant first diagnostic.

Follow-up ablations to consider:

- Class-weighted cross entropy.
- Temporal history or smoothing.
- Previous predicted phase as input.
- Soft probability context instead of hard one-hot context.
- Deeper oracle-vs-learned divergence analysis during rollout.
