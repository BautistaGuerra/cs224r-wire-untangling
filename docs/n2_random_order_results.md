# N=2 Balanced Random-Order BC Results

Date: 2026-05-23

## Why

The previous N=2 side-specific task was intentionally restricted: `stick0`
was always solved first and `stick1` second. This made phase-active MLP-BC
strong, but it also entangled object identity, side of the table, and temporal
phase. To increase task difficulty without changing the physical environment,
we introduced balanced high-level order variation:

- 50% of demos solve `[0, 1]`.
- 50% of demos solve `[1, 0]`.
- The observation-only policy still receives only raw env observations.
- The phase-active policy still receives only `phase` and current
  `active_stick`, not a full order ID.

This tests whether BC can handle a less scripted high-level sequence while
keeping the oracle-context baseline available.

## How

The new config `configs/stick_reorder_n2_random_order.yaml` keeps the same
`two_stick_side` environment and changes only the expert order schedule:

```yaml
expert:
  order_mode: balanced
  order_choices:
    - [0, 1]
    - [1, 0]
```

Demo collection now schedules orders per episode, retries failed attempts with
the same target order, enforces exact balance, and stores each demo's
`stick_order` as HDF5 metadata. Playback now reports success by order and uses
the scheduled order for oracle phase tracking.

## Results

Dataset sanity check for `data/stick_n2_random_order_demos.hdf5`:

- `200` successful demos, `0` failures.
- `100` demos with order `[0, 1]`.
- `100` demos with order `[1, 0]`.
- Every demo switches active stick exactly once.
- Every demo ends in success.

Closed-loop MLP-BC evaluation:

| Train Data | Eval Config | Conditioning | Success | Per-Order Success |
|---|---|---|---:|---|
| fixed order | fixed order | obs-only | 0-1% | `[0,1]`: 0-1% |
| fixed order | random order | obs-only | 0% | `[0,1]`: 0/50, `[1,0]`: 0/50 |
| random order | random order | obs-only | 68% | `[0,1]`: 35/50, `[1,0]`: 33/50 |
| random order | fixed order | obs-only | 63% | `[0,1]`: 63/100 |
| random order | random order | phase-active | 96% | `[0,1]`: 49/50, `[1,0]`: 47/50 |

## Interpretation

The random-order obs-only result was unexpectedly strong. The improvement is
not caused by evaluation leakage: the physical env config is unchanged, and
obs-only checkpoints do not consume the episode order. Instead, balanced order
acts like symmetry/data augmentation. In fixed-order demos, `stick0`, negative
Y, and "first object" are always correlated; random-order demos break that
shortcut and expose both sticks in both temporal roles.

This refines the phase/context story:

- Explicit oracle context is still strongest: phase-active reaches 96%.
- Dataset design matters: balanced order alone raises obs-only from near 0%
  to roughly 65%.
- The remaining gap suggests learned phase/active-stick context is still a
  useful next baseline before residual RL.

## Next Steps

1. Train a learned context model that predicts `phase` and `active_stick` from
   state or short history, then evaluate phase-active MLP-BC using learned
   labels instead of the scripted tracker.
2. Increase N=2 difficulty one axis at a time:
   - wider initial and goal ranges,
   - cross-side or crossing goals,
   - closer starts / interference cases,
   - retry-focused failure modes.
3. Apply residual RL only after selecting a task where BC is imperfect but not
   hopeless, ideally in the 40-80% success range.
4. Report success by order and failure phase for every harder N=2 variant.
