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

1. **Learn the context signal.** Train a predictor for `phase` and
   `active_stick` from state, then from short history if transition errors are
   common. Evaluate phase-active MLP-BC with learned labels instead of the
   scripted tracker.
2. **Test diffusion / flow matching on this variant.** Re-run DPFM on the
   balanced random-order dataset with:
   - observation only,
   - observation + phase,
   - observation + phase + active stick.
   If explicit context helps DPFM, that supports the hypothesis that sharp
   phase transitions are a major failure mode.
3. **Increase N=2 difficulty one axis at a time.** Keep reporting obs-only,
   oracle-context, learned-context, and diffusion baselines while adding:
   - wider initial and goal ranges,
   - cross-side or crossing goals,
   - closer starts / interference cases,
   - retry-focused failure modes after missed grasps.
4. **Use residual RL only on the right difficulty band.** Apply residual RL
   after selecting a task where the base policy is imperfect but not hopeless,
   ideally around 40-80% success. Track whether residual corrections
   concentrate in missed-grasp, placement, retry, or phase-transition failures.
5. **Keep phase-wise diagnostics standard.** For every harder N=2 variant,
   report total success, per-order success, failure phase, and representative
   rollout videos.
