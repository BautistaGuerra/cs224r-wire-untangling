# DPFM N=2 Phase-Context Experiments

Date: 2026-05-29

This document summarizes the recent flow-matching / diffusion-policy
experiments on the two-stick task. It is intended as source material for the
final report, especially the discussion of why phase-active context helps
one-step MLP-BC but is harder to use in a chunked DPFM policy.

## Setup

The N=2 environment uses the side-specific two-stick task from
`configs/stick_reorder_n2.yaml` and the balanced random-order variant from
`configs/stick_reorder_n2_random_order.yaml`. The fixed-order dataset always
solves the same stick first; the balanced-order dataset alternates which stick
is solved first while keeping the same physical task geometry.

We compare two policy inputs:

- `obs`: raw robot, stick, and goal observation only.
- `phase-active`: raw observation plus one-hot scripted expert phase and
  one-hot active-stick index.

For MLP-BC, phase-active context is one-step aligned:

```text
obs_t + phase_t + active_t -> action_t
```

For DPFM, the same context conditions an action chunk:

```text
obs_t + phase_t + active_t -> [action_t, ..., action_t+H-1]
```

We write DPFM settings as `h{H}_e{E}_i{I}`, where `H` is action chunk horizon,
`E` is the number of sampled actions executed before replanning, and `I` is the
number of flow integration steps.

## Reference Baselines

The main MLP-BC reference results on the N=2 side task are:

| Task | Conditioning | MLP-BC Success |
|---|---|---:|
| Fixed order | obs-only | 0-1% |
| Fixed order | phase-active | ~94% |
| Balanced random order | obs-only | 68% |
| Balanced random order | phase-active | 96% |

The large phase-active gain for MLP-BC shows that the scripted phase and active
stick provide a strong oracle context signal. The balanced random-order
obs-only result also shows that order randomization acts like useful data
augmentation by exposing both sticks in both temporal roles.

## DPFM Results So Far

All results below are closed-loop evaluations over 100 episodes unless marked
pending.

| Task | Conditioning | DPFM Setting | Success | Notes |
|---|---|---|---:|---|
| Fixed order | obs-only | `h8_e4_i10` | 16% | Better than fixed obs-only MLP, but still weak. |
| Fixed order | phase-active | `h8_e4_i10` | 11% | Unexpectedly far below phase-active MLP. |
| Random order | obs-only | `h8_e4_i10` | 56% | Decent, but below MLP obs-only. |
| Random order | phase-active | `h8_e4_i10` | 29% | Context helps vs fixed phase-active, but remains poor. |
| Fixed order | phase-active | `h8_e1_i10` playback | 31% | Same checkpoint, execute one action per replan. |
| Random order | phase-active | `h8_e1_i10` playback | 42% | Shows stale context during cached execution matters. |
| Fixed order | phase-active | `h4_e1_i10` | 31% | Shorter chunks did not improve fixed order beyond `h8_e1`. |
| Random order | phase-active | `h4_e1_i10` | 54% | Improvement over `h8_e4`, but still far below MLP. |
| Random order | obs-only | `h4_e1_i10` | 66% | Essentially matches MLP obs-only random order. |
| Fixed order | obs-only | `h4_e1_i10` | pending | Training/evaluation in progress. |
| Random order | phase-active boundary-aware | `h8_e1_i10` | 47% | Split chunks at `(phase, active_stick)` boundaries. |
| Random order | phase-active boundary-aware + runtime replan | `h8_e1_i10` | 52% | Same checkpoint, discard cached chunks on context change. |
| Random order | phase-active boundary-aware | `h4_e1_i10` | 45% | Shorter boundary-aware horizon did not help. |
| Random order | phase-active boundary-aware + padded-tail mask | `h4_e1_i10`, 300 epochs | 53% | Best boundary-aware result, but close to prior non-boundary `h4_e1`. |
| Random order | phase-active boundary-aware + padded-tail mask | `h4_e1_i10`, 1000 epochs | 35% | Longer training degraded closed-loop performance. |

The key result is that `h4_e1` obs-only random-order DPFM reaches 66%, close to
the 68% MLP obs-only baseline. This suggests the PR #14 DPFM architecture is
not fundamentally broken on the randomized N=2 task. The remaining failure is
specific to how phase-active context is represented for action chunks.

We found that boundary-aware chunking improved over the original random-order
phase-active `h8_e4_i10` checkpoint at 29%, but it did not close the gap to
phase-active MLP-BC. The best boundary-aware run was the masked `h4_e1_i10`
checkpoint at 53%, while the existing non-boundary phase-active `h4_e1_i10`
result was 54% and obs-only random-order `h4_e1_i10` reached 66%. We will treat
the boundary-aware phase-conditioning result as useful analysis, not as the
strongest behavior-cloning baseline for residual RL.

## Chunk-Boundary Diagnostic

A temporary local diagnostic measured how often DPFM action chunks cross phase
or active-stick boundaries in the demonstration data. For `h8_e4` on the N=2
datasets:

| Dataset | Phase Change in Executed Prefix | Phase Change in Full Horizon | Active-Stick Change in Executed Prefix | Active-Stick Change in Full Horizon |
|---|---:|---:|---:|---:|
| Fixed order | 15.8% | 36.9% | 1.2% | 2.8% |
| Random order | 15.8% | 36.9% | 1.2% | 2.8% |

For random-order `h4_e1`, the executed prefix is phase-consistent by
construction, but the full training horizon still crosses a phase boundary
15.8% of the time. This matters because DPFM is trained against the whole
predicted chunk, not only the first executed action.

## Interpretation

The phase-active MLP succeeds because each supervised action is paired with the
phase that generated it. The phase-active DPFM checkpoint is instead trained to
predict future action sequences from a single current phase label. Near phase
boundaries, this creates inconsistent supervision:

```text
input context: phase = APPROACH
target chunk:  APPROACH, APPROACH, DESCEND, DESCEND, ...
```

Thus the model can be penalized unless it predicts future actions from later
phases while still conditioned on the earlier phase. Reducing `execute_steps`
improves playback because the online phase tracker is refreshed more often.
Reducing `H` also helps somewhat because fewer targets cross boundaries. But
neither fully fixes the training mismatch.

The result pattern supports the following diagnosis:

1. Stale context during cached chunk execution hurts phase-active DPFM.
2. Mixed-phase training targets are a larger issue.
3. Observation-only DPFM is competitive because it avoids this explicit
   single-label-per-chunk mismatch.

## Candidate Fixes

### 1. Boundary-Aware Chunks

This is the next planned implementation because it keeps the current model
architecture. During training, split each episode into contiguous
`(phase, active_stick)` segments and sample chunks only within those segments.
If a chunk reaches a segment boundary, pad by repeating the last in-segment
action.

Example:

```text
phase:   A   A   A | D   D   D
action:  a0  a1  a2| a3  a4  a5
```

Current target from `t=0`:

```text
[a0, a1, a2, a3, a4, a5]
```

Boundary-aware target:

```text
[a0, a1, a2, a2, a2, a2]
```

This tells the model by supervision that a chunk conditioned on `phase=A`
should not learn future `D` actions. It does not require a learned phase
detector because training demos already store `phase` and `active_stick`.
Inference still uses the current scripted phase tracker after each observed
environment step.

Recommended first runs:

- phase-active fixed order, boundary-aware `h8_e1_i10`
- phase-active random order, boundary-aware `h8_e1_i10`
- phase-active fixed order, boundary-aware `h4_e1_i10`
- phase-active random order, boundary-aware `h4_e1_i10`

If boundary-aware `h8` works, it recovers more sequence modeling while avoiding
cross-phase supervision.

Implementation notes:

- Boundary-aware chunks are opt-in via `--phase-boundary-chunks` and are rejected
  unless `--conditioning phase-active` is also set.
- Default DPFM training still chunks by full episode and stores
  `chunking="episode"` in the checkpoint. Boundary-aware runs store
  `chunking="phase_boundary"`.
- Training and validation checkpoints now include chunking diagnostics: segment
  count, chunk count, padded-tail fraction, and mean/min/max segment length.
- At a segment boundary, the remaining target actions are padded by repeating the
  last action inside the current `(phase, active_stick)` segment. The masked
  version stores `loss_masking="padded_tail"` and excludes those repeated tail
  timesteps from the flow-matching loss. Default episode chunking still stores
  `loss_masking="none"` and keeps the existing episode-end padding behavior.
- Playback has an optional `--dpfm-replan-on-context-change` flag. When enabled,
  a phase-active DPFM policy discards a cached sampled chunk before the next
  cached action if the current tracked `(phase, active_stick)` differs from the
  context used to sample that chunk. This is evaluation behavior only and is not
  stored in checkpoints.

For the random-order `h4_e1_i10` boundary-aware checkpoint, we measured:

| Split | Segments | Chunks | Padded-Tail Chunks | Padded-Tail Fraction | Mean Segment Length | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2529 | 44526 | 7571 | 17.0% | 17.6 | 1 | 34 |
| Validation | 280 | 4929 | 840 | 17.0% | 17.6 | 8 | 32 |

We used this diagnostic to justify the padded-tail loss mask. Masking removed
synthetic repeated tail actions from the flow-matching loss, but the 300-epoch
masked run only reached 53%, and the 1000-epoch masked run dropped to 35%. Our
interpretation is that padded-tail loss was a real cleanup issue, but it was not
the dominant cause of the phase-active DPFM gap.

### 2. Per-Timestep Context Sequence

This is a more principled architecture change. Instead of conditioning the
whole chunk on one phase/active vector, provide a context vector for each
predicted timestep:

```text
global input: obs_t
local context: [(phase_t, active_t), ..., (phase_t+H-1, active_t+H-1)]
output: [action_t, ..., action_t+H-1]
```

In the temporal U-Net, the local context sequence could be concatenated to the
noisy action sequence along the channel dimension:

```text
noisy actions: (B, H, action_dim)
local context: (B, H, context_dim)
U-Net input:   (B, H, action_dim + context_dim)
U-Net output:  (B, H, action_dim)
```

Training is straightforward because demos contain future phase labels. Inference
is harder because future observations and future phase labels are not known
before executing the chunk. This would require either a learned future context
predictor, a model-predictive rollout, or executing very short chunks.

### 3. Learned Future Context Predictor

A learned context model could predict future phase and active-stick labels from
current state or short history:

```text
history/current obs -> [(phase_t, active_t), ..., (phase_t+H-1, active_t+H-1)]
```

This would make per-timestep context usable at inference, but introduces a
second learned component and new failure modes. A reasonable progression would
be:

1. Train/evaluate per-timestep DPFM with oracle future labels offline.
2. Train a context predictor from demos.
3. Evaluate DPFM using predicted future context.
4. If needed, train DPFM with noisy or predicted context to reduce train/test
   mismatch.

## Current Recommendation

We will use the N=2 DPFM implementation as the behavior-cloning base for
residual RL, with random-order obs-only `h4_e1_i10` as the strongest current
base checkpoint and masked boundary-aware phase-active `h4_e1_i10` as an
optional ablation. The obs-only checkpoint is the more pragmatic residual-RL
starting point because it has the best closed-loop DPFM success so far at 66%.

For the final report, we'll probably present boundary-aware chunks as a
negative/diagnostic result: they improved over the stale-context `h8_e4`
phase-active DPFM setting, but did not outperform the simpler non-boundary
`h4_e1` phase-active checkpoint or the obs-only `h4_e1` baseline. Our working
explanation is that a single current phase label is not an ideal conditioning
interface for a sampled future action trajectory. If time remains for BC-side
iteration, we will try continuous subgoal conditioning, but the immediate
project priority is residual RL on top of the strongest N=2 DPFM baseline.
