# DPFM Goal-Conditioning and Multi-Task Literature Notes

Date: 2026-05-30

These notes summarize the design decision after the boundary-aware chunking
experiments. The goal is to decide what is worth trying next for phase-active
DPFM on the two-stick random-order task.

## Current Finding

Boundary-aware chunks did not close the phase-active DPFM gap.

- Random-order phase-active boundary-aware `h8_e1_i10`: 47-52% success.
- Random-order phase-active boundary-aware `h4_e1_i10`: 45% success.
- Prior random-order phase-active non-boundary `h4_e1_i10`: 54% success.
- Prior random-order obs-only `h4_e1_i10`: 66% success.
- Phase-active MLP-BC random-order baseline: about 96% success.

The `h4_e1_i10` boundary-aware checkpoint diagnostics showed about 17% padded
tail chunks. That is high enough to be suspicious, but probably not high enough
by itself to explain the full MLP-vs-DPFM gap.

## Literature Takeaways

### Diffusion Policy

Diffusion Policy predicts an action horizon from recent observations, executes
only part of that horizon, then replans in closed loop. The paper separates
observation horizon, action prediction horizon, and action execution horizon,
which matches our `pred_horizon` and `execute_steps` setup.

Relevance: lowering `execute_steps` is a standard way to reduce stale context at
execution time. It does not fully solve training-time context mismatch when one
current phase label conditions target actions from future phases.

Reference:

- Cheng Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action
  Diffusion", IJRR 2025.
  https://journals.sagepub.com/doi/10.1177/02783649241273668

### Diffuser and Decision Diffuser

Diffuser frames planning as denoising a trajectory. Its denoising process can be
guided toward reward objectives or goal constraints at test time.

Decision Diffuser frames offline sequential decision making as conditional
generative modeling over trajectories. It conditions on returns, constraints, or
skills, and uses classifier-free guidance for flexible behavior composition.

Relevance: these methods condition on trajectory-level goals, constraints, or
skills. They do not rely on a transient per-step FSM label being stretched over
a whole future action chunk.

References:

- Michael Janner et al., "Planning with Diffusion for Flexible Behavior
  Synthesis", ICML 2022.
  https://proceedings.mlr.press/v162/janner22a.html
  https://diffusion-planning.github.io/
- Anurag Ajay et al., "Is Conditional Generative Modeling all you need for
  Decision-Making?", 2022.
  https://arxiv.org/abs/2211.15657

### Hierarchical Diffusion Policy

Hierarchical Diffusion Policy factorizes multi-task manipulation into a
high-level task planner and a low-level goal-conditioned diffusion policy. The
high-level policy predicts a next-best end-effector pose; the low-level policy
generates motion trajectories conditioned on that goal.

Relevance: this is the closest fit to our problem. The scripted expert already
has an implicit high-level state machine and per-phase target poses. We can use
those targets as continuous subgoal conditioning for DPFM instead of only
feeding phase labels.

Reference:

- Xiao Ma et al., "Hierarchical Diffusion Policy for Kinematics-Aware
  Multi-Task Robotic Manipulation", CVPR 2024.
  https://arxiv.org/abs/2403.03890

### Robot Latent Diffusion

RoLD decouples action trajectory encoding from control policy generation by
learning a latent action trajectory space and then diffusing in that latent
space. It reports improved multi-task manipulation performance.

Relevance: latent trajectory spaces are a plausible longer-term fix for
multi-task chunk modeling, but this is too large for the current project
timeline.

Reference:

- Wenhui Tan et al., "RoLD: Robot Latent Diffusion for Multi-task Policy
  Modeling", 2024.
  https://arxiv.org/abs/2403.07312

### Generalist and Multi-Task Robot Policies

Octo and recent multi-task DiT policies use language, goal images, or
multi-modal prompts as task conditioning. LeRobot's MultiTask DiT supports both
diffusion and flow-matching objectives for action generation conditioned on
language instructions.

Relevance: task conditioning should describe the intended task or goal. In our
setting, the closest low-cost analog is a structured subgoal vector computed
from the expert state machine and current observation.

References:

- Octo Model Team et al., "Octo: An Open-Source Generalist Robot Policy", 2024.
  https://arxiv.org/abs/2405.12213
- Hugging Face LeRobot MultiTask DiT documentation.
  https://huggingface.co/docs/lerobot/v0.5.1/multi_task_dit

### Flow Matching for Robot Policies and RL

Flow-matching robot policy papers use flow matching as an action trajectory
generator conditioned on observations. Newer work explores classifier-free
guidance and RL fine-tuning for flow-matching policies.

Relevance: guidance and RL fine-tuning are promising, but they should come after
the conditioning interface is fixed. RL should not be used to compensate for a
mis-specified behavior-cloning target.

References:

- Fan Zhang and Michael Gienger, "Robot Manipulation with Flow Matching", CoRL
  2024 workshop.
  https://openreview.net/pdf?id=l8DzhzIcEj
- Qinqing Zheng et al., "Guided Flows for Generative Modeling and Decision
  Making", 2023.
  https://arxiv.org/abs/2311.13443
- Samuel Pfrommer et al., "Reinforcement Learning for Flow-Matching Policies",
  2025.
  https://arxiv.org/abs/2507.15073
- David McAllister et al., "Flow Matching Policy Gradients", 2025.
  https://flowreinforce.github.io/

## Candidate Direction 1: Per-Timestep Phase/Active Context

At training time, this is feasible. The demos store `phase[t]` and
`active_stick[t]`, and these labels are captured before `action[t]` is executed.
For a chunk starting at `t`, the training target can include:

```text
actions:  a[t], ..., a[t+H-1]
context:  (phase[t], active[t]), ..., (phase[t+H-1], active[t+H-1])
```

The problem is inference. At deployment, only the current context is available
from the tracker and current observation. Future labels for `t+1 ... t+H-1` are
unknown until the policy executes actions and observes new states.

Possible inference approximations:

- Repeat the current `(phase, active_stick)` across the horizon.
- Use `execute_steps=1`, so only the first action uses a known-good context.
- Train a future phase/active predictor, which is likely too much new machinery.

Implementation estimate:

- Flattening future context sequence into global conditioning: medium.
- Proper per-timestep U-Net local conditioning channels: medium/high.
- Future context prediction: too large for the remaining timeline.

Decision: useful as a narrower ablation, but not the best next experiment.

## Candidate Direction 2: Continuous Subgoal Conditioning

Continuous subgoal conditioning replaces or augments `phase-active` with a
feature vector that describes the current local control objective.

Potential features:

- Active stick one-hot.
- Current phase one-hot.
- Target end-effector position, or delta from current EEF to target.
- Gripper command target.
- Relevant yaw error or yaw command.
- Active stick position and active goal position, if not already sufficiently
  represented in observation.

The scripted expert already computes phase-specific targets:

```text
APPROACH   -> above active stick, gripper open
DESCEND    -> grasp height, gripper open
GRASP      -> grasp pose, gripper close
LIFT       -> lift height, gripper close
TRANSPORT  -> above active goal, gripper close
PLACE      -> goal placement z, gripper close
RELEASE    -> goal placement z, gripper open
RETREAT    -> above goal, gripper open
```

Training can compute these features from demo observations plus demo phase and
active-stick labels. Inference can compute the same features from the current
observation plus the online phase tracker. This avoids needing future context
labels.

Implementation estimate:

- New conditioning mode, e.g. `phase-subgoal`: medium.
- Shared helper for subgoal feature computation: medium.
- Checkpoint metadata and inference wrapper support: small/medium.
- Tests for feature construction and checkpoint loading: small.

Decision: best next direction after a quick padded-tail mask cleanup. It is more
compatible with inference than per-timestep phase context and better aligned
with hierarchical/goal-conditioned diffusion policy literature.

## Candidate Direction 3: Padded-Tail Loss Mask

Boundary-aware chunking pads segment-end chunks by repeating the last in-segment
action. In v1, those padded timesteps still contribute to flow-matching loss.
This may teach the model to hold or stall near phase boundaries.

Implementation estimate: small.

Decision: do this as a cleanup experiment, but do not expect it alone to close
the full gap. The boundary-aware branch now stores `loss_masking="padded_tail"`
for masked runs and keeps default episode chunking at `loss_masking="none"`.

## Recommendation

1. Implement padded-tail loss masking for boundary-aware chunks.
2. Train/evaluate masked boundary-aware `h4_e1_i10` and possibly `h8_e1_i10`.
3. If masking does not produce a large improvement, implement continuous
   subgoal conditioning.
4. Defer per-timestep phase/active context unless we need a report ablation or
   have time to handle the inference mismatch carefully.
5. Defer RL fine-tuning until the behavior-cloning conditioning interface is
   cleaner.
