# Ablations

This document is a running log of ablations we have conducted, the conclusions
we can draw from them, and follow-up ablations that are worth testing next.

## Conducted

### Learned Low-Dimensional Context For Phase-Active BC

Question: can we replace oracle phase/active-stick context with a supervised
predictor trained from demonstrations?

Setup:

- Dataset: N=2 random-order expert demonstrations.
- Predictor input: current raw GymWrapper observation.
- Predictor targets: 8-way `phase` and 2-way `active_stick`.
- Policy using context: phase-active MLP-BC.
- Context used at rollout: hard argmax one-hot phase and active-stick labels.

Offline supervised metrics were strong:

- Train active-stick accuracy: `0.99957`
- Train phase accuracy: `0.99105`
- Train joint accuracy: `0.99062`
- Validation active-stick accuracy: `0.99739`
- Validation phase accuracy: `0.97735`
- Validation joint accuracy: `0.97495`

Closed-loop rollout performance dropped sharply:

- Learned-context phase-active BC: `47/100` success
  - Order `[0, 1]`: `25/50`
  - Order `[1, 0]`: `22/50`
  - Reward: `-188.534 +/- 134.035`
- Learned-context phase-active BC repeat: `46/100` success
  - Order `[0, 1]`: `20/50`
  - Order `[1, 0]`: `26/50`
  - Reward: `-184.777 +/- 130.559`
- Oracle-context phase-active BC: `99/100` success
  - Order `[0, 1]`: `50/50`
  - Order `[1, 0]`: `49/50`
  - Reward: `-58.003 +/- 16.042`

During learned-context rollout, learned-vs-oracle context disagreement was high:

- Phase disagreement: `53889/65872` (`81.8%`)
- Active-stick disagreement: `19151/65872` (`29.1%`)
- Joint disagreement: `54929/65872` (`83.4%`)

Interpretation:

Offline accuracy is measured on expert demonstration states, but learned-context
rollout is closed-loop. Once the predicted context causes a wrong action, BC can
visit states outside the expert distribution, and context prediction errors can
compound.

This result is best framed as an ablation of phase-active BC's dependence on
perfect context, not as a deployment-ready context estimator. The current
predictor consumes simulator state, including object poses, so it does not solve
the observation gap for physical rollout.

## Next Ablations

### Oracle-Controlled Rollout With Learned Context Logging

Run oracle-context phase-active BC, but log learned phase/active-stick
predictions in parallel without feeding them to the policy.

This tests whether the learned predictor generalizes to BC rollout observations
when its predictions are not allowed to affect the trajectory.

### Recurrent Context Predictor

Train an LSTM or other recurrent context predictor from observation history.
This tests whether temporal memory helps with phases that are not identifiable
from a single frame, especially `GRASP` and `RELEASE`.

Important caveat: if the recurrent predictor still consumes simulator state, it
is still a simulator-state ablation. It addresses memory/timing ambiguity, but
it does not make the method deployable on a real robot unless the inputs are
changed to deployable observations.

### History-Window MLP

Concatenate a short fixed window of recent raw observations and predict context
from that window. This is simpler than an LSTM and may capture enough progress
information for timer-like phases.

### Order / Task-Progress Conditioning

Add explicit order ID or task-progress features. This tests whether active-stick
errors come from ambiguity about intended order rather than visual or geometric
state estimation.

### Soft Context

Feed phase and active-stick probabilities to BC instead of hard argmax one-hot
labels. This tests whether uncertainty-aware context reduces transition errors.

### Class Weighting

Use class-weighted cross entropy for rare or short phases. This tests whether
transition-heavy phases are underfit by the current unweighted objective.

### Rollout-State Relabeling

Collect observations from BC rollouts and label them with the oracle tracker,
then train or fine-tune the context predictor on those states. This directly
targets the distribution shift seen in closed-loop learned-context rollout.
