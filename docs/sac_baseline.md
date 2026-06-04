# SAC Baseline

This baseline trains a standard Stable-Baselines3 SAC policy for the one-stick
task. The SAC actor outputs the full robosuite action directly.

DPFM is not used during SAC training or inference. The only offline signal is
expert HDF5 replay seeding from the same demonstration source used by residual
TD3.

## Result

The initial N=1 demo-seeded SAC run reached 33% success. This is substantially
below the residual-RL runs on top of a DPFM behavior-cloning base, supporting the
claim that learning the full manipulation policy from scratch remains difficult
even with expert replay seeding. The result motivates the residual-RL setup:
keep a competent DPFM BC policy as the base behavior, then learn targeted
corrections on top.

## Train Locally

```bash
python -m scripts.train_sac_baseline \
    --env-config configs/stick_reorder.yaml \
    --sac-config configs/sac_baseline.yaml \
    --num-sticks 1 \
    --demos-path data/stick_n1_orientation_demos.hdf5 \
    --checkpoint-dir checkpoints/sac_demo_seeded_n1
```

Short smoke run:

```bash
python -m scripts.train_sac_baseline \
    --total-timesteps 1000 \
    --demos-path data/stick_n1_orientation_demos.hdf5 \
    --checkpoint-dir checkpoints/sac_demo_seeded_n1_smoke \
    --no-wandb
```

Pure online SAC ablation:

```bash
python -m scripts.train_sac_baseline \
    --no-demo-seed \
    --checkpoint-dir checkpoints/sac_online_n1
```

Sparse reward ablation:

```bash
python -m scripts.train_sac_baseline \
    --no-reward-shaping \
    --checkpoint-dir checkpoints/sac_demo_seeded_n1_sparse
```

## Train On Modal

```bash
modal run --detach modal_train_sac_baseline.py \
    --demos-path /data/stick_n1_orientation_demos.hdf5 \
    --checkpoint-dir checkpoints/sac_demo_seeded_n1
```

Override Modal resources with the same environment variables used by the
residual-RL entrypoints:

```bash
MODAL_GPU=A100 MODAL_CPU=8 modal run --detach modal_train_sac_baseline.py \
    --demos-path /data/stick_n1_orientation_demos.hdf5 \
    --checkpoint-dir checkpoints/sac_demo_seeded_n1
```

If the demo file is only local:

```bash
modal run --detach modal_train_sac_baseline.py \
    --demos-path data/stick_n1_orientation_demos.hdf5 \
    --upload-demos
```

## Evaluate

`play_env.py` already supports SB3 SAC checkpoints:

```bash
python -m scripts.play_env \
    --config configs/stick_reorder.yaml \
    --sac_checkpoint checkpoints/sac_demo_seeded_n1/sac_final.zip \
    --num-sticks 1 \
    --episodes 100
```
