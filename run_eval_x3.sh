#!/bin/bash
set -euo pipefail

# Residual action dimension: 0.3
# WANDB run: https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/crezep81

OUT="dpfm_actlr5e-6_crtlr1e-4_h512_e0.3.txt"
: > "$OUT"

for i in 1 2 3; do
    echo "=== Run $i ===" | tee -a "$OUT"
    python -m scripts.play_env --config configs/stick_reorder.yaml \
       --rrl-checkpoint checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512_a0.3/td3_step460000.pt \
       --rrl-config configs/residual_td3.yaml \
       --dpfm_checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
       --dpfm-stochastic --num-sticks 1 --action-scale 0.3 --episodes 100 | tee -a "$OUT"
done


# Residual action dimension: 0.2
# WANDB run: https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/ctes47gj


OUT="dpfm_actlr5e-6_crtlr1e-4_h512_e0.2.txt"
: > "$OUT"

for i in 1 2 3; do
    echo "=== Run $i ===" | tee -a "$OUT"
    python -m scripts.play_env --config configs/stick_reorder.yaml \
       --rrl-checkpoint checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512/td3_step460000.pt \
       --rrl-config configs/residual_td3.yaml \
       --dpfm_checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
       --dpfm-stochastic --num-sticks 1 --action-scale 0.2 --episodes 100 | tee -a "$OUT"
done

# 4 enviuronment steps per 1 gradient update
# https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/idw7odys

OUT="dpfm_actlr5e-6_crtlr1e-4_h512_utd0.25.txt"
: > "$OUT"

for i in 1 2 3; do
    echo "=== Run $i ===" | tee -a "$OUT"
    python -m scripts.play_env --config configs/stick_reorder.yaml \
       --rrl-checkpoint checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512_utd0.25/td3_step920000.pt \
       --rrl-config configs/residual_td3.yaml \
       --dpfm_checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
       --dpfm-stochastic --num-sticks 1 --action-scale 0.2 --episodes 100 | tee -a "$OUT"
done

# No reward shaping
# https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/r0f00lyd
OUT="dpfm_actlr5e-6_crtlr1e-4_h512_noshape.txt"
: > "$OUT"

for i in 1 2 3; do
    echo "=== Run $i ===" | tee -a "$OUT"
    python -m scripts.play_env --config configs/stick_reorder.yaml \
       --rrl-checkpoint checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512_noshape/td3_step430000.pt \
       --rrl-config configs/residual_td3.yaml \
       --dpfm_checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
       --dpfm-stochastic --num-sticks 1 --action-scale 0.2 --no-reward-shaping --episodes 100 | tee -a "$OUT"
done

#
## Smaller actor/critic networks - 128 llayers instead of 512
## https://wandb.ai/alexta-uw/cs224r-wire-untangling/runs/r0f00lyd
#OUT="dpfm_actlr5e-6_crtlr1e-4_h512_noshape.txt"
#: > "$OUT"
#
#for i in 1 2 3; do
#    echo "=== Run $i ===" | tee -a "$OUT"
#    python -m scripts.play_env --config configs/stick_reorder.yaml \
#       --rrl-checkpoint checkpoints/dpfm_actlr5e-6_crtlr1e-4_h512_noshape/td3_step430000.pt \
#       --rrl-config configs/residual_td3.yaml \
#       --dpfm_checkpoint checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt \
#       --dpfm-stochastic --num-sticks 1 --action-scale 0.2 --episodes 100 | tee -a "$OUT"
#done
#
