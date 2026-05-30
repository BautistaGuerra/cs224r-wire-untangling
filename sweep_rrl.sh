#!/bin/bash
# Sweep residual RL training over action_scale × offline_fraction × dpfm_checkpoint.
# Runs up to 4 processes in parallel.

set -uo pipefail

BASE_DIR="checkpoints/sweep_rrl"
MAX_PARALLEL=4

DPFM_CHECKPOINTS=(
    "checkpoints/sweep_dpfm/norm_minmax_h8_e4_i20/flow_matching_policy.pt"
)
DPFM_NAMES=("minmax")

ACTION_SCALES=(0.1 0.2)
OFFLINE_FRACTIONS=(0.5)

running=0
mkdir -p "${BASE_DIR}"

for dp_idx in "${!DPFM_CHECKPOINTS[@]}"; do
    dpfm="${DPFM_CHECKPOINTS[$dp_idx]}"
    dp_name="${DPFM_NAMES[$dp_idx]}"
    for as in "${ACTION_SCALES[@]}"; do
        for of in "${OFFLINE_FRACTIONS[@]}"; do
            name="dpfm_${dp_name}_as${as}_of${of}"
            ckpt_dir="${BASE_DIR}/${name}"

            echo "========================================"
            echo "Launching: ${name}"
            echo "  dpfm: ${dpfm}"
            echo "  action_scale=${as} offline_fraction=${of}"
            echo "========================================"

            python -m scripts.train_residual_rl \
                --dpfm-checkpoint "$dpfm" \
                --action-scale "$as" \
                --offline-fraction "$of" \
                --checkpoint-dir "$ckpt_dir" \
                --num-sticks 1 \
                --demos-path data/demos.hdf5 \
                > "${ckpt_dir}.log" 2>&1 &

            running=$((running + 1))
            if [ "$running" -ge "$MAX_PARALLEL" ]; then
                echo "Waiting for batch of $MAX_PARALLEL to finish..."
                wait
                running=0
            fi
        done
    done
done

wait
echo ""
echo "========================================"
echo "Sweep complete. Results in ${BASE_DIR}/"
echo "========================================"
for log in "${BASE_DIR}"/*.log; do
    name="$(basename "$log" .log)"
    if grep -q "Final model saved" "$log" 2>/dev/null; then
        echo "  ${name}: DONE"
    else
        echo "  ${name}: INCOMPLETE (check ${log})"
    fi
done
