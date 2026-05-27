#!/bin/bash
# Sweep DPFM training over normalizer type, chunk horizon, and integration steps.
# After each training run, evaluates the checkpoint and saves results.
# Results are stored in checkpoints/sweep_dpfm/<config_name>/

set -euo pipefail

DEMOS="data/demos.hdf5"
BASE_DIR="checkpoints/sweep_dpfm"
EVAL_EPISODES=100

NORMALIZERS=("zscore" "minmax" "identity")
HORIZONS=(8 16)
EXEC_STEPS=(4 8)
INTEGRATION_STEPS=(10 20)

for norm in "${NORMALIZERS[@]}"; do
  for i in "${!HORIZONS[@]}"; do
    horizon="${HORIZONS[$i]}"
    exec="${EXEC_STEPS[$i]}"
    for integ in "${INTEGRATION_STEPS[@]}"; do
      name="norm_${norm}_h${horizon}_e${exec}_i${integ}"
      ckpt_dir="${BASE_DIR}/${name}"
      ckpt_path="${ckpt_dir}/flow_matching_policy.pt"
      results_file="${BASE_DIR}/${name}.txt"

      echo "========================================"
      echo "Training: ${name}"
      echo "========================================"
      python -m scripts.train_flow_matching \
        --demos-path "$DEMOS" \
        --action-normalizer "$norm" \
        --action-chunk-horizon "$horizon" \
        --execute-steps "$exec" \
        --integration-steps "$integ" \
        --checkpoint-dir "$ckpt_dir" \
        --no-wandb

      echo "----------------------------------------"
      echo "Evaluating: ${name} (${EVAL_EPISODES} episodes)"
      echo "----------------------------------------"
      python -m scripts.play_env \
        --dpfm_checkpoint "$ckpt_path" \
        --num-sticks 1 \
        --episodes "$EVAL_EPISODES" \
        --results-file "$results_file"
    done
  done
done

echo ""
echo "========================================"
echo "Sweep complete. Summary:"
echo "========================================"
for f in "${BASE_DIR}"/*.txt; do
  name="$(basename "$f" .txt)"
  rate="$(grep 'success_rate' "$f" | head -1)"
  echo "  ${name}: ${rate}"
done
