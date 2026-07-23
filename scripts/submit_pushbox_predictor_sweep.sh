#!/bin/bash
# Submit parallel ViT-small predictor-depth experiments and dependent decoders.
#
# Usage:
#   bash scripts/submit_pushbox_predictor_sweep.sh       # depths 8 and 10
#   bash scripts/submit_pushbox_predictor_sweep.sh 8 12  # custom depths

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
DEPTHS=("$@")
if [[ ${#DEPTHS[@]} -eq 0 ]]; then
    DEPTHS=(8 10)
fi

mkdir -p "$REPO/outputs"

for depth in "${DEPTHS[@]}"; do
    if [[ ! "$depth" =~ ^[1-9][0-9]*$ ]]; then
        echo "predictor depths must be positive integers, got: $depth" >&2
        exit 1
    fi

    model_dir="lewm_small_pred_d${depth}"
    run_label="lewm-small-pred-d${depth}"

    wm_submit=$(sbatch --parsable \
        --job-name="lewm-pb-small-d${depth}" \
        --output="$REPO/outputs/pushbox-wm-small-pred-d${depth}-%j.out" \
        --export="ALL,PRED_DEPTH=${depth},MODEL_DIR=${model_dir},RUN_LABEL=${run_label}" \
        "$REPO/scripts/train_pushbox_small.sbatch")
    wm_job=${wm_submit%%;*}

    decoder_submit=$(sbatch --parsable \
        --dependency="afterok:${wm_job}" \
        --job-name="lewm-pb-small-d${depth}-dec" \
        --output="$REPO/outputs/pushbox-small-pred-d${depth}-decoder-%j.out" \
        --export="ALL,MODEL_DIR=${model_dir},RUN_LABEL=${run_label}" \
        "$REPO/scripts/train_pushbox_small_decoder.sbatch")
    decoder_job=${decoder_submit%%;*}

    printf 'depth %s: WM job %s; decoder job %s (afterok:%s)\n' \
        "$depth" "$wm_job" "$decoder_job" "$wm_job"
done
