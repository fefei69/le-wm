#!/bin/bash
# Cache DINOv2-small, then submit the nested 1 h / 2 h / full-data curves.

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export HF_HOME="${HF_HOME:-/scratch/cw5167/huggingface}"
HOURS=("$@")
if [[ ${#HOURS[@]} -eq 0 ]]; then
    HOURS=(1 2 4)
fi

cd "$REPO"
source .venv/bin/activate
mkdir -p "$REPO/outputs" "$HF_HOME"

if ! python scripts/cache_dinov2_small.py --local-only; then
    echo "DINOv2-small is not cached; downloading it once before submission."
    python scripts/cache_dinov2_small.py
fi

for hours in "${HOURS[@]}"; do
    if [[ ! "$hours" =~ ^(1|2|4)$ ]]; then
        echo "curve hours must be 1, 2, or 4; got: $hours" >&2
        exit 1
    fi
    label="${hours}h"
    model_dir="dinowm_dinov2s_prop_${label}"
    run_label="dinowm-dinov2s-prop-${label}"
    submission=$(sbatch --parsable \
        --job-name="dinowm-pb-${label}" \
        --output="$REPO/outputs/pushbox-dinowm-${label}-%j.out" \
        --export="ALL,TRAIN_HOURS=${hours},CURVE_LABEL=${label},MODEL_DIR=${model_dir},RUN_LABEL=${run_label}" \
        "$REPO/scripts/train_pushbox_dinowm.sbatch")
    job_id=${submission%%;*}
    printf '%s curve: job %s; checkpoint pushbox/%s\n' \
        "$label" "$job_id" "$model_dir"
done
