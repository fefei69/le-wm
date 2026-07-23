#!/bin/bash
# Submit ViT-small PushBox LeWM training, then the decoder probe once it succeeds.

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm

wm_submit=$(sbatch --parsable "$REPO/scripts/train_pushbox_small.sbatch")
wm_job=${wm_submit%%;*}
decoder_submit=$(
    sbatch --parsable --dependency="afterok:${wm_job}" \
        "$REPO/scripts/train_pushbox_small_decoder.sbatch"
)
decoder_job=${decoder_submit%%;*}

echo "world model (ViT-small) job: $wm_job"
echo "decoder probe job:           $decoder_job (starts after job $wm_job succeeds)"
