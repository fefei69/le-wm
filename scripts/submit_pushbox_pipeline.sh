#!/bin/bash
# Submit PushBox LeWM training and start the image decoder only after success.

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm

wm_submit=$(sbatch --parsable "$REPO/scripts/train_pushbox.sbatch")
wm_job=${wm_submit%%;*}
decoder_submit=$(
    sbatch --parsable --dependency="afterok:${wm_job}" \
        "$REPO/scripts/train_pushbox_decoder.sbatch"
)
decoder_job=${decoder_submit%%;*}

echo "world model job: $wm_job"
echo "image decoder job: $decoder_job (starts after job $wm_job succeeds)"
