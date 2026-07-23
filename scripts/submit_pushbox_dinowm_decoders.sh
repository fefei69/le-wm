#!/bin/bash
# Submit post-hoc pixel decoders for the completed 1h/2h/4h DINO-WM curves.

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export HF_HOME="${HF_HOME:-/scratch/cw5167/huggingface}"
LABELS=("$@")
if [[ ${#LABELS[@]} -eq 0 ]]; then
    LABELS=(1h 2h 4h)
fi

cd "$REPO"
source .venv/bin/activate
mkdir -p "$REPO/outputs" "$HF_HOME"
python scripts/cache_dinov2_small.py --local-only

for label in "${LABELS[@]}"; do
    if [[ ! "$label" =~ ^(1h|2h|4h)$ ]]; then
        echo "curve label must be 1h, 2h, or 4h; got: $label" >&2
        exit 1
    fi
    model_dir="dinowm_dinov2s_prop_${label}"
    checkpoint="${STABLEWM_HOME}/checkpoints/pushbox/${model_dir}/weights_epoch_10.pt"
    out_dir="${STABLEWM_HOME}/checkpoints/pushbox/${model_dir}/decoder_vqvae"
    final_recon="${out_dir}/recon_step020000.png"
    if [[ ! -f "$checkpoint" ]]; then
        echo "missing checkpoint: $checkpoint" >&2
        exit 1
    fi
    if [[ -f "$final_recon" ]]; then
        echo "skip completed ${label} decoder: $out_dir"
        continue
    fi

    submission=$(sbatch --parsable \
        --job-name="dinowm-dec-${label}" \
        --output="$REPO/outputs/pushbox-dinowm-decoder-${label}-%j.out" \
        --export="ALL,CURVE_LABEL=${label}" \
        "$REPO/scripts/train_pushbox_dinowm_decoder.sbatch")
    job_id=${submission%%;*}
    printf '%s decoder: job %s; outputs %s\n' "$label" "$job_id" "$out_dir"
done
