#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=dinowm-decoder
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-dinowm-decoder-%j.out

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export HF_HOME="${HF_HOME:-/scratch/cw5167/huggingface}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

CURVE_LABEL="${CURVE_LABEL:?set CURVE_LABEL to 1h, 2h, or 4h}"
STEPS="${STEPS:-20000}"
if [[ ! "$CURVE_LABEL" =~ ^(1h|2h|4h)$ ]]; then
    echo "CURVE_LABEL must be 1h, 2h, or 4h; got: $CURVE_LABEL" >&2
    exit 1
fi
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer; got: $STEPS" >&2
    exit 1
fi

model_dir="dinowm_dinov2s_prop_${CURVE_LABEL}"
checkpoint="${STABLEWM_HOME}/checkpoints/pushbox/${model_dir}/weights_epoch_10.pt"
manifest="${STABLEWM_HOME}/checkpoints/pushbox/${model_dir}/split_manifest.json"
dataset="${STABLEWM_HOME}/datasets/pushbox_pilot_train.h5"
out_dir="${STABLEWM_HOME}/checkpoints/pushbox/${model_dir}/decoder_vqvae"
for path in "$checkpoint" "$manifest" "$dataset"; do
    if [[ ! -f "$path" ]]; then
        echo "missing required input: $path" >&2
        exit 1
    fi
done

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs" "$out_dir"
cd "$REPO"
source .venv/bin/activate

python scripts/cache_dinov2_small.py --local-only
export HF_HUB_OFFLINE=1

python train_dino_decoder.py \
    --checkpoint "pushbox/${model_dir}/weights_epoch_10.pt" \
    --dataset "$dataset" \
    --manifest "$manifest" \
    --steps "$STEPS" \
    --wandb-name "decoder-dinowm-dinov2s-prop-${CURVE_LABEL}" \
    --out "$out_dir"
