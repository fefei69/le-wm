#!/bin/bash
# Tandon account -> h200_tandon pool (starts sooner), public as fallback.
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=lewm-pushbox-small-decoder
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-small-decoder-%j.out

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Decode from the ViT-small WM. Same probe config as the tiny plateau baseline
# (proj latent, patch 14) so tiny-ep146 (28.0 dB) vs small is apples-to-apples.
# Resolve the highest available epoch so this is robust if the WM job stopped
# before epoch 150.
MODEL_DIR="${MODEL_DIR:-lewm_small}"
if [[ ! "$MODEL_DIR" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "MODEL_DIR contains unsupported characters: $MODEL_DIR" >&2
    exit 1
fi

ckpt_dir="${STABLEWM_HOME}/checkpoints/pushbox/${MODEL_DIR}"
latest=$(ls "${ckpt_dir}"/weights_epoch_*.pt 2>/dev/null \
         | sed -E 's#.*/weights_epoch_([0-9]+)\.pt#\1#' | sort -n | tail -1)

if [[ -z "${latest}" ]]; then
    echo "no ViT-small checkpoint in ${ckpt_dir}; did the WM job finish?" >&2
    exit 1
fi

checkpoint="pushbox/${MODEL_DIR}/weights_epoch_${latest}.pt"
out_dir="${ckpt_dir}/decoder_proj_ep${latest}_p14"
echo "decoding from epoch ${latest}: ${checkpoint} -> ${out_dir}"

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs"
cd "$REPO"
source .venv/bin/activate

python train_decoder.py \
    --checkpoint "${checkpoint}" \
    --dataset pushbox_pilot_train \
    --latent proj \
    --patch-size 14 \
    --steps 20000 \
    --wandb-name "decoder-proj-${MODEL_DIR}-ep${latest}-p14" \
    --out "${out_dir}"
