#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=lewm-decoder
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-lewm-decoder-%j.out

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Projected-latent decoder for one LeWM checkpoint (a data-scaling cell, e.g.
# lewm_small_25pct). Uses --latent proj so it renders the planning latent the
# rollout evolves, and reads validation_episode_indices from the run's manifest
# so it is scored on the world model's own held-out episodes. The final
# checkpoint epoch is discovered from disk.
STEPS="${STEPS:-20000}"
SEED="${SEED:-3072}"
PATCH_SIZE="${PATCH_SIZE:-14}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR, e.g. lewm_small_25pct}"
if [[ ! "$MODEL_DIR" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "unsupported MODEL_DIR: $MODEL_DIR" >&2
    exit 1
fi
for value in "$STEPS" "$PATCH_SIZE"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "STEPS and PATCH_SIZE must be positive integers; got: $value" >&2
        exit 1
    fi
done

model_root="${STABLEWM_HOME}/checkpoints/pushbox/${MODEL_DIR}"
manifest="${model_root}/split_manifest.json"
dataset="${DATASET:-pushbox_pilot_train}"
out_dir="${model_root}/decoder_proj"

final_epoch=-1
shopt -s nullglob
for f in "${model_root}"/weights_epoch_*.pt; do
    e="${f##*weights_epoch_}"; e="${e%.pt}"
    if [[ "$e" =~ ^[0-9]+$ ]] && (( 10#$e > final_epoch )); then
        final_epoch="$((10#$e))"
    fi
done
shopt -u nullglob
if (( final_epoch < 0 )); then
    echo "no weights_epoch_*.pt under ${model_root}" >&2
    exit 1
fi
checkpoint="${model_root}/weights_epoch_${final_epoch}.pt"
if [[ ! -f "$checkpoint" || ! -f "$manifest" ]]; then
    echo "missing checkpoint or manifest under ${model_root}" >&2
    exit 1
fi

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs" "$out_dir"
cd "$REPO"
source .venv/bin/activate

echo "decoding LeWM ${MODEL_DIR} at epoch ${final_epoch} (proj latent)"
python train_decoder.py \
    --checkpoint "pushbox/${MODEL_DIR}/weights_epoch_${final_epoch}.pt" \
    --dataset "$dataset" \
    --latent proj \
    --patch-size "$PATCH_SIZE" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --manifest "$manifest" \
    --wandb-name "decoder-${MODEL_DIR}-proj" \
    --out "$out_dir"
