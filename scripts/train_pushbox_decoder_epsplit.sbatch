#!/bin/bash
#SBATCH --account=torch_pr_519_general
#SBATCH --job-name=lewm-pushbox-decoder-epsplit
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-decoder-epsplit-%j.out

# Post-hoc pixel probe on the PushBox LeWM checkpoints, scored on the world
# model's own held-out episodes instead of a leaky frame-level split.
#
# The earlier decoder runs under checkpoints/pushbox/lewm/decoder* validated on a
# random 1% of FRAMES, so nearly every validation frame had an adjacent frame in
# training. Those val curves measure interpolation, not generalization; this job
# writes to separate directories so they remain available for comparison.
#
# Required:
#   EPOCH        world-model epoch to probe
#   LATENT       cls | proj   (proj is the planning latent the simulator renders)
# Optional:
#   PATCH_SIZE   decoder output patch size (default 16)
#   STEPS        optimizer steps (default 20000)
#
#   sbatch --export=ALL,EPOCH=150,LATENT=cls scripts/train_pushbox_decoder_epsplit.sbatch

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export HF_HOME="${HF_HOME:-/scratch/cw5167/huggingface}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

EPOCH="${EPOCH:?set EPOCH to the world-model epoch to probe}"
LATENT="${LATENT:?set LATENT to cls or proj}"
PATCH_SIZE="${PATCH_SIZE:-16}"
STEPS="${STEPS:-20000}"

if [[ ! "$EPOCH" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPOCH must be a positive integer; got: $EPOCH" >&2
    exit 1
fi
if [[ "$LATENT" != "cls" && "$LATENT" != "proj" ]]; then
    echo "LATENT must be cls or proj; got: $LATENT" >&2
    exit 1
fi
for value in "$PATCH_SIZE" "$STEPS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "PATCH_SIZE and STEPS must be positive integers; got: $value" >&2
        exit 1
    fi
done

relative_checkpoint="pushbox/lewm/weights_epoch_${EPOCH}.pt"
checkpoint="${STABLEWM_HOME}/checkpoints/${relative_checkpoint}"
dataset="${STABLEWM_HOME}/datasets/pushbox_pilot_train.h5"
out_dir="${STABLEWM_HOME}/checkpoints/pushbox/lewm/decoder_${LATENT}_ep${EPOCH}_p${PATCH_SIZE}_epsplit"
for path in "$checkpoint" "$dataset"; do
    if [[ ! -f "$path" ]]; then
        echo "missing required input: $path" >&2
        exit 1
    fi
done

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs" "$out_dir"
cd "$REPO"
source .venv/bin/activate

# --split-fraction/--split-num-steps/--seed reproduce config/train/pushbox.yaml
# (train_split 0.9, history_size 3 + num_preds 1, seed 3072), so the decoder is
# scored on exactly the episodes the world model never trained on.
python train_decoder.py \
    --checkpoint "$relative_checkpoint" \
    --dataset pushbox_pilot_train \
    --latent "$LATENT" \
    --patch-size "$PATCH_SIZE" \
    --steps "$STEPS" \
    --split-fraction 0.9 \
    --split-num-steps 4 \
    --seed 3072 \
    --wandb-name "decoder-${LATENT}-ep${EPOCH}-p${PATCH_SIZE}-epsplit" \
    --out "$out_dir"
