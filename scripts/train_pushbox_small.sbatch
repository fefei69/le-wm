#!/bin/bash
# Tandon account unlocks the h200_tandon pool (112-GPU cap) which starts far
# sooner than the public pool (24-GPU); h200_public kept as a fallback. The
# cluster enforces account<->partition matching, so these two must move together.
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=lewm-pushbox-small
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-wm-small-%j.out

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

dataset="${STABLEWM_HOME}/datasets/pushbox_pilot_train.h5"
if [[ ! -f "$dataset" ]]; then
    echo "missing dataset: $dataset" >&2
    exit 1
fi

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs"
cd "$REPO"
source .venv/bin/activate

# Larger ViT-small encoder (embed_dim 384). Checkpoints land in
# checkpoints/pushbox/lewm_small/ and do not touch the ViT-tiny run. Time is
# bumped vs train_pushbox.sbatch since the encoder is ~4x the tiny params.
# PRED_DEPTH and MODEL_DIR make controlled predictor-capacity sweeps possible;
# their defaults exactly preserve the completed depth-6 experiment.
PRED_DEPTH="${PRED_DEPTH:-6}"
MODEL_DIR="${MODEL_DIR:-lewm_small}"
RUN_LABEL="${RUN_LABEL:-lewm-small}"

if [[ ! "$PRED_DEPTH" =~ ^[1-9][0-9]*$ ]]; then
    echo "PRED_DEPTH must be a positive integer, got: $PRED_DEPTH" >&2
    exit 1
fi
if [[ ! "$MODEL_DIR" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "MODEL_DIR contains unsupported characters: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUN_LABEL contains unsupported characters: $RUN_LABEL" >&2
    exit 1
fi

echo "training ViT-small WM: predictor_depth=${PRED_DEPTH} model_dir=${MODEL_DIR}"
python train.py \
    --config-name=pushbox_small \
    model.predictor.depth="${PRED_DEPTH}" \
    output_model_name="pushbox/${MODEL_DIR}" \
    subdir="pushbox-${RUN_LABEL}-${SLURM_JOB_ID}"
