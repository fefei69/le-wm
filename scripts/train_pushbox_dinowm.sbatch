#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=dinowm-pushbox
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-dinowm-%j.out

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export HF_HOME="${HF_HOME:-/scratch/cw5167/huggingface}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

TRAIN_HOURS="${TRAIN_HOURS:-4}"
CURVE_LABEL="${CURVE_LABEL:-${TRAIN_HOURS}h}"
MODEL_DIR="${MODEL_DIR:-dinowm_dinov2s_prop_${CURVE_LABEL}}"
RUN_LABEL="${RUN_LABEL:-dinowm-dinov2s-prop-${CURVE_LABEL}}"
dataset="${STABLEWM_HOME}/datasets/pushbox_pilot_train.h5"

if [[ ! "$TRAIN_HOURS" =~ ^(1|2|4)(\.0)?$ ]]; then
    echo "TRAIN_HOURS must be 1, 2, or 4; got: $TRAIN_HOURS" >&2
    exit 1
fi
for value in "$CURVE_LABEL" "$MODEL_DIR" "$RUN_LABEL"; do
    if [[ ! "$value" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "unsupported identifier: $value" >&2
        exit 1
    fi
done
if [[ ! -f "$dataset" ]]; then
    echo "missing dataset: $dataset" >&2
    exit 1
fi
if [[ -f "${STABLEWM_HOME}/checkpoints/pushbox/${MODEL_DIR}/weights_epoch_10.pt" \
      && "${ALLOW_OVERWRITE:-0}" != 1 ]]; then
    echo "final checkpoint already exists for ${MODEL_DIR}; refusing to overwrite" >&2
    exit 1
fi

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs"
cd "$REPO"
source .venv/bin/activate

# Compute nodes should use the model prepared once by the submission script.
python scripts/cache_dinov2_small.py --local-only
export HF_HUB_OFFLINE=1

echo "training PushBox DINO-WM: curve=${CURVE_LABEL} target=${TRAIN_HOURS}h model=${MODEL_DIR}"
python train_dinowm.py \
    curve.train_hours="$TRAIN_HOURS" \
    curve.label="$CURVE_LABEL" \
    output_model_name="pushbox/${MODEL_DIR}" \
    subdir="pushbox-${RUN_LABEL}-${SLURM_JOB_ID}"
