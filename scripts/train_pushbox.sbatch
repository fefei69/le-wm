#!/bin/bash
#SBATCH --account=torch_pr_519_general
#SBATCH --job-name=lewm-pushbox
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-wm-%j.out

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

# Keep PushBox checkpoints separate from the existing PushT run. The dedicated
# config trains for 150 epochs, about 78k optimizer steps on this dataset.
python train.py \
    --config-name=pushbox \
    subdir="pushbox-lewm-${SLURM_JOB_ID}"
