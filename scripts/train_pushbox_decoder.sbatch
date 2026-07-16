#!/bin/bash
#SBATCH --account=torch_pr_519_general
#SBATCH --job-name=lewm-pushbox-decoder
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-decoder-%j.out

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

checkpoint="${STABLEWM_HOME}/checkpoints/pushbox/lewm/weights_epoch_150.pt"
if [[ ! -f "$checkpoint" ]]; then
    echo "missing world-model checkpoint: $checkpoint" >&2
    exit 1
fi

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs"
cd "$REPO"
source .venv/bin/activate

python train_decoder.py \
    --checkpoint pushbox/lewm/weights_epoch_150.pt \
    --dataset pushbox_pilot_train \
    --latent cls \
    --steps 20000 \
    --out "${STABLEWM_HOME}/checkpoints/pushbox/lewm/decoder"
