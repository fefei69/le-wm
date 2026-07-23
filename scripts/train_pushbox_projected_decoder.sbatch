#!/bin/bash
#SBATCH --account=torch_pr_519_general
#SBATCH --job-name=pushbox-proj-dec-sweep
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-proj-dec-sweep-%j.out

# Train projected-latent decoders in experiment-priority order.
#
# Stage 1 holds patch size 14 fixed while comparing the recommended world-model
# checkpoints: 146 (primary), 140, then 150.
# Stage 2 holds epoch 146 fixed while testing patch sizes 8 and 16. Together with
# the epoch-146/patch-14 run from stage 1, this isolates the output-patch effect.
#
# Submit:
#   sbatch scripts/train_pushbox_projected_decoder.sbatch
#
# Optional overrides:
#   sbatch --export=ALL,STEPS=1000 scripts/train_pushbox_projected_decoder.sbatch
#   sbatch --export=ALL,SKIP_COMPLETED=0 scripts/train_pushbox_projected_decoder.sbatch
#   sbatch --export=ALL,DRY_RUN=1 scripts/train_pushbox_projected_decoder.sbatch

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
CKPT_DIR="${STABLEWM_HOME}/checkpoints/pushbox/lewm"

DATASET="${DATASET:-pushbox_pilot_train}"
STEPS="${STEPS:-20000}"
SEED="${SEED:-3072}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Priority order from docs/real_world_exp.md.
CHECKPOINT_EPOCHS=(146 140 150)
PRIMARY_EPOCH=146
PRIMARY_PATCH_SIZE=14
PATCH_SIZE_TESTS=(8 16)

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs"
cd "$REPO"
source .venv/bin/activate

for epoch in "${CHECKPOINT_EPOCHS[@]}"; do
    checkpoint="${CKPT_DIR}/weights_epoch_${epoch}.pt"
    if [[ ! -f "$checkpoint" ]]; then
        echo "missing world-model checkpoint: $checkpoint" >&2
        exit 1
    fi
done

if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer, got: $STEPS" >&2
    exit 1
fi

printf -v final_recon 'recon_step%06d.png' "$STEPS"

train_decoder() {
    local epoch="$1"
    local patch_size="$2"
    local relative_checkpoint="pushbox/lewm/weights_epoch_${epoch}.pt"
    local out_dir="${CKPT_DIR}/decoder_proj_ep${epoch}_p${patch_size}"

    if [[ "$SKIP_COMPLETED" == "1" \
          && -f "${out_dir}/decoder.pt" \
          && -f "${out_dir}/${final_recon}" ]]; then
        echo "skip completed epoch=${epoch} patch=${patch_size}: ${out_dir}"
        return
    fi

    echo
    echo "============================================================"
    echo "training projected decoder: epoch=${epoch} patch=${patch_size}"
    echo "checkpoint: ${relative_checkpoint}"
    echo "output:     ${out_dir}"
    echo "============================================================"

    local command=(python train_decoder.py \
        --checkpoint "$relative_checkpoint" \
        --dataset "$DATASET" \
        --latent proj \
        --patch-size "$patch_size" \
        --steps "$STEPS" \
        --seed "$SEED" \
        --out "$out_dir")

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'dry run:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return
    fi

    "${command[@]}"
}

echo "Stage 1/2: checkpoint comparison at patch size ${PRIMARY_PATCH_SIZE}"
for epoch in "${CHECKPOINT_EPOCHS[@]}"; do
    train_decoder "$epoch" "$PRIMARY_PATCH_SIZE"
done

echo "Stage 2/2: patch-size comparison at epoch ${PRIMARY_EPOCH}"
for patch_size in "${PATCH_SIZE_TESTS[@]}"; do
    train_decoder "$PRIMARY_EPOCH" "$patch_size"
done

echo "all projected-decoder experiments completed"
