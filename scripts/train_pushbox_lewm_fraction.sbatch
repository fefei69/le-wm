#!/bin/bash
#SBATCH --account=torch_pr_595_tandon_advanced
#SBATCH --job-name=lewm-pushbox-frac
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/cw5167/workspace/le-wm/outputs/pushbox-lewm-frac-%j.out

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME=/scratch/cw5167/stable-wm
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
CONFIG_NAME="${CONFIG_NAME:-pushbox_small}"
MATCH_STEPS="${MATCH_STEPS:-0}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
CURVE_LABEL="${CURVE_LABEL:-frac${TRAIN_FRACTION}}"
MODEL_DIR="${MODEL_DIR:-lewm_small_${CURVE_LABEL}}"
RUN_LABEL="${RUN_LABEL:-lewm-small-${CURVE_LABEL}}"
dataset="${STABLEWM_HOME}/datasets/pushbox_pilot_train.h5"

if [[ ! "$TRAIN_FRACTION" =~ ^(0?\.[0-9]+|1(\.0+)?)$ ]]; then
    echo "TRAIN_FRACTION must be in (0, 1]; got: $TRAIN_FRACTION" >&2
    exit 1
fi
if [[ ! "$CONFIG_NAME" =~ ^pushbox(_small)?$ ]]; then
    echo "CONFIG_NAME must be pushbox or pushbox_small; got: $CONFIG_NAME" >&2
    exit 1
fi
if [[ ! "$MATCH_STEPS" =~ ^[01]$ ]]; then
    echo "MATCH_STEPS must be 0 or 1; got: $MATCH_STEPS" >&2
    exit 1
fi
if [[ ! "$CKPT_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "CKPT_INTERVAL must be a positive integer; got: $CKPT_INTERVAL" >&2
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
if compgen -G "${STABLEWM_HOME}/checkpoints/pushbox/${MODEL_DIR}/weights_epoch_*.pt" \
      >/dev/null && [[ "${ALLOW_OVERWRITE:-0}" != 1 ]]; then
    echo "checkpoints already exist for ${MODEL_DIR}; refusing to overwrite" >&2
    exit 1
fi

mkdir -p "$MPLCONFIGDIR" "$REPO/outputs"
cd "$REPO"
source .venv/bin/activate

match_args=()
if [[ "$MATCH_STEPS" == 1 ]]; then
    match_args+=(curve.match_optimizer_steps=true)
fi

# The run directory is deliberately job-id independent: train.py resumes from
# "$run_dir/<output_model_name>_weights.ckpt" when it exists, so a requeued job
# picks up where it stopped instead of restarting a multi-hour run. The W&B id
# interpolates from subdir and is stable for the same reason.
echo "training PushBox LeWM: config=${CONFIG_NAME} fraction=${TRAIN_FRACTION} model=${MODEL_DIR}"
python train.py \
    --config-name="$CONFIG_NAME" \
    curve.train_fraction="$TRAIN_FRACTION" \
    "${match_args[@]}" \
    checkpoint_interval="$CKPT_INTERVAL" \
    output_model_name="pushbox/${MODEL_DIR}" \
    subdir="pushbox-${RUN_LABEL}"
