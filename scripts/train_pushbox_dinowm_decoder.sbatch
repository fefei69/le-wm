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

# Accept either MODEL_DIR (data-scaling sweep, e.g. dinowm_dinov2s_prop_25pct)
# or CURVE_LABEL (the original 1h/2h/4h curves). The final checkpoint epoch is
# discovered from disk, so this works regardless of how many epochs the run
# took under matched optimizer steps.
STEPS="${STEPS:-20000}"
CURVE_LABEL="${CURVE_LABEL:-}"
MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
    if [[ -z "$CURVE_LABEL" ]]; then
        echo "set MODEL_DIR or CURVE_LABEL" >&2
        exit 1
    fi
    MODEL_DIR="dinowm_dinov2s_prop_${CURVE_LABEL}"
fi
if [[ ! "$MODEL_DIR" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "unsupported MODEL_DIR: $MODEL_DIR" >&2
    exit 1
fi
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer; got: $STEPS" >&2
    exit 1
fi

model_root="${STABLEWM_HOME}/checkpoints/pushbox/${MODEL_DIR}"
manifest="${model_root}/split_manifest.json"
dataset="${STABLEWM_HOME}/datasets/pushbox_pilot_train.h5"
out_dir="${model_root}/decoder_vqvae"

# The highest weights_epoch_<N>.pt is the fully trained world model.
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

echo "decoding DINO-WM ${MODEL_DIR} at epoch ${final_epoch}"
python train_dino_decoder.py \
    --checkpoint "pushbox/${MODEL_DIR}/weights_epoch_${final_epoch}.pt" \
    --dataset "$dataset" \
    --manifest "$manifest" \
    --steps "$STEPS" \
    --wandb-name "decoder-${MODEL_DIR}" \
    --out "$out_dir"
