#!/bin/bash
# Submit the PushBox data-scaling sweep: each world model trained on 1/4, 1/2,
# 3/4, and all of the training split, for real-world closed-loop comparison.
#
#   bash scripts/submit_pushbox_data_scaling.sh                  # all models
#   bash scripts/submit_pushbox_data_scaling.sh dinowm           # one model
#   bash scripts/submit_pushbox_data_scaling.sh lewm_small 0.25  # one cell
#
# The subsets are nested and shared across architectures: the same episode
# split (seed 3072) and subset seed (8317) mean every model at a given fraction
# trains on exactly the same episodes, and every model is validated on the same
# held-out 23 episodes. Set MATCH_STEPS=0 to train a fixed number of epochs
# instead of a fixed number of optimizer steps.

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
export STABLEWM_HOME="${STABLEWM_HOME:-/scratch/cw5167/stable-wm}"
export HF_HOME="${HF_HOME:-/scratch/cw5167/huggingface}"
MATCH_STEPS="${MATCH_STEPS:-1}"
WITH_DECODER="${WITH_DECODER:-1}"  # also queue a dependent pixel decoder per WM

ALL_MODELS=(dinowm lewm_small lewm)
DEFAULT_MODELS=(dinowm lewm_small)
FRACTIONS=(0.25 0.5 0.75 1.0)

models=()
fractions=()
for arg in "$@"; do
    case "$arg" in
        dinowm|lewm_small|lewm) models+=("$arg") ;;
        0.25|0.5|0.75|1.0) fractions+=("$arg") ;;
        all) models=("${ALL_MODELS[@]}") ;;
        *)
            echo "unknown argument: $arg" >&2
            echo "expected one of: ${ALL_MODELS[*]} all ${FRACTIONS[*]}" >&2
            exit 1
            ;;
    esac
done
[[ ${#models[@]} -eq 0 ]] && models=("${DEFAULT_MODELS[@]}")
[[ ${#fractions[@]} -eq 0 ]] && fractions=("${FRACTIONS[@]}")

cd "$REPO"
source .venv/bin/activate
mkdir -p "$REPO/outputs" "$HF_HOME"

# Cache the backbone once on a login node; compute nodes run with HF_HUB_OFFLINE.
for model in "${models[@]}"; do
    if [[ "$model" == dinowm ]]; then
        if ! python scripts/cache_dinov2_small.py --local-only; then
            echo "DINOv2-small is not cached; downloading it once before submission."
            python scripts/cache_dinov2_small.py
        fi
        break
    fi
done

label_for() {
    # 0.25 -> 25pct, 1.0 -> 100pct
    python - "$1" <<'PY'
import sys
print(f"{round(float(sys.argv[1]) * 100)}pct")
PY
}

for model in "${models[@]}"; do
    for fraction in "${fractions[@]}"; do
        label="$(label_for "$fraction")"
        case "$model" in
            dinowm)
                script=scripts/train_pushbox_dinowm.sbatch
                model_dir="dinowm_dinov2s_prop_${label}"
                run_label="dinowm-dinov2s-prop-${label}"
                exports="TRAIN_FRACTION=${fraction},MATCH_STEPS=${MATCH_STEPS}"
                ;;
            lewm_small|lewm)
                script=scripts/train_pushbox_lewm_fraction.sbatch
                config=$([[ "$model" == lewm_small ]] && echo pushbox_small || echo pushbox)
                model_dir="${model}_${label}"
                run_label="${model//_/-}-${label}"
                # Matched steps multiply the epoch count by 1/fraction; widen the
                # checkpoint interval so every run writes a similar number of files.
                interval=1
                if [[ "$MATCH_STEPS" == 1 ]]; then
                    interval="$(python -c "import sys; print(max(1, round(1 / float(sys.argv[1]))))" "$fraction")"
                fi
                exports="TRAIN_FRACTION=${fraction},MATCH_STEPS=${MATCH_STEPS}"
                exports+=",CONFIG_NAME=${config},CKPT_INTERVAL=${interval}"
                ;;
        esac

        submission=$(sbatch --parsable \
            --job-name="${model}-${label}" \
            --output="$REPO/outputs/pushbox-${run_label}-%j.out" \
            --export="ALL,${exports},CURVE_LABEL=${label},MODEL_DIR=${model_dir},RUN_LABEL=${run_label}" \
            "$REPO/$script")
        wm_job="${submission%%;*}"
        printf '%-11s %-7s wm  job %s -> checkpoints/pushbox/%s\n' \
            "$model" "$label" "$wm_job" "$model_dir"

        # Each world model gets its own pixel decoder, queued to start only after
        # the world model finishes (afterok). DINO-WM decodes frozen backbone
        # tokens; LeWM decodes its projected planning latent.
        [[ "$WITH_DECODER" == 1 ]] || continue
        case "$model" in
            dinowm) dec_script=scripts/train_pushbox_dinowm_decoder.sbatch ;;
            lewm_small|lewm) dec_script=scripts/train_pushbox_lewm_decoder.sbatch ;;
        esac
        dec_submission=$(sbatch --parsable \
            --dependency="afterok:${wm_job}" \
            --kill-on-invalid-dep=yes \
            --job-name="${model}-dec-${label}" \
            --output="$REPO/outputs/pushbox-${run_label}-decoder-%j.out" \
            --export="ALL,MODEL_DIR=${model_dir},CURVE_LABEL=${label}" \
            "$REPO/$dec_script")
        printf '%-11s %-7s dec job %s (after %s)\n' \
            "$model" "$label" "${dec_submission%%;*}" "$wm_job"
    done
done
