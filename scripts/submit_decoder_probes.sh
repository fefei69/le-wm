#!/bin/bash
# Submit one decoder-probe job per frozen WM checkpoint (Fig-10-style visualization).
#
# Usage:
#   bash scripts/submit_decoder_probes.sh            # epochs 0 1 2 3
#   bash scripts/submit_decoder_probes.sh 4 5        # custom epoch list
#
# Epoch 0 = random-init encoder; the weights file is created on the fly if missing.
# Each job writes to $CKPT_DIR/decoder_ep<N>/ (never the shared default dir).

set -euo pipefail

REPO=/scratch/cw5167/workspace/le-wm
CKPT_DIR=/scratch/cw5167/stable-wm/checkpoints/lewm

EPOCHS=("$@")
if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=(0 1 2 3)
fi

# random-init checkpoint for the "0 steps" panel
if [[ " ${EPOCHS[*]} " == *" 0 "* && ! -f "$CKPT_DIR/weights_epoch_0.pt" ]]; then
    echo "creating random-init $CKPT_DIR/weights_epoch_0.pt"
    (cd "$REPO" && source .venv/bin/activate && python - "$CKPT_DIR" <<'PY'
import json, sys, torch
from pathlib import Path
from hydra.utils import instantiate

ckpt_dir = Path(sys.argv[1])
cfg = json.loads((ckpt_dir / "config.json").read_text())
torch.manual_seed(3072)
model = instantiate(cfg)
torch.save(model.state_dict(), ckpt_dir / "weights_epoch_0.pt")
print("saved", ckpt_dir / "weights_epoch_0.pt")
PY
    )
fi

for ep in "${EPOCHS[@]}"; do
    ckpt="$CKPT_DIR/weights_epoch_${ep}.pt"
    out="$CKPT_DIR/decoder_ep${ep}"

    if [[ ! -f "$ckpt" ]]; then
        echo "skip epoch ${ep}: $ckpt not found" >&2
        continue
    fi
    # recon at the final step only exists if a previous run finished
    if [[ -f "$out/recon_step020000.png" ]]; then
        echo "skip epoch ${ep}: completed run already in $out"
        continue
    fi

    sbatch --job-name="lewm-dec-ep${ep}" \
        "$REPO/scripts/train_decoder.sbatch" \
        --checkpoint "lewm/weights_epoch_${ep}.pt" \
        --out "$out"
done
