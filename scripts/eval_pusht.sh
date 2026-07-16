#!/usr/bin/env bash
set -euo pipefail

# Checkpoint path relative to $STABLEWM_HOME, without the `_object.ckpt`
# suffix. Override with POLICY=<checkpoint-path> when evaluating another model.
POLICY="${POLICY:-quentinll/lewm-pusht}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

python eval.py --config-name=pusht "policy=${POLICY}" "$@"
