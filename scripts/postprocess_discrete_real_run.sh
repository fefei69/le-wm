#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: scripts/postprocess_discrete_real_run.sh RUN_PATH [TRIAL_ID] [extra options]"
}

if (($# < 1)); then
    usage >&2
    exit 2
fi

run_path="$1"
shift
trial_id=1
if (($#)) && [[ "$1" != --* ]]; then
    trial_id="$1"
    shift
fi

exec "$repo_root/.venv/bin/python" \
    "$repo_root/scripts/postprocess_real_run.py" \
    --run "$run_path" \
    --trial-id "$trial_id" \
    "$@"
