#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
collector_repo="${WM_DATA_COLLECTION_REPO:-$(cd "$repo_root/../wm_data_collection" && pwd)}"
recorder="$repo_root/scripts/record_real_constant_xy.py"
analyzer="$repo_root/scripts/analyze_real_latent_rollout.py"
analysis_python="$repo_root/.venv/bin/python"

usage() {
    cat <<'EOF'
Usage: scripts/run_real_plus_x_latent_probe.sh [--preflight|--dry-run|--execute] [RECORDER_ARGS...]

The default is a no-robot preflight. Pass --execute explicitly to enable the
live recording; all other arguments are forwarded unchanged to the recorder.
After a successful live recording, the launcher analyzes the resulting run.

Environment:
  WM_DATA_COLLECTION_REPO  Override the sibling wm_data_collection repository.
EOF
}

mode=preflight
recorder_args=()
while (($#)); do
    case "$1" in
        --execute)
            mode=execute
            ;;
        --preflight|--dry-run)
            mode=preflight
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            recorder_args+=("$@")
            break
            ;;
        *)
            recorder_args+=("$1")
            ;;
    esac
    shift
done

if [[ ! -f "$collector_repo/scripts/collect_keyboard_xy.py" ]]; then
    echo "wm_data_collection repository not found: $collector_repo" >&2
    exit 1
fi
if [[ ! -f "$recorder" ]]; then
    echo "real constant-XY recorder not found: $recorder" >&2
    exit 1
fi
if [[ ! -f "$analyzer" ]]; then
    echo "real latent-rollout analyzer not found: $analyzer" >&2
    exit 1
fi
if [[ ! -x /usr/bin/python3.12 ]]; then
    echo "ROS-compatible /usr/bin/python3.12 was not found" >&2
    exit 1
fi
if [[ ! -x "$analysis_python" ]]; then
    echo "analysis Python was not found: $analysis_python" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found on PATH" >&2
    exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
run_dir="$repo_root/real_robot_runs/plus_x_latent_probe_${stamp}_$$"
if [[ -e "$run_dir" ]]; then
    echo "refusing to reuse an existing run directory: $run_dir" >&2
    exit 1
fi

mode_args=(--preflight)
if [[ "$mode" == execute ]]; then
    mode_args=(--execute)
    echo "LIVE ROBOT MODE: after the arm reaches the start pose, press p once to record."
else
    echo "PREFLIGHT MODE: no display, camera, robot connection, or motion will occur."
fi
echo "Run directory: $run_dir"

# ROS Jazzy exposes its Python 3.12 modules through this generated environment.
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
fi

# Keep the hardware process in the collector's ROS-compatible environment.
# Launcher-owned arguments come last so callers cannot override the unique run
# directory or accidentally bypass the selected safe/live mode.
(
    cd "$collector_repo"
    uv run \
        --isolated \
        --python /usr/bin/python3.12 \
        --project "$collector_repo" \
        python "$recorder" \
        "${recorder_args[@]}" \
        --collector-repo "$collector_repo" \
        --output-dir "$run_dir" \
        "${mode_args[@]}"
)

if [[ "$mode" != execute ]]; then
    echo "Preflight passed; no run was recorded or analyzed."
    exit 0
fi

if [[ ! -d "$run_dir" || ! -f "$run_dir/metadata.json" || ! -d "$run_dir/frames" ]]; then
    echo "recorder exited without a complete run at: $run_dir" >&2
    exit 1
fi

# Prevent ROS/collector Python paths from leaking into the model environment.
(
    cd "$repo_root"
    env -u PYTHONPATH "$analysis_python" "$analyzer" --real-run "$run_dir"
)

echo "Real +X recording and latent analysis complete: $run_dir"
