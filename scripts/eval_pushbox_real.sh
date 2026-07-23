#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
collector_repo="${WM_DATA_COLLECTION_REPO:-$(cd "$repo_root/../wm_data_collection" && pwd)}"

# Live robot configuration.  Cartesian positions and workspace bounds are in
# metres in the robot base frame.  The evaluator fixes the end-effector
# orientation to [0, pi/2, 0], matching the data-collection setup.
robot_args=(
    --follower-ip 192.168.1.3
    --start-x 0.14
    --start-y 0.0185
    --safe-z 0.15
    --fixed-z 0.03
    --x-min 0.0183
    --x-max 0.45
    --y-min -0.26
    --y-max 0.26
    --trajectory-check-samples 10
    --camera-check-seconds 5
    --settled-linear-speed 0.1
)

# LeWM planner defaults.  Arguments supplied at launch are forwarded after
# these values, so an explicit override such as --action-mode continuous or
# --iterations 30 wins.
planner_args=(
    --action-mode keyboard
    --horizon 5
    --action-cap 0.005
    --num-samples 1024
    --iterations 20
    --elite-count 128
    --max-actions 50
)

usage() {
    cat <<'EOF'
Usage: scripts/eval_pushbox_real.sh [wrapper options] [evaluator options]

Wrapper options:
  --dry-run               Never connect to or move the robot
  --execute               Enable live robot execution (the launcher default)
  --preflight             Check artifacts and planner, then exit without hardware
  -h, --help              Show this help

Planner defaults:
  LeWM CEM, keyboard actions, horizon 5, 1024 samples x 20 iterations,
  128 elites, 5 mm action cap, and 50 maximum actions.

Other options are forwarded to real_robot_eval.py after these defaults, so
explicit expert overrides such as --action-mode or --iterations still win.
EOF
}

# This launcher defaults to live execution.  --dry-run is handled here rather
# than forwarded to Python, so the same configured pose/bounds can be tested
# without connecting to the arm.  If both mode flags are supplied, the last
# one wins.
live_mode=true
preflight=false
forwarded_args=()
while (($#)); do
    case "$1" in
        --execute)
            live_mode=true
            shift
            ;;
        --dry-run)
            live_mode=false
            shift
            ;;
        --preflight)
            preflight=true
            forwarded_args+=("$1")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            forwarded_args+=("$1")
            shift
            ;;
    esac
done

mode_args=()
if [[ "$live_mode" == true ]]; then
    mode_args+=(--execute)
    if [[ "$preflight" == false ]]; then
        echo "LIVE ROBOT MODE: the arm will move to the configured start pose after safety checks."
    fi
else
    echo "DRY RUN MODE: no robot connection or motion will occur."
fi

if [[ ! -f "$collector_repo/scripts/collect_keyboard_xy.py" ]]; then
    echo "wm_data_collection repository not found: $collector_repo" >&2
    exit 1
fi
if [[ ! -x /usr/bin/python3.12 ]]; then
    echo "ROS-compatible /usr/bin/python3.12 was not found" >&2
    exit 1
fi
if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
    echo "le-wm virtualenv not found; create/sync .venv first" >&2
    exit 1
fi

# ROS Jazzy installs its Python 3.12 modules through this environment.
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    # ROS installs this generated environment file outside the repository.
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
fi

cd "$collector_repo"
exec uv run \
    --isolated \
    --python /usr/bin/python3.12 \
    --project "$collector_repo" \
    python "$repo_root/real_robot_eval.py" \
    --collector-repo "$collector_repo" \
    "${mode_args[@]}" \
    "${robot_args[@]}" \
    "${planner_args[@]}" \
    "${forwarded_args[@]}"
