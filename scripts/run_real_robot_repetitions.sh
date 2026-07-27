#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Easy-to-edit defaults. Environment variables with the same names also work.
case_name="${CASE_NAME:-test01}"
method_name="${METHOD_NAME:-discrete_mcts}"
repetitions="${REPETITIONS:-5}"
fps="${FPS:-}"
mode=""
skip_postprocess=false
extra_args=()

set_mode() {
    if [[ -n "$mode" && "$mode" != "$1" ]]; then
        echo "choose exactly one of --execute, --dry-run, or --preflight" >&2
        exit 2
    fi
    mode="$1"
}

usage() {
    cat <<EOF
Usage: scripts/run_real_robot_repetitions.sh [options] [-- evaluator options]

Run the same named real-robot case/method repeatedly. Each repetition creates a
new timestamped run directory containing trial_001, then the existing runner
postprocesses it before this script asks you to prepare the next repetition.

Options:
  --case NAME            Case from config/real_robot_experiments.json
                         (default: $case_name)
  --method NAME          Method from the registry (default: $method_name)
  -n, --repetitions N    Number of independent runs (default: $repetitions)
  --execute              Live robot mode; required for real experiments
  --dry-run              Camera/planner session without robot motion
  --preflight            One no-hardware configuration check
  --fps FPS              Override postprocessed video FPS
  --skip-postprocess     Preserve runs without automatic analysis
  -h, --help             Show this help

Example:
  scripts/run_real_robot_repetitions.sh --case test01 \\
      --method discrete_mcts --repetitions 5 --execute

Do not use --trial-id as a repetition number. Every independent dataset-driven
run contains trial_001; repetitions are separated by their run directories.
EOF
}

while (($#)); do
    case "$1" in
        --case)
            (($# >= 2)) || { echo "--case requires a value" >&2; exit 2; }
            case_name="$2"
            shift 2
            ;;
        --case=*)
            case_name="${1#*=}"
            shift
            ;;
        --method)
            (($# >= 2)) || { echo "--method requires a value" >&2; exit 2; }
            method_name="$2"
            shift 2
            ;;
        --method=*)
            method_name="${1#*=}"
            shift
            ;;
        -n|--repetitions)
            (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
            repetitions="$2"
            shift 2
            ;;
        --repetitions=*)
            repetitions="${1#*=}"
            shift
            ;;
        --fps)
            (($# >= 2)) || { echo "--fps requires a value" >&2; exit 2; }
            fps="$2"
            shift 2
            ;;
        --fps=*)
            fps="${1#*=}"
            shift
            ;;
        --execute)
            set_mode execute
            shift
            ;;
        --dry-run)
            set_mode dry-run
            shift
            ;;
        --preflight)
            set_mode preflight
            shift
            ;;
        --skip-postprocess)
            skip_postprocess=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$mode" ]]; then
    echo "choose exactly one of --execute, --dry-run, or --preflight" >&2
    exit 2
fi
if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
    echo "--repetitions must be a positive integer" >&2
    exit 2
fi
if [[ -z "$case_name" || -z "$method_name" ]]; then
    echo "--case and --method must be non-empty" >&2
    exit 2
fi
if [[ "$mode" == "preflight" && "$repetitions" != 1 ]]; then
    echo "--preflight validates once; forcing repetitions from $repetitions to 1"
    repetitions=1
fi

python="$repo_root/.venv/bin/python"
runner="$repo_root/scripts/run_real_robot_experiment.py"
if [[ ! -x "$python" ]]; then
    echo "le-wm virtualenv not found: $python" >&2
    exit 1
fi
if [[ ! -f "$runner" ]]; then
    echo "experiment runner not found: $runner" >&2
    exit 1
fi

command=(
    "$python"
    "$runner"
    --case "$case_name"
    --method "$method_name"
    "--$mode"
)
if [[ -n "$fps" ]]; then
    command+=(--fps "$fps")
fi
if [[ "$skip_postprocess" == true ]]; then
    command+=(--skip-postprocess)
fi
if ((${#extra_args[@]})); then
    command+=(-- "${extra_args[@]}")
fi

echo "Repeated real-robot evaluation"
echo "  case:        $case_name"
echo "  method:      $method_name"
echo "  repetitions: $repetitions"
echo "  mode:        $mode"
if [[ "$mode" == "execute" ]]; then
    echo "LIVE ROBOT MODE: each repetition can move the arm after normal safety checks."
fi

for ((rep = 1; rep <= repetitions; rep++)); do
    echo
    echo "=== Repetition $rep / $repetitions: $case_name x $method_name ==="
    if [[ "$mode" == "execute" ]]; then
        read -r -p "Restore the box to the case initial state, clear the workspace, then press Enter (Ctrl-C aborts): "
    else
        read -r -p "Press Enter to start this repetition (Ctrl-C aborts): "
    fi
    "${command[@]}"
    echo "=== Repetition $rep completed ==="
done

echo
echo "All $repetitions repetitions completed successfully."
