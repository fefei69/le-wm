#!/usr/bin/env python3
"""Run one named real-robot case/method and postprocess its new run record."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "real_robot_experiments.json"
POSTPROCESSOR = REPO_ROOT / "scripts" / "postprocess_discrete_real_run.sh"
PROTECTED_EVALUATOR_FLAGS = {
    "--dataset-episode",
    "--goal-image",
    "--goal-video",
    "--goal-video-frame",
    "--initial-step",
    "--goal-step",
    "--output-dir",
    "--world-model",
    "--solver",
    "--planner",
    "--execute",
    "--dry-run",
    "--preflight",
}


@dataclass(frozen=True)
class Experiment:
    case_name: str
    method_name: str
    case: dict[str, Any]
    method: dict[str, Any]
    repo: Path
    launcher: Path
    output_root: Path
    trial_id: int
    postprocess_args: tuple[str, ...]


def _string_list(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return tuple(value)


def validate_extra_eval_args(
    values: Sequence[str], *, label: str
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    protected = [
        value
        for value in normalized
        if value.split("=", 1)[0] in PROTECTED_EVALUATOR_FLAGS
    ]
    if protected:
        raise ValueError(
            f"{label} cannot override experiment identity/mode flags: "
            + ", ".join(protected)
        )
    return normalized


def load_registry(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    try:
        registry = json.loads(config_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"experiment config not found: {config_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError("experiment config must use schema_version 1")
    for key in ("cases", "methods"):
        if not isinstance(registry.get(key), dict) or not registry[key]:
            raise ValueError(f"experiment config {key} must be a non-empty object")
    defaults = registry.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("experiment config defaults must be an object")
    return registry


def resolve_experiment(
    registry: dict[str, Any],
    *,
    case_name: str,
    method_name: str,
    trial_id_override: int | None = None,
    repo_root: Path = REPO_ROOT,
) -> Experiment:
    try:
        case = registry["cases"][case_name]
    except KeyError:
        raise ValueError(f"unknown case {case_name!r}") from None
    try:
        method = registry["methods"][method_name]
    except KeyError:
        raise ValueError(f"unknown method {method_name!r}") from None
    if not isinstance(case, dict) or not isinstance(method, dict):
        raise ValueError("selected case and method must be objects")

    dataset_episode = case.get("dataset_episode")
    if not isinstance(dataset_episode, str) or not dataset_episode:
        raise ValueError("case dataset_episode must be a non-empty string")
    try:
        initial_step = int(case["initial_step"])
        goal_step = int(case["goal_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("case initial_step and goal_step must be integers") from exc
    if initial_step < 0 or goal_step <= initial_step:
        raise ValueError("case requires 0 <= initial_step < goal_step")

    adapter = method.get("adapter")
    world_model = method.get("world_model")
    solver = method.get("solver")
    if adapter not in {"lewm", "discrete"}:
        raise ValueError("method adapter must be 'lewm' or 'discrete'")
    if not isinstance(world_model, str) or not world_model:
        raise ValueError("method world_model must be a non-empty string")
    if not isinstance(solver, str) or not solver:
        raise ValueError("method solver must be a non-empty string")
    if adapter == "lewm":
        if world_model not in {"lewm", "dinowm"}:
            raise ValueError("lewm adapter world_model must be lewm or dinowm")
        if solver not in {"cem", "categorical-cem"}:
            raise ValueError("lewm adapter solver must be cem or categorical-cem")
    else:
        if solver not in {"discrete-cem", "mcts"}:
            raise ValueError(
                "discrete adapter solver must be discrete-cem or mcts"
            )

    raw_repo = method.get("repo")
    if not isinstance(raw_repo, str) or not raw_repo:
        raise ValueError("method repo must be a non-empty path string")
    repo = Path(raw_repo).expanduser()
    if not repo.is_absolute():
        repo = repo_root / repo
    repo = repo.resolve()
    launcher = repo / "scripts" / "eval_pushbox_real.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"method launcher not found: {launcher}")
    episode_path = Path(dataset_episode).expanduser()
    if not episode_path.is_absolute():
        episode_path = repo / episode_path
    if not episode_path.is_file():
        raise FileNotFoundError(
            f"case dataset episode is unavailable in {repo}: {episode_path}"
        )

    defaults = registry.get("defaults", {})
    raw_trial_id = (
        trial_id_override
        if trial_id_override is not None
        else case.get("trial_id", defaults.get("trial_id", 1))
    )
    try:
        trial_id = int(raw_trial_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("trial_id must be an integer") from exc
    if trial_id < 1:
        raise ValueError("trial_id must be positive")
    postprocess_args = _string_list(
        defaults.get("postprocess_args", []),
        label="defaults.postprocess_args",
    )
    validate_extra_eval_args(
        _string_list(method.get("extra_args", []), label="method.extra_args"),
        label="method.extra_args",
    )

    normalized_case = dict(case)
    normalized_case["initial_step"] = initial_step
    normalized_case["goal_step"] = goal_step
    return Experiment(
        case_name=case_name,
        method_name=method_name,
        case=normalized_case,
        method=dict(method),
        repo=repo,
        launcher=launcher,
        output_root=repo / "real_robot_runs",
        trial_id=trial_id,
        postprocess_args=postprocess_args,
    )


def build_evaluator_command(
    experiment: Experiment,
    *,
    mode: str,
    extra_args: Sequence[str] = (),
) -> list[str]:
    if mode not in {"execute", "dry-run", "preflight"}:
        raise ValueError("mode must be execute, dry-run, or preflight")
    adapter = experiment.method["adapter"]
    if adapter == "lewm":
        method_args = [
            "--world-model",
            experiment.method["world_model"],
            "--solver",
            experiment.method["solver"],
        ]
    else:
        method_args = ["--planner", experiment.method["solver"]]
    method_args.extend(
        validate_extra_eval_args(
            _string_list(
                experiment.method.get("extra_args", []),
                label="method.extra_args",
            ),
            label="method.extra_args",
        )
    )
    case_args = [
        "--dataset-episode",
        str(experiment.case["dataset_episode"]),
        "--initial-step",
        str(experiment.case["initial_step"]),
        "--goal-step",
        str(experiment.case["goal_step"]),
        "--output-dir",
        str(experiment.output_root),
    ]
    mode_args = {
        "execute": ["--execute"],
        "dry-run": ["--dry-run"],
        "preflight": ["--dry-run", "--preflight"],
    }[mode]
    return [
        "bash",
        str(experiment.launcher),
        *method_args,
        *case_args,
        *mode_args,
        *[str(value) for value in extra_args],
    ]


def snapshot_run_directories(output_root: Path) -> set[Path]:
    if not output_root.exists():
        return set()
    if not output_root.is_dir():
        raise NotADirectoryError(output_root)
    return {
        child.resolve()
        for child in output_root.iterdir()
        if child.is_dir()
    }


def identify_new_run(before: set[Path], output_root: Path) -> Path:
    after = snapshot_run_directories(output_root)
    candidates = sorted(
        path
        for path in after - before
        if (path / "metadata.json").is_file()
        and (path / "events.jsonl").is_file()
    )
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(
            "expected exactly one new real-robot run directory, found "
            f"{len(candidates)}: {rendered}"
        )
    return candidates[0]


def write_provenance(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case", dest="case_name")
    parser.add_argument("--method", dest="method_name")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list configured cases and methods, then exit",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", dest="mode", action="store_const", const="execute")
    mode.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run")
    mode.add_argument("--preflight", dest="mode", action="store_const", const="preflight")
    parser.add_argument("--trial-id", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="leave the new run unprocessed",
    )
    parser.add_argument(
        "extra_eval_args",
        nargs=argparse.REMAINDER,
        help="extra evaluator arguments after --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.config)
        if args.list:
            print("Cases:")
            for name, value in registry["cases"].items():
                print(
                    f"  {name}: {value.get('dataset_episode')} "
                    f"[{value.get('initial_step')} -> {value.get('goal_step')}]"
                )
            print("Methods:")
            for name, value in registry["methods"].items():
                print(
                    f"  {name}: wm={value.get('world_model')}, "
                    f"solver={value.get('solver')}, repo={value.get('repo')}"
                )
            return 0
        if not args.case_name or not args.method_name:
            parser.error("--case and --method are required unless --list is used")
        if args.mode is None:
            parser.error("choose exactly one of --execute, --dry-run, or --preflight")
        if args.fps is not None and (
            not math.isfinite(args.fps) or args.fps <= 0
        ):
            parser.error("--fps must be finite and positive")
        extra_args = list(args.extra_eval_args)
        if extra_args[:1] == ["--"]:
            extra_args = extra_args[1:]
        extra_args = list(
            validate_extra_eval_args(extra_args, label="extra evaluator arguments")
        )
        experiment = resolve_experiment(
            registry,
            case_name=args.case_name,
            method_name=args.method_name,
            trial_id_override=args.trial_id,
        )
        command = build_evaluator_command(
            experiment, mode=args.mode, extra_args=extra_args
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Experiment: case={experiment.case_name}, "
        f"method={experiment.method_name}, mode={args.mode}"
    )
    print(
        f"World model: {experiment.method['world_model']}; "
        f"solver: {experiment.method['solver']}"
    )
    print("Command:")
    print("  " + shlex.join(command))
    if args.mode == "execute":
        print(
            "LIVE ROBOT EXPERIMENT: the evaluator will connect to and move the arm "
            "after its normal safety checks."
        )
    sys.stdout.flush()

    before = snapshot_run_directories(experiment.output_root)
    started_at = datetime.now().astimezone().isoformat()
    completed = subprocess.run(command, cwd=experiment.repo, check=False)
    if args.mode == "preflight":
        return int(completed.returncode)

    try:
        run_path = identify_new_run(before, experiment.output_root)
    except RuntimeError as exc:
        print(f"experiment run detection failed: {exc}", file=sys.stderr)
        return int(completed.returncode or 1)
    provenance_path = run_path / "experiment.json"
    provenance = {
        "schema_version": 1,
        "case_name": experiment.case_name,
        "method_name": experiment.method_name,
        "case": experiment.case,
        "method": experiment.method,
        "mode": args.mode,
        "trial_id": experiment.trial_id,
        "config_path": str(args.config.expanduser().resolve()),
        "repo": str(experiment.repo),
        "run_path": str(run_path),
        "command": command,
        "started_at": started_at,
        "evaluator_completed_at": datetime.now().astimezone().isoformat(),
        "evaluator_exit_code": int(completed.returncode),
        "postprocess": None,
    }
    write_provenance(provenance_path, provenance)
    print(f"Run record: {run_path}")
    print(f"Experiment provenance: {provenance_path}")
    if completed.returncode:
        print(
            "evaluator exited unsuccessfully; preserving the run without "
            "postprocessing",
            file=sys.stderr,
        )
        return int(completed.returncode)
    if args.skip_postprocess:
        print("Postprocessing skipped by request.")
        return 0

    postprocess_command = [
        "bash",
        str(POSTPROCESSOR),
        str(run_path),
        str(experiment.trial_id),
        *experiment.postprocess_args,
    ]
    if args.fps is not None:
        postprocess_command.extend(["--fps", str(args.fps)])
    print("Postprocess:")
    print("  " + shlex.join(postprocess_command))
    postprocessed = subprocess.run(
        postprocess_command, cwd=REPO_ROOT, check=False
    )
    analysis_path = (
        run_path
        / "analysis"
        / f"overview_trial_{experiment.trial_id:03d}"
    )
    provenance["postprocess"] = {
        "command": postprocess_command,
        "completed_at": datetime.now().astimezone().isoformat(),
        "exit_code": int(postprocessed.returncode),
        "analysis_path": str(analysis_path),
    }
    write_provenance(provenance_path, provenance)
    if postprocessed.returncode:
        print(
            f"postprocessing failed; run is preserved at {run_path}",
            file=sys.stderr,
        )
        return int(postprocessed.returncode)
    print(f"Analysis: {analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
