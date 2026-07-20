"""Interactive goal-conditioned evaluation on the real PushBox robot.

The controller deliberately starts in dry-run mode.  Passing ``--execute`` is
required before any Trossen connection or robot motion occurs.  The hardware
loop runs in the ROS-compatible collector environment and delegates LeWM/CEM
inference to :mod:`real_robot_planner` in this repository's virtualenv.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import imageio.v3 as iio
import numpy as np


ACTION_CAP_M = 0.010 # 0.010
TICK_S = 0.2
UI_SLEEP_MAX_S = 0.005
RESET_STEP_M = 0.005
RESET_TOLERANCE_M = 0.002
MODEL_IMAGE_SHAPE = (224, 224, 3)
DEFAULT_PLANNER_TIMEOUT_S = 2.0
DEFAULT_MAX_PLAN_AGE_S = 2.0
DEFAULT_MAX_OBSERVATION_DELAY_S = 0.3
DEFAULT_MOTION_SETTLE_TIMEOUT_S = 1.0
DEFAULT_XY_TRACKING_TOLERANCE_M = 0.005
DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.1
DEFAULT_SETTLED_LINEAR_SPEED_M_S = 0.1
DEFAULT_TRACKING_SETTLE_GRACE_S = 0.05


def validate_planned_action(action: Any, cap_m: float = ACTION_CAP_M) -> np.ndarray:
    """Validate one physical XY delta without silently accepting unsafe output."""
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"planned action must have shape (2,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("planned action must be finite")
    norm = float(np.linalg.norm(array))
    if norm > float(cap_m) + 1e-7:
        raise ValueError(
            f"planned action norm {norm:.6f} m exceeds cap {float(cap_m):.6f} m"
        )
    return array.astype(np.float32)


def transform_planner_action(
    action: Any, *, negate: bool, cap_m: float = ACTION_CAP_M
) -> np.ndarray:
    """Map between planner and robot XY conventions for a sign-flip diagnostic.

    Negation is its own inverse, so this is also used to map a clipped physical
    action back into the planner convention for transition-history feedback.
    """
    transformed = validate_planned_action(action, cap_m=cap_m)
    if negate:
        transformed = -transformed
    return transformed.astype(np.float32, copy=False)


def validate_planner_result(
    result: dict[str, Any], *, cap_m: float, horizon: int
) -> tuple[np.ndarray, np.ndarray, float, float, float, bool]:
    """Fail closed unless every planner value used for control is well formed."""
    action = validate_planned_action(result["action"], cap_m)
    plan = np.asarray(result["plan"], dtype=np.float64)
    if plan.shape != (int(horizon), 2):
        raise ValueError(
            f"planned horizon must have shape ({int(horizon)}, 2), got {plan.shape}"
        )
    if not np.isfinite(plan).all():
        raise ValueError("planned horizon must be finite")
    if np.any(np.linalg.norm(plan, axis=1) > float(cap_m) + 1e-7):
        raise ValueError("planned horizon contains an action above the cap")
    cost = float(result["cost"])
    goal_distance = float(result["goal_distance"])
    solve_time_s = float(result["solve_time_s"])
    if not np.isfinite([cost, goal_distance, solve_time_s]).all():
        raise ValueError("planner cost, goal distance, and solve time must be finite")
    if cost < 0.0 or goal_distance < 0.0 or solve_time_s < 0.0:
        raise ValueError("planner cost, goal distance, and solve time must be non-negative")
    return (
        action,
        plan.astype(np.float32),
        cost,
        goal_distance,
        solve_time_s,
        bool(result.get("at_goal", False)),
    )


def validate_measured_pose(
    pose: Any,
    *,
    target_xy: np.ndarray,
    fixed_z: float,
    orientation: np.ndarray,
    xy_tolerance_m: float,
    z_tolerance_m: float,
    orientation_tolerance_rad: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
    require_tracking: bool,
) -> np.ndarray:
    """Validate measured Cartesian state before another absolute command."""
    measured = np.asarray(pose, dtype=np.float64).reshape(-1)
    if measured.size < 6 or not np.isfinite(measured[:6]).all():
        raise ValueError("measured Cartesian pose must contain six finite values")
    measured = measured[:6]
    if abs(measured[2] - float(fixed_z)) > float(z_tolerance_m):
        raise ValueError(
            f"fixed-Z error is {(measured[2] - float(fixed_z)) * 1000:+.1f} mm"
        )
    angle_error = (measured[3:6] - np.asarray(orientation) + np.pi) % (2 * np.pi) - np.pi
    if float(np.linalg.norm(angle_error)) > float(orientation_tolerance_rad):
        raise ValueError("measured orientation is outside tolerance")
    if bounds is not None:
        if not (
            bounds[0][0] <= measured[0] <= bounds[0][1]
            and bounds[1][0] <= measured[1] <= bounds[1][1]
        ):
            raise ValueError("measured X/Y pose is outside commissioned bounds")
    if require_tracking:
        error = float(np.linalg.norm(measured[:2] - np.asarray(target_xy)))
        if error > float(xy_tolerance_m):
            raise ValueError(
                f"X/Y tracking error {error * 1000:.1f} mm exceeds "
                f"{float(xy_tolerance_m) * 1000:.1f} mm"
            )
    return measured


def motion_completion_status(
    pose: Any,
    velocity: Any,
    *,
    target_xy: np.ndarray,
    xy_tolerance_m: float,
    settled_linear_speed_m_s: float,
) -> tuple[bool, float, float]:
    """Report whether an autonomous Cartesian command has actually finished."""
    measured = np.asarray(pose, dtype=np.float64).reshape(-1)
    cartesian_velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1)
    if measured.size < 2 or not np.isfinite(measured[:2]).all():
        raise ValueError("measured Cartesian pose does not contain finite X/Y")
    if cartesian_velocity.size < 3 or not np.isfinite(
        cartesian_velocity[:3]
    ).all():
        raise ValueError("measured Cartesian velocity is invalid")
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target X/Y must contain two finite values")
    tracking_error_m = float(np.linalg.norm(measured[:2] - target))
    linear_speed_m_s = float(np.linalg.norm(cartesian_velocity[:3]))
    complete = (
        tracking_error_m <= float(xy_tolerance_m)
        and linear_speed_m_s <= float(settled_linear_speed_m_s)
    )
    return complete, tracking_error_m, linear_speed_m_s


def snapshot_follows_settle(
    receipt_monotonic_ns: int, settled_monotonic_ns: int | None
) -> bool:
    """Only post-settle camera receipts are valid planner observations."""
    return settled_monotonic_ns is None or int(receipt_monotonic_ns) > int(
        settled_monotonic_ns
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunRecorder:
    """Small crash-readable run record: metadata, frames, and flushed JSONL events."""

    def __init__(self, root: Path, metadata: dict[str, Any]) -> None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.path = root.expanduser().resolve() / f"{stamp}_{os.getpid()}"
        self.frames_path = self.path / "frames"
        self.frames_path.mkdir(parents=True, exist_ok=False)
        (self.path / "metadata.json").write_text(
            json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n"
        )
        self._events = (self.path / "events.jsonl").open("a", buffering=1)

    def event(self, kind: str, **values: Any) -> None:
        payload = {
            "type": kind,
            "wall_time": datetime.now().astimezone().isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            **values,
        }
        self._events.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        self._events.flush()

    def save_frame(self, name: str, image: np.ndarray) -> str:
        relative = Path("frames", f"{name}.png")
        iio.imwrite(
            self.path / relative,
            np.asarray(image, dtype=np.uint8),
            compress_level=1,
        )
        return str(relative)

    def close(self) -> None:
        self._events.close()


@dataclass(frozen=True)
class UIEvents:
    quit_requested: bool = False
    goal_requested: bool = False
    preview_requested: bool = False
    autonomous_toggle: bool = False
    pause_requested: bool = False
    reset_requested: bool = False
    focus_lost: bool = False
    outcome: str | None = None


def prioritize_ui_events(events: UIEvents) -> tuple[str, str | None] | None:
    """Choose one event so a safety event cannot be undone in the same poll."""
    if events.quit_requested:
        return ("quit", None)
    if events.focus_lost:
        return ("pause", "window focus lost")
    if events.pause_requested:
        return ("pause", "operator pause")
    if events.outcome is not None:
        return ("outcome", events.outcome)
    if events.reset_requested:
        return ("reset", None)
    if events.goal_requested:
        return ("goal", None)
    if events.preview_requested:
        return ("preview", None)
    if events.autonomous_toggle:
        return ("autonomous", None)
    return None


class RealRobotUI:
    """Pygame adapter; current and goal frames are shown side by side."""

    def __init__(self, held_input: Any, size: int = 224, ui_fps: int = 30) -> None:
        self.held_input = held_input
        self.size = int(size)
        self.ui_fps = int(ui_fps)
        self._pygame: Any = None
        self._screen: Any = None
        self._font: Any = None
        self._focused = True
        self._manual_release_required = False

    def start(self) -> None:
        import pygame

        pygame.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((self.size * 2, self.size + 64))
        pygame.display.set_caption("LeWM real PushBox evaluation")
        self._font = pygame.font.Font(None, 22)
        self._focused = bool(pygame.key.get_focused())

    def poll(self) -> UIEvents:
        if self._pygame is None:
            raise RuntimeError("UI has not been started")
        pygame = self._pygame
        values = {
            "quit_requested": False,
            "goal_requested": False,
            "preview_requested": False,
            "autonomous_toggle": False,
            "pause_requested": False,
            "reset_requested": False,
            "focus_lost": False,
            "outcome": None,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                values["quit_requested"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    values["quit_requested"] = True
                elif event.key == pygame.K_g:
                    values["goal_requested"] = True
                elif event.key == pygame.K_v:
                    values["preview_requested"] = True
                elif event.key == pygame.K_p:
                    values["autonomous_toggle"] = True
                elif event.key == pygame.K_SPACE:
                    values["pause_requested"] = True
                elif event.key == pygame.K_r:
                    values["reset_requested"] = True
                elif event.key == pygame.K_s:
                    values["outcome"] = "success"
                elif event.key == pygame.K_f:
                    values["outcome"] = "failure"
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.held_input.set_speed_level(int(event.key - pygame.K_0))
            elif event.type == pygame.WINDOWFOCUSLOST:
                self.held_input.focus_lost()
                self._focused = False
                self._manual_release_required = True
                values["focus_lost"] = True
            elif event.type == pygame.WINDOWFOCUSGAINED:
                self.held_input.focus_regained()
                self._focused = True
        return UIEvents(**values)

    @property
    def focused(self) -> bool:
        return self._focused

    def suppress_manual_until_release(self) -> None:
        self._manual_release_required = True

    def manual_action(self) -> np.ndarray:
        if self._pygame is None:
            raise RuntimeError("UI has not been started")
        keys = self._pygame.key.get_pressed()
        held = {
            "up": keys[self._pygame.K_UP],
            "down": keys[self._pygame.K_DOWN],
            "left": keys[self._pygame.K_LEFT],
            "right": keys[self._pygame.K_RIGHT],
        }
        if self._manual_release_required:
            if any(held.values()):
                return np.zeros(2, dtype=np.float32)
            self._manual_release_required = False
        return self.held_input.update_held(
            held
        )

    def render(
        self,
        current: np.ndarray,
        goal: np.ndarray | None,
        status: str,
    ) -> None:
        if self._pygame is None or self._screen is None or self._font is None:
            raise RuntimeError("UI has not been started")
        pygame = self._pygame
        current_surface = pygame.image.frombuffer(
            np.ascontiguousarray(current).tobytes(),
            (self.size, self.size),
            "RGB",
        )
        self._screen.blit(current_surface, (0, 0))
        if goal is None:
            goal_surface = pygame.Surface((self.size, self.size))
            goal_surface.fill((20, 20, 20))
        else:
            goal_surface = pygame.image.frombuffer(
                np.ascontiguousarray(goal).tobytes(),
                (self.size, self.size),
                "RGB",
            )
        self._screen.blit(goal_surface, (self.size, 0))
        overlay = pygame.Surface((self.size * 2, 64), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 230))
        overlay.blit(self._font.render("CURRENT", True, (255, 255, 255)), (8, 4))
        overlay.blit(
            self._font.render("GOAL", True, (255, 255, 255)),
            (self.size + 8, 4),
        )
        overlay.blit(self._font.render(status, True, (220, 255, 220)), (8, 30))
        self._screen.blit(overlay, (0, self.size))
        pygame.display.flip()

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._font = None


class PlannerProcess:
    """Own the LeWM subprocess and serialize requests on one worker thread."""

    def __init__(self, args: argparse.Namespace, repo_root: Path) -> None:
        self.args = args
        self.repo_root = repo_root
        self.process: subprocess.Popen | None = None
        self.connection: Any = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner")
        self.socket_path = Path(
            tempfile.gettempdir(), f"lewm-pushbox-{os.getpid()}-{time.time_ns()}.sock"
        )
        self.authkey = secrets.token_bytes(32)

    def start(self) -> None:
        python = self.repo_root / ".venv/bin/python"
        if not python.is_file():
            raise FileNotFoundError(f"LeWM Python environment not found: {python}")
        command = [
            str(python),
            str(self.repo_root / "real_robot_planner.py"),
            "--socket",
            str(self.socket_path),
            "--authkey",
            self.authkey.hex(),
            "--checkpoint",
            self.args.checkpoint,
            "--dataset",
            str(self.args.dataset),
            "--device",
            self.args.device,
            "--horizon",
            str(self.args.horizon),
            "--num-samples",
            str(self.args.num_samples),
            "--iterations",
            str(self.args.iterations),
            "--elite-count",
            str(self.args.elite_count),
            "--action-cap",
            str(self.args.action_cap),
            "--action-mode",
            self.args.action_mode,
            "--goal-tolerance",
            str(self.args.goal_tolerance),
            "--seed",
            str(self.args.seed),
        ]
        if self.args.artifact_manifest is not None:
            command.extend(["--artifact-manifest", str(self.args.artifact_manifest)])
        environment = os.environ.copy()
        # ROS Jazzy contributes Python 3.12 modules to PYTHONPATH.  The planner
        # runs under Python 3.10 and must not see binary modules from that ABI.
        environment.pop("PYTHONPATH", None)
        environment.setdefault(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir(), "lewm-matplotlib"))
        )
        Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(command, cwd=self.repo_root, env=environment)
        deadline = time.monotonic() + self.args.planner_start_timeout
        while not self.socket_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"planner exited during startup with code {self.process.returncode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for planner startup")
            time.sleep(0.05)
        self.connection = Client(
            str(self.socket_path), family="AF_UNIX", authkey=self.authkey
        )

    def _round_trip(self, request: dict[str, Any]) -> dict[str, Any]:
        self.connection.send(request)
        if not self.connection.poll(self.args.planner_timeout):
            raise TimeoutError(
                f"planner did not respond within {self.args.planner_timeout:.3f}s"
            )
        response = self.connection.recv()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "planner request failed"))
        return response

    def plan(
        self,
        current: np.ndarray,
        goal: np.ndarray,
        *,
        reset: bool,
        previous_action: np.ndarray | None,
    ) -> Future:
        return self.executor.submit(
            self._round_trip,
            {
                "op": "plan",
                "current": np.asarray(current, dtype=np.uint8).copy(),
                "goal": np.asarray(goal, dtype=np.uint8).copy(),
                "reset": bool(reset),
                "previous_action": (
                    None
                    if previous_action is None
                    else np.asarray(previous_action, dtype=np.float32).copy()
                ),
            },
        )

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        if self.connection is not None:
            self.connection.close()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.socket_path.unlink(missing_ok=True)


def _build_parser(repo_root: Path) -> argparse.ArgumentParser:
    collector_default = repo_root.parent / "wm_data_collection"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-repo", type=Path, default=collector_default)
    parser.add_argument(
        "--transform-profile",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--camera-params",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repo_root / "stable-wm/datasets/pushbox_pilot_train.h5",
    )
    parser.add_argument(
        "--checkpoint", default="pushbox/lewm/weights_epoch_146.pt"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--follower-ip", default="192.168.1.3")
    parser.add_argument("--fixed-z", type=float, default=0.03)
    parser.add_argument("--safe-z", type=float, default=0.15)
    parser.add_argument("--start-x", type=float, default=0.14)
    parser.add_argument("--start-y", type=float, default=0.0185)
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--allow-unbounded-xy", action="store_true")
    parser.add_argument("--trajectory-check-samples", type=int, default=10)
    parser.add_argument("--fixed-z-warning-tolerance", type=float, default=0.002)
    parser.add_argument("--skip-camera-check", action="store_true")
    parser.add_argument("--camera-check-seconds", type=float, default=5.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "validate artifacts and planner/collector modules, then exit without "
            "display, camera, or robot access"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--action-cap", type=float, default=ACTION_CAP_M)
    parser.add_argument(
        "--action-mode",
        choices=("continuous", "keyboard"),
        default="keyboard",
        help=(
            "keyboard restricts CEM to zero plus the exact 8-way 2.5/5/10 mm "
            "actions present in training"
        ),
    )
    parser.add_argument("--action-deadband", type=float, default=0.00025)
    parser.add_argument(
        "--negate-planner-action",
        action="store_true",
        help=(
            "diagnostic only: multiply autonomous planner XY actions by -1 at "
            "the robot boundary; startup, reset, and manual arrows are unchanged"
        ),
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elite-count", type=int, default=32)
    parser.add_argument("--goal-tolerance", type=float, default=0.0)
    parser.add_argument("--planner-start-timeout", type=float, default=120.0)
    parser.add_argument("--planner-timeout", type=float, default=DEFAULT_PLANNER_TIMEOUT_S)
    parser.add_argument("--max-plan-age", type=float, default=DEFAULT_MAX_PLAN_AGE_S)
    parser.add_argument(
        "--max-observation-delay",
        type=float,
        default=DEFAULT_MAX_OBSERVATION_DELAY_S,
        help=(
            "maximum delay from verified arm settling to the camera frame used "
            "for the next plan"
        ),
    )
    parser.add_argument(
        "--motion-settle-timeout",
        type=float,
        default=DEFAULT_MOTION_SETTLE_TIMEOUT_S,
        help=(
            "maximum time after an autonomous/hold command to reach the target "
            "and fall below --settled-linear-speed"
        ),
    )
    parser.add_argument(
        "--xy-tracking-tolerance",
        type=float,
        default=DEFAULT_XY_TRACKING_TOLERANCE_M,
    )
    parser.add_argument(
        "--orientation-tolerance",
        type=float,
        default=DEFAULT_ORIENTATION_TOLERANCE_RAD,
    )
    parser.add_argument(
        "--settled-linear-speed",
        type=float,
        default=DEFAULT_SETTLED_LINEAR_SPEED_M_S,
    )
    parser.add_argument(
        "--tracking-settle-grace",
        type=float,
        default=DEFAULT_TRACKING_SETTLE_GRACE_S,
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=repo_root / "config/real_robot_eval.json",
    )
    parser.add_argument("--allow-artifact-mismatch", action="store_true")
    parser.add_argument("--verify-dataset-hash", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "real_robot_runs"
    )
    return parser


def _workspace_bounds(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    values = (args.x_min, args.x_max, args.y_min, args.y_max)
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        parser.error("provide all four of --x-min/--x-max/--y-min/--y-max")
    if all(value is not None for value in values):
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            parser.error("workspace bounds must be finite")
        bounds = ((args.x_min, args.x_max), (args.y_min, args.y_max))
        if args.x_min >= args.x_max or args.y_min >= args.y_max:
            parser.error("workspace bounds must be strictly increasing")
        if not args.x_min <= args.start_x <= args.x_max:
            parser.error("start X is outside the configured bounds")
        if not args.y_min <= args.start_y <= args.y_max:
            parser.error("start Y is outside the configured bounds")
        return bounds
    if args.execute and not args.allow_unbounded_xy:
        parser.error(
            "--execute requires commissioned XY bounds, or the explicit "
            "--allow-unbounded-xy override"
        )
    return None


def _validate_artifact_contract(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    repo_root: Path,
    transform_profile: Path,
    camera_params: Path,
) -> dict[str, Any] | None:
    manifest_path = args.artifact_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        if args.allow_artifact_mismatch:
            print(f"WARNING: artifact manifest not found: {manifest_path}")
            args.artifact_manifest = None
            return None
        parser.error(
            f"artifact manifest not found: {manifest_path}; use "
            "--allow-artifact-mismatch only for explicit commissioning"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid artifact manifest {manifest_path}: {exc}")
    if manifest.get("schema_version") != 1 or manifest.get("manifest_type") != "real_robot_eval":
        parser.error("unsupported real-robot artifact manifest")

    artifact_root = Path(
        os.environ.get("STABLEWM_HOME", str(repo_root / "stable-wm"))
    ).expanduser().resolve()
    expected_dataset = (artifact_root / manifest["dataset"]["path"]).resolve()
    expected_checkpoint = (artifact_root / manifest["world_model"]["checkpoint"]).resolve()
    supplied_dataset = args.dataset.expanduser().resolve()
    checkpoint_arg = Path(args.checkpoint).expanduser()
    supplied_checkpoint = (
        checkpoint_arg.resolve()
        if checkpoint_arg.is_file()
        else (artifact_root / "checkpoints" / checkpoint_arg).resolve()
    )
    mismatches = []
    if supplied_dataset != expected_dataset:
        mismatches.append(f"dataset {supplied_dataset} != {expected_dataset}")
    if supplied_checkpoint != expected_checkpoint:
        mismatches.append(f"checkpoint {supplied_checkpoint} != {expected_checkpoint}")

    runtime = manifest["runtime"]
    if runtime.get("frameskip") != 1 or float(runtime.get("tick_hz", 0.0)) != 5.0:
        mismatches.append("manifest does not describe the required 5 Hz/frameskip-1 model")
    if runtime.get("action_dim") != 2 or runtime.get("image_shape_chw") != [3, 224, 224]:
        mismatches.append("manifest action or image dimensions are incompatible")
    if float(runtime.get("maximum_action_norm_m", -1.0)) != ACTION_CAP_M:
        mismatches.append("manifest action cap differs from the collection contract")

    collector = manifest.get("collector", {})
    expected_transform_hash = collector.get("transform_profile_sha256")
    expected_camera_hash = collector.get("camera_params_sha256")
    if expected_transform_hash and _sha256(transform_profile) != expected_transform_hash:
        mismatches.append("transform profile hash does not match the model contract")
    if expected_camera_hash and _sha256(camera_params) != expected_camera_hash:
        mismatches.append("camera parameter hash does not match the model contract")

    if mismatches:
        message = "; ".join(mismatches)
        if not args.allow_artifact_mismatch:
            parser.error(
                f"artifact compatibility check failed: {message}; use "
                "--allow-artifact-mismatch only for explicit commissioning"
            )
        print(f"WARNING: artifact compatibility override: {message}")
        args.artifact_manifest = None
        return None

    if not expected_checkpoint.is_file() or not expected_dataset.is_file():
        parser.error("manifest checkpoint or dataset file is missing")
    if _sha256(expected_checkpoint) != manifest["world_model"]["checkpoint_sha256"]:
        parser.error("checkpoint SHA-256 does not match the artifact manifest")
    if args.execute or args.verify_dataset_hash:
        print("Verifying dataset SHA-256 (this reads the complete dataset)...")
        if _sha256(expected_dataset) != manifest["dataset"]["sha256"]:
            parser.error("dataset SHA-256 does not match the artifact manifest")

    args.dataset = supplied_dataset
    args.artifact_manifest = manifest_path
    return manifest


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repo_root = Path(__file__).resolve().parent
    collector_repo = args.collector_repo.expanduser().resolve()
    if not (collector_repo / "scripts/collect_keyboard_xy.py").is_file():
        raise FileNotFoundError(f"collector repository not found: {collector_repo}")
    args.transform_profile = (
        args.transform_profile.expanduser().resolve()
        if args.transform_profile is not None
        else collector_repo / "config/transform-profile.json"
    )
    args.camera_params = (
        args.camera_params.expanduser().resolve()
        if args.camera_params is not None
        else collector_repo / "config/camera-params.yaml"
    )
    args.dataset = args.dataset.expanduser().resolve()
    sys.path.insert(0, str(collector_repo))

    from scripts.collect_keyboard_xy import (
        CollectorConfig,
        FIXED_ORIENTATION,
        FollowerArmAdapter,
        accepted_target,
        advance_tick_deadline,
        move_arm_home_then_zero,
    )
    from scripts.pygame_input import HeldInput
    from scripts.ros_camera import (
        HEALTH_MAX_AGE_S,
        LatestImageStore,
        RosImageSubscriber,
        measure_stream_health,
    )

    if not 0.0 < args.action_cap <= ACTION_CAP_M:
        parser.error(f"--action-cap must be in (0, {ACTION_CAP_M}]")
    if not 0.0 <= args.action_deadband <= args.action_cap:
        parser.error("--action-deadband must be in [0, --action-cap]")
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    positive_values = {
        "--planner-start-timeout": args.planner_start_timeout,
        "--planner-timeout": args.planner_timeout,
        "--max-plan-age": args.max_plan_age,
        "--max-observation-delay": args.max_observation_delay,
        "--motion-settle-timeout": args.motion_settle_timeout,
        "--xy-tracking-tolerance": args.xy_tracking_tolerance,
        "--orientation-tolerance": args.orientation_tolerance,
        "--settled-linear-speed": args.settled_linear_speed,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if not np.isfinite(args.tracking_settle_grace) or args.tracking_settle_grace < 0.0:
        parser.error("--tracking-settle-grace must be finite and non-negative")
    if args.motion_settle_timeout <= TICK_S + args.tracking_settle_grace:
        parser.error(
            "--motion-settle-timeout must exceed the 0.2 s command time plus "
            "--tracking-settle-grace"
        )
    bounds = _workspace_bounds(args, parser)
    config = CollectorConfig.from_paths(
        args.transform_profile,
        args.camera_params,
        follower_ip=args.follower_ip,
        fixed_z=args.fixed_z,
        safe_z=args.safe_z,
        fixed_z_warning_tolerance=args.fixed_z_warning_tolerance,
        start_xy=(args.start_x, args.start_y),
        trajectory_check_samples=args.trajectory_check_samples,
        enforce_xy_limits=False,
    )
    if config.transform_profile.output_size != MODEL_IMAGE_SHAPE[0]:
        parser.error(
            f"transform profile must output {MODEL_IMAGE_SHAPE[0]}x{MODEL_IMAGE_SHAPE[1]}"
        )
    artifact_manifest = _validate_artifact_contract(
        args,
        parser,
        repo_root,
        args.transform_profile,
        args.camera_params,
    )

    mode_name = "EXECUTE" if args.execute else "DRY RUN"
    print(f"Starting real PushBox evaluation in {mode_name} mode")
    if args.negate_planner_action:
        print(
            "WARNING: NEGATED PLANNER-ACTION DIAGNOSTIC is enabled. "
            "Autonomous robot XY commands will be planner_action * -1; "
            "manual, reset, and startup motion are unchanged."
        )
    if args.execute and bounds is None:
        print(
            "WARNING: autonomous XY limits are disabled. Keep the physical "
            "emergency stop in hand."
        )

    planner = PlannerProcess(args, repo_root)
    try:
        planner.start()
    except Exception:
        planner.close()
        raise
    if args.preflight:
        print(
            "preflight passed: artifact contract, model, dataset, planner, and "
            "collector modules are ready; display, camera, and robot were not accessed"
        )
        planner.close()
        return 0

    image_store = LatestImageStore()
    ros = RosImageSubscriber(image_store)
    ui = RealRobotUI(HeldInput(config.magnitudes))
    arm: Any = None
    recorder: RunRecorder | None = None
    home_positions = np.array(
        [0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.0]
    )
    target_xy = np.array(config.start_xy, dtype=np.float64)
    normal_exit = False

    try:
        recorder = RunRecorder(
            args.output_dir,
            {
                "argv": vars(args),
                "mode": mode_name,
                "artifact_manifest": artifact_manifest,
                "collector_repo": collector_repo,
                "transform_profile_sha256": _sha256(args.transform_profile),
                "camera_params_sha256": _sha256(args.camera_params),
            },
        )
        print(f"Run record: {recorder.path}")
        recorder.event("run_started", mode=mode_name)

        if not args.skip_camera_check:
            report = measure_stream_health(
                duration_s=args.camera_check_seconds, max_age_s=HEALTH_MAX_AGE_S
            )
            print(report.summary())
            if not report.healthy:
                print("camera stream failed the health gate; robot was not connected")
                recorder.event("camera_health_failed", summary=report.summary())
                return 1

        # Validate both runtime dependencies and the local display before any
        # robot connection or startup trajectory.
        ros.start()
        ui.start()

        if args.execute:
            arm = FollowerArmAdapter.connect(config)
            arm.home(home_positions)
            arm.send_cartesian(
                np.r_[target_xy, config.safe_z, FIXED_ORIENTATION],
                goal_time=2.0,
                blocking=True,
            )
            arm.send_cartesian(
                np.r_[target_xy, config.fixed_z, FIXED_ORIENTATION],
                goal_time=2.0,
                blocking=True,
            )
            startup_pose = validate_measured_pose(
                arm.pose(),
                target_xy=target_xy,
                fixed_z=config.fixed_z,
                orientation=FIXED_ORIENTATION,
                xy_tolerance_m=args.xy_tracking_tolerance,
                z_tolerance_m=config.fixed_z_warning_tolerance,
                orientation_tolerance_rad=args.orientation_tolerance,
                bounds=bounds,
                require_tracking=True,
            )
            startup_velocity = np.asarray(arm.velocities(), dtype=np.float64).reshape(-1)
            if (
                startup_velocity.size < 3
                or not np.isfinite(startup_velocity[:3]).all()
                or np.linalg.norm(startup_velocity[:3]) > args.settled_linear_speed
            ):
                raise RuntimeError("arm did not settle after the startup trajectory")
            recorder.event("startup_pose_verified", pose=startup_pose)

        print(
            "Controls: arrows=manual, 1/2/3=speed, g=capture goal, v=preview, "
            "p=start/pause autonomy, SPACE=pause, r=reset, s/f=outcome, q=quit"
        )

        preview = np.zeros(MODEL_IMAGE_SHAPE, dtype=np.uint8)
        preview_sequence: int | None = None
        plan_sequence: int | None = None
        goal: np.ndarray | None = None
        autonomous = False
        preview_once = False
        resetting = False
        quit_requested = False
        pending: Future | None = None
        pending_generation = -1
        generation = 0
        reset_planner = False
        action_count = 0
        trial_id = 0
        trial_complete = True
        latest_action = np.zeros(2, dtype=np.float32)
        latest_plan = np.zeros((args.horizon, 2), dtype=np.float32)
        latest_cost: float | None = None
        latest_solve_s: float | None = None
        latest_goal_distance: float | None = None
        next_plan_time = 0.0
        previous_executed_action: np.ndarray | None = None
        faulted = False
        fault_reason = ""
        tracking_check_after = 0.0
        motion_settle_deadline = 0.0
        motion_command_ns: int | None = None
        settled_observation_after_ns: int | None = None
        last_pose: np.ndarray | None = None
        last_command_ns: int | None = None
        pending_started_at = 0.0
        pending_frame_receipt_ns = 0
        pending_frame_sequence = -1
        pending_current: np.ndarray | None = None
        pending_command_to_capture_s: float | None = None
        pending_settle_to_capture_s: float | None = None

        def mark_motion_pending(command_ns: int | None = None) -> None:
            nonlocal tracking_check_after, motion_settle_deadline
            nonlocal motion_command_ns, settled_observation_after_ns
            issued_ns = time.monotonic_ns() if command_ns is None else int(command_ns)
            issued_s = issued_ns / 1e9
            motion_command_ns = issued_ns
            tracking_check_after = (
                issued_s + TICK_S + args.tracking_settle_grace
            )
            motion_settle_deadline = issued_s + args.motion_settle_timeout
            settled_observation_after_ns = None

        def read_pose(*, require_tracking: bool) -> np.ndarray:
            if not args.execute or arm is None:
                return np.r_[target_xy, config.fixed_z, FIXED_ORIENTATION]
            measured = validate_measured_pose(
                arm.pose(),
                target_xy=target_xy,
                fixed_z=config.fixed_z,
                orientation=FIXED_ORIENTATION,
                xy_tolerance_m=args.xy_tracking_tolerance,
                z_tolerance_m=config.fixed_z_warning_tolerance,
                orientation_tolerance_rad=args.orientation_tolerance,
                bounds=bounds,
                require_tracking=require_tracking,
            )
            return measured

        def hold_measured_pose() -> None:
            nonlocal target_xy, last_pose
            if not args.execute or arm is None:
                return
            measured = np.asarray(arm.pose(), dtype=np.float64).reshape(-1)
            if measured.size < 6 or not np.isfinite(measured[:6]).all():
                raise RuntimeError("cannot hold an invalid measured pose")
            measured = measured[:6]
            arm.send_cartesian(measured)
            command_ns = time.monotonic_ns()
            target_xy = measured[:2].copy()
            last_pose = measured
            mark_motion_pending(command_ns)

        def pause(reason: str, *, hold: bool = False) -> None:
            nonlocal autonomous, preview_once, resetting, generation
            was_active = autonomous or preview_once or resetting
            if was_active or hold:
                print(f"paused: {reason}")
            autonomous = False
            preview_once = False
            resetting = False
            generation += 1
            ui.suppress_manual_until_release()
            if hold:
                hold_measured_pose()
            recorder.event("paused", reason=reason, held_measured_pose=hold)

        def latch_fault(reason: str) -> None:
            nonlocal faulted, fault_reason
            if faulted:
                return
            faulted = True
            fault_reason = reason
            # Faults fail closed: do not issue a corrective or recovery command.
            pause(reason, hold=False)
            print(f"FAULT (restart required): {reason}")
            recorder.event("fault", reason=reason)

        next_tick = time.monotonic()
        next_render = next_tick
        while not quit_requested:
            events = ui.poll()
            intent = prioritize_ui_events(events)
            if intent is not None and intent[0] == "quit":
                if not trial_complete:
                    recorder.event("trial_outcome", trial_id=trial_id, outcome="abort")
                pause("operator quit", hold=not faulted)
                quit_requested = True
                continue
            if intent is not None and intent[0] == "pause":
                pause(intent[1] or "operator pause", hold=True)
                continue
            if intent is not None and intent[0] == "outcome":
                if goal is None or trial_complete:
                    print("no active trial to label")
                else:
                    pause(f"trial marked {intent[1]}", hold=True)
                    trial_complete = True
                    recorder.event(
                        "trial_outcome", trial_id=trial_id, outcome=intent[1]
                    )
            elif intent is not None and intent[0] == "reset":
                if faulted:
                    print(f"reset blocked by fault: {fault_reason}")
                else:
                    pause("reset requested", hold=True)
                    resetting = True
            elif intent is not None and intent[0] == "goal":
                if faulted:
                    print(f"goal capture blocked by fault: {fault_reason}")
                else:
                    if not trial_complete:
                        recorder.event(
                            "trial_outcome", trial_id=trial_id, outcome="replaced"
                        )
                    pause("capturing a new goal", hold=True)
                    snapshot = image_store.snapshot()
                    if snapshot is None or not LatestImageStore.accept(
                        snapshot, None, time.monotonic_ns(), HEALTH_MAX_AGE_S
                    ):
                        print("goal capture rejected: camera frame is missing or stale")
                    else:
                        goal = config.transform_profile.apply(snapshot.rgb).copy()
                        if goal.shape != MODEL_IMAGE_SHAPE:
                            latch_fault("transformed goal has an invalid shape")
                            continue
                        trial_id += 1
                        trial_complete = False
                        action_count = 0
                        reset_planner = True
                        previous_executed_action = None
                        last_command_ns = None
                        latest_action[:] = 0.0
                        goal_path = recorder.save_frame(f"trial_{trial_id:03d}_goal", goal)
                        recorder.event(
                            "trial_started",
                            trial_id=trial_id,
                            goal_frame=goal_path,
                            camera_sequence=snapshot.sequence,
                        )
                        print(
                            "goal captured. Reposition/reset the scene, then press v "
                            "for a preview or p to arm autonomous execution."
                        )
            elif intent is not None and intent[0] == "preview":
                if faulted:
                    print(f"preview blocked by fault: {fault_reason}")
                elif goal is None or trial_complete:
                    print("cannot preview: capture an active goal first")
                else:
                    pause("preview requested", hold=True)
                    preview_once = True
                    reset_planner = True
                    previous_executed_action = None
                    last_command_ns = None
                    next_plan_time = (
                        tracking_check_after if args.execute else 0.0
                    )
            elif intent is not None and intent[0] == "autonomous":
                if autonomous:
                    pause("autonomous execution toggled off", hold=True)
                elif faulted:
                    print(f"autonomous execution blocked by fault: {fault_reason}")
                elif not ui.focused:
                    print("cannot start: evaluator window is not focused")
                elif goal is None:
                    print("cannot start: press g to capture a goal first")
                elif trial_complete:
                    print("cannot start: capture a new goal to begin a trial")
                elif action_count >= args.max_actions:
                    print("cannot start: trial action budget is exhausted")
                else:
                    pause("arming autonomous execution", hold=True)
                    autonomous = True
                    preview_once = False
                    reset_planner = True
                    previous_executed_action = None
                    last_command_ns = None
                    next_plan_time = (
                        tracking_check_after if args.execute else 0.0
                    )
                    print(f"autonomous execution armed ({mode_name})")
                    recorder.event("autonomy_armed", trial_id=trial_id)

            camera_error = ros.take_error()
            if camera_error is not None:
                print(f"camera subscriber error: {camera_error}")
                latch_fault("camera subscriber error")

            snapshot = image_store.snapshot()
            if snapshot is not None and snapshot.sequence != preview_sequence:
                preview = config.transform_profile.apply(snapshot.rgb)
                if preview.shape != MODEL_IMAGE_SHAPE:
                    latch_fault("transformed preview has an invalid shape")
                    continue
                preview_sequence = snapshot.sequence

            if pending is not None and pending.done():
                completed_generation = pending_generation
                try:
                    result = pending.result()
                except Exception as exc:
                    if completed_generation == generation:
                        print(f"planner failed: {exc}")
                        latch_fault(f"planner failure: {exc}")
                    else:
                        recorder.event("stale_planner_error_ignored", error=str(exc))
                else:
                    if completed_generation == generation:
                        now_ns = time.monotonic_ns()
                        plan_age_s = (now_ns - pending_frame_receipt_ns) / 1e9
                        latest_snapshot = image_store.snapshot()
                        stream_is_fresh = LatestImageStore.accept(
                            latest_snapshot, None, now_ns, HEALTH_MAX_AGE_S
                        )
                        if plan_age_s > args.max_plan_age:
                            latch_fault(
                                f"plan age {plan_age_s:.3f}s exceeds "
                                f"{args.max_plan_age:.3f}s"
                            )
                        elif not stream_is_fresh:
                            latch_fault("camera stream became stale during planning")
                        else:
                            try:
                                (
                                    latest_action,
                                    latest_plan,
                                    latest_cost,
                                    latest_goal_distance,
                                    latest_solve_s,
                                    at_goal,
                                ) = validate_planner_result(
                                    result,
                                    cap_m=args.action_cap,
                                    horizon=args.horizon,
                                )
                            except (KeyError, TypeError, ValueError) as exc:
                                latch_fault(f"unsafe planner output: {exc}")

                        if faulted:
                            pass
                        elif at_goal and autonomous:
                            pause("goal tolerance reached", hold=False)
                            trial_complete = True
                            recorder.event(
                                "trial_outcome", trial_id=trial_id, outcome="at_goal"
                            )
                        elif at_goal and preview_once:
                            preview_once = False
                            preview_robot_action = transform_planner_action(
                                latest_action,
                                negate=args.negate_planner_action,
                                cap_m=args.action_cap,
                            )
                            recorder.event(
                                "preview",
                                trial_id=trial_id,
                                action=latest_action,
                                robot_action=preview_robot_action,
                                planner_action_negated=args.negate_planner_action,
                                plan=latest_plan,
                                cost=latest_cost,
                                goal_distance=latest_goal_distance,
                                solve_time_s=latest_solve_s,
                                plan_age_s=plan_age_s,
                                settle_to_capture_s=pending_settle_to_capture_s,
                                at_goal=True,
                            )
                            print("preview reports that the current scene is already at goal")
                        elif autonomous:
                            planner_action = latest_action.copy()
                            if np.linalg.norm(planner_action) < args.action_deadband:
                                planner_action[:] = 0.0
                            robot_requested_action = transform_planner_action(
                                planner_action,
                                negate=args.negate_planner_action,
                                cap_m=args.action_cap,
                            )
                            target, accepted = accepted_target(
                                target_xy, robot_requested_action, bounds
                            )
                            if (
                                np.linalg.norm(robot_requested_action) > 0.0
                                and np.linalg.norm(accepted) == 0.0
                            ):
                                pause(
                                    "planned action blocked by workspace limit",
                                    hold=False,
                                )
                            else:
                                measured_before = None
                                executed = np.zeros(2, dtype=np.float32)
                                command_ns = time.monotonic_ns()
                                if args.execute and arm is not None:
                                    try:
                                        measured_before = read_pose(require_tracking=True)
                                    except (RuntimeError, ValueError) as exc:
                                        latch_fault(f"pose verification failed: {exc}")
                                    if not faulted:
                                        arm.send_cartesian(
                                            np.r_[target, config.fixed_z, FIXED_ORIENTATION]
                                        )
                                        target_xy = target
                                        executed = accepted.astype(np.float32, copy=True)
                                        command_ns = time.monotonic_ns()
                                        mark_motion_pending(command_ns)
                                if not faulted:
                                    planner_feedback_action = transform_planner_action(
                                        executed,
                                        negate=args.negate_planner_action,
                                        cap_m=args.action_cap,
                                    )
                                    previous_executed_action = planner_feedback_action
                                    action_count += 1
                                    last_command_ns = command_ns
                                    next_plan_time = tracking_check_after
                                    frame_path = recorder.save_frame(
                                        f"trial_{trial_id:03d}_step_{action_count:03d}",
                                        pending_current,
                                    )
                                    recorder.event(
                                        "autonomous_step",
                                        trial_id=trial_id,
                                        step=action_count,
                                        observation_frame=frame_path,
                                        camera_sequence=pending_frame_sequence,
                                        plan_age_s=plan_age_s,
                                        command_to_capture_s=pending_command_to_capture_s,
                                        settle_to_capture_s=pending_settle_to_capture_s,
                                        requested_action=planner_action,
                                        robot_requested_action=robot_requested_action,
                                        accepted_action=accepted,
                                        executed_action=executed,
                                        planner_feedback_action=planner_feedback_action,
                                        planner_action_negated=args.negate_planner_action,
                                        plan=latest_plan,
                                        cost=latest_cost,
                                        goal_distance=latest_goal_distance,
                                        solve_time_s=latest_solve_s,
                                        measured_pose_before=measured_before,
                                        target_xy=target_xy,
                                    )
                                    settle_delay = (
                                        "n/a"
                                        if pending_settle_to_capture_s is None
                                        else f"{pending_settle_to_capture_s:.3f}s"
                                    )
                                    print(
                                        f"step {action_count}: planner={planner_action.tolist()}, "
                                        f"robot_requested={robot_requested_action.tolist()}, "
                                        f"executed={executed.tolist()}, solve={latest_solve_s:.3f}s, "
                                        f"age={plan_age_s:.3f}s, "
                                        f"obs_after_settle={settle_delay}, "
                                        f"goal_distance={latest_goal_distance:.6g}"
                                    )
                                    if action_count >= args.max_actions:
                                        pause(
                                            "maximum trial action count reached",
                                            hold=False,
                                        )
                                        trial_complete = True
                                        recorder.event(
                                            "trial_outcome",
                                            trial_id=trial_id,
                                            outcome="budget_exhausted",
                                        )
                        elif preview_once:
                            preview_once = False
                            preview_robot_action = transform_planner_action(
                                latest_action,
                                negate=args.negate_planner_action,
                                cap_m=args.action_cap,
                            )
                            recorder.event(
                                "preview",
                                trial_id=trial_id,
                                action=latest_action,
                                robot_action=preview_robot_action,
                                planner_action_negated=args.negate_planner_action,
                                plan=latest_plan,
                                cost=latest_cost,
                                goal_distance=latest_goal_distance,
                                solve_time_s=latest_solve_s,
                                plan_age_s=plan_age_s,
                                settle_to_capture_s=pending_settle_to_capture_s,
                            )
                            print(
                                f"preview planner={latest_action.tolist()}, "
                                f"robot_would_execute={preview_robot_action.tolist()}, "
                                f"solve={latest_solve_s:.3f}s, "
                                f"goal_distance={latest_goal_distance:.6g}"
                            )
                pending = None
                pending_current = None

            now = time.monotonic()
            if now >= next_tick:
                next_tick, _ = advance_tick_deadline(next_tick, now)
                if args.execute and arm is not None:
                    try:
                        awaiting_planner_settle = (
                            (autonomous or preview_once)
                            and settled_observation_after_ns is None
                        )
                        last_pose = read_pose(
                            require_tracking=(
                                now >= tracking_check_after
                                and not awaiting_planner_settle
                            )
                        )
                    except (RuntimeError, ValueError) as exc:
                        latch_fault(f"pose verification failed: {exc}")

                if not faulted and not autonomous and not preview_once:
                    manual_action = ui.manual_action()
                    if np.linalg.norm(manual_action) > 0.0:
                        resetting = False
                        target, _ = accepted_target(target_xy, manual_action, bounds)
                        if args.execute and arm is not None:
                            try:
                                last_pose = read_pose(require_tracking=True)
                            except (RuntimeError, ValueError) as exc:
                                latch_fault(f"pose verification failed: {exc}")
                            if not faulted:
                                arm.send_cartesian(
                                    np.r_[target, config.fixed_z, FIXED_ORIENTATION]
                                )
                                target_xy = target
                                mark_motion_pending()
                    elif resetting:
                        start = np.array(config.start_xy, dtype=np.float64)
                        current_xy = (
                            last_pose[:2]
                            if args.execute and last_pose is not None
                            else target_xy
                        )
                        remaining = start - current_xy
                        distance = float(np.linalg.norm(remaining))
                        if distance <= RESET_TOLERANCE_M:
                            if args.execute and arm is not None:
                                try:
                                    last_pose = read_pose(require_tracking=True)
                                except (RuntimeError, ValueError) as exc:
                                    latch_fault(f"pose verification failed: {exc}")
                            if not faulted:
                                resetting = False
                                target_xy = current_xy.copy()
                                print("arm reset reached configured start X/Y")
                                recorder.event("reset_complete", measured_xy=current_xy)
                        else:
                            if distance > RESET_STEP_M:
                                remaining *= RESET_STEP_M / distance
                            target, _ = accepted_target(current_xy, remaining, bounds)
                            if args.execute and arm is not None:
                                try:
                                    last_pose = read_pose(require_tracking=True)
                                except (RuntimeError, ValueError) as exc:
                                    latch_fault(f"pose verification failed: {exc}")
                                if not faulted:
                                    arm.send_cartesian(
                                        np.r_[target, config.fixed_z, FIXED_ORIENTATION]
                                    )
                                    target_xy = target
                                    mark_motion_pending()

            # Begin completion checks after the nominal 0.2-second trajectory,
            # then plan only from a camera receipt newer than verified settling.
            now = time.monotonic()
            wants_plan = not faulted and (autonomous or preview_once)
            if (
                wants_plan
                and goal is not None
                and pending is None
                and now >= next_plan_time
            ):
                ready_for_snapshot = True
                if args.execute and arm is not None:
                    if settled_observation_after_ns is None:
                        try:
                            measured = read_pose(require_tracking=False)
                            velocity = arm.velocities()
                            complete, tracking_error_m, linear_speed_m_s = (
                                motion_completion_status(
                                    measured,
                                    velocity,
                                    target_xy=target_xy,
                                    xy_tolerance_m=args.xy_tracking_tolerance,
                                    settled_linear_speed_m_s=args.settled_linear_speed,
                                )
                            )
                        except (RuntimeError, ValueError) as exc:
                            latch_fault(f"motion completion check failed: {exc}")
                        else:
                            if complete:
                                last_pose = measured
                                settled_observation_after_ns = time.monotonic_ns()
                                settle_time_s = (
                                    None
                                    if motion_command_ns is None
                                    else (
                                        settled_observation_after_ns
                                        - motion_command_ns
                                    )
                                    / 1e9
                                )
                                recorder.event(
                                    "motion_settled",
                                    trial_id=trial_id,
                                    step=action_count,
                                    tracking_error_m=tracking_error_m,
                                    linear_speed_m_s=linear_speed_m_s,
                                    settle_time_s=settle_time_s,
                                )
                            elif now >= motion_settle_deadline:
                                latch_fault(
                                    "arm did not settle before timeout: "
                                    f"tracking error {tracking_error_m * 1000:.1f} mm, "
                                    f"linear speed {linear_speed_m_s:.4f} m/s"
                                )
                        # Even when settling is verified in this iteration, wait
                        # for a camera receipt strictly newer than the verification.
                        ready_for_snapshot = False

                if ready_for_snapshot and not faulted:
                    snapshot = image_store.snapshot()
                    now_ns = time.monotonic_ns()
                    if snapshot is None:
                        pause("camera frame unavailable", hold=False)
                    elif not snapshot_follows_settle(
                        snapshot.receipt_monotonic_ns,
                        settled_observation_after_ns,
                    ):
                        pass
                    elif not LatestImageStore.accept(
                        snapshot, plan_sequence, now_ns, HEALTH_MAX_AGE_S
                    ):
                        age_s = (now_ns - snapshot.receipt_monotonic_ns) / 1e9
                        if age_s > HEALTH_MAX_AGE_S:
                            pause("camera frame stale", hold=False)
                    else:
                        current = config.transform_profile.apply(snapshot.rgb)
                        if current.shape != MODEL_IMAGE_SHAPE:
                            latch_fault("transformed planner frame has an invalid shape")
                        else:
                            command_to_capture_s = (
                                None
                                if last_command_ns is None
                                else (
                                    snapshot.receipt_monotonic_ns - last_command_ns
                                )
                                / 1e9
                            )
                            settle_to_capture_s = (
                                None
                                if settled_observation_after_ns is None
                                else (
                                    snapshot.receipt_monotonic_ns
                                    - settled_observation_after_ns
                                )
                                / 1e9
                            )
                            if command_to_capture_s is not None and (
                                command_to_capture_s <= 0.0
                            ):
                                latch_fault(
                                    "planner observation was captured before its command"
                                )
                            elif settle_to_capture_s is not None and (
                                settle_to_capture_s <= 0.0
                                or settle_to_capture_s > args.max_observation_delay
                            ):
                                latch_fault(
                                    "settle-to-observation delay "
                                    f"{settle_to_capture_s:.3f}s is outside (0, "
                                    f"{args.max_observation_delay:.3f}s]"
                                )
                            else:
                                plan_sequence = snapshot.sequence
                                pending_generation = generation
                                pending_started_at = now
                                pending_frame_receipt_ns = (
                                    snapshot.receipt_monotonic_ns
                                )
                                pending_frame_sequence = snapshot.sequence
                                pending_current = current.copy()
                                pending_command_to_capture_s = command_to_capture_s
                                pending_settle_to_capture_s = settle_to_capture_s
                                pending = planner.plan(
                                    current,
                                    goal,
                                    reset=reset_planner,
                                    previous_action=previous_executed_action,
                                )
                                reset_planner = False
                                previous_executed_action = None

            if (
                pending is not None
                and pending_generation == generation
                and time.monotonic() - pending_started_at > args.planner_timeout
            ):
                latch_fault(
                    f"planner request exceeded {args.planner_timeout:.3f}s deadline"
                )

            now = time.monotonic()
            if now >= next_render:
                state = (
                    "FAULT"
                    if faulted
                    else (
                        "PLANNING"
                        if pending is not None and pending_generation == generation
                        else (
                            "AUTO"
                            if autonomous
                            else (
                                "PREVIEW"
                                if preview_once
                                else ("RESET" if resetting else "PAUSED")
                            )
                        )
                    )
                )
                latency = "-" if latest_solve_s is None else f"{latest_solve_s:.2f}s"
                display_robot_action = transform_planner_action(
                    latest_action,
                    negate=args.negate_planner_action,
                    cap_m=args.action_cap,
                )
                action_status = (
                    f"plan=({latest_action[0]:+.4f},{latest_action[1]:+.4f}) "
                    f"cmd=({display_robot_action[0]:+.4f},{display_robot_action[1]:+.4f})"
                    if args.negate_planner_action
                    else f"a=({latest_action[0]:+.4f},{latest_action[1]:+.4f})"
                )
                status = (
                    f"{mode_name} {state} step={action_count}/{args.max_actions} "
                    f"{action_status} solve={latency}"
                )
                ui.render(preview, goal, status)
                next_render = now + 1.0 / ui.ui_fps

            sleep_s = min(next_tick, next_render) - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(min(sleep_s, UI_SLEEP_MAX_S))

        if args.execute and arm is not None:
            if faulted:
                print(
                    "Fault is latched; automatic home-to-zero recovery was skipped. "
                    "Use the physical E-stop and inspect the arm."
                )
                recorder.event("automatic_recovery_skipped", reason=fault_reason)
            else:
                move_arm_home_then_zero(arm, home_positions)
                recorder.event("shutdown_motion_completed")
        normal_exit = True
    finally:
        ui.close()
        ros.stop()
        planner.close()
        if recorder is not None:
            recorder.event(
                "run_finished",
                normal_exit=normal_exit,
                faulted=locals().get("faulted", False),
                fault_reason=locals().get("fault_reason", ""),
            )
            recorder.close()
        if args.execute and arm is not None and not normal_exit:
            print(
                "Evaluator exited on an error; automatic recovery motion was not "
                "attempted. Use the physical E-stop and inspect the arm."
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent
    parser = _build_parser(repo_root)
    args = parser.parse_args(argv)
    return run(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
