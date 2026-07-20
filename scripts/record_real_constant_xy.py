#!/usr/bin/env python3
"""Record a deterministic constant-XY rollout on the real PushBox arm.

The model is deliberately absent from the control loop.  A live run records
state 0 before motion and state k immediately after action k, producing
``frame_000.png -> action 1 -> ... -> action N -> frame_NNN.png``.

Without ``--execute`` this command is a preflight only: it does not open a
display, start ROS, connect to the arm, or move hardware.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import imageio.v3 as iio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTOR_REPO = REPO_ROOT.parent / "wm_data_collection"
DEFAULT_REFERENCE_IMAGE = REPO_ROOT / (
    "real_robot_runs/20260717_165123_1069809/frames/trial_001_step_007.png"
)
DEFAULT_GOAL_IMAGE = REPO_ROOT / (
    "real_robot_runs/20260717_165123_1069809/frames/trial_001_goal.png"
)
DEFAULT_MANIFEST = REPO_ROOT / "config/real_robot_eval.json"
DEFAULT_START_X_M = 0.1898409503403485
DEFAULT_START_Y_M = 0.02137995091021234
DEFAULT_ACTION_X_M = 0.005
DEFAULT_ACTION_Y_M = 0.0
DEFAULT_STEPS = 50
DEFAULT_TICK_S = 0.2
MAX_ACTION_NORM_M = 0.010
MODEL_SHAPE = (224, 224, 3)
RAW_SHAPE = (480, 640, 3)
SCHEDULE_TOLERANCE_S = 0.02
UI_SLEEP_MAX_S = 0.005


class OperatorAbort(RuntimeError):
    """An operator stopped an otherwise valid exact rollout."""


class SafetyFault(RuntimeError):
    """A failed safety/recording invariant; no recovery motion is allowed."""


@dataclass(frozen=True)
class Geometry:
    start_xy: np.ndarray
    action_xy: np.ndarray
    targets_xy: np.ndarray
    bounds: tuple[tuple[float, float], tuple[float, float]]
    boundary_margin_m: float

    @property
    def steps(self) -> int:
        return int(self.targets_xy.shape[0] - 1)

    @property
    def final_xy(self) -> np.ndarray:
        return self.targets_xy[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_geometry(
    *,
    start_xy: Sequence[float],
    action_xy: Sequence[float],
    steps: int,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    boundary_margin_m: float,
) -> Geometry:
    """Precompute every absolute target and fail before hardware is touched."""
    start = np.asarray(start_xy, dtype=np.float64)
    action = np.asarray(action_xy, dtype=np.float64)
    if start.shape != (2,) or action.shape != (2,):
        raise ValueError("start_xy and action_xy must each contain two values")
    values = np.r_[start, action, x_bounds, y_bounds, boundary_margin_m]
    if not np.isfinite(values).all():
        raise ValueError("trajectory geometry must be finite")
    if int(steps) < 1:
        raise ValueError("--steps must be positive")
    action_norm = float(np.linalg.norm(action))
    if not 0.0 < action_norm <= MAX_ACTION_NORM_M + 1e-12:
        raise ValueError(
            f"action norm must be in (0, {MAX_ACTION_NORM_M:g}] m, got {action_norm:g}"
        )
    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError("workspace bounds must be strictly increasing")
    if boundary_margin_m < 0.0:
        raise ValueError("boundary margin must be non-negative")

    indices = np.arange(int(steps) + 1, dtype=np.float64)[:, None]
    targets = start[None] + indices * action[None]
    lower = np.array(
        [x_bounds[0] + boundary_margin_m, y_bounds[0] + boundary_margin_m]
    )
    upper = np.array(
        [x_bounds[1] - boundary_margin_m, y_bounds[1] - boundary_margin_m]
    )
    if np.any(lower > upper):
        raise ValueError("boundary margin consumes the configured workspace")
    outside = np.flatnonzero(np.any((targets < lower) | (targets > upper), axis=1))
    if outside.size:
        index = int(outside[0])
        raise ValueError(
            f"absolute target {index} {targets[index].tolist()} is outside the "
            f"workspace with {boundary_margin_m * 1000:.1f} mm margin"
        )
    return Geometry(
        start_xy=start,
        action_xy=action,
        targets_xy=targets,
        bounds=(x_bounds, y_bounds),
        boundary_margin_m=float(boundary_margin_m),
    )


def load_source_event(reference_image: Path) -> dict[str, Any] | None:
    events_path = reference_image.parent.parent / "events.jsonl"
    if not events_path.is_file():
        return None
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        observation = event.get("observation_frame")
        if observation and Path(observation).stem == reference_image.stem:
            return event
    return None


def require_rgb(path: Path, shape: tuple[int, int, int] = MODEL_SHAPE) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = np.asarray(iio.imread(path))
    if image.shape != shape or image.dtype != np.uint8:
        raise ValueError(f"{path} must be RGB uint8 with shape {shape}, got {image.shape} {image.dtype}")
    return image


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_rgb_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    if not frames:
        return
    import imageio_ffmpeg

    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{float(fps):g}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            array = np.asarray(frame, dtype=np.uint8)
            if array.shape != first.shape:
                raise ValueError("all video frames must have the same shape")
            process.stdin.write(np.ascontiguousarray(array).tobytes())
    finally:
        process.stdin.close()
    if process.wait(timeout=30) != 0:
        raise RuntimeError(f"ffmpeg failed to write {path}")


class ProbeRecorder:
    """Crash-readable metadata/events plus non-blocking PNG writes."""

    def __init__(
        self,
        path: Path,
        metadata: dict[str, Any],
        reference: np.ndarray,
        reference_path: Path,
        goal: np.ndarray | None,
        goal_path: Path | None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.frames_path = self.path / "frames"
        self.raw_frames_path = self.path / "raw_frames"
        self.reference_path = self.path / "reference"
        self.frames_path.mkdir(parents=True, exist_ok=False)
        self.raw_frames_path.mkdir()
        self.reference_path.mkdir()
        shutil.copy2(reference_path, self.reference_path / "source.png")
        if goal is not None and goal_path is not None:
            shutil.copy2(goal_path, self.reference_path / "task_goal.png")
        self.metadata = dict(metadata)
        write_json_atomic(self.path / "metadata.json", self.metadata)
        self._events = (self.path / "events.jsonl").open("a", buffering=1)
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="probe-png")
        self._writes: list[Future[Any]] = []
        self.model_frames: list[np.ndarray] = []
        self.raw_frames: list[np.ndarray] = []
        self.closed = False

    def event(self, kind: str, **values: Any) -> None:
        payload = {
            "type": kind,
            "wall_time": datetime.now().astimezone().isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            **values,
        }
        self._events.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")
        self._events.flush()

    @staticmethod
    def _write_raw_state(raw_path: Path, raw: np.ndarray) -> None:
        iio.imwrite(raw_path, raw, compress_level=1)

    def check_pending_writes(self, *, wait: bool = False) -> None:
        """Surface background raw-frame failures before issuing more motion."""
        pending: list[Future[Any]] = []
        for future in self._writes:
            if wait or future.done():
                future.result()
            else:
                pending.append(future)
        self._writes = pending

    def state(self, step: int, model: np.ndarray, raw: np.ndarray, **values: Any) -> None:
        model_copy = np.asarray(model, dtype=np.uint8).copy()
        raw_copy = np.asarray(raw, dtype=np.uint8).copy()
        if model_copy.shape != MODEL_SHAPE or raw_copy.shape != RAW_SHAPE:
            raise ValueError("recorder received an invalid model or raw frame")
        filename = f"frame_{int(step):03d}.png"
        self.model_frames.append(model_copy)
        self.raw_frames.append(raw_copy)
        # The exact 224x224 model input is the critical artifact.  Make it
        # durable synchronously before a following motion can be issued; the
        # larger raw PNG is allowed to finish in the background.
        iio.imwrite(self.frames_path / filename, model_copy, compress_level=1)
        self._writes.append(
            self._writer.submit(
                self._write_raw_state,
                self.raw_frames_path / filename,
                raw_copy,
            )
        )
        self.event(
            "state_captured",
            step=int(step),
            frame=str(Path("frames") / filename),
            raw_frame=str(Path("raw_frames") / filename),
            **values,
        )

    def close(self, *, status: str, error: str | None = None) -> None:
        if self.closed:
            return
        self.closed = True
        write_error: str | None = None
        try:
            self.check_pending_writes(wait=True)
            self._writer.shutdown(wait=True)
            if self.model_frames:
                fps = float(self.metadata["trajectory"]["fps"])
                write_rgb_video(self.path / "real_model_view.mp4", self.model_frames, fps)
                write_rgb_video(self.path / "real_camera.mp4", self.raw_frames, fps)
        except Exception as exc:  # retain the control-loop result and report both
            write_error = f"{type(exc).__name__}: {exc}"
            self._writer.shutdown(wait=True)
        self.event(
            "run_closed",
            status=status,
            captured_state_count=len(self.model_frames),
            error=error,
            artifact_error=write_error,
        )
        self._events.close()
        self.metadata.update(
            {
                "status": status if write_error is None else "artifact_error",
                "closed_at": datetime.now().astimezone().isoformat(),
                "captured_state_count": len(self.model_frames),
                "error": error,
                "artifact_error": write_error,
            }
        )
        write_json_atomic(self.path / "metadata.json", self.metadata)
        if write_error is not None:
            raise RuntimeError(write_error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-repo", type=Path, default=DEFAULT_COLLECTOR_REPO)
    parser.add_argument("--transform-profile", type=Path)
    parser.add_argument("--camera-params", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-image", type=Path, default=DEFAULT_REFERENCE_IMAGE)
    parser.add_argument("--goal-image", type=Path, default=DEFAULT_GOAL_IMAGE)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="connect to and move the real arm")
    mode.add_argument("--preflight", action="store_true", help="validate only; never access camera/display/arm")
    parser.add_argument("--follower-ip", default="192.168.1.3")
    parser.add_argument("--start-x", type=float, default=DEFAULT_START_X_M)
    parser.add_argument("--start-y", type=float, default=DEFAULT_START_Y_M)
    parser.add_argument("--action-x-m", type=float, default=DEFAULT_ACTION_X_M)
    parser.add_argument("--action-y-m", type=float, default=DEFAULT_ACTION_Y_M)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--tick-seconds", type=float, default=DEFAULT_TICK_S)
    parser.add_argument("--fixed-z", type=float, default=0.03)
    parser.add_argument("--safe-z", type=float, default=0.15)
    parser.add_argument("--x-min", type=float, default=0.0183)
    parser.add_argument("--x-max", type=float, default=0.45)
    parser.add_argument("--y-min", type=float, default=-0.26)
    parser.add_argument("--y-max", type=float, default=0.26)
    parser.add_argument("--boundary-margin", type=float, default=0.005)
    parser.add_argument("--trajectory-check-samples", type=int, default=10)
    parser.add_argument("--camera-check-seconds", type=float, default=5.0)
    parser.add_argument("--skip-camera-check", action="store_true")
    parser.add_argument("--xy-tracking-tolerance", type=float, default=0.002)
    parser.add_argument("--z-tolerance", type=float, default=0.002)
    parser.add_argument("--orientation-tolerance", type=float, default=0.1)
    parser.add_argument("--max-linear-speed", type=float, default=0.07)
    parser.add_argument("--reference-start-tolerance", type=float, default=0.002)
    return parser


def resolve_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    args.collector_repo = args.collector_repo.expanduser().resolve()
    args.transform_profile = (
        args.transform_profile.expanduser().resolve()
        if args.transform_profile is not None
        else args.collector_repo / "config/transform-profile.json"
    )
    args.camera_params = (
        args.camera_params.expanduser().resolve()
        if args.camera_params is not None
        else args.collector_repo / "config/camera-params.yaml"
    )
    args.manifest = args.manifest.expanduser().resolve()
    args.reference_image = args.reference_image.expanduser().resolve()
    args.goal_image = args.goal_image.expanduser().resolve() if args.goal_image else None
    if args.output_dir is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        args.output_dir = REPO_ROOT / "real_robot_runs" / f"constant_plus_x_{stamp}_{os.getpid()}"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()

    positive = {
        "--tick-seconds": args.tick_seconds,
        "--fixed-z": args.fixed_z,
        "--safe-z": args.safe_z,
        "--camera-check-seconds": args.camera_check_seconds,
        "--xy-tracking-tolerance": args.xy_tracking_tolerance,
        "--z-tolerance": args.z_tolerance,
        "--orientation-tolerance": args.orientation_tolerance,
        "--max-linear-speed": args.max_linear_speed,
        "--reference-start-tolerance": args.reference_start_tolerance,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and positive")
    if not math.isclose(args.tick_seconds, DEFAULT_TICK_S, abs_tol=1e-9):
        parser.error("--tick-seconds must remain 0.2 s to match the training data")
    if args.safe_z <= args.fixed_z:
        parser.error("--safe-z must be above --fixed-z")
    if args.trajectory_check_samples < 1:
        parser.error("--trajectory-check-samples must be positive")
    return args


def validate_contract(args: argparse.Namespace, geometry: Geometry) -> dict[str, Any]:
    if not args.collector_repo.is_dir():
        raise FileNotFoundError(f"collector repository not found: {args.collector_repo}")
    for path in (args.transform_profile, args.camera_params, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = require_rgb(args.reference_image)
    goal = require_rgb(args.goal_image) if args.goal_image is not None else None
    manifest = json.loads(args.manifest.read_text())

    expected_transform = manifest["collector"]["transform_profile_sha256"]
    expected_camera = manifest["collector"]["camera_params_sha256"]
    if sha256(args.transform_profile) != expected_transform:
        raise ValueError("transform profile hash does not match the real-eval manifest")
    if sha256(args.camera_params) != expected_camera:
        raise ValueError("camera parameter hash does not match the real-eval manifest")
    runtime = manifest["runtime"]
    if not math.isclose(1.0 / args.tick_seconds, float(runtime["tick_hz"]), abs_tol=1e-9):
        raise ValueError("recording frequency does not match the checkpoint contract")
    if float(np.linalg.norm(geometry.action_xy)) > float(runtime["maximum_action_norm_m"]):
        raise ValueError("constant action exceeds the manifest action cap")

    source_event = load_source_event(args.reference_image)
    if source_event is None:
        raise ValueError("could not resolve the reference image's autonomous-step event")
    measured = np.asarray(source_event.get("measured_pose_before", []), dtype=np.float64)
    if measured.size < 6 or not np.isfinite(measured[:6]).all():
        raise ValueError("reference event has no finite six-axis measured_pose_before")
    mismatch = float(np.linalg.norm(measured[:2] - geometry.start_xy))
    if mismatch > args.reference_start_tolerance:
        raise ValueError(
            f"configured start differs from reference pose by {mismatch * 1000:.1f} mm"
        )
    if abs(float(measured[2]) - args.fixed_z) > args.z_tolerance:
        raise ValueError(
            f"configured fixed Z differs from the reference by "
            f"{abs(float(measured[2]) - args.fixed_z) * 1000:.1f} mm"
        )
    fixed_orientation = np.array([0.0, np.pi / 2.0, 0.0], dtype=np.float64)
    orientation_error = (measured[3:6] - fixed_orientation + np.pi) % (2 * np.pi) - np.pi
    if float(np.linalg.norm(orientation_error)) > args.orientation_tolerance:
        raise ValueError("configured downward orientation does not match the reference pose")

    world_spec = Path(manifest["world_model"]["checkpoint"])
    world_path = REPO_ROOT / "stable-wm" / world_spec
    if not world_path.is_file():
        raise FileNotFoundError(f"epoch-146 world checkpoint not found: {world_path}")
    actual_world_hash = sha256(world_path)
    if actual_world_hash != manifest["world_model"]["checkpoint_sha256"]:
        raise ValueError("world checkpoint hash does not match the real-eval manifest")

    normalization = manifest["normalization"]["action"]
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.asarray(normalization["std"], dtype=np.float64)
    normalized_action = (geometry.action_xy - mean) / std
    return {
        "reference": reference,
        "goal": goal,
        "manifest": manifest,
        "source_event": source_event,
        "reference_start_error_m": mismatch,
        "world_path": world_path,
        "world_hash": actual_world_hash,
        "normalized_action": normalized_action,
    }


def print_preflight(args: argparse.Namespace, geometry: Geometry, contract: dict[str, Any]) -> None:
    x_margin = args.x_max - geometry.final_xy[0]
    print("Constant +XY real-robot probe preflight PASSED")
    print(f"  reference state: {args.reference_image}")
    print(f"  task goal:       {args.goal_image}")
    print(f"  checkpoint:      {contract['world_path'].name} (SHA-256 verified)")
    print(f"  start XY:        {geometry.start_xy.tolist()} m")
    print(f"  action:          {geometry.action_xy.tolist()} m x {geometry.steps}")
    print(f"  final XY:        {geometry.final_xy.tolist()} m")
    print(f"  displacement:    {(geometry.action_xy * geometry.steps).tolist()} m")
    print(f"  duration:        {geometry.steps * args.tick_seconds:.1f} s at {1 / args.tick_seconds:.1f} Hz")
    print(f"  +X hard-bound margin at final state: {x_margin * 1000:.2f} mm")
    print(f"  output states:   {geometry.steps + 1} (frame_000 through frame_{geometry.steps:03d})")
    if not args.execute:
        print("PREFLIGHT ONLY: camera, display, and robot were not accessed.")
    else:
        print(
            "LIVE SAFETY: keep the physical E-stop in hand and clear the complete "
            f"tool/box path through X={geometry.final_xy[0]:.6f} m."
        )


def _validate_velocity(velocity: Any, maximum: float) -> np.ndarray:
    measured = np.asarray(velocity, dtype=np.float64).reshape(-1)
    if measured.size < 6 or not np.isfinite(measured[:6]).all():
        raise SafetyFault("measured Cartesian velocity is invalid")
    speed = float(np.linalg.norm(measured[:3]))
    if speed > maximum:
        raise SafetyFault(
            f"arm linear speed {speed:.4f} m/s exceeds {maximum:.4f} m/s"
        )
    return measured[:6]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = resolve_arguments(parser.parse_args(argv), parser)
    geometry = build_geometry(
        start_xy=(args.start_x, args.start_y),
        action_xy=(args.action_x_m, args.action_y_m),
        steps=args.steps,
        x_bounds=(args.x_min, args.x_max),
        y_bounds=(args.y_min, args.y_max),
        boundary_margin_m=args.boundary_margin,
    )

    # Collector imports stay lazy so a preflight never imports ROS or Trossen.
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(args.collector_repo))
    from scripts.collect_keyboard_xy import (
        CollectorConfig,
        FIXED_ORIENTATION,
        FollowerArmAdapter,
        move_arm_home_then_zero,
    )
    from scripts.pygame_input import HeldInput
    from scripts.ros_camera import (
        HEALTH_MAX_AGE_S,
        LatestImageStore,
        RosImageSubscriber,
        measure_stream_health,
    )
    from real_robot_eval import RealRobotUI, validate_measured_pose

    config = CollectorConfig.from_paths(
        args.transform_profile,
        args.camera_params,
        follower_ip=args.follower_ip,
        fixed_z=args.fixed_z,
        safe_z=args.safe_z,
        fixed_z_warning_tolerance=args.z_tolerance,
        start_xy=tuple(geometry.start_xy),
        x_bounds=(args.x_min, args.x_max),
        y_bounds=(args.y_min, args.y_max),
        trajectory_check_samples=args.trajectory_check_samples,
        enforce_xy_limits=True,
    )
    if config.transform_profile.output_size != MODEL_SHAPE[0]:
        raise ValueError("transform profile must produce 224x224 images")
    contract = validate_contract(args, geometry)
    print_preflight(args, geometry, contract)
    if not args.execute:
        return 0

    metadata = {
        "schema_version": 1,
        "experiment": "real_constant_xy_latent_probe",
        "status": "initialized",
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": "EXECUTE",
        "semantics": {
            "frame_000": "real transformed camera state before action 1",
            "frame_k": "real transformed camera state after action k",
            "frame_count": geometry.steps + 1,
        },
        "trajectory": {
            "start_xy_m": geometry.start_xy,
            "final_xy_m": geometry.final_xy,
            "action_xy_m": geometry.action_xy,
            "normalized_action": contract["normalized_action"],
            "steps": geometry.steps,
            "tick_seconds": args.tick_seconds,
            "fps": 1.0 / args.tick_seconds,
            "nominal_targets_xy_m": geometry.targets_xy,
        },
        "robot": {
            "follower_ip": args.follower_ip,
            "fixed_z_m": args.fixed_z,
            "safe_z_m": args.safe_z,
            "fixed_orientation_rpy": FIXED_ORIENTATION,
            "workspace_bounds_m": geometry.bounds,
            "boundary_margin_m": geometry.boundary_margin_m,
            "trajectory_check_samples": args.trajectory_check_samples,
        },
        "safety": {
            "xy_tracking_tolerance_m": args.xy_tracking_tolerance,
            "z_tolerance_m": args.z_tolerance,
            "orientation_tolerance_rad": args.orientation_tolerance,
            "max_linear_speed_m_s": args.max_linear_speed,
            "schedule_tolerance_s": SCHEDULE_TOLERANCE_S,
            "camera_max_age_s": HEALTH_MAX_AGE_S,
        },
        "reference": {
            "source_original": str(args.reference_image),
            "source_copy": "reference/source.png",
            "source_sha256": sha256(args.reference_image),
            "task_goal_original": str(args.goal_image) if args.goal_image else None,
            "task_goal_copy": "reference/task_goal.png" if args.goal_image else None,
            "task_goal_sha256": sha256(args.goal_image) if args.goal_image else None,
            "source_event": contract["source_event"],
            "configured_to_reference_start_error_m": contract["reference_start_error_m"],
        },
        "artifacts": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "world_checkpoint": "pushbox/lewm/weights_epoch_146.pt",
            "world_checkpoint_path": str(contract["world_path"]),
            "world_checkpoint_sha256": contract["world_hash"],
            "transform_profile": str(args.transform_profile),
            "transform_profile_sha256": sha256(args.transform_profile),
            "camera_params": str(args.camera_params),
            "camera_params_sha256": sha256(args.camera_params),
        },
    }

    recorder: ProbeRecorder | None = None
    image_store = LatestImageStore()
    ros = RosImageSubscriber(image_store)
    ui = RealRobotUI(HeldInput(config.magnitudes))
    arm: Any = None
    current_model = contract["reference"].copy()
    last_valid_pose: np.ndarray | None = None
    faulted = False
    status = "fault"
    error_text: str | None = None
    permit_recovery = False
    ros_started = False
    ui_started = False
    home_positions = np.array(
        [0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )

    def check_pose(target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal last_valid_pose
        try:
            pose = validate_measured_pose(
                arm.pose(),
                target_xy=target_xy,
                fixed_z=args.fixed_z,
                orientation=FIXED_ORIENTATION,
                xy_tolerance_m=args.xy_tracking_tolerance,
                z_tolerance_m=args.z_tolerance,
                orientation_tolerance_rad=args.orientation_tolerance,
                bounds=geometry.bounds,
                require_tracking=True,
            )
        except Exception as exc:
            raise SafetyFault(f"pose verification failed: {exc}") from exc
        velocity = _validate_velocity(arm.velocities(), args.max_linear_speed)
        last_valid_pose = pose.copy()
        return pose, velocity

    def poll_running() -> None:
        events = ui.poll()
        if events.quit_requested:
            raise OperatorAbort("operator pressed q")
        if events.focus_lost:
            raise OperatorAbort("preview window lost focus")
        if events.pause_requested:
            raise OperatorAbort("operator pressed SPACE")
        if events.autonomous_toggle:
            raise OperatorAbort("operator pressed p while running")
        camera_error = ros.take_error()
        if camera_error is not None:
            raise SafetyFault(f"camera subscriber error: {camera_error}")

    def wait_until(deadline: float, step: int) -> None:
        next_render = 0.0
        while True:
            poll_running()
            now = time.monotonic()
            if now >= deadline:
                break
            if now >= next_render:
                ui.render(
                    current_model,
                    contract["goal"],
                    f"RUNNING {step:03d}/{geometry.steps:03d}  SPACE/q=abort",
                )
                next_render = now + 1.0 / 30.0
            time.sleep(min(UI_SLEEP_MAX_S, max(0.0, deadline - now)))
        lateness = time.monotonic() - deadline
        if lateness > SCHEDULE_TOLERANCE_S:
            raise SafetyFault(
                f"5 Hz deadline missed by {lateness * 1000:.1f} ms "
                f"(limit {SCHEDULE_TOLERANCE_S * 1000:.0f} ms)"
            )

    def wait_for_ack(message: str, *, allow_recovery: bool) -> bool:
        """Wait for an intentional q after completion/abort/fault."""
        while True:
            events = ui.poll()
            ui.render(current_model, contract["goal"], message)
            if events.quit_requested:
                return allow_recovery
            time.sleep(1.0 / 30.0)

    try:
        recorder = ProbeRecorder(
            args.output_dir,
            metadata,
            contract["reference"],
            args.reference_image,
            contract["goal"],
            args.goal_image,
        )
        print(f"Run directory: {recorder.path}")
        recorder.event("run_started")

        if not args.skip_camera_check:
            report = measure_stream_health(
                duration_s=args.camera_check_seconds, max_age_s=HEALTH_MAX_AGE_S
            )
            print(report.summary())
            recorder.event("camera_health", healthy=report.healthy, summary=report.summary())
            if not report.healthy:
                raise SafetyFault("camera stream failed its health gate; arm was not connected")

        ros.start()
        ros_started = True
        ui.start()
        ui_started = True

        camera_deadline = time.monotonic() + 3.0
        while True:
            camera_error = ros.take_error()
            if camera_error is not None:
                raise SafetyFault(f"camera subscriber error before staging: {camera_error}")
            snapshot = image_store.snapshot()
            if snapshot is not None and LatestImageStore.accept(
                snapshot, None, time.monotonic_ns(), HEALTH_MAX_AGE_S
            ):
                current_model = config.transform_profile.apply(snapshot.rgb)
                break
            if time.monotonic() >= camera_deadline:
                raise SafetyFault("timed out waiting for a fresh live-camera frame")
            time.sleep(0.01)

        print("Moving the arm to the configured start pose; keep the physical E-stop in hand.")
        arm = FollowerArmAdapter.connect(config)
        recorder.event("arm_connected", follower_ip=args.follower_ip)
        arm.home(home_positions)
        arm.send_cartesian(
            np.r_[geometry.start_xy, args.safe_z, FIXED_ORIENTATION],
            goal_time=2.0,
            blocking=True,
        )
        arm.send_cartesian(
            np.r_[geometry.start_xy, args.fixed_z, FIXED_ORIENTATION],
            goal_time=2.0,
            blocking=True,
        )
        startup_pose, startup_velocity = check_pose(geometry.start_xy)
        recorder.event(
            "startup_pose_verified", pose=startup_pose, velocity=startup_velocity
        )

        print("Arm is holding at the reference XY. Arrange the scene to match the SOURCE pane.")
        print("Verify the entire +X path is clear, then press p once to record; q cancels.")
        while True:
            camera_error = ros.take_error()
            if camera_error is not None:
                raise SafetyFault(f"camera subscriber error before arming: {camera_error}")
            snapshot = image_store.snapshot()
            fresh = snapshot is not None and LatestImageStore.accept(
                snapshot, None, time.monotonic_ns(), HEALTH_MAX_AGE_S
            )
            if fresh:
                assert snapshot is not None
                current_model = config.transform_profile.apply(snapshot.rgb)
            events = ui.poll()
            ui.render(
                current_model,
                contract["reference"],
                "PAUSED: right=SOURCE  p=record  q=cancel",
            )
            if events.quit_requested:
                status = "cancelled"
                permit_recovery = True
                recorder.event("cancelled_before_recording")
                break
            if events.autonomous_toggle:
                if not ui.focused:
                    print("Start ignored: focus the preview window first.")
                elif not fresh:
                    print("Start ignored: the camera frame is stale.")
                else:
                    break
            time.sleep(1.0 / 30.0)

        if status != "cancelled":
            recorder.event("recording_armed")
            previous_sequence: int | None = None
            previous_command_ns: int | None = None

            def capture_state(step: int, target_xy: np.ndarray) -> None:
                nonlocal previous_sequence, current_model
                camera_error = ros.take_error()
                if camera_error is not None:
                    raise SafetyFault(f"camera subscriber error: {camera_error}")
                capture_ns = time.monotonic_ns()
                snapshot = image_store.snapshot()
                if not LatestImageStore.accept(
                    snapshot, previous_sequence, capture_ns, HEALTH_MAX_AGE_S
                ):
                    raise SafetyFault("camera frame is missing, stale, or not new")
                assert snapshot is not None
                if (
                    previous_command_ns is not None
                    and snapshot.receipt_monotonic_ns <= previous_command_ns
                ):
                    raise SafetyFault("state frame was received before its preceding action")
                pose, velocity = check_pose(target_xy)
                model = config.transform_profile.apply(snapshot.rgb)
                if model.shape != MODEL_SHAPE or model.dtype != np.uint8:
                    raise SafetyFault("camera transform produced an invalid model frame")
                current_model = model
                previous_sequence = snapshot.sequence
                age_s = (capture_ns - snapshot.receipt_monotonic_ns) / 1e9
                command_to_observation_s = (
                    None
                    if previous_command_ns is None
                    else (snapshot.receipt_monotonic_ns - previous_command_ns) / 1e9
                )
                command_to_capture_call_s = (
                    None
                    if previous_command_ns is None
                    else (capture_ns - previous_command_ns) / 1e9
                )
                tracking_error = pose[:2] - target_xy
                recorder.state(
                    step,
                    model,
                    snapshot.rgb,
                    target_xy_m=target_xy,
                    measured_pose=pose,
                    measured_velocity=velocity,
                    tracking_error_xy_m=tracking_error,
                    tracking_error_norm_m=float(np.linalg.norm(tracking_error)),
                    camera_sequence=snapshot.sequence,
                    image_source_timestamp_ns=snapshot.source_timestamp_ns,
                    image_receipt_monotonic_ns=snapshot.receipt_monotonic_ns,
                    capture_monotonic_ns=capture_ns,
                    image_age_s=age_s,
                    preceding_command_monotonic_ns=previous_command_ns,
                    # Match real-eval/dataset semantics: observation time is
                    # the camera receipt, not the later Python capture call.
                    command_to_capture_s=command_to_observation_s,
                    command_to_snapshot_receipt_s=command_to_observation_s,
                    command_to_capture_call_s=command_to_capture_call_s,
                )
                print(
                    f"state {step:03d}/{geometry.steps:03d}: "
                    f"measured_xy={pose[:2].tolist()}, "
                    f"tracking={np.linalg.norm(tracking_error) * 1000:.2f} mm"
                )

            command_times_ns: list[int] = []

            def send_action(action_index: int, scheduled_ns: int | None) -> int:
                nonlocal previous_command_ns
                # Consume stop-priority input and camera faults at the last
                # possible point before every Cartesian command.
                poll_running()
                recorder.check_pending_writes(wait=False)
                candidate_ns = time.monotonic_ns()
                schedule_lateness_s = (
                    None
                    if scheduled_ns is None
                    else (candidate_ns - scheduled_ns) / 1e9
                )
                if (
                    schedule_lateness_s is not None
                    and schedule_lateness_s > SCHEDULE_TOLERANCE_S
                ):
                    raise SafetyFault(
                        f"command {action_index} missed its absolute 5 Hz schedule by "
                        f"{schedule_lateness_s * 1000:.1f} ms"
                    )
                if previous_command_ns is not None:
                    interval_s = (candidate_ns - previous_command_ns) / 1e9
                    if abs(interval_s - args.tick_seconds) > SCHEDULE_TOLERANCE_S:
                        raise SafetyFault(
                            f"command interval {interval_s:.4f} s left the "
                            f"{args.tick_seconds:.2f} +/- {SCHEDULE_TOLERANCE_S:.2f} s window"
                        )
                else:
                    interval_s = None
                target = geometry.targets_xy[action_index]
                command_ns = candidate_ns
                arm.send_cartesian(
                    np.r_[target, args.fixed_z, FIXED_ORIENTATION],
                    goal_time=args.tick_seconds,
                    blocking=False,
                )
                recorder.event(
                    "action_commanded",
                    action_index=action_index,
                    requested_action_xy_m=geometry.action_xy,
                    absolute_target_xy_m=target,
                    command_monotonic_ns=command_ns,
                    command_return_monotonic_ns=time.monotonic_ns(),
                    scheduled_command_monotonic_ns=scheduled_ns,
                    schedule_lateness_s=schedule_lateness_s,
                    previous_command_interval_s=interval_s,
                    goal_time_s=args.tick_seconds,
                )
                previous_command_ns = command_ns
                command_times_ns.append(command_ns)
                return command_ns

            # Establish both model and raw state 0 before authorizing action 1.
            capture_state(0, geometry.targets_xy[0])
            recorder.check_pending_writes(wait=True)
            first_command_ns = send_action(1, scheduled_ns=None)
            tick_ns = round(args.tick_seconds * 1e9)

            # Absolute deadlines prevent capture/logging overhead from
            # accumulating into a slower and slower control rate.  At each
            # deadline, capture state k and (except at the final state) send
            # action k+1 against the same fixed schedule epoch.
            for state_step in range(1, geometry.steps + 1):
                scheduled_ns = first_command_ns + state_step * tick_ns
                wait_until(scheduled_ns / 1e9, state_step)
                capture_state(state_step, geometry.targets_xy[state_step])
                if state_step < geometry.steps:
                    send_action(state_step + 1, scheduled_ns=scheduled_ns)

            status = "completed"
            permit_recovery = True
            intervals = np.diff(np.asarray(command_times_ns, dtype=np.int64)) / 1e9
            recorder.event(
                "recording_completed",
                states=geometry.steps + 1,
                command_count=len(command_times_ns),
                command_interval_mean_s=(float(intervals.mean()) if intervals.size else None),
                command_interval_min_s=(float(intervals.min()) if intervals.size else None),
                command_interval_max_s=(float(intervals.max()) if intervals.size else None),
                first_to_final_state_s=(time.monotonic_ns() - first_command_ns) / 1e9,
            )
            # The recorded sequence is complete; this hold cannot alter its data.
            if last_valid_pose is not None:
                arm.send_cartesian(last_valid_pose, goal_time=args.tick_seconds, blocking=False)
                recorder.event("final_pose_held", pose=last_valid_pose)
            permit_recovery = wait_for_ack(
                "COMPLETED/HOLDING: q = lift, home, and analyze", allow_recovery=True
            )

    except OperatorAbort as exc:
        status = "aborted"
        error_text = str(exc)
        print(f"ABORTED: {exc}")
        if recorder is not None:
            recorder.event("operator_abort", reason=str(exc))
        if arm is not None:
            try:
                measured = np.asarray(arm.pose(), dtype=np.float64).reshape(-1)[:6]
                if measured.shape != (6,) or not np.isfinite(measured).all():
                    raise ValueError("cannot hold an invalid measured pose")
                measured = validate_measured_pose(
                    measured,
                    target_xy=measured[:2],
                    fixed_z=args.fixed_z,
                    orientation=FIXED_ORIENTATION,
                    xy_tolerance_m=args.xy_tracking_tolerance,
                    z_tolerance_m=args.z_tolerance,
                    orientation_tolerance_rad=args.orientation_tolerance,
                    bounds=geometry.bounds,
                    require_tracking=False,
                )
                arm.send_cartesian(measured, goal_time=args.tick_seconds, blocking=False)
                last_valid_pose = measured.copy()
                if recorder is not None:
                    recorder.event("abort_pose_held", pose=measured)
                if ui_started:
                    permit_recovery = wait_for_ack(
                        "ABORTED/HOLDING: q again = safe retreat", allow_recovery=True
                    )
            except Exception as hold_exc:
                faulted = True
                status = "fault"
                permit_recovery = False
                error_text += f"; hold failed: {hold_exc}"
                if recorder is not None:
                    recorder.event("abort_hold_fault", reason=str(hold_exc))
    except Exception as exc:
        faulted = True
        status = "fault"
        error_text = f"{type(exc).__name__}: {exc}"
        print(f"FAULT (no automatic recovery): {error_text}")
        if recorder is not None:
            recorder.event("fault", reason=error_text)
        if ui_started:
            try:
                wait_for_ack("FAULT: E-STOP/inspect, then q (NO recovery)", allow_recovery=False)
            except Exception:
                pass
    except KeyboardInterrupt:
        status = "aborted"
        error_text = "KeyboardInterrupt"
        print("Interrupted; no automatic recovery will be attempted.")
        permit_recovery = False
    finally:
        if permit_recovery and not faulted and arm is not None and last_valid_pose is not None:
            try:
                current = np.asarray(arm.pose(), dtype=np.float64).reshape(-1)[:6]
                current = validate_measured_pose(
                    current,
                    target_xy=last_valid_pose[:2],
                    fixed_z=args.fixed_z,
                    orientation=FIXED_ORIENTATION,
                    xy_tolerance_m=args.xy_tracking_tolerance,
                    z_tolerance_m=args.z_tolerance,
                    orientation_tolerance_rad=args.orientation_tolerance,
                    bounds=geometry.bounds,
                    require_tracking=True,
                )
                recovery_velocity = _validate_velocity(
                    arm.velocities(), args.max_linear_speed
                )
                retreat_xy = current[:2].copy()
                recorder.event("safe_retreat_started", xy=retreat_xy) if recorder else None
                recorder.event(
                    "pre_retreat_pose_verified",
                    pose=current,
                    velocity=recovery_velocity,
                ) if recorder else None
                arm.send_cartesian(
                    np.r_[retreat_xy, args.safe_z, FIXED_ORIENTATION],
                    goal_time=2.0,
                    blocking=True,
                )
                move_arm_home_then_zero(arm, home_positions)
                recorder.event("safe_retreat_completed") if recorder else None
            except Exception as recovery_exc:
                faulted = True
                status = "fault"
                permit_recovery = False
                recovery_text = f"safe retreat failed: {recovery_exc}"
                error_text = f"{error_text}; {recovery_text}" if error_text else recovery_text
                print(f"FAULT: {recovery_text}")
                recorder.event("recovery_fault", reason=recovery_text) if recorder else None
        elif faulted:
            print("Fault is latched. Use the physical E-stop and inspect the arm; no home command was sent.")

        if ros_started:
            ros.stop()
        if ui_started:
            ui.close()
        if recorder is not None:
            try:
                recorder.close(status=status, error=error_text)
            except Exception as artifact_exc:
                print(f"artifact write failed: {artifact_exc}")
                status = "fault"

    if status == "completed":
        print(f"Real rollout complete: {args.output_dir}")
        return 0
    if status == "cancelled":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
