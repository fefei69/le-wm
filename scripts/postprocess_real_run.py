#!/usr/bin/env python3
"""Render a real-robot trial, its planning objective, and pixel tracking QA.

The video places every raw planner observation beside the fixed trial goal.
The plot uses the planner's recorded cost field, not a training loss. Each
video frame and plot sample therefore refer to the same autonomous_step.

Pixel tracking is an additional, deliberately lightweight check.  The box is
the centroid of its large red patch; the visual EE proxy is the centroid of
the small red pusher tip, stabilized by temporal continuity.  Missing points
remain missing and never prevent the primary post-processing artifacts from
being produced.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")

import imageio.v2 as imageio
import imageio.v3 as iio
import matplotlib
import cv2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SHAPE = (224, 224, 3)
HEADER_HEIGHT = 36
BOX_TIERS = (  # (red minimum, red-vs-green/blue margin, minimum blob area)
    (100, 50, 60),
    (90, 35, 40),
)
EE_RED_MIN = 20
EE_RED_GREEN_MARGIN = 8
EE_RED_BLUE_MARGIN = 6
EE_MIN_BLOB_PX = 3
EE_MAX_BLOB_PX = 100
EE_MAX_BLOB_WIDTH = 15
EE_MAX_BLOB_HEIGHT = 25
EE_DARK_MAX = 70
EE_MIN_DARK_CONTEXT = 0.18
EE_MAX_STEP_PX = 12.0
EE_REINITIALIZE_AFTER_MISSES = 5


@dataclass(frozen=True)
class TrialStep:
    step: int
    elapsed_s: float
    observation_path: Path
    cost: float
    goal_distance: float
    solve_time_s: float


@dataclass(frozen=True)
class TrialData:
    run_path: Path
    trial_id: int
    goal_path: Path
    steps: tuple[TrialStep, ...]
    outcome: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrackingData:
    goal_box: np.ndarray
    goal_ee: np.ndarray
    box_centroids: np.ndarray
    ee_centroids: np.ndarray
    box_goal_errors: np.ndarray
    ee_goal_errors: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="real_robot_runs/RUN_ID directory",
    )
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: RUN/analysis/overview_trial_NNN)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="output video FPS (default: artifact tick_hz, otherwise 5)",
    )
    return parser.parse_args(argv)


def resolve_run_path(run: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a path or a bare ID from this repository's real_robot_runs."""
    expanded = run.expanduser()
    candidates = (
        [expanded]
        if expanded.is_absolute()
        else [
            Path.cwd() / expanded,
            repo_root / expanded,
            repo_root / "real_robot_runs" / expanded,
        ]
    )
    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_dir():
            return resolved
    attempted = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(f"run directory not found; tried: {attempted}")


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"event at {path}:{line_number} is not an object")
        events.append(event)
    return events


def resolve_run_artifact(run_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = run_path / path
    path = path.resolve()
    if not path.is_relative_to(run_path):
        raise ValueError(f"{label} escapes the run directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def finite_nonnegative(event: dict[str, Any], key: str, *, step: int) -> float:
    try:
        value = float(event[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"step {step} has no valid {key}") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"step {step} {key} must be finite and non-negative")
    return value


def load_trial(run: Path, trial_id: int) -> TrialData:
    run_path = resolve_run_path(run)
    if trial_id < 1:
        raise ValueError("--trial-id must be positive")
    metadata_path = run_path / "metadata.json"
    events_path = run_path / "events.jsonl"
    if not metadata_path.is_file() or not events_path.is_file():
        raise FileNotFoundError("--run must contain metadata.json and events.jsonl")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    events = read_events(events_path)

    starts = [
        event
        for event in events
        if event.get("type") == "trial_started"
        and int(event.get("trial_id", -1)) == trial_id
    ]
    if len(starts) != 1:
        raise ValueError(
            f"expected one trial_started event for trial {trial_id}, found {len(starts)}"
        )
    goal_path = resolve_run_artifact(
        run_path, starts[0].get("goal_frame"), label="goal_frame"
    )

    step_events = [
        event
        for event in events
        if event.get("type") == "autonomous_step"
        and int(event.get("trial_id", -1)) == trial_id
    ]
    if not step_events:
        raise ValueError(f"trial {trial_id} has no autonomous_step events")
    observed_steps = [int(event.get("step", -1)) for event in step_events]
    expected_steps = list(range(1, len(step_events) + 1))
    if observed_steps != expected_steps:
        raise ValueError(
            f"trial {trial_id} steps must be contiguous from 1, got {observed_steps}"
        )

    first_monotonic_ns = step_events[0].get("monotonic_ns")
    use_monotonic_time = isinstance(first_monotonic_ns, int)
    records: list[TrialStep] = []
    for index, event in enumerate(step_events):
        step = index + 1
        monotonic_ns = event.get("monotonic_ns")
        if use_monotonic_time:
            if not isinstance(monotonic_ns, int) or monotonic_ns < first_monotonic_ns:
                raise ValueError(f"step {step} has invalid monotonic_ns")
            elapsed_s = (monotonic_ns - first_monotonic_ns) / 1e9
        else:
            elapsed_s = float(index)
        records.append(
            TrialStep(
                step=step,
                elapsed_s=float(elapsed_s),
                observation_path=resolve_run_artifact(
                    run_path,
                    event.get("observation_frame"),
                    label=f"step {step} observation_frame",
                ),
                cost=finite_nonnegative(event, "cost", step=step),
                goal_distance=finite_nonnegative(
                    event, "goal_distance", step=step
                ),
                solve_time_s=finite_nonnegative(
                    event, "solve_time_s", step=step
                ),
            )
        )

    outcomes = [
        str(event.get("outcome"))
        for event in events
        if event.get("type") == "trial_outcome"
        and int(event.get("trial_id", -1)) == trial_id
    ]
    return TrialData(
        run_path=run_path,
        trial_id=trial_id,
        goal_path=goal_path,
        steps=tuple(records),
        outcome=outcomes[-1] if outcomes else None,
        metadata=metadata,
    )


def load_rgb(path: Path) -> np.ndarray:
    frame = np.asarray(iio.imread(path))
    if frame.shape != IMAGE_SHAPE or frame.dtype != np.uint8:
        raise ValueError(
            f"expected RGB uint8 {IMAGE_SHAPE}, got {frame.shape} {frame.dtype}: {path}"
        )
    return frame


def valid_point(point: np.ndarray) -> bool:
    return point.shape == (2,) and bool(np.isfinite(point).all())


def detect_box_centroid(image: np.ndarray) -> np.ndarray:
    """Return the large red box-patch centroid ``(u, v)`` or NaNs."""
    image_i16 = image.astype(np.int16, copy=False)
    r, g, b = image_i16[..., 0], image_i16[..., 1], image_i16[..., 2]
    for red_min, margin, minimum_area in BOX_TIERS:
        mask = (
            (r > red_min)
            & (r - g > margin)
            & (r - b > margin)
        ).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if count <= 1:
            continue
        component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        if stats[component, cv2.CC_STAT_AREA] < minimum_area:
            continue
        return centroids[component].astype(np.float32)
    return np.full(2, np.nan, dtype=np.float32)


def gripper_tip_candidates(image: np.ndarray) -> np.ndarray:
    """Return plausible red pusher-tip components as ``(u, v, area)`` rows."""
    image_i16 = image.astype(np.int16, copy=False)
    r, g, b = image_i16[..., 0], image_i16[..., 1], image_i16[..., 2]
    red = (
        (r >= EE_RED_MIN)
        & (r - g >= EE_RED_GREEN_MARGIN)
        & (r - b >= EE_RED_BLUE_MARGIN)
    ).astype(np.uint8)
    dark = image_i16.mean(axis=2) < EE_DARK_MAX
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        red, connectivity=8
    )
    candidates: list[tuple[float, float, float]] = []
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        u, v = centroids[component]
        if not (
            EE_MIN_BLOB_PX <= area <= EE_MAX_BLOB_PX
            and width <= EE_MAX_BLOB_WIDTH
            and height <= EE_MAX_BLOB_HEIGHT
            and 20 <= u <= 205
            and 48 <= v <= 130
        ):
            continue
        x0, x1 = max(0, x - 10), min(image.shape[1], x + width + 10)
        y0, y1 = max(0, y - 10), min(image.shape[0], y + height + 10)
        if dark[y0:y1, x0:x1].mean() < EE_MIN_DARK_CONTEXT:
            continue
        candidates.append((float(u), float(v), float(area)))
    return np.asarray(candidates, dtype=np.float32).reshape(-1, 3)


def detect_goal_gripper_tip(image: np.ndarray) -> np.ndarray:
    """Select the normally-lowest plausible pusher tip in a standalone goal."""
    candidates = gripper_tip_candidates(image)
    if not len(candidates):
        return np.full(2, np.nan, dtype=np.float32)
    return candidates[int(np.argmax(candidates[:, 1])), :2].copy()


def track_gripper_tip(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Track the visual pusher tip with a conservative continuity gate."""
    centroids = np.full((len(frames), 2), np.nan, dtype=np.float32)
    previous: np.ndarray | None = None
    consecutive_misses = 0
    for index, frame in enumerate(frames):
        candidates = gripper_tip_candidates(frame)
        selected: np.ndarray | None = None
        if len(candidates):
            if previous is None:
                selected = candidates[int(np.argmax(candidates[:, 1])), :2]
            else:
                distances = np.linalg.norm(candidates[:, :2] - previous, axis=1)
                closest = int(np.argmin(distances))
                if distances[closest] <= EE_MAX_STEP_PX:
                    selected = candidates[closest, :2]
        if selected is None:
            consecutive_misses += 1
            if consecutive_misses >= EE_REINITIALIZE_AFTER_MISSES:
                previous = None
            continue
        centroids[index] = selected
        previous = selected.copy()
        consecutive_misses = 0
    return centroids


def goal_errors(points: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Return Euclidean pixel errors, preserving missing measurements as NaN."""
    errors = np.full(len(points), np.nan, dtype=np.float32)
    if not valid_point(goal):
        return errors
    detected = np.isfinite(points).all(axis=1)
    errors[detected] = np.linalg.norm(points[detected] - goal, axis=1)
    return errors


def track_trial(trial: TrialData) -> TrackingData:
    """Compute optional image-space box and EE tracks for a trial."""
    goal = load_rgb(trial.goal_path)
    frames = [load_rgb(record.observation_path) for record in trial.steps]
    goal_box = detect_box_centroid(goal)
    goal_ee = detect_goal_gripper_tip(goal)
    box_centroids = np.asarray(
        [detect_box_centroid(frame) for frame in frames], dtype=np.float32
    )
    ee_centroids = track_gripper_tip(frames)
    return TrackingData(
        goal_box=goal_box,
        goal_ee=goal_ee,
        box_centroids=box_centroids,
        ee_centroids=ee_centroids,
        box_goal_errors=goal_errors(box_centroids, goal_box),
        ee_goal_errors=goal_errors(ee_centroids, goal_ee),
    )


def compose_video_frame(
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    step: int,
    cost: float,
) -> np.ndarray:
    observation = np.asarray(observation)
    goal = np.asarray(goal)
    if observation.shape != IMAGE_SHAPE or observation.dtype != np.uint8:
        raise ValueError("observation must be RGB uint8 224x224")
    if goal.shape != IMAGE_SHAPE or goal.dtype != np.uint8:
        raise ValueError("goal must be RGB uint8 224x224")

    canvas = np.zeros(
        (IMAGE_SHAPE[0] + HEADER_HEIGHT, IMAGE_SHAPE[1] * 2, 3),
        dtype=np.uint8,
    )
    canvas[HEADER_HEIGHT:, : IMAGE_SHAPE[1]] = observation
    canvas[HEADER_HEIGHT:, IMAGE_SHAPE[1] :] = goal
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((8, 11), f"RAW OBSERVATION  step {step:03d}", fill=(255, 255, 255))
    draw.text(
        (IMAGE_SHAPE[1] + 8, 11),
        f"GOAL  planner cost {cost:.5g}",
        fill=(255, 255, 255),
    )
    return np.asarray(image)


def default_fps(metadata: dict[str, Any]) -> float:
    manifest = metadata.get("artifact_manifest")
    if isinstance(manifest, dict):
        runtime = manifest.get("runtime")
        if isinstance(runtime, dict):
            try:
                tick_hz = float(runtime["tick_hz"])
            except (KeyError, TypeError, ValueError):
                tick_hz = 0.0
            if np.isfinite(tick_hz) and tick_hz > 0.0:
                return tick_hz
    return 5.0


def write_video(path: Path, trial: TrialData, fps: float) -> None:
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("--fps must be finite and positive")
    goal = load_rgb(trial.goal_path)
    with imageio.get_writer(
        path,
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for record in trial.steps:
            observation = load_rgb(record.observation_path)
            writer.append_data(
                compose_video_frame(
                    observation,
                    goal,
                    step=record.step,
                    cost=record.cost,
                )
            )


def annotate_tracking(
    frame: np.ndarray,
    *,
    box: np.ndarray,
    ee: np.ndarray,
) -> np.ndarray:
    """Overlay the box cross and EE circle used by the tracking check."""
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    if valid_point(box):
        u, v = (float(value) for value in box)
        draw.line((u - 5, v, u + 5, v), fill=(70, 255, 90), width=2)
        draw.line((u, v - 5, u, v + 5), fill=(70, 255, 90), width=2)
    if valid_point(ee):
        u, v = (float(value) for value in ee)
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(40, 220, 240), width=2)
    return np.asarray(image)


def format_error(value: float, label: str) -> str:
    return f"{label} {value:.1f}px" if np.isfinite(value) else f"{label} MISS"


def compose_tracking_frame(
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    step: int,
    observation_box: np.ndarray,
    observation_ee: np.ndarray,
    goal_box: np.ndarray,
    goal_ee: np.ndarray,
    box_error: float,
    ee_error: float,
) -> np.ndarray:
    observation_marked = annotate_tracking(
        observation, box=observation_box, ee=observation_ee
    )
    goal_marked = annotate_tracking(goal, box=goal_box, ee=goal_ee)
    canvas = np.zeros(
        (IMAGE_SHAPE[0] + HEADER_HEIGHT, IMAGE_SHAPE[1] * 2, 3), dtype=np.uint8
    )
    canvas[HEADER_HEIGHT:, : IMAGE_SHAPE[1]] = observation_marked
    canvas[HEADER_HEIGHT:, IMAGE_SHAPE[1] :] = goal_marked
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((8, 4), f"TRACKING CHECK  step {step:03d}", fill=(255, 255, 255))
    draw.text((8, 20), "box +   EE o", fill=(255, 255, 255))
    draw.text(
        (IMAGE_SHAPE[1] + 8, 4),
        format_error(box_error, "box"),
        fill=(255, 255, 255),
    )
    draw.text(
        (IMAGE_SHAPE[1] + 8, 20),
        format_error(ee_error, "EE"),
        fill=(255, 255, 255),
    )
    return np.asarray(image)


def write_tracking_video(
    path: Path,
    trial: TrialData,
    tracking: TrackingData,
    fps: float,
) -> None:
    goal = load_rgb(trial.goal_path)
    with imageio.get_writer(
        path,
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for index, record in enumerate(trial.steps):
            writer.append_data(
                compose_tracking_frame(
                    load_rgb(record.observation_path),
                    goal,
                    step=record.step,
                    observation_box=tracking.box_centroids[index],
                    observation_ee=tracking.ee_centroids[index],
                    goal_box=tracking.goal_box,
                    goal_ee=tracking.goal_ee,
                    box_error=float(tracking.box_goal_errors[index]),
                    ee_error=float(tracking.ee_goal_errors[index]),
                )
            )


def write_metrics(path: Path, trial: TrialData) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["step", "elapsed_s", "planner_cost", "goal_distance", "solve_time_s"]
        )
        for record in trial.steps:
            writer.writerow(
                [
                    record.step,
                    record.elapsed_s,
                    record.cost,
                    record.goal_distance,
                    record.solve_time_s,
                ]
            )


def write_tracking_metrics(
    path: Path, trial: TrialData, tracking: TrackingData
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "elapsed_s",
                "box_u_px",
                "box_v_px",
                "box_goal_error_px",
                "ee_u_px",
                "ee_v_px",
                "ee_goal_error_px",
            ]
        )
        for index, record in enumerate(trial.steps):
            writer.writerow(
                [
                    record.step,
                    record.elapsed_s,
                    *tracking.box_centroids[index],
                    tracking.box_goal_errors[index],
                    *tracking.ee_centroids[index],
                    tracking.ee_goal_errors[index],
                ]
            )


def write_loss_plot(path: Path, trial: TrialData) -> None:
    elapsed = np.asarray([record.elapsed_s for record in trial.steps])
    costs = np.asarray([record.cost for record in trial.steps])
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.plot(elapsed, costs, marker="o", markersize=3, linewidth=1.5)
    axis.set_xlabel("elapsed wall time from first planned action (s)")
    axis.set_ylabel("recorded planner cost")
    title = f"Planning objective over time — trial {trial.trial_id:03d}"
    if trial.outcome is not None:
        title += f" ({trial.outcome})"
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_tracking_plot(
    path: Path, trial: TrialData, tracking: TrackingData
) -> None:
    elapsed = np.asarray([record.elapsed_s for record in trial.steps])
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True)
    series = (
        ("Box", tracking.box_goal_errors, "#3d8b57"),
        ("EE / red pusher tip", tracking.ee_goal_errors, "#268b9c"),
    )
    for axis, (label, errors, color) in zip(axes, series):
        detected = np.isfinite(errors)
        if detected.any():
            axis.plot(
                elapsed[detected], errors[detected], marker="o",
                markersize=3, linewidth=1.4, color=color,
            )
        else:
            axis.text(
                0.5, 0.5, "no valid goal-relative detections",
                ha="center", va="center", transform=axis.transAxes,
            )
        axis.set_ylabel(f"{label} error (px)")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("elapsed wall time from first planned action (s)")
    title = f"Optional pixel tracking check — trial {trial.trial_id:03d}"
    if trial.outcome is not None:
        title += f" ({trial.outcome})"
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def optional_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def tracking_metric_summary(
    points: np.ndarray, goal: np.ndarray, errors: np.ndarray
) -> dict[str, Any]:
    detected = np.isfinite(points).all(axis=1)
    valid_errors = errors[np.isfinite(errors)]
    return {
        "goal_detected": valid_point(goal),
        "detection_rate": float(detected.mean()),
        "start_error_px": optional_float(float(errors[0])),
        "best_error_px": (
            optional_float(float(valid_errors.min())) if len(valid_errors) else None
        ),
        "final_error_px": optional_float(float(errors[-1])),
        "last_valid_error_px": (
            optional_float(float(valid_errors[-1])) if len(valid_errors) else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    trial = load_trial(args.run, args.trial_id)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else trial.run_path
        / "analysis"
        / f"overview_trial_{trial.trial_id:03d}"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.mkdir(parents=True, exist_ok=False)
    fps = default_fps(trial.metadata) if args.fps is None else float(args.fps)

    video_path = output_path / "raw_vs_goal.mp4"
    plot_path = output_path / "planning_loss.png"
    metrics_path = output_path / "planning_metrics.csv"
    tracking_video_path = output_path / "tracking_check.mp4"
    tracking_plot_path = output_path / "tracking_errors.png"
    tracking_metrics_path = output_path / "tracking_metrics.csv"
    write_video(video_path, trial, fps)
    write_loss_plot(plot_path, trial)
    write_metrics(metrics_path, trial)
    tracking = track_trial(trial)
    write_tracking_video(tracking_video_path, trial, tracking, fps)
    write_tracking_plot(tracking_plot_path, trial, tracking)
    write_tracking_metrics(tracking_metrics_path, trial, tracking)
    box_tracking_summary = tracking_metric_summary(
        tracking.box_centroids, tracking.goal_box, tracking.box_goal_errors
    )
    ee_tracking_summary = tracking_metric_summary(
        tracking.ee_centroids, tracking.goal_ee, tracking.ee_goal_errors
    )
    summary = {
        "schema_version": 2,
        "source_run": str(trial.run_path),
        "trial_id": trial.trial_id,
        "outcome": trial.outcome,
        "step_count": len(trial.steps),
        "video_fps": fps,
        "semantics": {
            "raw_frame": "observation used to produce this autonomous action",
            "goal_frame": "fixed goal recorded by trial_started",
            "planning_loss": "autonomous_step.cost, the planner objective, not training loss",
            "elapsed_s": "autonomous_step monotonic time relative to the first step",
            "box_tracking": "centroid of the large red box patch in each saved image",
            "ee_tracking": "centroid of the small red pusher tip with temporal continuity; optional QA only",
            "tracking_endpoint": "last saved pre-action observation, not a guaranteed post-final-action frame",
        },
        "tracking": {
            "units": "image_pixels",
            "box": box_tracking_summary,
            "ee": ee_tracking_summary,
        },
        "artifacts": {
            "video": video_path.name,
            "planning_loss_plot": plot_path.name,
            "planning_metrics_csv": metrics_path.name,
            "tracking_check_video": tracking_video_path.name,
            "tracking_errors_plot": tracking_plot_path.name,
            "tracking_metrics_csv": tracking_metrics_path.name,
        },
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"processed {len(trial.steps)} steps into {output_path}")
    print(f"video: {video_path}")
    print(f"planning loss: {plot_path}")
    print(
        "tracking check: "
        f"box detection {box_tracking_summary['detection_rate']:.1%}, "
        f"last error {box_tracking_summary['last_valid_error_px']}; "
        f"EE detection {ee_tracking_summary['detection_rate']:.1%}, "
        f"last error {ee_tracking_summary['last_valid_error_px']}"
    )
    print(f"tracking video: {tracking_video_path}")
    print(f"tracking plot: {tracking_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
