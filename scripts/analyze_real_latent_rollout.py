#!/usr/bin/env python3
"""Analyze a recorded constant-XY real-robot rollout in LeWM latent space.

The recorder contract is one RGB state before motion followed by one RGB state
after every action::

    frame_000.png -> action 1 -> ... -> action N -> frame_NNN.png

Every real frame is encoded independently.  Distance to the recorded task
goal is the headline metric when that image is available; distance to the
final real frame is also reported as a trajectory-consistency diagnostic.  A
predictor rollout from frame 000 uses the constant action recorded in metadata,
allowing real and predicted trajectories to be compared with the same
projected-latent MSE used by :mod:`real_robot_planner`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")

import imageio.v2 as imageio
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import stable_worldmodel as swm
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_robot_planner import CEMConfig, PushBoxPlanner


DEFAULT_MANIFEST = REPO_ROOT / "config/real_robot_eval.json"
DEFAULT_WORLD_CHECKPOINT = "pushbox/lewm/weights_epoch_146.pt"
FRAME_PATTERN = re.compile(r"frame_(\d{3})\.png")
MAX_ACTION_NORM_M = 0.010


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        "--real-run",
        dest="run",
        type=Path,
        required=True,
        help="recorder run containing metadata.json, events.jsonl, and frames/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: RUN/analysis)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--world-checkpoint", default=DEFAULT_WORLD_CHECKPOINT)
    parser.add_argument(
        "--task-goal",
        "--goal-image",
        dest="task_goal",
        type=Path,
        help="optional task-goal RGB image; overrides a goal path in metadata/events",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="annotated-video FPS (default: recorder metadata, otherwise 5)",
    )
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device; 'auto' selects CUDA when available",
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_checkpoint_path(spec: str) -> Path:
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    candidate = REPO_ROOT / "stable-wm/checkpoints" / spec
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"world-model checkpoint not found: {spec}")


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} line {line_number}") from error
        if not isinstance(event, dict):
            raise ValueError(f"event at {path}:{line_number} is not an object")
        events.append(event)
    if not events:
        raise ValueError(f"no events found in {path}")
    return events


def load_contiguous_frames(frames_path: Path) -> tuple[list[Path], list[np.ndarray]]:
    indexed: list[tuple[int, Path]] = []
    for path in frames_path.glob("frame_*.png"):
        match = FRAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected frame filename (expected frame_NNN.png): {path}")
        indexed.append((int(match.group(1)), path))
    indexed.sort()
    if len(indexed) < 2:
        raise ValueError(f"expected at least frame_000.png and frame_001.png in {frames_path}")
    indices = [index for index, _ in indexed]
    expected = list(range(len(indexed)))
    if indices != expected:
        raise ValueError(f"frames are not contiguous from zero: got {indices}, expected {expected}")

    paths = [path for _, path in indexed]
    frames: list[np.ndarray] = []
    for path in paths:
        frame = iio.imread(path)
        if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"{path} must be RGB uint8 224x224, got {frame.shape} {frame.dtype}"
            )
        frames.append(frame)
    return paths, frames


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def finite_xy(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (2,) or not np.isfinite(array).all():
        return None
    return array.astype(np.float32)


def metadata_constant_action(
    metadata: dict[str, Any], events: list[dict[str, Any]]
) -> tuple[np.ndarray, str]:
    vector_keys = (
        "constant_action_m",
        "raw_action_m",
        "action_xy_m",
        "action_m",
        "commanded_action_m",
    )
    for mapping in walk_dicts(metadata):
        for key in vector_keys:
            if key in mapping:
                action = finite_xy(mapping[key])
                if action is not None:
                    return action, f"metadata:{key}"
        for x_key, y_key in (
            ("action_x_m", "action_y_m"),
            ("delta_x_m", "delta_y_m"),
            ("dx_m", "dy_m"),
        ):
            if x_key in mapping and y_key in mapping:
                action = finite_xy([mapping[x_key], mapping[y_key]])
                if action is not None:
                    return action, f"metadata:{x_key},{y_key}"

    event_actions: list[np.ndarray] = []
    for event in events:
        for key in ("accepted_action", "executed_action", "commanded_action", "action"):
            if key in event:
                action = finite_xy(event[key])
                if action is not None:
                    event_actions.append(action)
                    break
    if not event_actions:
        raise ValueError("constant action was not found in metadata or events")
    reference = event_actions[0]
    if any(not np.allclose(action, reference, rtol=0.0, atol=1e-8) for action in event_actions):
        raise ValueError("event actions are not constant and metadata has no constant action")
    return reference, "events"


def metadata_scalar(metadata: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for mapping in walk_dicts(metadata):
        for key in keys:
            if key not in mapping:
                continue
            try:
                value = float(mapping[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None


def path_from_value(value: Any) -> str | None:
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, dict):
        for key in ("path", "image", "frame", "file"):
            if key in value and isinstance(value[key], (str, os.PathLike)):
                return os.fspath(value[key])
    return None


def metadata_task_goal(metadata: dict[str, Any], events: list[dict[str, Any]]) -> str | None:
    keys = (
        "task_goal_copy",
        "task_goal_image",
        "task_goal_frame",
        "goal_image",
        "goal_frame",
    )
    for mapping in walk_dicts(metadata):
        for key in keys:
            if key in mapping:
                path = path_from_value(mapping[key])
                if path is not None:
                    return path
    for event in events:
        for key in ("task_goal_image", "task_goal_frame"):
            if key in event:
                path = path_from_value(event[key])
                if path is not None:
                    return path
        event_type = str(event.get("type", "")).lower()
        if "goal" in event_type:
            for key in ("goal_image", "goal_frame", "frame", "image"):
                if key in event:
                    path = path_from_value(event[key])
                    if path is not None:
                        return path
    return None


def resolve_recorded_path(value: str | os.PathLike[str], run_path: Path) -> Path:
    candidate = Path(value).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [run_path / candidate, REPO_ROOT / candidate]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"recorded path does not exist: {value}")


def event_frame_index(event: dict[str, Any]) -> int | None:
    for key in ("frame_index", "state_step", "state_index"):
        value = event.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    for key in ("frame", "frame_path", "image", "image_path", "observation_frame"):
        value = event.get(key)
        if not isinstance(value, str):
            continue
        match = FRAME_PATTERN.search(Path(value).name)
        if match is not None:
            return int(match.group(1))
    return None


def finite_pose(value: Any) -> list[float] | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 2 or not np.isfinite(array[:2]).all():
        return None
    return array.tolist()


def frame_event_data(events: list[dict[str, Any]], frame_count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [{} for _ in range(frame_count)]
    for event in events:
        index = event_frame_index(event)
        pose_key: str | None = None
        if index is None and isinstance(event.get("step"), int):
            step = int(event["step"])
            if "measured_pose_after" in event or "capture_pose" in event:
                index = step
            elif "measured_pose_before" in event:
                index = max(0, step - 1)
        if index is None or not 0 <= index < frame_count:
            continue
        record = records[index]
        for key in (
            "capture_pose",
            "measured_pose_after",
            "measured_pose",
            "pose",
            "measured_pose_before",
        ):
            if key in event:
                pose = finite_pose(event[key])
                if pose is not None:
                    record["pose"] = pose
                    pose_key = key
                    break
        if pose_key is not None:
            record["pose_source"] = pose_key
        for source, target in (
            ("target_xy_m", "target_xy_m"),
            ("tracking_error_xy_m", "tracking_error_xy_m"),
            ("tracking_error_norm_m", "tracking_error_norm_m"),
            ("measured_velocity", "measured_velocity"),
            ("image_age_s", "image_age_s"),
            ("command_to_capture_s", "command_to_capture_s"),
        ):
            if source in event:
                record[target] = event[source]
        for source, target in (
            ("capture_monotonic_ns", "capture_monotonic_ns"),
            ("camera_monotonic_ns", "capture_monotonic_ns"),
            ("monotonic_ns", "capture_monotonic_ns"),
            ("wall_time", "wall_time"),
            ("camera_sequence", "camera_sequence"),
        ):
            if source in event and target not in record:
                record[target] = event[source]
    return records


def encode_frames(
    planner: PushBoxPlanner, frames: list[np.ndarray], batch_size: int
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            pixels = torch.cat(
                [planner._prepare_image(frame) for frame in frames[start : start + batch_size]],
                dim=0,
            )
            embedding = planner.model.encode({"pixels": pixels})["emb"][:, -1]
            if not torch.isfinite(embedding).all():
                raise RuntimeError("real-frame encoder produced a non-finite embedding")
            embeddings.append(embedding)
    return torch.cat(embeddings, dim=0)


def predict_constant_action(
    planner: PushBoxPlanner, initial_frame: np.ndarray, action: np.ndarray, steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = planner._prepare_image(initial_frame).unsqueeze(1)
    raw_actions = torch.as_tensor(
        action, dtype=torch.float32, device=planner.device
    ).reshape(1, 1, 1, 2)
    raw_actions = raw_actions.expand(1, 1, steps, 2).clone()
    normalized = (raw_actions - planner.action_mean.unsqueeze(1)) / (
        planner.action_std.unsqueeze(1)
    )
    with torch.inference_mode():
        rollout = planner.model.rollout(
            {"pixels": pixels}, normalized, history_size=planner.history_size
        )
    embeddings = rollout["predicted_emb"][0, 0]
    if embeddings.shape != (steps + 1, real_latent_dim(planner)):
        raise RuntimeError(f"unexpected predicted latent shape: {tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("predictor produced a non-finite embedding")
    return embeddings, normalized[0, 0]


def real_latent_dim(planner: PushBoxPlanner) -> int:
    return int(planner.model.predictor.pos_embedding.shape[-1])


def mse_to_goal(latents: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.square(latents.astype(np.float64) - goal.astype(np.float64)).mean(axis=1)


def cosine_distance(latents: np.ndarray, goal: np.ndarray) -> np.ndarray:
    latents = latents.astype(np.float64)
    goal = goal.astype(np.float64)
    denominator = np.linalg.norm(latents, axis=1) * np.linalg.norm(goal)
    similarity = (latents @ goal) / np.maximum(denominator, np.finfo(np.float64).eps)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def improvement(distance: np.ndarray) -> np.ndarray:
    result = np.full(len(distance), np.nan, dtype=np.float64)
    result[1:] = distance[:-1] - distance[1:]
    return result


def normalized_progress(distance: np.ndarray) -> np.ndarray:
    if distance[0] <= np.finfo(np.float64).eps:
        return np.zeros_like(distance, dtype=np.float64)
    return (distance[0] - distance) / distance[0]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else None


def distance_summary(distance: np.ndarray) -> dict[str, Any]:
    changes = distance[1:] - distance[:-1]
    tolerance = max(1e-9, float(distance[0]) * 1e-6)
    improving = changes < -tolerance
    worsening = changes > tolerance
    flat = ~(improving | worsening)
    longest_worsening = 0
    current_streak = 0
    for value in worsening:
        current_streak = current_streak + 1 if value else 0
        longest_worsening = max(longest_worsening, current_streak)
    steps = np.arange(len(distance), dtype=np.float64)
    return {
        "start_mse": float(distance[0]),
        "final_mse": float(distance[-1]),
        "minimum_mse": float(distance.min()),
        "maximum_mse": float(distance.max()),
        "improving_steps": int(improving.sum()),
        "worsening_steps": int(worsening.sum()),
        "flat_steps": int(flat.sum()),
        "improving_fraction": float(improving.mean()),
        "spearman_step_vs_negative_mse": correlation(
            average_ranks(steps), average_ranks(-distance)
        ),
        "maximum_backtrack_mse": float(np.maximum(changes, 0.0).max(initial=0.0)),
        "total_backtrack_mse": float(np.maximum(changes, 0.0).sum()),
        "longest_worsening_streak": longest_worsening,
        "classification_tolerance_mse": tolerance,
    }


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def save_metrics_csv(
    path: Path,
    frame_paths: list[Path],
    action: np.ndarray,
    frame_records: list[dict[str, Any]],
    real: np.ndarray,
    predicted: np.ndarray,
    final_goal: np.ndarray,
    task_goal: np.ndarray | None,
) -> dict[str, np.ndarray]:
    latent_dim = real.shape[1]
    real_final = mse_to_goal(real, final_goal)
    predicted_final = mse_to_goal(predicted, final_goal)
    prediction_error = np.square(predicted.astype(np.float64) - real.astype(np.float64)).mean(axis=1)
    metrics: dict[str, np.ndarray] = {
        "real_to_final_goal_mse": real_final,
        "real_to_final_goal_rmse": np.sqrt(real_final),
        "real_to_final_goal_l2": np.sqrt(real_final * latent_dim),
        "real_to_final_goal_cosine_distance": cosine_distance(real, final_goal),
        "real_final_goal_improvement_mse": improvement(real_final),
        "real_final_goal_progress": normalized_progress(real_final),
        "predicted_to_final_goal_mse": predicted_final,
        "predicted_to_final_goal_rmse": np.sqrt(predicted_final),
        "predicted_final_goal_improvement_mse": improvement(predicted_final),
        "predicted_final_goal_progress": normalized_progress(predicted_final),
        "predicted_to_real_mse": prediction_error,
        "real_latent_norm": np.linalg.norm(real.astype(np.float64), axis=1),
        "predicted_latent_norm": np.linalg.norm(predicted.astype(np.float64), axis=1),
    }
    if task_goal is not None:
        real_task = mse_to_goal(real, task_goal)
        predicted_task = mse_to_goal(predicted, task_goal)
        metrics.update(
            {
                "real_to_task_goal_mse": real_task,
                "real_task_goal_improvement_mse": improvement(real_task),
                "real_task_goal_progress": normalized_progress(real_task),
                "predicted_to_task_goal_mse": predicted_task,
                "predicted_task_goal_improvement_mse": improvement(predicted_task),
                "predicted_task_goal_progress": normalized_progress(predicted_task),
            }
        )

    fixed_fields = [
        "step",
        "frame",
        "nominal_x_m",
        "nominal_y_m",
        "measured_x_m",
        "measured_y_m",
        "measured_dx_m",
        "measured_dy_m",
        "target_x_m",
        "target_y_m",
        "tracking_error_x_m",
        "tracking_error_y_m",
        "tracking_error_norm_m",
        "image_age_s",
        "command_to_capture_s",
        "capture_monotonic_ns",
        "wall_time",
        "camera_sequence",
    ]
    fieldnames = [*fixed_fields, *metrics.keys()]
    initial_pose = frame_records[0].get("pose")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for step, frame_path in enumerate(frame_paths):
            record = frame_records[step]
            pose = record.get("pose")
            target = finite_xy(record.get("target_xy_m"))
            tracking_error = finite_xy(record.get("tracking_error_xy_m"))
            row: dict[str, Any] = {
                "step": step,
                "frame": str(frame_path),
                "nominal_x_m": float(action[0]) * step,
                "nominal_y_m": float(action[1]) * step,
                "measured_x_m": pose[0] if pose is not None else None,
                "measured_y_m": pose[1] if pose is not None else None,
                "measured_dx_m": (
                    pose[0] - initial_pose[0]
                    if pose is not None and initial_pose is not None
                    else None
                ),
                "measured_dy_m": (
                    pose[1] - initial_pose[1]
                    if pose is not None and initial_pose is not None
                    else None
                ),
                "target_x_m": target[0] if target is not None else None,
                "target_y_m": target[1] if target is not None else None,
                "tracking_error_x_m": (
                    tracking_error[0] if tracking_error is not None else None
                ),
                "tracking_error_y_m": (
                    tracking_error[1] if tracking_error is not None else None
                ),
                "tracking_error_norm_m": record.get("tracking_error_norm_m"),
                "image_age_s": record.get("image_age_s"),
                "command_to_capture_s": record.get("command_to_capture_s"),
                "capture_monotonic_ns": record.get("capture_monotonic_ns"),
                "wall_time": record.get("wall_time"),
                "camera_sequence": record.get("camera_sequence"),
            }
            row.update({key: float(values[step]) for key, values in metrics.items()})
            writer.writerow({key: csv_value(value) for key, value in row.items()})
    return metrics


def save_plot(
    path: Path,
    metrics: dict[str, np.ndarray],
    action: np.ndarray,
    frame_records: list[dict[str, Any]],
) -> None:
    steps = np.arange(len(metrics["real_to_final_goal_mse"]))
    has_task_goal = "real_to_task_goal_mse" in metrics
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    if has_task_goal:
        axes[0].plot(steps, metrics["real_to_task_goal_mse"], label="real encoded")
        axes[0].plot(
            steps, metrics["predicted_to_task_goal_mse"], label="predicted"
        )
        axes[0].set_ylabel("MSE to task goal")
        axes[0].legend()
        axes[1].plot(steps, metrics["real_to_final_goal_mse"], label="real encoded")
        axes[1].plot(
            steps,
            metrics["predicted_to_final_goal_mse"],
            label="predicted from frame 000",
        )
        axes[1].set_ylabel("MSE to final-real state")
        axes[1].legend()
    else:
        axes[0].plot(steps, metrics["real_to_final_goal_mse"], label="real encoded")
        axes[0].plot(
            steps,
            metrics["predicted_to_final_goal_mse"],
            label="predicted from frame 000",
        )
        axes[0].set_ylabel("MSE to final-real state")
        axes[0].legend()
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].plot(
            steps,
            metrics["real_final_goal_improvement_mse"],
            label="positive = closer",
        )
        axes[1].set_ylabel("real per-step improvement")
        axes[1].legend()

    axes[2].plot(steps, metrics["predicted_to_real_mse"], color="tab:red")
    axes[2].set_ylabel("predicted-to-real MSE")
    axes[2].set_xlabel("state step (after N constant actions)")
    measured_x = np.asarray(
        [
            record.get("pose", [np.nan, np.nan])[0]
            if record.get("pose") is not None
            else np.nan
            for record in frame_records
        ],
        dtype=np.float64,
    )
    if np.isfinite(measured_x).sum() >= 2:
        twin = axes[2].twinx()
        twin.plot(steps, measured_x - measured_x[np.flatnonzero(np.isfinite(measured_x))[0]], color="0.5", alpha=0.55)
        twin.set_ylabel("measured X displacement (m)", color="0.4")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"Real constant action [{float(action[0]):+.4f}, {float(action[1]):+.4f}] m"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def draw_curve(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    real_distance: np.ndarray,
    predicted_distance: np.ndarray,
    current_step: int,
) -> None:
    left, top, right, bottom = bounds
    draw.rectangle(bounds, outline=(100, 100, 100))
    combined = np.concatenate([real_distance, predicted_distance])
    low = float(np.nanmin(combined))
    high = float(np.nanmax(combined))
    if high <= low:
        high = low + 1.0

    def points(values: np.ndarray) -> list[tuple[int, int]]:
        denominator = max(1, len(values) - 1)
        return [
            (
                int(left + (right - left) * index / denominator),
                int(bottom - (bottom - top) * (float(value) - low) / (high - low)),
            )
            for index, value in enumerate(values)
        ]

    real_points = points(real_distance)
    predicted_points = points(predicted_distance)
    draw.line(real_points, fill=(70, 150, 255), width=2)
    draw.line(predicted_points, fill=(255, 160, 60), width=2)
    x = int(left + (right - left) * current_step / max(1, len(real_distance) - 1))
    draw.line((x, top, x, bottom), fill=(230, 230, 230), width=1)
    draw.text((left + 4, top + 3), "real", fill=(70, 150, 255))
    draw.text((left + 42, top + 3), "pred", fill=(255, 160, 60))
    draw.text((right - 92, top + 3), "goal MSE", fill="white")


def save_annotated_video(
    path: Path,
    frames: list[np.ndarray],
    action: np.ndarray,
    frame_records: list[dict[str, Any]],
    metrics: dict[str, np.ndarray],
    task_goal_frame: np.ndarray | None,
    fps: float,
) -> None:
    final_goal_image = Image.fromarray(frames[-1]).resize((192, 192), Image.Resampling.BILINEAR)
    task_goal_image = (
        Image.fromarray(task_goal_frame).resize((192, 192), Image.Resampling.BILINEAR)
        if task_goal_frame is not None
        else None
    )
    has_task_goal = "real_to_task_goal_mse" in metrics
    real_distance = (
        metrics["real_to_task_goal_mse"]
        if has_task_goal
        else metrics["real_to_final_goal_mse"]
    )
    predicted_distance = (
        metrics["predicted_to_task_goal_mse"]
        if has_task_goal
        else metrics["predicted_to_final_goal_mse"]
    )
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_log_level="error",
    ) as video:
        for step, frame in enumerate(frames):
            canvas = Image.new("RGB", (896, 512), (20, 20, 20))
            current = Image.fromarray(frame).resize((416, 416), Image.Resampling.BILINEAR)
            canvas.paste(current, (16, 48))
            canvas.paste(final_goal_image, (448, 48))
            if task_goal_image is not None:
                canvas.paste(task_goal_image, (656, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 16), "REAL CAMERA", fill="white")
            draw.text((448, 16), "FINAL REAL STATE", fill="white")
            if task_goal_image is not None:
                draw.text((656, 16), "TASK GOAL", fill="white")
            record = frame_records[step]
            pose = record.get("pose")
            delta_key = (
                "real_task_goal_improvement_mse"
                if has_task_goal
                else "real_final_goal_improvement_mse"
            )
            progress_key = (
                "real_task_goal_progress"
                if has_task_goal
                else "real_final_goal_progress"
            )
            delta = metrics[delta_key][step]
            progress = metrics[progress_key][step]
            lines = [
                f"state {step:03d}/{len(frames) - 1:03d}  (after {step} actions)",
                f"nominal dXY [{float(action[0]) * step:+.4f}, {float(action[1]) * step:+.4f}] m",
                (
                    f"measured XY [{pose[0]:+.4f}, {pose[1]:+.4f}] m"
                    if pose is not None
                    else "measured XY unavailable"
                ),
                f"real -> {'TASK goal' if has_task_goal else 'final state'} MSE {real_distance[step]:.6f}",
                (
                    f"step improvement {delta:+.6f}  (positive is closer)"
                    if math.isfinite(float(delta))
                    else "step improvement n/a"
                ),
                f"normalized progress {progress * 100:+.1f}%",
                f"pred -> {'TASK goal' if has_task_goal else 'final state'} MSE {predicted_distance[step]:.6f}",
                f"pred -> real MSE {metrics['predicted_to_real_mse'][step]:.6f}",
                f"real -> final-state MSE {metrics['real_to_final_goal_mse'][step]:.6f}",
            ]
            y = 254
            for line in lines:
                draw.text((448, y), line, fill="white")
                y += 16
            draw_curve(
                draw,
                (448, 398, 880, 496),
                real_distance,
                predicted_distance,
                step,
            )
            video.append_data(np.asarray(canvas))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_path = args.run.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_path / "analysis"
    )
    metadata_path = run_path / "metadata.json"
    events_path = run_path / "events.jsonl"
    frames_path = run_path / "frames"
    manifest_path = args.manifest.expanduser().resolve()
    world_path = world_checkpoint_path(args.world_checkpoint)

    for path in (metadata_path, events_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not frames_path.is_dir():
        raise FileNotFoundError(frames_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if args.encode_batch_size < 1:
        raise ValueError("--encode-batch-size must be positive")

    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    events = read_events(events_path)
    frame_paths, frames = load_contiguous_frames(frames_path)
    steps = len(frames) - 1

    recorded_steps = metadata_scalar(metadata, ("steps", "num_steps", "action_steps"))
    if recorded_steps is not None and int(recorded_steps) != steps:
        raise ValueError(
            f"metadata records {recorded_steps:g} steps but found {steps} frame transitions"
        )
    action, action_source = metadata_constant_action(metadata, events)
    action_norm = float(np.linalg.norm(action))
    if not 0.0 < action_norm <= MAX_ACTION_NORM_M + 1e-7:
        raise ValueError(
            f"recorded action norm must be in (0, {MAX_ACTION_NORM_M:g}] m, "
            f"got {action_norm:g}"
        )
    trajectory = metadata.get("trajectory", {})
    nominal_targets = np.asarray(
        trajectory.get("nominal_targets_xy_m", []), dtype=np.float64
    )
    if nominal_targets.shape != (steps + 1, 2) or not np.isfinite(nominal_targets).all():
        raise ValueError(
            "metadata trajectory.nominal_targets_xy_m must contain one finite XY "
            "target per recorded frame"
        )
    expected_targets = nominal_targets[0] + np.arange(steps + 1)[:, None] * action
    # ``finite_xy`` intentionally returns float32 because this is the exact
    # tensor sent to the model.  Allow its sub-nanometer representation error
    # to accumulate over the recorded trajectory.
    if not np.allclose(nominal_targets, expected_targets, rtol=0.0, atol=1e-8):
        raise ValueError("metadata nominal targets do not match its constant action")

    fps = args.fps
    if fps is None:
        fps = metadata_scalar(metadata, ("fps", "control_fps", "tick_hz"))
    if fps is None:
        tick_s = metadata_scalar(metadata, ("tick_s", "tick_seconds", "control_period_s"))
        fps = 1.0 / tick_s if tick_s is not None and tick_s > 0 else 5.0
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video FPS must be finite and positive")

    task_goal_path: Path | None = None
    if args.task_goal is not None:
        task_goal_path = args.task_goal.expanduser().resolve()
        if not task_goal_path.is_file():
            raise FileNotFoundError(task_goal_path)
    else:
        recorded_goal = metadata_task_goal(metadata, events)
        if recorded_goal is not None:
            task_goal_path = resolve_recorded_path(recorded_goal, run_path)
    task_goal_frame: np.ndarray | None = None
    if task_goal_path is not None:
        task_goal_frame = iio.imread(task_goal_path)
        if task_goal_frame.shape != (224, 224, 3) or task_goal_frame.dtype != np.uint8:
            raise ValueError(
                f"task goal must be RGB uint8 224x224, got "
                f"{task_goal_frame.shape} {task_goal_frame.dtype}"
            )

    manifest = json.loads(manifest_path.read_text())
    expected_hash = manifest["world_model"]["checkpoint_sha256"]
    actual_hash = sha256(world_path)
    if actual_hash != expected_hash:
        raise ValueError("world-model hash does not match the real-eval manifest")
    if world_path.name != "weights_epoch_146.pt":
        raise ValueError(f"this probe requires epoch 146, got {world_path.name}")
    artifacts = metadata.get("artifacts", {})
    recorded_world_hash = artifacts.get("world_checkpoint_sha256")
    if recorded_world_hash is not None and recorded_world_hash != actual_hash:
        raise ValueError("recorder and analyzer world-model hashes do not match")
    recorded_world = artifacts.get("world_checkpoint")
    if recorded_world is not None and Path(recorded_world).name != world_path.name:
        raise ValueError("recorder and analyzer world-model checkpoints do not match")
    recorded_manifest_hash = artifacts.get("manifest_sha256")
    if recorded_manifest_hash is not None and recorded_manifest_hash != sha256(manifest_path):
        raise ValueError("recorder and analyzer manifest hashes do not match")
    normalization = manifest["normalization"]["action"]
    action_mean = np.asarray(normalization["mean"], dtype=np.float32)
    action_std = np.asarray(normalization["std"], dtype=np.float32)

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = swm.wm.utils.load_pretrained(args.world_checkpoint)
    planner = PushBoxPlanner(
        model,
        action_mean,
        action_std,
        config=CEMConfig(),
        device=device,
    )
    real_tensor = encode_frames(planner, frames, args.encode_batch_size)
    predicted_tensor, normalized_actions = predict_constant_action(
        planner, frames[0], action, steps
    )
    recorded_normalized = finite_xy(trajectory.get("normalized_action"))
    if recorded_normalized is None or not np.allclose(
        recorded_normalized,
        normalized_actions[0].detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("recorder and analyzer normalized actions do not match")
    task_goal_tensor = (
        encode_frames(planner, [task_goal_frame], 1)[0]
        if task_goal_frame is not None
        else None
    )

    real = real_tensor.detach().cpu().numpy().astype(np.float32)
    predicted = predicted_tensor.detach().cpu().numpy().astype(np.float32)
    final_goal = real[-1].copy()
    task_goal = (
        task_goal_tensor.detach().cpu().numpy().astype(np.float32)
        if task_goal_tensor is not None
        else None
    )
    frame_records = frame_event_data(events, len(frames))

    output_path.mkdir(parents=True, exist_ok=False)
    np.save(output_path / "real_latents.npy", real)
    np.save(output_path / "predicted_latents.npy", predicted)
    np.save(output_path / "final_real_goal_latent.npy", final_goal)
    np.save(
        output_path / "normalized_actions.npy",
        normalized_actions.detach().cpu().numpy().astype(np.float32),
    )
    if task_goal is not None:
        np.save(output_path / "task_goal_latent.npy", task_goal)

    metrics = save_metrics_csv(
        output_path / "metrics.csv",
        frame_paths,
        action,
        frame_records,
        real,
        predicted,
        final_goal,
        task_goal,
    )
    save_plot(output_path / "goal_distance.png", metrics, action, frame_records)
    save_annotated_video(
        output_path / "annotated.mp4",
        frames,
        action,
        frame_records,
        metrics,
        task_goal_frame,
        fps,
    )

    real_final_summary = distance_summary(metrics["real_to_final_goal_mse"])
    predicted_final_summary = distance_summary(metrics["predicted_to_final_goal_mse"])
    measured_x = np.asarray(
        [
            record.get("pose", [np.nan, np.nan])[0]
            if record.get("pose") is not None
            else np.nan
            for record in frame_records
        ],
        dtype=np.float64,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "semantics": {
            "frame_index": "state after this many recorded constant actions",
            "primary_goal": "projected encoder latent of the final real frame",
            "goal_distance": "mean((state_projected_latent - goal_projected_latent)^2)",
            "real_encoding": "each real frame encoded independently",
            "prediction": "autoregressive epoch-146 rollout from real frame 000",
            "positive_improvement": "previous goal MSE minus current goal MSE",
            "final_primary_distance_is_zero_by_construction": True,
        },
        "run": {
            "path": str(run_path),
            "metadata": str(metadata_path),
            "events": str(events_path),
            "frame_count": len(frames),
            "steps": steps,
            "fps": fps,
        },
        "action": {
            "source": action_source,
            "raw_action_m": action.tolist(),
            "normalized_action": normalized_actions[0].detach().cpu().tolist(),
            "nominal_cumulative_displacement_m": (action * steps).tolist(),
        },
        "world_model": {
            "checkpoint": args.world_checkpoint,
            "resolved_path": str(world_path),
            "sha256": actual_hash,
            "epoch": 146,
            "device": str(device),
            "history_size": planner.history_size,
            "latent_dim": real.shape[1],
        },
        "goals": {
            "trajectory_endpoint_frame": str(frame_paths[-1]),
            "task_goal_image": str(task_goal_path) if task_goal_path is not None else None,
        },
        "metrics": {
            "real_to_final_real_goal": real_final_summary,
            "predicted_to_final_real_goal": predicted_final_summary,
            "predicted_to_real": {
                "initial_mse": float(metrics["predicted_to_real_mse"][0]),
                "mean_mse": float(metrics["predicted_to_real_mse"].mean()),
                "final_mse": float(metrics["predicted_to_real_mse"][-1]),
                "maximum_mse": float(metrics["predicted_to_real_mse"].max()),
            },
            "pearson_measured_x_vs_negative_primary_mse": correlation(
                measured_x, -metrics["real_to_final_goal_mse"]
            ),
        },
        "artifacts": {
            "metrics_csv": "metrics.csv",
            "summary_json": "summary.json",
            "plot": "goal_distance.png",
            "annotated_video": "annotated.mp4",
            "real_latents": "real_latents.npy",
            "predicted_latents": "predicted_latents.npy",
            "final_real_goal_latent": "final_real_goal_latent.npy",
        },
    }
    if task_goal is not None:
        summary["metrics"]["real_to_task_goal"] = distance_summary(
            metrics["real_to_task_goal_mse"]
        )
        summary["metrics"]["predicted_to_task_goal"] = distance_summary(
            metrics["predicted_to_task_goal_mse"]
        )
        summary["artifacts"]["task_goal_latent"] = "task_goal_latent.npy"
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    print(f"analyzed {len(frames)} real states ({steps} actions) into {output_path}")
    print(f"constant action: {action.tolist()} m ({action_source})")
    if task_goal is not None:
        real_task_summary = summary["metrics"]["real_to_task_goal"]
        print(
            "real -> TASK goal MSE: "
            f"start={real_task_summary['start_mse']:.6f}, "
            f"final={real_task_summary['final_mse']:.6f}, "
            f"min={real_task_summary['minimum_mse']:.6f}, "
            f"improving={real_task_summary['improving_steps']}/{steps}"
        )
    print(
        "real -> final-real state MSE: "
        f"start={metrics['real_to_final_goal_mse'][0]:.6f}, "
        f"final={metrics['real_to_final_goal_mse'][-1]:.6f}, "
        f"improving={real_final_summary['improving_steps']}/{steps}, "
        f"Spearman={real_final_summary['spearman_step_vs_negative_mse']!r}"
    )
    print(
        "predicted -> real MSE: "
        f"mean={metrics['predicted_to_real_mse'].mean():.6f}, "
        f"final={metrics['predicted_to_real_mse'][-1]:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
