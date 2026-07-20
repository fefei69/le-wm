#!/usr/bin/env python3
"""Replay recorded real-robot CEM plans and decode their latent trajectories."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import stable_pretraining as spt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_decoder import CLSDecoder
from real_robot_planner import CEMConfig, PushBoxPlanner


DEFAULT_DECODER = REPO_ROOT / (
    "stable-wm/checkpoints/pushbox/lewm/decoder_proj_ep146_p8/decoder.pt"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument(
        "--plan-step",
        type=int,
        action="append",
        help="recorded action step to decode; repeat as needed (default: every step)",
    )
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--world-checkpoint")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default="auto", help="torch device; auto selects CUDA when available"
    )
    parser.add_argument(
        "--allow-decoder-mismatch",
        action="store_true",
        help="decode with a mismatched checkpoint basis and watermark outputs as invalid",
    )
    return parser.parse_args(argv)


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


def resolve_world_checkpoint(spec: str) -> Path:
    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    candidate = REPO_ROOT / "stable-wm/checkpoints" / spec
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"world-model checkpoint not found: {spec}")


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    if array.shape != (224, 224, 3):
        raise ValueError(f"expected 224x224 RGB image, got {array.shape}: {path}")
    return array


def load_decoder(path: Path, device: torch.device) -> tuple[CLSDecoder, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"state_dict", "args"} <= checkpoint.keys():
        raise ValueError("decoder checkpoint must contain state_dict and args")
    decoder_args = dict(checkpoint["args"])
    if decoder_args.get("latent") != "proj":
        raise ValueError("CEM predictions require a projected-latent decoder")
    state_dict = checkpoint["state_dict"]
    decoder = CLSDecoder(
        cls_dim=int(state_dict["cls_proj.0.weight"].shape[1]),
        img_size=int(decoder_args["img_size"]),
        patch_size=int(decoder_args["patch_size"]),
        dim=int(decoder_args["dim"]),
        heads=int(decoder_args["heads"]),
        depth=int(decoder_args["depth"]),
    ).to(device)
    decoder.load_state_dict(state_dict, strict=True)
    decoder.eval().requires_grad_(False)
    return decoder, decoder_args


def decode_latents(
    decoder: CLSDecoder, latents: torch.Tensor, batch_size: int
) -> np.ndarray:
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.as_tensor(
        stats["mean"], dtype=torch.float32, device=latents.device
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        stats["std"], dtype=torch.float32, device=latents.device
    ).view(1, 3, 1, 1)
    frames: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, latents.shape[0], batch_size):
            reconstruction = decoder(latents[start : start + batch_size])
            rgb = (reconstruction * std + mean).clamp(0.0, 1.0)
            frames.append(
                rgb.permute(0, 2, 3, 1)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
    return np.concatenate(frames, axis=0)


def encode_frames(
    planner: PushBoxPlanner, frames: list[np.ndarray], batch_size: int = 32
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            pixels = torch.cat(
                [planner._prepare_image(frame) for frame in frames[start : start + batch_size]],
                dim=0,
            )
            output.append(planner.model.encode({"pixels": pixels})["emb"][:, -1])
    encoded = torch.cat(output)
    if not torch.isfinite(encoded).all():
        raise RuntimeError("encoder produced a non-finite latent")
    return encoded


def labeled_tile(frame: np.ndarray, label: str, *, invalid: bool = False) -> Image.Image:
    image = Image.fromarray(frame)
    height = 34 if invalid else 24
    tile = Image.new("RGB", (image.width, image.height + height), "black")
    tile.paste(image, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((5, image.height + 5), label, fill="white")
    if invalid:
        draw.text((5, image.height + 18), "INVALID DECODER/CHECKPOINT MISMATCH", fill="red")
    return tile


def save_contact_sheet(
    path: Path,
    source: np.ndarray,
    goal: np.ndarray,
    decoded_current: np.ndarray,
    decoded_goal: np.ndarray,
    decoded_predictions: np.ndarray,
    *,
    invalid: bool,
) -> None:
    entries = [
        (source, "real current RGB"),
        (goal, "real goal RGB"),
        (decoded_current, "decoded encoder current"),
        (decoded_goal, "decoded encoder goal"),
    ]
    entries.extend(
        (frame, f"predicted state {index:02d}")
        for index, frame in enumerate(decoded_predictions, start=1)
    )
    tiles = [labeled_tile(frame, label, invalid=invalid) for frame, label in entries]
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new(
        "RGB", (columns * tiles[0].width, rows * tiles[0].height), (32, 32, 32)
    )
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(path)


def save_latent_heatmap(path: Path, latents: np.ndarray, ylabel: str) -> None:
    limit = float(np.percentile(np.abs(latents), 99.0))
    limit = max(limit, np.finfo(np.float32).eps)
    fig, axis = plt.subplots(figsize=(14, max(3.0, 0.22 * len(latents))))
    image = axis.imshow(latents, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_xlabel("projected latent dimension")
    axis.set_ylabel(ylabel)
    fig.colorbar(image, ax=axis, label="latent value")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_goal_plot(
    path: Path, replayed: np.ndarray, recorded: np.ndarray, steps: np.ndarray
) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(steps, replayed, marker="o", markersize=3, label="re-encoded from PNG")
    axis.plot(steps, recorded, linestyle="--", label="recorded during live planning")
    axis.set_xlabel("real MPC step")
    axis.set_ylabel("current-to-goal latent MSE")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_robot_tracking(
    output_path: Path,
    steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compare each commanded delta with the next measured pre-plan pose."""
    if len(steps) < 2 or not all("measured_pose_before" in event for event in steps):
        return None
    poses = np.asarray(
        [event["measured_pose_before"][:2] for event in steps], dtype=np.float64
    )
    commanded = np.asarray(
        [event["executed_action"] for event in steps[:-1]], dtype=np.float64
    )
    measured = np.diff(poses, axis=0)
    residual = measured - commanded
    step_ids = np.arange(1, len(steps))
    with (output_path / "robot_tracking.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "command_x_m",
                "command_y_m",
                "measured_delta_x_m",
                "measured_delta_y_m",
                "residual_x_m",
                "residual_y_m",
            ]
        )
        for step, command, delta, error in zip(
            step_ids, commanded, measured, residual
        ):
            writer.writerow([step, *command, *delta, *error])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for axis, dimension, label in zip(axes, range(2), ("X", "Y")):
        axis.plot(step_ids, commanded[:, dimension] * 1000.0, label="commanded")
        axis.plot(
            step_ids,
            measured[:, dimension] * 1000.0,
            linestyle="--",
            label="measured before next plan",
        )
        axis.set_ylabel(f"{label} delta (mm)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("real MPC step")
    fig.tight_layout()
    fig.savefig(output_path / "robot_tracking.png", dpi=160)
    plt.close(fig)

    correlations = [
        float(np.corrcoef(commanded[:, index], measured[:, index])[0, 1])
        for index in range(2)
    ]
    return {
        "comparable_motion_count": len(commanded),
        "commanded_sum_xy_m": commanded.sum(axis=0).tolist(),
        "measured_sum_xy_m": measured.sum(axis=0).tolist(),
        "command_measured_correlation_xy": correlations,
        "median_absolute_residual_xy_m": np.median(
            np.abs(residual), axis=0
        ).tolist(),
    }


def append_context(
    planner: PushBoxPlanner,
    current: torch.Tensor,
    previous_action: np.ndarray | None,
) -> None:
    current = current.reshape(1, 1, -1)
    if planner._latent_context is None:
        if previous_action is not None:
            raise ValueError("first replay observation cannot have a previous action")
        planner._latent_context = current
        return
    if previous_action is None:
        raise ValueError("missing previous action while rebuilding planner history")
    action = torch.as_tensor(
        previous_action, dtype=torch.float32, device=planner.device
    ).reshape(1, 1, 2)
    normalized = (action - planner.action_mean) / planner.action_std
    planner._executed_actions = (
        normalized
        if planner._executed_actions is None
        else torch.cat([planner._executed_actions, normalized], dim=1)
    )
    planner._latent_context = torch.cat([planner._latent_context, current], dim=1)
    planner._latent_context = planner._latent_context[:, -planner.history_size :]
    required = planner._latent_context.shape[1] - 1
    planner._executed_actions = (
        planner._executed_actions[:, -required:] if required else None
    )


def write_plan_metrics(
    path: Path, latents: np.ndarray, actions: np.ndarray, goal: np.ndarray
) -> np.ndarray:
    distances = np.square(latents.astype(np.float64) - goal).mean(axis=1)
    deltas = np.zeros(len(latents), dtype=np.float64)
    deltas[1:] = np.linalg.norm(np.diff(latents.astype(np.float64), axis=0), axis=1)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["state", "latent_source", "action_x_m", "action_y_m", "goal_mse", "latent_norm", "latent_delta"]
        )
        for state in range(len(latents)):
            action = [float("nan"), float("nan")] if state == 0 else actions[state - 1]
            writer.writerow(
                [
                    state,
                    "encoder" if state == 0 else "predictor",
                    action[0],
                    action[1],
                    distances[state],
                    np.linalg.norm(latents[state].astype(np.float64)),
                    deltas[state],
                ]
            )
    return distances


def save_index(path: Path, run_name: str, summaries: list[dict[str, Any]]) -> None:
    sections = []
    for item in summaries:
        step = int(item["step"])
        sections.append(
            f"<section><h2>Plan step {step:03d}</h2>"
            f"<p>first action: {html.escape(str(item['first_action_m']))}; "
            f"recorded cost: {item['recorded_cost']:.6f}; "
            f"replayed selected-plan cost: {item['replayed_cost']:.6f}; "
            f"one-step error: {item['one_step_prediction_mse']}</p>"
            f"<a href='plan_{step:03d}/contact_sheet.png'>"
            f"<img src='plan_{step:03d}/contact_sheet.png'></a></section>"
        )
    document = f"""<!doctype html><meta charset='utf-8'>
<title>CEM latent replay {html.escape(run_name)}</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:auto}}img{{max-width:100%}}section{{border-top:1px solid #aaa;padding:1rem 0}}</style>
<h1>CEM latent replay: {html.escape(run_name)}</h1>
<p><a href='actual_goal_distance.png'>actual goal distance</a> · <a href='actual_latent_heatmap.png'>actual latent heatmap</a> · <a href='robot_tracking.png'>robot command tracking</a> · <a href='summary.csv'>summary CSV</a></p>
{''.join(sections)}
"""
    path.write_text(document)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_path = args.run.expanduser().resolve()
    metadata_path = run_path / "metadata.json"
    events_path = run_path / "events.jsonl"
    if not metadata_path.is_file() or not events_path.is_file():
        raise FileNotFoundError("--run must contain metadata.json and events.jsonl")
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be positive")

    metadata = json.loads(metadata_path.read_text())
    run_args = metadata.get("argv", {})
    world_spec = args.world_checkpoint or run_args.get("checkpoint")
    if not world_spec:
        raise ValueError("world checkpoint is absent from both CLI and run metadata")
    dataset_path = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else Path(run_args["dataset"]).expanduser().resolve()
    )
    world_path = resolve_world_checkpoint(world_spec)
    decoder_path = args.decoder.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else REPO_ROOT / "outputs" / f"real_cem_latent_replay_{run_path.name}"
    )
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    horizon = int(run_args["horizon"])
    planner = PushBoxPlanner.from_artifacts(
        checkpoint=str(world_path),
        dataset_path=dataset_path,
        config=CEMConfig(
            horizon=horizon,
            action_cap_m=float(run_args["action_cap"]),
            action_mode="continuous",
        ),
        device=str(device),
        artifact_manifest=None,
    )
    decoder, decoder_args = load_decoder(decoder_path, device)
    decoder_world = str(decoder_args.get("checkpoint", ""))
    decoder_matches = Path(decoder_world).name == world_path.name
    if not decoder_matches and not args.allow_decoder_mismatch:
        raise ValueError(
            f"decoder was trained for {decoder_world}, but run used {world_path.name}; "
            "train/use a matching projected-latent decoder"
        )
    if int(decoder.cls_proj[0].in_features) != int(
        planner.model.predictor.pos_embedding.shape[-1]
    ):
        raise ValueError("decoder and world model latent dimensions differ")

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    steps = [
        event
        for event in events
        if event.get("type") == "autonomous_step"
        and int(event.get("trial_id", -1)) == args.trial_id
    ]
    steps.sort(key=lambda event: int(event["step"]))
    if not steps:
        raise ValueError(f"run has no autonomous steps for trial {args.trial_id}")
    expected = list(range(1, len(steps) + 1))
    actual_step_ids = [int(event["step"]) for event in steps]
    if actual_step_ids != expected:
        raise ValueError("autonomous step ids are not contiguous from one")
    selected = set(args.plan_step or expected)
    invalid_selected = sorted(selected - set(expected))
    if invalid_selected:
        raise ValueError(f"requested plan steps are absent: {invalid_selected}")

    trial_started = next(
        event
        for event in events
        if event.get("type") == "trial_started"
        and int(event.get("trial_id", -1)) == args.trial_id
    )
    goal_path = run_path / trial_started["goal_frame"]
    goal_frame = load_rgb(goal_path)
    observation_paths = [run_path / event["observation_frame"] for event in steps]
    real_frames = [load_rgb(path) for path in observation_paths]
    real_latents_t = encode_frames(planner, real_frames)
    goal_latent_t = encode_frames(planner, [goal_frame])[0]
    real_latents = real_latents_t.detach().cpu().numpy().astype(np.float32)
    goal_latent = goal_latent_t.detach().cpu().numpy().astype(np.float32)

    output_path.mkdir(parents=True)
    shutil.copy2(goal_path, output_path / "goal.png")
    np.save(output_path / "actual_encoded_latents.npy", real_latents)
    np.save(output_path / "goal_encoded_latent.npy", goal_latent)
    save_latent_heatmap(output_path / "actual_latent_heatmap.png", real_latents, "real MPC step")
    replayed_goal = np.square(real_latents.astype(np.float64) - goal_latent).mean(axis=1)
    recorded_goal = np.asarray([event["goal_distance"] for event in steps], dtype=np.float64)
    step_ids = np.asarray(expected)
    save_goal_plot(output_path / "actual_goal_distance.png", replayed_goal, recorded_goal, step_ids)
    robot_tracking = save_robot_tracking(output_path, steps)

    planner.reset()
    summaries: list[dict[str, Any]] = []
    plan_records: list[tuple[dict[str, Any], torch.Tensor, np.ndarray]] = []
    previous_action: np.ndarray | None = None
    with torch.inference_mode():
        for index, event in enumerate(steps):
            append_context(planner, real_latents_t[index], previous_action)
            plan = np.asarray(event["plan"], dtype=np.float32)
            if plan.shape != (horizon, 2):
                raise ValueError(f"step {index + 1} has invalid plan shape {plan.shape}")
            trajectory = planner._rollout_latents(
                planner._latent_context,
                torch.as_tensor(plan, device=device).unsqueeze(0),
            )[0]
            replayed_cost = float(
                planner._rollout_cost(
                    planner._latent_context,
                    goal_latent_t.reshape(1, 1, -1),
                    torch.as_tensor(plan, device=device).unsqueeze(0),
                )[0].item()
            )
            one_step = (
                None
                if index + 1 >= len(real_latents_t)
                else float(
                    (trajectory[1] - real_latents_t[index + 1]).square().mean().item()
                )
            )
            predicted_distances = (
                trajectory - goal_latent_t.reshape(1, -1)
            ).square().mean(dim=1)
            actual_next_distance = (
                None
                if index + 1 >= len(replayed_goal)
                else float(replayed_goal[index + 1])
            )
            current_distance = float(replayed_goal[index])
            predicted_one_step_distance = float(predicted_distances[1].item())
            summary = {
                "step": index + 1,
                "first_action_m": plan[0].tolist(),
                "plan_sum_xy_m": plan.astype(np.float64).sum(axis=0).tolist(),
                "recorded_cost": float(event["cost"]),
                "replayed_cost": replayed_cost,
                "recorded_minus_replayed_cost": float(event["cost"])
                - replayed_cost,
                "recorded_goal_distance": float(event["goal_distance"]),
                "reencoded_goal_distance": current_distance,
                "predicted_one_step_goal_distance": predicted_one_step_distance,
                "predicted_terminal_goal_distance": float(
                    predicted_distances[-1].item()
                ),
                "actual_next_goal_distance": actual_next_distance,
                "predicted_one_step_improvement": current_distance
                - predicted_one_step_distance,
                "actual_next_improvement": (
                    None
                    if actual_next_distance is None
                    else current_distance - actual_next_distance
                ),
                "one_step_prediction_mse": one_step,
            }
            summaries.append(summary)
            if index + 1 in selected:
                plan_records.append((summary, trajectory.detach(), plan))
            previous_action = np.asarray(event["planner_feedback_action"], dtype=np.float32)

    decode_tensors = [goal_latent_t.reshape(1, -1)]
    decode_tensors.extend(record[1] for record in plan_records)
    decoded_all = decode_latents(
        decoder, torch.cat(decode_tensors, dim=0), args.decode_batch_size
    )
    decoded_goal = decoded_all[0]
    decode_offset = 1
    selected_summaries: list[dict[str, Any]] = []
    for summary, trajectory_t, plan in plan_records:
        step = int(summary["step"])
        directory = output_path / f"plan_{step:03d}"
        frames_path = directory / "decoded_predictions"
        frames_path.mkdir(parents=True)
        shutil.copy2(observation_paths[step - 1], directory / "source.png")
        shutil.copy2(goal_path, directory / "goal.png")
        trajectory = trajectory_t.cpu().numpy().astype(np.float32)
        np.save(directory / "latent_trajectory.npy", trajectory)
        np.save(directory / "selected_actions.npy", plan)
        distances = write_plan_metrics(
            directory / "latent_metrics.csv", trajectory, plan, goal_latent
        )
        decoded = decoded_all[decode_offset : decode_offset + len(trajectory)]
        decode_offset += len(trajectory)
        iio.imwrite(directory / "decoded_current_encoder.png", decoded[0], compress_level=1)
        iio.imwrite(directory / "decoded_goal_encoder.png", decoded_goal, compress_level=1)
        for predicted_step, frame in enumerate(decoded[1:], start=1):
            iio.imwrite(
                frames_path / f"predicted_{predicted_step:03d}.png",
                frame,
                compress_level=1,
            )
        save_contact_sheet(
            directory / "contact_sheet.png",
            real_frames[step - 1],
            goal_frame,
            decoded[0],
            decoded_goal,
            decoded[1:],
            invalid=not decoder_matches,
        )
        summary["predicted_goal_distances"] = distances.tolist()
        (directory / "summary.json").write_text(
            json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n"
        )
        selected_summaries.append(summary)

    with (output_path / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "action_x_m",
                "action_y_m",
                "recorded_cost",
                "replayed_cost",
                "recorded_minus_replayed_cost",
                "recorded_goal_distance",
                "reencoded_goal_distance",
                "predicted_one_step_goal_distance",
                "actual_next_goal_distance",
                "predicted_terminal_goal_distance",
                "predicted_one_step_improvement",
                "actual_next_improvement",
                "one_step_prediction_mse",
                "plan_sum_x_m",
                "plan_sum_y_m",
            ]
        )
        for item in summaries:
            writer.writerow(
                [
                    item["step"],
                    *item["first_action_m"],
                    item["recorded_cost"],
                    item["replayed_cost"],
                    item["recorded_minus_replayed_cost"],
                    item["recorded_goal_distance"],
                    item["reencoded_goal_distance"],
                    item["predicted_one_step_goal_distance"],
                    item["actual_next_goal_distance"],
                    item["predicted_terminal_goal_distance"],
                    item["predicted_one_step_improvement"],
                    item["actual_next_improvement"],
                    item["one_step_prediction_mse"],
                    *item["plan_sum_xy_m"],
                ]
            )

    save_index(output_path / "index.html", run_path.name, selected_summaries)
    replay_metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_run": str(run_path),
        "trial_id": args.trial_id,
        "selected_plan_steps": sorted(selected),
        "world_checkpoint": str(world_path),
        "decoder": str(decoder_path),
        "decoder_training_checkpoint": decoder_world,
        "decoder_matches_world_model": decoder_matches,
        "scientifically_valid_decoding": decoder_matches,
        "history_size": planner.history_size,
        "horizon": horizon,
        "robot_tracking": robot_tracking,
        "semantics": {
            "decoded_current_encoder": "decoder output from the recorded observation's projected encoder latent",
            "decoded_predictions": "open-loop recursive predictions under the recorded selected CEM plan",
            "prediction_count": horizon,
            "real_comparison": "only predicted state 1 is directly comparable; MPC replanned before later real states",
        },
    }
    (output_path / "metadata.json").write_text(
        json.dumps(jsonable(replay_metadata), indent=2, sort_keys=True) + "\n"
    )
    print(
        f"replayed {len(steps)} plans and decoded {len(selected)} selected horizons "
        f"into {output_path}"
    )
    print(f"open {output_path / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
