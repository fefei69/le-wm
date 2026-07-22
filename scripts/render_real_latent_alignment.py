#!/usr/bin/env python3
"""Render time-aligned real frames, decoded z, and decoded one-step z_hat."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_real_cem_latents import (
    DEFAULT_DECODER,
    decode_latents,
    load_decoder,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="real_robot_runs/RUN directory containing saved latent artifacts",
    )
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default="auto", help="torch device; auto selects CUDA if available"
    )
    return parser.parse_args(argv)


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


def load_rgb(path: Path) -> np.ndarray:
    frame = np.asarray(iio.imread(path))
    if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
        raise ValueError(f"expected RGB uint8 224x224 frame, got {frame.shape} {frame.dtype}: {path}")
    return frame


def load_latent(path: Path) -> np.ndarray:
    latent = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if latent.ndim != 1 or latent.size < 1 or not np.isfinite(latent).all():
        raise ValueError(f"invalid latent artifact: {path}")
    return latent


def resolve_trial_steps(
    run_path: Path, events: list[dict[str, Any]], trial_id: int
) -> list[dict[str, Any]]:
    steps = [
        event
        for event in events
        if event.get("type") == "autonomous_step"
        and int(event.get("trial_id", -1)) == trial_id
    ]
    steps.sort(key=lambda event: int(event["step"]))
    if not steps:
        raise ValueError(f"run has no autonomous steps for trial {trial_id}")
    identifiers = [int(event["step"]) for event in steps]
    if identifiers != list(range(1, len(steps) + 1)):
        raise ValueError("autonomous steps must be contiguous from one")
    required = ("observation_frame", "encoded_latent", "predicted_next_latent")
    for event in steps:
        missing = [key for key in required if not event.get(key)]
        if missing:
            raise ValueError(
                f"step {event['step']} is missing {missing}; this run predates live "
                "z/z_hat recording, so use replay_real_cem_latents.py instead"
            )
        for key in required:
            if not (run_path / event[key]).is_file():
                raise FileNotFoundError(run_path / event[key])
    return steps


def save_decoded_frames(
    output_path: Path, decoded_z: np.ndarray, decoded_z_hat: np.ndarray
) -> None:
    z_path = output_path / "decoded_z"
    z_hat_path = output_path / "decoded_z_hat"
    z_path.mkdir(parents=True, exist_ok=True)
    z_hat_path.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(decoded_z):
        iio.imwrite(z_path / f"t_{index:03d}.png", frame, compress_level=1)
    for index, frame in enumerate(decoded_z_hat, start=1):
        iio.imwrite(z_hat_path / f"t_{index:03d}.png", frame, compress_level=1)


def render_alignment(
    path: Path,
    raw: list[np.ndarray],
    decoded_z: np.ndarray,
    decoded_z_hat: np.ndarray,
    *,
    tile_size: int,
) -> None:
    count = len(raw)
    if len(decoded_z) != count or len(decoded_z_hat) != count:
        raise ValueError("raw, z, and z_hat sequences must have equal lengths")
    gutter = 190
    header = 30
    row_gap = 26
    width = gutter + count * tile_size
    height = header + 3 * (tile_size + row_gap)
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    row_labels = (
        "raw observation[t]",
        "decoded encoded z[t]",
        "decoded predicted z_hat[t]",
    )
    for column in range(count):
        x = gutter + column * tile_size
        draw.text((x + 4, 8), f"t={column:03d}", fill="white")
    for row, label in enumerate(row_labels):
        y = header + row * (tile_size + row_gap)
        draw.text((8, y + tile_size // 2 - 6), label, fill="white")
    resampling = Image.Resampling.BILINEAR
    for column, frame in enumerate(raw):
        x = gutter + column * tile_size
        canvas.paste(Image.fromarray(frame).resize((tile_size, tile_size), resampling), (x, header))
        z_y = header + tile_size + row_gap
        canvas.paste(
            Image.fromarray(decoded_z[column]).resize((tile_size, tile_size), resampling),
            (x, z_y),
        )
        prediction_y = header + 2 * (tile_size + row_gap)
        if column == 0:
            draw.rectangle(
                (x, prediction_y, x + tile_size - 1, prediction_y + tile_size - 1),
                outline=(110, 110, 110),
                width=2,
            )
            draw.multiline_text(
                (x + 10, prediction_y + tile_size // 2 - 16),
                "empty\n(no prior action)",
                fill=(180, 180, 180),
                align="center",
            )
        else:
            # z_hat produced with action[column - 1] predicts observation[column].
            canvas.paste(
                Image.fromarray(decoded_z_hat[column - 1]).resize(
                    (tile_size, tile_size), resampling
                ),
                (x, prediction_y),
            )
        draw.text((x + 4, z_y + tile_size + 5), "encoder", fill=(210, 210, 210))
        prediction_label = "none" if column == 0 else f"from action[{column - 1:03d}]"
        draw.text(
            (x + 4, prediction_y + tile_size + 5),
            prediction_label,
            fill=(210, 210, 210),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, compress_level=1)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tile_size < 64:
        raise ValueError("--tile-size must be at least 64")
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be positive")
    run_path = args.run.expanduser().resolve()
    metadata_path = run_path / "metadata.json"
    events_path = run_path / "events.jsonl"
    if not metadata_path.is_file() or not events_path.is_file():
        raise FileNotFoundError("--run must contain metadata.json and events.jsonl")
    metadata = json.loads(metadata_path.read_text())
    events = read_events(events_path)
    steps = resolve_trial_steps(run_path, events, args.trial_id)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else run_path / "analysis" / f"latent_alignment_trial_{args.trial_id:03d}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    raw = [load_rgb(run_path / event["observation_frame"]) for event in steps]
    encoded = np.stack(
        [load_latent(run_path / event["encoded_latent"]) for event in steps]
    )
    predicted = np.stack(
        [load_latent(run_path / event["predicted_next_latent"]) for event in steps]
    )
    if encoded.shape != predicted.shape:
        raise ValueError(
            f"encoded/predicted latent shapes differ: {encoded.shape} vs {predicted.shape}"
        )

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    decoder_path = args.decoder.expanduser().resolve()
    decoder, decoder_args = load_decoder(decoder_path, device)
    world_checkpoint = str(metadata.get("argv", {}).get("checkpoint", ""))
    decoder_checkpoint = str(decoder_args.get("checkpoint", ""))
    if not world_checkpoint or Path(world_checkpoint).name != Path(decoder_checkpoint).name:
        raise ValueError(
            f"decoder was trained for {decoder_checkpoint}, but run used "
            f"{world_checkpoint or 'an unspecified checkpoint'}"
        )
    if int(decoder.cls_proj[0].in_features) != encoded.shape[1]:
        raise ValueError("decoder input dimension does not match saved latents")

    all_latents = torch.from_numpy(np.concatenate([encoded, predicted], axis=0)).to(
        device
    )
    decoded = decode_latents(decoder, all_latents, args.decode_batch_size)
    decoded_z = decoded[: len(steps)]
    decoded_z_hat = decoded[len(steps) :]
    save_decoded_frames(output_path, decoded_z, decoded_z_hat)
    figure_path = output_path / "latent_alignment.png"
    render_alignment(
        figure_path,
        raw,
        decoded_z,
        decoded_z_hat,
        tile_size=args.tile_size,
    )

    errors = np.square(
        predicted[:-1].astype(np.float64) - encoded[1:].astype(np.float64)
    ).mean(axis=1)
    with (output_path / "one_step_prediction_error.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observation_time", "source_action_step", "latent_mse"])
        for time_index, error in enumerate(errors, start=1):
            writer.writerow([time_index, time_index, float(error)])
    analysis_metadata = {
        "source_run": str(run_path),
        "trial_id": args.trial_id,
        "observation_count": len(steps),
        "latent_dim": encoded.shape[1],
        "decoder": str(decoder_path),
        "world_checkpoint": world_checkpoint,
        "alignment": {
            "row_1": "raw observation[t]",
            "row_2": "decoder(z[t]) where z[t] is encoded from row 1",
            "row_3": (
                "empty at t=0; for t>0 decoder(z_hat[t]) predicted from "
                "z[t-1] and accepted action[t-1]"
            ),
            "terminal_prediction": (
                "the last saved z_hat predicts a state after the final recorded "
                "observation and is saved separately but not placed in the figure"
            ),
        },
        "one_step_latent_mse": {
            "count": len(errors),
            "mean": float(errors.mean()) if len(errors) else None,
            "median": float(np.median(errors)) if len(errors) else None,
            "maximum": float(errors.max()) if len(errors) else None,
        },
        "artifacts": {
            "figure": figure_path.name,
            "decoded_z": "decoded_z/",
            "decoded_z_hat": "decoded_z_hat/",
            "one_step_prediction_error": "one_step_prediction_error.csv",
        },
    }
    (output_path / "metadata.json").write_text(
        json.dumps(analysis_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote time-aligned latent figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
