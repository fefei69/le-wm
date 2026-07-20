#!/usr/bin/env python3
"""Decode an autoregressive PushBox LeWM rollout from one saved RGB frame."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")

import imageio.v2 as imageio
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import stable_pretraining as spt
import stable_worldmodel as swm
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from image_decoder import CLSDecoder
from real_robot_planner import CEMConfig, PushBoxPlanner


DEFAULT_INPUT = REPO_ROOT / (
    "real_robot_runs/20260717_165123_1069809/frames/trial_001_step_007.png"
)
DEFAULT_DECODER = REPO_ROOT / (
    "stable-wm/checkpoints/pushbox/lewm/decoder_proj_ep146_p8/decoder.pt"
)
DEFAULT_MANIFEST = REPO_ROOT / "config/real_robot_eval.json"
DEFAULT_OUTPUT = REPO_ROOT / (
    "outputs/latent_rollout_ep146_p8_plus_x_5mm_50"
)
DEFAULT_WORLD_CHECKPOINT = "pushbox/lewm/weights_epoch_146.pt"
MAX_ACTION_NORM_M = 0.010


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--world-checkpoint", default=DEFAULT_WORLD_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--action-x-m",
        type=float,
        default=0.005,
        help="raw base-frame X delta in meters per predictor step",
    )
    parser.add_argument(
        "--action-y-m",
        type=float,
        default=0.0,
        help="raw base-frame Y delta in meters per predictor step",
    )
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--decode-batch-size", type=int, default=8)
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


def load_source_event(input_path: Path) -> dict[str, Any] | None:
    events_path = input_path.parent.parent / "events.jsonl"
    if not events_path.is_file():
        return None
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        observation = event.get("observation_frame")
        if observation and Path(observation).stem == input_path.stem:
            return event
    return None


def load_decoder(path: Path, device: torch.device) -> tuple[CLSDecoder, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"state_dict", "args"} <= checkpoint.keys():
        raise ValueError("decoder checkpoint must contain state_dict and args")
    decoder_args = dict(checkpoint["args"])
    if decoder_args.get("latent") != "proj":
        raise ValueError("autoregressive rollout requires a projected-latent decoder")
    state_dict = checkpoint["state_dict"]
    cls_dim = int(state_dict["cls_proj.0.weight"].shape[1])
    decoder = CLSDecoder(
        cls_dim=cls_dim,
        img_size=int(decoder_args["img_size"]),
        patch_size=int(decoder_args["patch_size"]),
        dim=int(decoder_args["dim"]),
        heads=int(decoder_args["heads"]),
        depth=int(decoder_args["depth"]),
    ).to(device)
    decoder.load_state_dict(state_dict, strict=True)
    decoder.eval().requires_grad_(False)
    return decoder, decoder_args


def denormalize(reconstruction: torch.Tensor) -> torch.Tensor:
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.as_tensor(
        stats["mean"], dtype=reconstruction.dtype, device=reconstruction.device
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        stats["std"], dtype=reconstruction.dtype, device=reconstruction.device
    ).view(1, 3, 1, 1)
    return (reconstruction * std + mean).clamp(0.0, 1.0)


def decode_latents(
    decoder: CLSDecoder, latents: torch.Tensor, batch_size: int
) -> np.ndarray:
    decoded: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, latents.shape[0], batch_size):
            reconstruction = decoder(latents[start : start + batch_size])
            rgb = denormalize(reconstruction)
            frames = (
                rgb.permute(0, 2, 3, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()
            )
            decoded.append(frames)
    return np.concatenate(decoded, axis=0)


def labeled_tile(frame: np.ndarray, label: str, label_height: int = 24) -> Image.Image:
    image = Image.fromarray(frame)
    tile = Image.new("RGB", (image.width, image.height + label_height), "black")
    tile.paste(image, (0, 0))
    ImageDraw.Draw(tile).text((5, image.height + 5), label, fill="white")
    return tile


def save_contact_sheet(
    path: Path, source: np.ndarray, decoded: np.ndarray, selected_steps: list[int]
) -> None:
    entries = [(source, "source RGB")]
    entries.extend(
        (
            decoded[step],
            f"step {step:03d} ({'encoder' if step == 0 else 'predicted'})",
        )
        for step in selected_steps
    )
    tiles = [labeled_tile(frame, label) for frame, label in entries]
    columns = 5
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new(
        "RGB", (columns * tiles[0].width, rows * tiles[0].height), (32, 32, 32)
    )
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    sheet.save(path)


def latent_metrics(latents: np.ndarray) -> dict[str, np.ndarray]:
    norms = np.linalg.norm(latents, axis=1)
    delta = np.zeros(len(latents), dtype=np.float64)
    delta[1:] = np.linalg.norm(np.diff(latents, axis=0), axis=1)
    drift = np.linalg.norm(latents - latents[0], axis=1)
    denominator = np.maximum(norms * norms[0], np.finfo(np.float64).eps)
    cosine = (latents @ latents[0]) / denominator
    return {"norm": norms, "delta_l2": delta, "drift_l2": drift, "cosine_to_z0": cosine}


def save_metrics_csv(path: Path, metrics: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "latent_source", *metrics.keys()])
        for step in range(len(metrics["norm"])):
            writer.writerow(
                [
                    step,
                    "encoder" if step == 0 else "predictor",
                    *(float(values[step]) for values in metrics.values()),
                ]
            )


def save_diagnostics(path: Path, metrics: dict[str, np.ndarray]) -> None:
    steps = np.arange(len(metrics["norm"]))
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(steps, metrics["norm"])
    axes[0].set_ylabel("latent norm")
    axes[1].plot(steps, metrics["delta_l2"], label="consecutive L2")
    axes[1].plot(steps, metrics["drift_l2"], label="L2 from z0")
    axes[1].set_ylabel("latent distance")
    axes[1].legend()
    axes[2].plot(steps, metrics["cosine_to_z0"])
    axes[2].set_ylabel("cosine to z0")
    axes[2].set_xlabel("state step (after N actions)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    decoder_path = args.decoder.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    world_path = world_checkpoint_path(args.world_checkpoint)

    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.history_size < 1:
        raise ValueError("--history-size must be positive")
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be positive")
    if not np.isfinite([args.action_x_m, args.action_y_m, args.fps]).all():
        raise ValueError("action and fps values must be finite")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    raw_action = np.asarray([args.action_x_m, args.action_y_m], dtype=np.float32)
    if float(np.linalg.norm(raw_action)) > MAX_ACTION_NORM_M + 1e-7:
        raise ValueError(f"action norm exceeds the {MAX_ACTION_NORM_M:g} m data cap")
    if not input_path.is_file() or not decoder_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("input, decoder, and manifest files must exist")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    source = iio.imread(input_path)
    if source.shape != (224, 224, 3) or source.dtype != np.uint8:
        raise ValueError(f"input must be RGB uint8 224x224, got {source.shape} {source.dtype}")

    manifest = json.loads(manifest_path.read_text())
    expected_world_hash = manifest["world_model"]["checkpoint_sha256"]
    actual_world_hash = sha256(world_path)
    if actual_world_hash != expected_world_hash:
        raise ValueError("world-model hash does not match the real-eval manifest")
    normalization = manifest["normalization"]["action"]
    action_mean = np.asarray(normalization["mean"], dtype=np.float32)
    action_std = np.asarray(normalization["std"], dtype=np.float32)

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    world_model = swm.wm.utils.load_pretrained(args.world_checkpoint)
    planner = PushBoxPlanner(
        world_model,
        action_mean,
        action_std,
        config=CEMConfig(),
        device=device,
    )
    if planner.history_size != args.history_size:
        raise ValueError(
            f"model history is {planner.history_size}, requested {args.history_size}"
        )
    decoder, decoder_args = load_decoder(decoder_path, device)
    decoder_world = decoder_args.get("checkpoint")
    if decoder_world and Path(decoder_world).name != world_path.name:
        raise ValueError(
            f"decoder was trained for {decoder_world}, not {args.world_checkpoint}"
        )

    pixels = planner._prepare_image(source).unsqueeze(1)
    raw_actions = torch.as_tensor(raw_action, device=device).reshape(1, 1, 1, 2)
    raw_actions = raw_actions.expand(1, 1, args.steps, 2).clone()
    normalized_actions = (raw_actions - planner.action_mean.unsqueeze(1)) / (
        planner.action_std.unsqueeze(1)
    )
    with torch.inference_mode():
        rollout = planner.model.rollout(
            {"pixels": pixels}, normalized_actions, history_size=args.history_size
        )
    latents = rollout["predicted_emb"][0, 0]
    expected_shape = (args.steps + 1, int(decoder.cls_proj[0].in_features))
    if tuple(latents.shape) != expected_shape:
        raise RuntimeError(
            f"unexpected latent shape {tuple(latents.shape)}, expected {expected_shape}"
        )
    if not torch.isfinite(latents).all():
        raise RuntimeError("rollout produced non-finite latents")

    decoded = decode_latents(decoder, latents, args.decode_batch_size)
    if decoded.shape != (args.steps + 1, 224, 224, 3):
        raise RuntimeError(f"unexpected decoded shape: {decoded.shape}")

    output_path.mkdir(parents=True, exist_ok=False)
    frames_path = output_path / "frames"
    frames_path.mkdir()
    shutil.copy2(input_path, output_path / "source.png")
    for step, frame in enumerate(decoded):
        kind = "encoder" if step == 0 else "predicted"
        iio.imwrite(frames_path / f"frame_{step:03d}_{kind}.png", frame, compress_level=1)

    with imageio.get_writer(
        output_path / "rollout.mp4",
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_log_level="error",
    ) as video:
        for frame in decoded:
            video.append_data(frame)

    selected_steps = sorted(
        {step for step in (0, 1, 2, 5, 10, 20, 30, 40, 50) if step <= args.steps}
    )
    save_contact_sheet(output_path / "contact_sheet.png", source, decoded, selected_steps)

    latent_array = latents.detach().cpu().numpy().astype(np.float32)
    np.save(output_path / "latents.npy", latent_array)
    np.save(
        output_path / "actions.npy",
        np.repeat(raw_action[None], args.steps, axis=0).astype(np.float32),
    )
    np.save(
        output_path / "normalized_actions.npy",
        normalized_actions[0, 0].detach().cpu().numpy().astype(np.float32),
    )
    metrics = latent_metrics(latent_array.astype(np.float64))
    save_metrics_csv(output_path / "metrics.csv", metrics)
    save_diagnostics(output_path / "latent_diagnostics.png", metrics)

    source_event = load_source_event(input_path)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "semantics": {
            "frame_000": "decoder output from the projected encoder latent",
            "frame_001_to_final": "decoder output from recursive predictor latents",
            "frame_index": "state after this many constant actions",
            "decode_reencode": False,
        },
        "source": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "event": source_event,
        },
        "world_model": {
            "checkpoint": args.world_checkpoint,
            "resolved_path": str(world_path),
            "sha256": actual_world_hash,
            "history_size": planner.history_size,
        },
        "decoder": {
            "path": str(decoder_path),
            "sha256": sha256(decoder_path),
            "training_metadata": decoder_args,
        },
        "rollout": {
            "steps": args.steps,
            "raw_action_m": raw_action.tolist(),
            "normalized_action": normalized_actions[0, 0, 0].detach().cpu().tolist(),
            "nominal_cumulative_displacement_m": (raw_action * args.steps).tolist(),
            "control_fps": args.fps,
            "device": str(device),
            "latent_shape": list(latent_array.shape),
        },
        "artifacts": {
            "decoded_frame_count": len(decoded),
            "video": "rollout.mp4",
            "contact_sheet": "contact_sheet.png",
            "diagnostics": "latent_diagnostics.png",
            "metrics": "metrics.csv",
        },
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print(f"wrote {len(decoded)} decoded frames to {output_path}")
    print(f"raw action per step: {raw_action.tolist()} m")
    print(
        "normalized action: "
        f"{normalized_actions[0, 0, 0].detach().cpu().tolist()}"
    )
    print(
        f"latent norm: start={metrics['norm'][0]:.4f}, "
        f"min={metrics['norm'].min():.4f}, max={metrics['norm'].max():.4f}, "
        f"final={metrics['norm'][-1]:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
