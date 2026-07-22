#!/usr/bin/env python3
"""D3: decode what the world model imagines a backward plan accomplishes.

Takes real backward-action mistakes from the replay logs (steps where the
planner chose ~180 degrees against a ground-truth push), rebuilds the exact
planner context, rolls the world model forward under (a) the ground-truth
action and (b) the planner's chosen action, each held for the horizon, and
decodes every predicted latent with the trained CLS decoder.  The montage
shows, per example: the real future frames, the model's imagined future under
the ground-truth push, and its imagined future under the backward plan, with
terminal CEM costs annotated — making the hallucination directly visible.

GPU cost: ~2 minutes.

Usage::

    python scripts/diag_decode_rollout.py [--examples 4]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")

import h5py
import numpy as np
from PIL import Image, ImageDraw
import stable_pretraining as spt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_decoder import CLSDecoder
from scripts.diag_token_ranking import build_planner, teacher_forced_context

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY = 3
GOAL_OFFSET = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(REPO_ROOT / "stable-wm/datasets/pushbox_pilot_train.h5"))
    parser.add_argument("--checkpoint", default="pushbox/lewm/weights_epoch_146.pt")
    parser.add_argument("--decoder", type=Path,
                        default=REPO_ROOT / "stable-wm/checkpoints/pushbox/lewm/decoder_proj_ep146_p14/decoder.pt")
    parser.add_argument("--replay", type=Path, default=REPO_ROOT / "outputs/planner_vs_gt/sliding")
    parser.add_argument("--box-track", type=Path,
                        default=REPO_ROOT / "outputs/coverage/pushbox_pilot_train/box_track.npz")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/diagnostics/decoded_rollouts")
    return parser.parse_args()


def load_decoder(path: Path, device: torch.device) -> CLSDecoder:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    decoder_args = dict(checkpoint["args"])
    if decoder_args.get("latent") != "proj":
        raise ValueError("rollout decoding requires a projected-latent decoder")
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
    return decoder


def decode(decoder: CLSDecoder, latents: torch.Tensor) -> np.ndarray:
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.as_tensor(stats["mean"], device=latents.device).view(1, 3, 1, 1)
    std = torch.as_tensor(stats["std"], device=latents.device).view(1, 3, 1, 1)
    frames = []
    with torch.inference_mode():
        for start in range(0, len(latents), 8):
            rgb = (decoder(latents[start : start + 8]) * std + mean).clamp(0, 1)
            frames.append(
                rgb.permute(0, 2, 3, 1).mul(255).round().to(torch.uint8).cpu().numpy()
            )
    return np.concatenate(frames)


def find_backward_mistakes(replay_dir: Path, moving: np.ndarray, ep_offset: np.ndarray,
                           count: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    picks = []
    for file in sorted(replay_dir.glob("ep_*.npz")):
        data = np.load(file)
        episode = int(file.stem.split("_")[1])
        gt, planned, ts = data["gt_action"], data["planned_action"], data["t"]
        for i in range(len(ts)):
            gt_n, plan_n = np.linalg.norm(gt[i]), np.linalg.norm(planned[i])
            if gt_n < 1e-6 or plan_n < 1e-6:
                continue
            cosine = float(gt[i] @ planned[i] / (gt_n * plan_n))
            t = int(ts[i])
            row = int(ep_offset[episode]) + t
            if cosine < -0.9 and moving[row] and t >= HISTORY - 1:
                picks.append((episode, t, gt[i].copy(), planned[i].copy()))
    stride = max(1, len(picks) // count)
    return picks[::stride][:count]


def labeled_row(frames: np.ndarray, label: str, pad: int = 2) -> Image.Image:
    height, width = frames.shape[1:3]
    strip = Image.new("RGB", ((width + pad) * len(frames) + 120, height), "black")
    draw = ImageDraw.Draw(strip)
    draw.text((4, height // 2 - 6), label, fill="white")
    for i, frame in enumerate(frames):
        strip.paste(Image.fromarray(frame), (120 + i * (width + pad), 0))
    return strip


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    track = np.load(args.box_track)
    centroids = track["centroids"]
    moving = np.zeros(len(centroids), dtype=bool)
    deltas = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    valid = ~np.isnan(deltas)
    moving[1:][valid] = deltas[valid] > 1.0

    planner, _ = build_planner(args)
    decoder = load_decoder(args.decoder, torch.device(args.device))

    with h5py.File(args.dataset, "r") as f:
        ep_offset = f["ep_offset"][:]
        cases = find_backward_mistakes(args.replay, moving, ep_offset, args.examples)
        print(f"decoding {len(cases)} backward-mistake cases")
        for episode, t, gt_action, planned_action in cases:
            offset = int(ep_offset[episode])
            frames = f["pixels"][offset + t - HISTORY + 1 : offset + t + 1]
            actions = f["action"][offset + t - HISTORY + 1 : offset + t].astype(np.float32)
            real_future = f["pixels"][offset + t : offset + t + args.horizon + 1]
            goal = f["pixels"][offset + t + GOAL_OFFSET]

            rows = [labeled_row(real_future, "real")]
            costs = {}
            with torch.inference_mode():
                goal_emb = planner._encode(goal)
                for name, action in (("gt", gt_action), ("planner", planned_action)):
                    teacher_forced_context(planner, frames, actions)
                    plan = (
                        torch.as_tensor(action, dtype=torch.float32, device=planner.device)
                        .view(1, 1, 2)
                        .repeat(1, args.horizon, 1)
                    )
                    rollout = planner._rollout_latents(planner._latent_context, plan)[0]
                    costs[name] = float(
                        (rollout[-1] - goal_emb[0, -1]).square().mean().item()
                    )
                    rows.append(
                        labeled_row(decode(decoder, rollout), f"{name}\n[{action[0]:+.3f},{action[1]:+.3f}]\ncost {costs[name]:.3f}")
                    )
            rows.append(labeled_row(goal[None], "goal t+25"))

            width = max(row.width for row in rows)
            sheet = Image.new("RGB", (width, sum(row.height + 4 for row in rows)), "black")
            y = 0
            for row in rows:
                sheet.paste(row, (0, y))
                y += row.height + 4
            out = args.output_dir / f"ep{episode:03d}_t{t:04d}.png"
            sheet.save(out)
            print(f"  {out}  gt-cost {costs['gt']:.4f} vs planner-cost {costs['planner']:.4f}")


if __name__ == "__main__":
    main()
