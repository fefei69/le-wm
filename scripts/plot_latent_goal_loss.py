#!/usr/bin/env python3
"""Plot the planner's latent goal loss over time on recorded episodes.

For every episode video in ``datasets_videos/<session>/ep_*.mp4`` this script
encodes each frame with the LeWM checkpoint the real-robot planner uses and
computes the exact quantity CEM minimizes at the terminal step (and reports as
``goal_distance``): ``MSE(z_t, z_goal)`` over the projected CLS latent, with
the goal taken as the episode's final frame.  Frame preprocessing mirrors
``real_robot_planner.PushBoxPlanner`` (ImageNet normalization, 224x224).

If the latent is a usable planning signal, curves should decrease as frames
approach the goal; per-episode Spearman rho(loss, t) summarizes monotonicity.

Latents are episode-independent, so per-episode losses are cached to
``latent_goal_loss.npz`` in the output directory; reruns only redo the plots.

Usage::

    python scripts/plot_latent_goal_loss.py [--checkpoint pushbox/lewm/weights_epoch_146.pt]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # checkpoint targets jepa.JEPA

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=REPO_ROOT / "datasets_videos")
    parser.add_argument("--checkpoint", default="pushbox/lewm/weights_epoch_146.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/latent_goal_loss")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=256, help="frames per encoder forward")
    return parser.parse_args()


def build_transform() -> v2.Compose:
    stats_ = spt.data.dataset_stats.ImageNet
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=stats_["mean"], std=stats_["std"]),
            v2.Resize((224, 224)),
        ]
    )


@torch.inference_mode()
def episode_goal_loss(
    model: torch.nn.Module,
    transform: v2.Compose,
    frames: np.ndarray,
    device: torch.device,
    batch: int,
) -> np.ndarray:
    """Per-frame MSE(z_t, z_goal) with the last frame as goal."""
    chunks = []
    for start in range(0, len(frames), batch):
        pixels = torch.stack([transform(f) for f in frames[start : start + batch]])
        pixels = pixels.unsqueeze(1).to(device)  # (B, T=1, C, H, W)
        emb = model.encode({"pixels": pixels})["emb"][:, 0]
        chunks.append(emb)
    latents = torch.cat(chunks)
    if not torch.isfinite(latents).all():
        raise RuntimeError("encoder produced non-finite latents")
    loss = (latents - latents[-1]).square().mean(dim=-1)
    return loss.cpu().numpy().astype(np.float64)


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def collect_losses(args: argparse.Namespace) -> dict[str, np.ndarray]:
    videos = sorted(args.videos.glob("*/ep_*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no episode videos under {args.videos}")
    model = swm.wm.utils.load_pretrained(args.checkpoint)
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    transform = build_transform()
    losses: dict[str, np.ndarray] = {}
    for i, video in enumerate(videos):
        frames = read_video(video)
        key = f"{video.parent.name}/{video.stem}"
        losses[key] = episode_goal_loss(
            model, transform, frames, torch.device(args.device), args.batch
        )
        if i % 20 == 0:
            print(f"  encoded {i + 1}/{len(videos)}: {key} ({len(frames)} frames)", flush=True)
    return losses


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "latent_goal_loss.npz"
    if cache.exists():
        losses = dict(np.load(cache))
        print(f"loaded cached losses for {len(losses)} episodes from {cache}")
    else:
        losses = collect_losses(args)
        np.savez_compressed(cache, **losses)

    keys = sorted(losses)
    curves = [losses[k] for k in keys]
    rho = np.array([stats.spearmanr(np.arange(len(c)), c).statistic for c in curves])
    initial = np.array([c[0] for c in curves])
    floor = np.array([c[-6:-1].mean() for c in curves])  # near-goal plateau, goal frame excluded

    # resample every curve to normalized time for the summary band
    grid = np.linspace(0.0, 1.0, 200)
    resampled = np.stack(
        [np.interp(grid, np.linspace(0, 1, len(c)), c) for c in curves]
    )
    quantiles = np.percentile(resampled, [25, 50, 75], axis=0)

    # last 10 s aligned at the goal
    tail_len = 50
    tails = np.stack([c[-tail_len:] for c in curves if len(c) >= tail_len])
    tail_q = np.percentile(tails, [25, 50, 75], axis=0)

    report = [
        f"episodes: {len(curves)}   checkpoint: {args.checkpoint}",
        f"loss at episode start: median {np.median(initial):.4f} "
        f"(IQR {np.percentile(initial, 25):.4f}-{np.percentile(initial, 75):.4f})",
        f"near-goal plateau (mean of last 5 non-goal frames): median {np.median(floor):.4f}",
        f"start/plateau contrast ratio: median {np.median(initial / np.maximum(floor, 1e-12)):.2f}x",
        f"Spearman rho(loss, t): median {np.median(rho):.3f}; "
        f"{np.mean(rho < -0.5):.0%} of episodes below -0.5, {np.mean(rho > 0):.0%} above 0",
    ]
    text = "\n".join(report)
    (args.output_dir / "summary.txt").write_text(text + "\n")
    print("\n" + text + "\n")

    ink, muted, accent, band = "#39485e", "#7a8699", "#3d6bb0", "#c9d9ef"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.edgecolor": muted, "axes.labelcolor": ink,
        "text.color": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    fig.suptitle(
        f"Planner latent goal loss MSE(z_t, z_goal), goal = last frame ({len(curves)} episodes)",
        color=ink,
    )

    ax = axes[0, 0]
    for c in curves:
        ax.plot(np.linspace(0, 1, len(c)), c, color=accent, alpha=0.06, lw=0.7)
    ax.fill_between(grid, quantiles[0], quantiles[2], color=band, alpha=0.8, lw=0)
    ax.plot(grid, quantiles[1], color=ink, lw=1.6)
    ax.set_title("all episodes, normalized time (line: median, band: IQR)")
    ax.set_xlabel("episode progress")
    ax.set_ylabel("latent MSE to goal")

    ax = axes[0, 1]
    steps = np.arange(-tail_len + 1, 1)
    for c in tails:
        ax.plot(steps, c, color=accent, alpha=0.06, lw=0.7)
    ax.fill_between(steps, tail_q[0], tail_q[2], color=band, alpha=0.8, lw=0)
    ax.plot(steps, tail_q[1], color=ink, lw=1.6)
    ax.set_title(f"final {tail_len} steps (10 s), aligned at goal")
    ax.set_xlabel("steps before goal frame")
    ax.set_ylabel("latent MSE to goal")

    ax = axes[1, 0]
    ax.hist(rho, bins=30, range=(-1, 1), color=accent, alpha=0.85)
    ax.axvline(0, color=muted, lw=1, ls="--")
    ax.set_title("per-episode Spearman rho(loss, t)  (-1 = monotone decrease)")
    ax.set_xlabel("rho")
    ax.set_ylabel("episodes")

    ax = axes[1, 1]
    order = np.argsort(rho)
    picks = [order[int(p * (len(order) - 1))] for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    shades = ["#1f3e66", "#2f5a92", "#3d6bb0", "#6f96c9", "#a3bede"]
    for shade, idx in zip(shades, picks):
        c = curves[idx]
        label = f"{keys[idx].split('/')[0][-6:]}/{keys[idx].split('/')[1]} (rho {rho[idx]:+.2f})"
        ax.plot(np.linspace(0, 1, len(c)), c, color=shade, lw=1.3, label=label)
    ax.set_title("representative episodes across the rho range")
    ax.set_xlabel("episode progress")
    ax.set_ylabel("latent MSE to goal")
    ax.legend(fontsize=7, frameon=False)

    for ax in axes.flat:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = args.output_dir / "latent_goal_loss.png"
    fig.savefig(fig_path, dpi=160)
    print(f"figure: {fig_path}\nsummary: {args.output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
