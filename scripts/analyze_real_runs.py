#!/usr/bin/env python3
"""Forensic comparison of the 2026-07-21 real-robot runs (success vs failure).

For each run under ``real_robot_runs/``: parses ``events.jsonl`` (per-step
executed actions, planner goal distance, EE poses), tracks the box in every
saved frame and in the goal image (red-patch segmentation, same detector as
the coverage analysis), and computes:

- true task progress: box -> goal-box pixel distance over time,
- planner-visible progress: latent goal distance over time,
- model surprise: realized one-step prediction error ||z_hat[t] - z[t+1]||^2
  from the stored latents, against the offline validation baseline,
- coverage: where the run's box positions and the goal's box position sit in
  the training-data box-density map (percentile of training density),
- action statistics (direction mix, forward/backward split).

Outputs a comparison figure, a goal/start montage, and a markdown report in
``outputs/real_runs_20260721/``.

Usage::

    python scripts/analyze_real_runs.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "20260721_182051_3575856_test01": True,
    "20260721_183103_3584574_test02": True,
    "20260721_191430_3622865": False,
    "20260721_191641_3625162": False,
    "20260721_191833_3627097": False,
    "20260721_192115_3629765": False,
    "20260721_192540_3633741_test03": True,
}
TIERS = ((100, 50, 60), (90, 35, 40))


def detect_box(image: np.ndarray) -> tuple[float, float] | None:
    array = image.astype(np.int16)
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    for r_min, margin, min_blob in TIERS:
        mask = (r > r_min) & (r - g > margin) & (r - b > margin)
        labels, count = ndimage.label(mask)
        if count == 0:
            continue
        sizes = ndimage.sum_labels(np.ones_like(labels), labels, np.arange(1, count + 1))
        best = int(np.argmax(sizes)) + 1
        if sizes[best - 1] < min_blob:
            continue
        ys, xs = np.nonzero(labels == best)
        return float(xs.mean()), float(ys.mean())
    return None


def load_run(run_dir: Path) -> dict:
    steps = []
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["type"] == "autonomous_step":
            steps.append(event)
        elif event["type"] == "trial_outcome":
            outcome = event["outcome"]
    goal_image = np.asarray(Image.open(run_dir / "frames/trial_001_goal.png"))
    goal_box = detect_box(goal_image)
    boxes, frames = [], sorted((run_dir / "frames").glob("trial_001_step_*.png"))
    for frame_path in frames:
        boxes.append(detect_box(np.asarray(Image.open(frame_path))))
    z_dir, zh_dir = run_dir / "latents/z", run_dir / "latents/z_hat"
    z = np.stack([np.load(p) for p in sorted(z_dir.glob("*.npy"))])
    z_hat = np.stack([np.load(p) for p in sorted(zh_dir.glob("*.npy"))])
    surprise = ((z_hat[:-1] - z[1:]) ** 2).mean(axis=1)
    return {
        "outcome": outcome,
        "goal_distance": np.array([s["goal_distance"] for s in steps]),
        "actions": np.array([s["executed_action"] for s in steps]),
        "ee": np.array([s["measured_pose_before"][:2] for s in steps if s["measured_pose_before"]]),
        "goal_box": goal_box,
        "boxes": boxes,
        "surprise": surprise,
        "goal_image": goal_image,
        "first_frame": np.asarray(Image.open(frames[0])) if frames else None,
    }


def box_to_goal_px(record: dict) -> np.ndarray:
    goal = record["goal_box"]
    return np.array([
        np.hypot(b[0] - goal[0], b[1] - goal[1]) if (b and goal) else np.nan
        for b in record["boxes"]
    ])


def training_density() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    track = np.load(REPO_ROOT / "outputs/coverage/pushbox_pilot_train/box_track.npz")
    pts = track["centroids"]
    pts = pts[~np.isnan(pts[:, 0])]
    hist, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=56, range=[[0, 224], [0, 224]])
    return hist, xe, ye


def density_percentile(hist, xe, ye, u, v) -> float:
    """Training visitation count at (u, v) as a percentile of visited bins."""
    i = np.clip(np.searchsorted(xe, u) - 1, 0, hist.shape[0] - 1)
    j = np.clip(np.searchsorted(ye, v) - 1, 0, hist.shape[1] - 1)
    value = hist[i, j]
    visited = hist[hist > 0]
    return float((visited < value).mean() * 100.0)


def main() -> None:
    out_dir = REPO_ROOT / "outputs/real_runs_20260721"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist, xe, ye = training_density()
    baseline = json.load(open(REPO_ROOT / "outputs/diagnostics/action_sensitivity.json"))
    base_mse = baseline["one_step_mse_true"]

    records = {}
    for name in RUNS:
        records[name] = load_run(REPO_ROOT / "real_robot_runs" / name)
        records[name]["success"] = RUNS[name]

    ink, muted = "#39485e", "#7a8699"
    ok_color, bad_color = "#2e7d4f", "#c0392b"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.edgecolor": muted, "axes.labelcolor": ink,
        "text.color": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    lines = []

    for name, rec in records.items():
        color = ok_color if rec["success"] else bad_color
        label = name[-6:]
        axes[0, 0].plot(rec["goal_distance"], color=color, lw=1.2, alpha=0.9)
        axes[0, 1].plot(box_to_goal_px(rec), color=color, lw=1.2, alpha=0.9)
        axes[0, 2].plot(rec["surprise"], color=color, lw=1.0, alpha=0.8)
        boxes = np.array([b for b in rec["boxes"] if b])
        axes[1, 0].plot(boxes[:, 0], boxes[:, 1], color=color, lw=1.2)
        axes[1, 0].plot(*boxes[0], "o", color=color, ms=5, mfc="white")
        if rec["goal_box"]:
            axes[1, 0].plot(*rec["goal_box"], "*", color=color, ms=13)
        percentiles = [
            density_percentile(hist, xe, ye, b[0], b[1]) for b in rec["boxes"] if b
        ]
        rec["run_cov"] = float(np.median(percentiles))
        rec["goal_cov"] = (
            density_percentile(hist, xe, ye, *rec["goal_box"]) if rec["goal_box"] else np.nan
        )
        rec["net_box_px"] = float(np.hypot(*(boxes[-1] - boxes[0]))) if len(boxes) > 1 else 0.0
        rec["fwd_frac"] = float((rec["actions"][:, 0] > 1e-6).mean())
        rec["back_frac"] = float((rec["actions"][:, 0] < -1e-6).mean())
        lines.append(
            f"| {name} | {'OK' if rec['success'] else 'FAIL'} | {len(rec['goal_distance'])} "
            f"| {rec['goal_distance'][0]:.2f} -> {rec['goal_distance'][-1]:.2f} "
            f"| {box_to_goal_px(rec)[0]:.0f} -> {np.nanmin(box_to_goal_px(rec)):.0f} px "
            f"| {rec['net_box_px']:.0f} px | {rec['surprise'].mean() / base_mse:.1f}x "
            f"| {rec['run_cov']:.0f} / {rec['goal_cov']:.0f} "
            f"| {rec['fwd_frac']:.0%}/{rec['back_frac']:.0%} |"
        )

    axes[0, 0].set_title("planner latent goal distance per step")
    axes[0, 0].set_xlabel("step")
    axes[0, 1].set_title("TRUE progress: box -> goal-box distance (px)")
    axes[0, 1].set_xlabel("step")
    axes[0, 2].axhline(base_mse, color=muted, ls="--", lw=1)
    axes[0, 2].set_title("one-step surprise ||z_hat - z_next||^2 (dashed: val baseline)")
    axes[0, 2].set_xlabel("step")

    ax = axes[1, 0]
    ax.imshow(np.log1p(hist.T), cmap="Blues", origin="upper", extent=(0, 224, 224, 0), alpha=0.9)
    ax.set_title("box paths (o=start, *=goal) on training density")
    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)

    for ax, key, title in ((axes[1, 1], True, "successes"), (axes[1, 2], False, "failures")):
        acts = np.concatenate([r["actions"] for r in records.values() if r["success"] == key])
        moving = np.linalg.norm(acts, axis=1) > 1e-6
        ang = np.arctan2(acts[moving, 1], acts[moving, 0])
        counts, _ = np.histogram(ang, bins=np.linspace(-np.pi, np.pi, 9))
        ax.bar(np.linspace(-np.pi, np.pi, 8, endpoint=False) + np.pi / 8, counts,
               width=0.7, color=ok_color if key else bad_color, alpha=0.8)
        ax.set_title(f"executed action directions: {title} (0 = +x fwd)")
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(["180", "-90", "0 fwd", "+90", "180"])

    for ax in axes.flat:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Real robot runs 2026-07-21: successes (green) vs failures (red)", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "comparison.png", dpi=160)

    tiles = []
    for name, rec in records.items():
        pair = np.concatenate([rec["first_frame"], rec["goal_image"]], axis=1)
        img = Image.fromarray(pair)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        tag = f"{name[-6:]} {'OK' if rec['success'] else 'FAIL'} goal_cov={rec['goal_cov']:.0f}%"
        draw.text((4, 4), tag, fill=(255, 255, 0))
        tiles.append(np.asarray(img))
    Image.fromarray(np.concatenate(tiles, axis=0)).save(out_dir / "start_vs_goal_montage.png")

    header = (
        "| run | result | steps | latent gd first->last | box->goal first->best "
        "| net box motion | surprise vs baseline | coverage %ile run/goal | fwd/back actions |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    (out_dir / "comparison_table.md").write_text(header + "\n".join(lines) + "\n")
    print(header + "\n".join(lines))
    print(f"\nfigure: {out_dir / 'comparison.png'}")
    print(f"montage: {out_dir / 'start_vs_goal_montage.png'}")


if __name__ == "__main__":
    main()
