#!/usr/bin/env python3
"""Compare replayed planner actions against ground-truth teleop actions.

Consumes the per-episode ``ep_XXX.npz`` files written by
``scripts/replay_planner_vs_gt.py`` and renders, per goal mode:

- a 9-class direction confusion matrix (stay + 8 keyboard directions) between
  the teleoperator's action and the planner's replanned action at that step,
- the angular-error distribution over steps where both chose to move,
- direction accuracy binned by latent goal distance,
- example frames with both actions drawn as arrows.

Ground-truth 0.010 m actions exceed the planner's 0.005 m cap; they keep their
direction class here (magnitude comparisons exclude them).

Usage::

    python scripts/analyze_planner_vs_gt.py [--results outputs/planner_vs_gt]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]

# angular order: stay, then 8 directions counter-clockwise from +x
DIRECTION_LABELS = ["stay", "→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
STAY_EPS_M = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "outputs/planner_vs_gt")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data/pushbox_pilot_train.h5"))
    parser.add_argument("--modes", nargs="+", default=["sliding", "fixed"])
    parser.add_argument("--examples", type=int, default=8, help="example frames per mode")
    return parser.parse_args()


def direction_class(actions: np.ndarray) -> np.ndarray:
    """0 = stay, 1..8 = 45-degree sectors counter-clockwise from +x."""
    norms = np.linalg.norm(actions, axis=-1)
    sectors = np.round(np.arctan2(actions[..., 1], actions[..., 0]) / (np.pi / 4)).astype(int) % 8
    return np.where(norms < STAY_EPS_M, 0, sectors + 1)


def angular_error_deg(gt: np.ndarray, planned: np.ndarray) -> np.ndarray:
    delta = np.abs(direction_class(gt) - direction_class(planned)) % 8
    return np.minimum(delta, 8 - delta) * 45.0


def load_mode(mode_dir: Path) -> dict[str, np.ndarray]:
    files = sorted(mode_dir.glob("ep_*.npz"))
    if not files:
        raise FileNotFoundError(f"no replay results in {mode_dir}")
    merged: dict[str, list] = {}
    for file in files:
        data = np.load(file)
        episode = int(file.stem.split("_")[1])
        count = len(data["t"])
        for key in data.files:
            merged.setdefault(key, []).append(data[key])
        merged.setdefault("episode", []).append(np.full(count, episode))
    result = {key: np.concatenate(value) for key, value in merged.items()}
    result["n_episodes"] = np.int64(len(files))
    return result


def confusion(gt_cls: np.ndarray, plan_cls: np.ndarray) -> np.ndarray:
    matrix = np.zeros((9, 9), dtype=np.int64)
    np.add.at(matrix, (gt_cls, plan_cls), 1)
    return matrix


def draw_confusion(ax, matrix: np.ndarray, title: str) -> None:
    rows = matrix.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(matrix / rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            if matrix[i, j]:
                share = matrix[i, j] / rows[i, 0]
                color = "white" if share > 0.55 else "#39485e"
                ax.text(j, i, f"{share:.0%}" if share >= 0.005 else "<1%",
                        ha="center", va="center", fontsize=6.5, color=color)
    ax.set_xticks(range(9), DIRECTION_LABELS)
    ax.set_yticks(range(9), DIRECTION_LABELS)
    ax.set_xlabel("planner action")
    ax.set_ylabel("ground-truth action")
    ax.set_title(title)


def pick_examples(
    gt: np.ndarray, planned: np.ndarray, moving: np.ndarray, count: int
) -> np.ndarray:
    """Balanced sample of moving steps: half agreements, half disagreements."""
    errors = angular_error_deg(gt, planned)
    agree = np.nonzero(moving & (errors == 0))[0]
    disagree = np.nonzero(moving & (errors >= 90))[0]
    rng = np.random.default_rng(0)
    half = count // 2
    picks = []
    for pool, want in ((agree, half), (disagree, count - half)):
        if len(pool):
            picks.append(rng.choice(pool, min(want, len(pool)), replace=False))
    return np.sort(np.concatenate(picks))


def draw_example(ax, frame: np.ndarray, gt: np.ndarray, planned: np.ndarray, caption: str) -> None:
    ax.imshow(frame)
    center = np.array([112.0, 190.0])
    for vector, color, width in ((gt, "#2e7d4f", 3.0), (planned, "#c56a4c", 1.8)):
        norm = np.linalg.norm(vector)
        if norm < STAY_EPS_M:
            ax.plot(*center, "o", mfc="none", mec=color, ms=10, mew=width)
            continue
        # dataset frame: +x is away from the camera (up in image), +y is left
        arrow = np.array([-vector[1], -vector[0]]) / norm * 36.0
        ax.annotate("", xy=center + arrow, xytext=center,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=width))
    ax.set_title(caption, fontsize=7)
    ax.set_axis_off()


def main() -> None:
    args = parse_args()
    ink, muted = "#39485e", "#7a8699"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.edgecolor": muted, "axes.labelcolor": ink,
        "text.color": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    dataset = h5py.File(args.dataset, "r")
    ep_offset = dataset["ep_offset"][:]
    report_lines = []

    for mode in args.modes:
        data = load_mode(args.results / mode)
        gt, planned = data["gt_action"], data["planned_action"]
        gt_cls, plan_cls = direction_class(gt), direction_class(planned)
        moving = (gt_cls != 0) & (plan_cls != 0)
        errors = angular_error_deg(gt, planned)
        gt_over_cap = np.linalg.norm(gt, axis=1) > 0.005 + 1e-7

        direction_acc = float((gt_cls == plan_cls).mean())
        marginal_chance = float(
            (np.bincount(gt_cls, minlength=9) / len(gt_cls)
             * np.bincount(plan_cls, minlength=9) / len(plan_cls)).sum()
        )
        kappa = (direction_acc - marginal_chance) / (1.0 - marginal_chance)
        within45 = float((errors[moving] <= 45).mean())
        both_mag = moving & ~gt_over_cap
        mag_agree = float(
            np.isclose(
                np.linalg.norm(gt[both_mag], axis=1),
                np.linalg.norm(planned[both_mag], axis=1),
                rtol=0.05,
            ).mean()
        )
        summary = (
            f"[{mode}] episodes {int(data['n_episodes'])}, steps {len(gt)}: "
            f"direction accuracy {direction_acc:.1%} (chance {marginal_chance:.1%}, "
            f"kappa {kappa:.2f}); within 45 deg when both move {within45:.1%}; "
            f"gt stay-rate {float((gt_cls == 0).mean()):.1%} vs planner {float((plan_cls == 0).mean()):.1%}; "
            f"magnitude match when direction-comparable {mag_agree:.1%}; "
            f"gt over cap {float(gt_over_cap.mean()):.1%} of steps"
        )
        report_lines.append(summary)
        print(summary)

        fig = plt.figure(figsize=(13, 7.6))
        grid = fig.add_gridspec(2, 4, height_ratios=(1.35, 1.0))
        fig.suptitle(
            f"Planner vs ground truth — {mode} goal "
            f"({int(data['n_episodes'])} validation episodes, {len(gt)} replans)",
            color=ink,
        )

        ax = fig.add_subplot(grid[0, 0:2])
        draw_confusion(ax, confusion(gt_cls, plan_cls), "direction confusion (row-normalized)")

        ax = fig.add_subplot(grid[0, 2])
        bins = np.arange(-22.5, 181, 45)
        ax.hist(errors[moving], bins=bins, color="#3d6bb0", alpha=0.85)
        ax.set_xticks((0, 45, 90, 135, 180))
        ax.set_title("angular error, both moving")
        ax.set_xlabel("degrees")
        ax.set_ylabel("steps")

        ax = fig.add_subplot(grid[0, 3])
        distance = data["goal_distance"]
        edges = np.quantile(distance, np.linspace(0, 1, 9))
        centers, accuracy = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (distance >= lo) & (distance <= hi)
            if mask.sum() > 20:
                centers.append(0.5 * (lo + hi))
                accuracy.append((gt_cls[mask] == plan_cls[mask]).mean())
        ax.plot(centers, accuracy, "o-", color="#3d6bb0", ms=4)
        ax.axhline(marginal_chance, color=muted, lw=1, ls="--")
        ax.set_title("direction accuracy vs goal distance")
        ax.set_xlabel("latent MSE to goal")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)

        picks = pick_examples(gt, planned, moving, 4)
        for slot, index in enumerate(picks[:4]):
            ax = fig.add_subplot(grid[1, slot])
            episode, t = int(data["episode"][index]), int(data["t"][index])
            frame = dataset["pixels"][int(ep_offset[episode]) + t]
            caption = (
                f"ep {episode} t={t}  gt {DIRECTION_LABELS[gt_cls[index]]} "
                f"vs plan {DIRECTION_LABELS[plan_cls[index]]} ({errors[index]:.0f}°)"
            )
            draw_example(ax, frame, gt[index], planned[index], caption)
        handles = [
            plt.Line2D([], [], color="#2e7d4f", lw=2.5, label="ground truth"),
            plt.Line2D([], [], color="#c56a4c", lw=1.8, label="planner"),
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

        for ax in fig.axes:
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = args.results / f"planner_vs_gt_{mode}.png"
        fig.savefig(out, dpi=160)
        print(f"figure: {out}")

    (args.results / "summary.txt").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
