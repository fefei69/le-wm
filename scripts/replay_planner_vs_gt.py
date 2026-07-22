#!/usr/bin/env python3
"""Replay the real-robot CEM planner over dataset episodes and log its actions.

MPC-style offline replay on the held-out validation episodes of
``pushbox_pilot_train.h5``: at every step the planner sees the real recorded
frame, replans with exactly the ``scripts/eval_pushbox_real.sh`` configuration
(keyboard vocabulary, horizon 10, action cap 0.005 m, 1024 samples, 10 CEM
iterations, elite 32, epoch-146 checkpoint), and its chosen first action is
logged next to the teleoperator's ground-truth action.  The latent context is
teacher-forced with the ground-truth executed actions, mirroring what the
robot would have experienced along the same trajectory.

Ground-truth actions with |a| > cap (the 0.010 m keyboard magnitude) cannot be
fed back as ``previous_action`` — the planner validates against its cap — so
the context is reset there and the step is flagged ``fresh``, like the start
of a deployment run.

Goal modes: ``sliding`` (goal = frame t+offset, clamped to episode end) and
``fixed`` (goal = last frame of the episode).  Results are written
incrementally, one ``ep_XXX.npz`` per episode, so interrupted runs resume.

Usage::

    python scripts/replay_planner_vs_gt.py --goal-mode sliding
    python scripts/replay_planner_vs_gt.py --goal-mode fixed
    python scripts/replay_planner_vs_gt.py --goal-mode sliding --max-episodes 1 --max-steps 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stable_worldmodel as swm

from real_robot_planner import (
    CEMConfig,
    PushBoxPlanner,
    compute_training_action_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-mode", choices=("sliding", "fixed"), required=True)
    parser.add_argument("--goal-offset", type=int, default=25, help="sliding-goal lookahead in steps")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "stable-wm/datasets/pushbox_pilot_train.h5"))
    parser.add_argument("--checkpoint", default="pushbox/lewm/weights_epoch_146.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/planner_vs_gt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=None, help="debug: limit episode count")
    parser.add_argument("--max-steps", type=int, default=None, help="debug: limit steps per episode")
    return parser.parse_args()


def build_planner(args: argparse.Namespace) -> tuple[PushBoxPlanner, np.ndarray]:
    # Mirrors PushBoxPlanner.from_artifacts, but also returns the validation
    # episode indices so the replay can restrict itself to held-out data.
    dataset = swm.data.load_dataset(
        args.dataset,
        transform=None,
        keys_to_load=["action"],
        keys_to_cache=["action"],
        num_steps=4,
        frameskip=1,
    )
    mean, std, val_episodes = compute_training_action_stats(dataset)
    print(f"validation episodes: {val_episodes.tolist()}")
    config = CEMConfig(
        horizon=10,
        num_samples=1024,
        iterations=10,
        elite_count=32,
        action_cap_m=0.005,
        action_mode="keyboard",
    )
    model = swm.wm.utils.load_pretrained(args.checkpoint)
    planner = PushBoxPlanner(model, mean, std, config=config, device=args.device)
    return planner, val_episodes


def replay_episode(
    planner: PushBoxPlanner,
    pixels: np.ndarray,
    actions: np.ndarray,
    goal_mode: str,
    goal_offset: int,
    max_steps: int | None,
) -> dict[str, np.ndarray]:
    length = len(pixels)
    cap = planner.config.action_cap_m
    steps = range(length - 1) if max_steps is None else range(min(length - 1, max_steps))
    records: dict[str, list] = {k: [] for k in (
        "t", "gt_action", "planned_action", "cost", "goal_distance", "fresh", "solve_time_s",
    )}
    fresh = True
    planner.reset()
    for t in steps:
        if t > 0:
            previous = actions[t - 1]
            if float(np.linalg.norm(previous)) > cap + 1e-7:
                planner.reset()
                fresh = True
        goal_index = min(t + goal_offset, length - 1) if goal_mode == "sliding" else length - 1
        result = planner.plan(
            pixels[t],
            pixels[goal_index],
            previous_action=None if fresh else actions[t - 1],
        )
        records["t"].append(t)
        records["gt_action"].append(actions[t])
        records["planned_action"].append(result["action"])
        records["cost"].append(result["cost"])
        records["goal_distance"].append(result["goal_distance"])
        records["fresh"].append(fresh)
        records["solve_time_s"].append(result["solve_time_s"])
        fresh = bool(result.get("at_goal", False))  # planner self-resets at goal
    return {key: np.asarray(value) for key, value in records.items()}


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / args.goal_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    planner, val_episodes = build_planner(args)
    episodes = val_episodes.tolist()
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    with h5py.File(args.dataset, "r") as f:
        ep_offset, ep_len = f["ep_offset"][:], f["ep_len"][:]
        started = time.monotonic()
        for n, episode in enumerate(episodes):
            target = out_dir / f"ep_{episode:03d}.npz"
            if target.exists():
                print(f"[{n + 1}/{len(episodes)}] ep {episode}: cached, skipping", flush=True)
                continue
            sel = slice(int(ep_offset[episode]), int(ep_offset[episode] + ep_len[episode]))
            pixels = f["pixels"][sel]
            actions = f["action"][sel].astype(np.float32)
            with torch.inference_mode():
                records = replay_episode(
                    planner, pixels, actions, args.goal_mode, args.goal_offset, args.max_steps
                )
            np.savez_compressed(target, **records)
            done = len(records["t"])
            rate = records["solve_time_s"].mean()
            print(
                f"[{n + 1}/{len(episodes)}] ep {episode}: {done} plans, "
                f"{rate:.2f}s/plan, elapsed {time.monotonic() - started:.0f}s",
                flush=True,
            )
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
