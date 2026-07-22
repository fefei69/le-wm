#!/usr/bin/env python3
"""D2: rank all 17 keyboard action tokens by CEM cost at clear-push states.

At states where the teleoperator was demonstrably pushing the box (contact
mask from the coverage analysis), rebuild the planner's exact latent context
(teacher-forced, like the deployed MPC), hold each of the 17 keyboard tokens
constant over the full horizon, and score every rollout with the planner's own
``_rollout_cost``.  If the world model + cost ranked actions sensibly, the
ground-truth push direction should beat its 180-degree opposite at these
states; the aggregate tells us how often the cost actively prefers backing
away from a push in progress.

Requires ``outputs/coverage/pushbox_pilot_train/box_track.npz`` (box contact
mask) and, for state selection only, episodes listed in the replay output are
not needed.  GPU cost: ~2 minutes.

Usage::

    python scripts/diag_token_ranking.py [--states 50]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stable_worldmodel as swm

from real_robot_planner import (
    CEMConfig,
    PushBoxPlanner,
    compute_training_action_stats,
    keyboard_action_vocabulary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY = 3
GOAL_OFFSET = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(REPO_ROOT / "stable-wm/datasets/pushbox_pilot_train.h5"))
    parser.add_argument("--checkpoint", default="pushbox/lewm/weights_epoch_146.pt")
    parser.add_argument("--box-track", type=Path,
                        default=REPO_ROOT / "outputs/coverage/pushbox_pilot_train/box_track.npz")
    parser.add_argument("--states", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/diagnostics/token_ranking.npz")
    return parser.parse_args()


def build_planner(args: argparse.Namespace) -> tuple[PushBoxPlanner, np.ndarray]:
    dataset = swm.data.load_dataset(
        args.dataset, transform=None, keys_to_load=["action"], keys_to_cache=["action"],
        num_steps=4, frameskip=1,
    )
    mean, std, val_episodes = compute_training_action_stats(dataset)
    config = CEMConfig(
        horizon=args.horizon, num_samples=1024, iterations=10, elite_count=32,
        action_cap_m=0.005, action_mode="keyboard",
    )
    model = swm.wm.utils.load_pretrained(args.checkpoint)
    planner = PushBoxPlanner(model, mean, std, config=config, device=args.device)
    return planner, val_episodes


def teacher_forced_context(
    planner: PushBoxPlanner, frames: np.ndarray, actions: np.ndarray
) -> None:
    """Feed frames[t-2..t] with executed actions, mirroring deployed MPC state."""
    planner.reset()
    embeddings = [planner._encode(frame) for frame in frames]
    planner._latent_context = torch.cat(embeddings, dim=1)
    normed = (
        torch.as_tensor(actions, dtype=torch.float32, device=planner.device).unsqueeze(0)
        - planner.action_mean
    ) / planner.action_std
    planner._executed_actions = normed


def select_states(
    f: h5py.File, val_episodes: np.ndarray, moving: np.ndarray, count: int, seed: int
) -> list[tuple[int, int]]:
    ep_offset, ep_len = f["ep_offset"][:], f["ep_len"][:]
    actions = f["action"][:]
    rng = np.random.default_rng(seed)
    pool = []
    for episode in val_episodes.tolist():
        offset, length = int(ep_offset[episode]), int(ep_len[episode])
        for t in range(HISTORY - 1, length - GOAL_OFFSET):
            row = offset + t
            history_ok = np.all(
                np.linalg.norm(actions[row - HISTORY + 1 : row], axis=1) <= 0.005 + 1e-7
            )
            gt_moves = np.linalg.norm(actions[row]) > 1e-6
            if moving[row] and history_ok and gt_moves:
                pool.append((episode, t))
    picks = rng.choice(len(pool), min(count, len(pool)), replace=False)
    return [pool[i] for i in np.sort(picks)]


def main() -> None:
    args = parse_args()
    moving_mask = build_moving_mask(args)
    planner, val_episodes = build_planner(args)
    vocab = keyboard_action_vocabulary(0.005)  # (17, 2): zero + 8 dirs x {2.5, 5} mm

    with h5py.File(args.dataset, "r") as f:
        states = select_states(f, val_episodes, moving_mask, args.states, args.seed)
        print(f"selected {len(states)} clear-push validation states")
        ep_offset = f["ep_offset"][:]
        all_costs, all_gt, all_meta = [], [], []
        plans = torch.as_tensor(vocab, dtype=torch.float32, device=planner.device)
        plans = plans.unsqueeze(1).repeat(1, args.horizon, 1)  # (17, H, 2)
        with torch.inference_mode():
            for episode, t in states:
                offset = int(ep_offset[episode])
                frames = f["pixels"][offset + t - HISTORY + 1 : offset + t + 1]
                actions = f["action"][offset + t - HISTORY + 1 : offset + t].astype(np.float32)
                goal = f["pixels"][offset + t + GOAL_OFFSET]
                teacher_forced_context(planner, frames, actions)
                goal_emb = planner._encode(goal)
                costs = planner._rollout_cost(
                    planner._latent_context, goal_emb, plans
                ).cpu().numpy()
                all_costs.append(costs)
                all_gt.append(f["action"][offset + t].astype(np.float32))
                all_meta.append((episode, t))

    costs = np.stack(all_costs)          # (N, 17)
    gt = np.stack(all_gt)                # (N, 2)
    meta = np.asarray(all_meta)

    directions = vocab[1:9] / np.linalg.norm(vocab[1:9], axis=1, keepdims=True)
    gt_unit = gt / np.linalg.norm(gt, axis=1, keepdims=True)
    gt_dir = np.argmax(gt_unit @ directions.T, axis=1)          # 0..7
    fwd_5mm = 1 + 8 + gt_dir                                    # token ids: 5 mm ring is 9..16
    back_5mm = 1 + 8 + (gt_dir + 4) % 8
    fwd_cost = costs[np.arange(len(costs)), fwd_5mm]
    back_cost = costs[np.arange(len(costs)), back_5mm]
    argmin_token = costs.argmin(axis=1)
    ranks = (costs < costs[np.arange(len(costs)), fwd_5mm][:, None]).sum(axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, costs=costs, gt_action=gt, meta=meta, vocab=vocab,
        fwd_token=fwd_5mm, back_token=back_5mm, argmin_token=argmin_token,
    )
    print(f"forward(5mm, GT direction) beats backward(5mm): "
          f"{(fwd_cost < back_cost).mean():.1%} of states (blind chance 50%)")
    print(f"cost argmin IS the GT-direction 5mm token: {(argmin_token == fwd_5mm).mean():.1%}")
    print(f"median rank of GT-direction token among 17 (0=best): {np.median(ranks):.1f}")
    spread = costs.std(axis=1) / costs.mean(axis=1)
    print(f"cost spread across tokens (std/mean): median {np.median(spread):.4f} "
          f"-> tiny values mean the cost barely distinguishes actions")
    print(f"saved: {args.output}")


def build_moving_mask(args: argparse.Namespace) -> np.ndarray:
    track = np.load(args.box_track)
    centroids = track["centroids"]
    moving = np.zeros(len(centroids), dtype=bool)
    deltas = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    valid = ~np.isnan(deltas)
    moving[1:][valid] = deltas[valid] > 1.0
    return moving


if __name__ == "__main__":
    main()
