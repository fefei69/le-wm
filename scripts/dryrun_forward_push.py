#!/usr/bin/env python3
"""Offline dry-run of the forward-push robot test.

For each strictly-forward push segment extracted by the coverage analysis
(``outputs/forward_push/forward_segments.json``), simulate the start of a real
trial exactly as ``eval_pushbox_real.sh`` would run it: fresh planner context,
current = segment start frame, goal = segment end frame, full CEM with the
deployed settings (keyboard vocabulary, horizon 10, cap 0.005, 1024 samples,
10 iterations).  Reports whether the first planned action points forward (+x),
i.e. whether the planner would begin pushing the box the right way.

Usage::

    python scripts/dryrun_forward_push.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diag_token_ranking import build_planner

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(REPO_ROOT / "stable-wm/datasets/pushbox_pilot_train.h5"))
    parser.add_argument("--checkpoint", default="pushbox/lewm/weights_epoch_146.pt")
    parser.add_argument("--segments", type=Path, default=REPO_ROOT / "outputs/forward_push/forward_segments.json")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/forward_push/dryrun_results.json")
    return parser.parse_args()


def classify(action: np.ndarray) -> str:
    if np.linalg.norm(action) < 1e-6:
        return "stay"
    angle = np.degrees(np.arctan2(action[1], action[0]))
    if abs(angle) <= 22.5:
        return "forward(+x)"
    if abs(angle) >= 157.5:
        return "backward(-x)"
    return "lateral" if abs(abs(angle) - 90) <= 22.5 else (
        "diag-fwd" if abs(angle) < 90 else "diag-back"
    )


def main() -> None:
    args = parse_args()
    segments = json.loads(args.segments.read_text())
    planner, _ = build_planner(args)

    results = []
    with h5py.File(args.dataset, "r") as f:
        ep_offset = f["ep_offset"][:]
        for seg in segments:
            offset = int(ep_offset[seg["episode"]])
            current = f["pixels"][offset + seg["t_start"]]
            goal = f["pixels"][offset + seg["t_end"]]
            planner.reset()
            with torch.inference_mode():
                result = planner.plan(current, goal)
            action = np.asarray(result["action"], dtype=np.float64)
            plan = np.asarray(result["plan"], dtype=np.float64)
            results.append({
                "segment": f"ep{seg['episode']:03d}_t{seg['t_start']:03d}",
                "net_px": seg["net_px"],
                "first_action": action.tolist(),
                "first_class": classify(action),
                "plan_mean_x": float(plan[:, 0].mean()),
                "plan_fwd_steps": int(sum(classify(a) in ("forward(+x)", "diag-fwd") for a in plan)),
                "goal_distance": result["goal_distance"],
                "cost": result["cost"],
            })
            print(f"{results[-1]['segment']}: first={results[-1]['first_class']:12s} "
                  f"a={action.round(4).tolist()} plan_fwd {results[-1]['plan_fwd_steps']}/{len(plan)} "
                  f"gd={result['goal_distance']:.3f}")

    classes = [r["first_class"] for r in results]
    fwd = sum(c in ("forward(+x)", "diag-fwd") for c in classes)
    back = sum(c in ("backward(-x)", "diag-back") for c in classes)
    print(f"\nfirst action forward-ish: {fwd}/{len(results)}  backward-ish: {back}/{len(results)}")
    print(f"mean planned x over horizon > 0 (net forward plan): "
          f"{sum(r['plan_mean_x'] > 0 for r in results)}/{len(results)}")
    args.output.write_text(json.dumps(results, indent=1) + "\n")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
