#!/usr/bin/env python3
"""D1: action-shuffle sensitivity of the PushBox world model (paper metric).

Implements the "action shuffle ratio" from *Hallucination in World Models is
Predictable and Preventable* (evaluation metric iii): one-step teacher-forced
prediction MSE with shuffled actions relative to true actions.  A ratio of
~1.0 means the dynamics model ignores actions (action-marginalized
hallucination); the paper flags ratios <= 1.1.

Extensions beyond the paper, aimed at the observed backward-action failures:

- negated-action ratio: can the model distinguish a push from its reverse?
- zero-action ratio: does replacing the true action with "stay" even matter?
- per-magnitude breakdown of the final context action (2.5/5/10 mm tokens).

Runs on the held-out validation episodes only.  GPU cost: ~1 minute.

Usage::

    python scripts/diag_action_sensitivity.py [--windows 2000]
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

import stable_pretraining as spt
import stable_worldmodel as swm
from torchvision.transforms import v2

from real_robot_planner import compute_training_action_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(REPO_ROOT / "stable-wm/datasets/pushbox_pilot_train.h5"))
    parser.add_argument("--checkpoint", default="pushbox/lewm/weights_epoch_146.pt")
    parser.add_argument("--windows", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/diagnostics/action_sensitivity.json")
    return parser.parse_args()


def build_transform() -> v2.Compose:
    stats = spt.data.dataset_stats.ImageNet
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=stats["mean"], std=stats["std"]),
        v2.Resize((224, 224)),
    ])


@torch.inference_mode()
def predict_batch(model, ctx_emb: torch.Tensor, actions_norm: torch.Tensor) -> torch.Tensor:
    """One-step prediction from a 3-frame context: returns predicted z[t+3]."""
    act_emb = model.action_encoder(actions_norm)
    return model.predict(ctx_emb, act_emb)[:, -1]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    dataset = swm.data.load_dataset(
        args.dataset, transform=None, keys_to_load=["action"], keys_to_cache=["action"],
        num_steps=4, frameskip=1,
    )
    mean, std, val_episodes = compute_training_action_stats(dataset)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=args.device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=args.device)

    model = swm.wm.utils.load_pretrained(args.checkpoint).to(args.device).eval()
    model.requires_grad_(False)
    transform = build_transform()

    with h5py.File(args.dataset, "r") as f:
        ep_offset, ep_len = f["ep_offset"][:], f["ep_len"][:]
        candidates = []
        for episode in val_episodes.tolist():
            offset, length = int(ep_offset[episode]), int(ep_len[episode])
            for t in range(length - HISTORY):
                candidates.append(offset + t)
        picks = np.sort(rng.choice(len(candidates), min(args.windows, len(candidates)), replace=False))
        starts = np.asarray(candidates)[picks]

        errors: dict[str, list] = {k: [] for k in (
            "true", "shuffled", "negated", "zeroed", "repeat_last",
        )}
        last_action_raw: list[np.ndarray] = []
        for chunk_start in range(0, len(starts), args.batch):
            chunk = starts[chunk_start : chunk_start + args.batch]
            frames = np.stack([f["pixels"][s : s + HISTORY + 1] for s in chunk])
            raw_actions = np.stack([f["action"][s : s + HISTORY] for s in chunk]).astype(np.float32)
            raw_actions = np.nan_to_num(raw_actions, nan=0.0)
            pixels = torch.stack([
                torch.stack([transform(img) for img in window]) for window in frames
            ]).to(args.device)
            emb = model.encode({"pixels": pixels})["emb"]  # (B, 4, D)
            ctx_emb, target = emb[:, :HISTORY], emb[:, HISTORY]
            actions = torch.as_tensor(raw_actions, device=args.device)
            normed = (actions - mean_t) / std_t

            variants = {
                "true": normed,
                "shuffled": normed[torch.randperm(len(normed), device=args.device)],
                "negated": (-actions - mean_t) / std_t,
                "zeroed": (torch.zeros_like(actions) - mean_t) / std_t,
            }
            for name, acts in variants.items():
                prediction = predict_batch(model, ctx_emb, acts)
                errors[name].append(
                    (prediction - target).square().mean(dim=-1).cpu().numpy()
                )
            errors["repeat_last"].append(
                (ctx_emb[:, -1] - target).square().mean(dim=-1).cpu().numpy()
            )
            last_action_raw.append(raw_actions[:, -1])

    error_arrays = {k: np.concatenate(v) for k, v in errors.items()}
    last_action = np.concatenate(last_action_raw)
    magnitudes = np.linalg.norm(last_action, axis=1)
    true_err = error_arrays["true"]

    def ratio(name: str, mask: np.ndarray | None = None) -> float:
        sel = slice(None) if mask is None else mask
        return float(error_arrays[name][sel].mean() / true_err[sel].mean())

    report = {
        "windows": int(len(true_err)),
        "one_step_mse_true": float(true_err.mean()),
        "one_step_mse_repeat_last_frame": float(error_arrays["repeat_last"].mean()),
        "shuffle_ratio": ratio("shuffled"),
        "negate_ratio": ratio("negated"),
        "zero_ratio": ratio("zeroed"),
        "paper_threshold_actions_ignored": 1.1,
        "per_final_action_magnitude": {},
    }
    bands = {"stay(<1mm)": magnitudes < 1e-4, "2.5mm": np.isclose(magnitudes, 0.0025, atol=5e-4),
             "5mm": np.isclose(magnitudes, 0.005, atol=5e-4), "10mm": np.isclose(magnitudes, 0.010, atol=1e-3)}
    for band, mask in bands.items():
        if mask.sum() >= 20:
            report["per_final_action_magnitude"][band] = {
                "windows": int(mask.sum()),
                "shuffle_ratio": ratio("shuffled", mask),
                "negate_ratio": ratio("negated", mask),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    verdict = "ACTIONS LARGELY IGNORED" if report["shuffle_ratio"] <= 1.1 else "actions are used"
    print(f"\nshuffle ratio {report['shuffle_ratio']:.3f} -> {verdict} (paper flags <= 1.1)")


if __name__ == "__main__":
    main()
