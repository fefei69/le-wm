#!/usr/bin/env python3
"""Upload completed PushBox DINO-WM CSV histories to Weights & Biases."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import wandb


RUNS = (
    {
        "label": "1h",
        "job_id": "14576573",
        "run_id": "dinowm1h14576573",
        "registry_dir": Path(
            "/scratch/cw5167/.cache/stable-pretraining/runs/20260722/"
            "144803/30dfa5bfa71e"
        ),
    },
    {
        "label": "2h",
        "job_id": "14576574",
        "run_id": "dinowm2h14576574",
        "registry_dir": Path(
            "/scratch/cw5167/.cache/stable-pretraining/runs/20260722/"
            "144814/60b6c7c67357"
        ),
    },
    {
        "label": "4h",
        "job_id": "14576575",
        "run_id": "dinowm4h14576575",
        "registry_dir": Path(
            "/scratch/cw5167/.cache/stable-pretraining/runs/20260722/"
            "144825/c30772893593"
        ),
    },
)

LOSS_METRICS = {
    "fit/loss_step",
    "fit/pixels_loss_step",
    "fit/proprio_loss_step",
    "fit/loss_epoch",
    "fit/pixels_loss_epoch",
    "fit/proprio_loss_epoch",
    "validate/loss_step",
    "validate/pixels_loss_step",
    "validate/proprio_loss_step",
    "validate/loss_epoch",
    "validate/pixels_loss_epoch",
    "validate/proprio_loss_epoch",
}


def read_history(path: Path) -> list[tuple[int, dict[str, float | int]]]:
    """Merge sparse RegistryLogger rows sharing the same optimizer step."""

    by_step: dict[int, dict[str, float | int]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("step"):
                continue
            step = int(float(row["step"]))
            record = by_step.setdefault(step, {})
            for key, raw_value in row.items():
                if key not in LOSS_METRICS | {"epoch"} or not raw_value:
                    continue
                value = float(raw_value)
                if not math.isfinite(value):
                    continue
                record[key] = int(value) if key == "epoch" else value
    return sorted(by_step.items())


def latest_metric(history, key: str) -> float | None:
    for _, record in reversed(history):
        if key in record:
            return float(record[key])
    return None


def upload_one(spec, args) -> str:
    label = spec["label"]
    checkpoint_dir = Path(
        f"/scratch/cw5167/stable-wm/checkpoints/pushbox/"
        f"dinowm_dinov2s_prop_{label}"
    )
    metrics_path = spec["registry_dir"] / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics for {label}: {metrics_path}")

    history = read_history(metrics_path)
    config = {
        "model_family": "DINO-WM",
        "curve_label": label,
        "loss_space": "DINOv2 patch features",
    }
    notes = (
        "Loss history imported from the completed run. DINO feature MSE and "
        "LeWM latent MSE use different representations/scales, so compare "
        "trends rather than raw loss magnitudes."
    )
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=spec["run_id"],
        name=f"pushbox-dinowm-dinov2s-prop-{label}",
        group="pushbox-dinowm-learning-curves",
        job_type="world-model-training",
        tags=[
            "pushbox",
            "dinowm",
            "dinov2-small",
            "proprio",
            "learning-curve",
            "csv-import",
            label,
        ],
        notes=notes,
        config=config,
        resume="allow",
        reinit="finish_previous",
        dir=args.wandb_dir,
        save_code=False,
        settings=wandb.Settings(
            disable_code=True,
            disable_git=True,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_machine_info=True,
        ),
    )
    for step, record in history:
        run.log(record, step=step)

    for key in (
        "validate/loss_epoch",
        "validate/pixels_loss_epoch",
        "validate/proprio_loss_epoch",
        "fit/loss_epoch",
        "fit/pixels_loss_epoch",
        "fit/proprio_loss_epoch",
    ):
        value = latest_metric(history, key)
        if value is not None:
            run.summary[f"final/{key}"] = value
    run.summary["comparison_warning"] = (
        "Raw DINO feature MSE is not numerically comparable to LeWM latent MSE."
    )
    url = run.url
    run.finish()
    return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="cw5167-nyu")
    parser.add_argument("--project", default="le-wm")
    parser.add_argument("--wandb-dir", default="/tmp/cw5167-dinowm-wandb")
    args = parser.parse_args()
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)

    for spec in RUNS:
        url = upload_one(spec, args)
        print(f"{spec['label']}: {url}")


if __name__ == "__main__":
    main()
