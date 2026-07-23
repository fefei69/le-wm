"""Train a frozen-DINOv2, action/proprio-conditioned PushBox world model."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from dinowm_utils import compute_column_stats, select_nested_episode_subset
from utils import (
    SaveCkptCallback,
    ZScoreNormalizer,
    balanced_episode_split,
    get_img_preprocessor,
)


def _strip_feature_range(tensor, feature_range):
    """Remove one modality from the final feature dimension."""

    lo, hi = feature_range
    return torch.cat([tensor[..., :lo], tensor[..., hi:]], dim=-1)


def dinowm_forward(self, batch, stage, cfg):
    """Predict next DINO patches and proprioception from history and actions."""

    self.model.backbone.eval()
    for key in self.model.extra_encoders:
        value = torch.nan_to_num(batch[key], 0.0)
        if value.ndim == 4 and value.shape[-2] == 1:
            value = value.squeeze(-2)
        if value.ndim != 3:
            raise ValueError(
                f"Expected {key!r} with shape (B,T,D), got {tuple(value.shape)}"
            )
        batch[key] = value

    batch = self.model.encode(batch, target="emb", is_video=False)
    context = batch["emb"][:, : cfg.wm.history_size]
    prediction = self.model.predict(context)
    target = batch["emb"][:, cfg.wm.num_preds :].detach()

    pixels_dim = batch["pixels_emb"].shape[-1]
    batch["pixels_loss"] = F.mse_loss(
        prediction[..., :pixels_dim], target[..., :pixels_dim]
    )

    start = pixels_dim
    action_range = None
    for key in self.model.extra_encoders:
        width = batch[f"{key}_emb"].shape[-1]
        feature_range = (start, start + width)
        if key == "action":
            action_range = feature_range
        else:
            batch[f"{key}_loss"] = F.mse_loss(
                prediction[..., start : start + width],
                target[..., start : start + width],
            )
        start += width

    if action_range is None:
        raise ValueError("DINO-WM requires an action encoder")
    batch["actionless_emb"] = _strip_feature_range(batch["emb"], action_range)
    batch["actionless_prev_emb"] = _strip_feature_range(context, action_range)
    batch["actionless_pred_emb"] = _strip_feature_range(
        prediction, action_range
    )
    batch["actionless_target_emb"] = _strip_feature_range(target, action_range)
    batch["loss"] = F.mse_loss(
        batch["actionless_pred_emb"], batch["actionless_target_emb"]
    )
    if not torch.isfinite(batch["loss"]):
        raise ValueError("Non-finite DINO-WM loss encountered")

    losses = {
        f"{stage}/{key}": value.detach()
        for key, value in batch.items()
        if key == "loss" or key.endswith("_loss")
    }
    self.log_dict(
        losses,
        on_step=True,
        on_epoch=True,
        sync_dist=True,
        batch_size=batch["pixels"].shape[0],
    )
    return batch


def resolve_dataset_name(name: str) -> str:
    """Resolve explicit and repository-relative datasets before dispatch."""

    path = Path(name).expanduser()
    if path.is_absolute():
        return str(path)
    local_path = Path(hydra.utils.get_original_cwd(), path)
    return str(local_path.resolve()) if local_path.exists() else name


def _normalizer(dataset, column, episode_indices):
    stats = compute_column_stats(dataset, column, episode_indices)
    transform = spt.data.transforms.WrapTorchTransform(
        ZScoreNormalizer(stats.mean, stats.std), source=column, target=column
    )
    serializable = {
        "mean": stats.mean.squeeze(0).tolist(),
        "std": stats.std.squeeze(0).tolist(),
        "num_rows": stats.num_rows,
    }
    return transform, serializable


def _write_metadata(path: Path, cfg, manifest) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "training_config.yaml").open("w") as handle:
        OmegaConf.save(cfg, handle)
    with (path / "split_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


@hydra.main(
    version_base=None, config_path="./config/train", config_name="dinowm"
)
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    encoding_keys = list(cfg.wm.encoding.keys())
    dataset = swm.data.load_dataset(
        resolve_dataset_name(cfg.dataset_name),
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        keys_to_load=["pixels", *encoding_keys],
        keys_to_cache=encoding_keys,
    )

    episode_split = balanced_episode_split(
        dataset,
        train_fraction=cfg.train_split,
        seed=cfg.seed,
        search_trials=cfg.split_search_trials,
    )
    train_subset = select_nested_episode_subset(
        dataset,
        episode_split.train_episode_indices,
        target_hours=cfg.curve.train_hours,
        sample_hz=cfg.curve.sample_hz,
        seed=cfg.curve.subset_seed,
    )
    if len(train_subset.clip_indices) < cfg.batch_size:
        raise ValueError(
            f"Training subset has only {len(train_subset.clip_indices)} clips"
        )

    transforms = [get_img_preprocessor("pixels", "pixels", cfg.image_size)]
    normalization = {}
    for column in encoding_keys:
        transform, stats = _normalizer(
            dataset, column, train_subset.episode_indices
        )
        transforms.append(transform)
        normalization[column] = stats
    dataset.transform = spt.data.transforms.Compose(*transforms)

    validation_frames = int(
        np.asarray(dataset.lengths)[episode_split.val_episode_indices].sum()
    )
    manifest = {
        "dataset": str(cfg.dataset_name),
        "sample_hz": float(cfg.curve.sample_hz),
        "nominal_train_hours": float(cfg.curve.train_hours),
        "actual_train_hours": train_subset.hours,
        "train_frames": train_subset.num_frames,
        "train_clips": int(len(train_subset.clip_indices)),
        "train_episode_indices": train_subset.episode_indices.tolist(),
        "validation_hours": validation_frames
        / float(cfg.curve.sample_hz)
        / 3600.0,
        "validation_frames": validation_frames,
        "validation_clips": int(len(episode_split.val_clip_indices)),
        "validation_episode_indices": episode_split.val_episode_indices.tolist(),
        "split_seed": int(cfg.seed),
        "subset_seed": int(cfg.curve.subset_seed),
        "normalization": normalization,
    }
    print(
        "DINO-WM curve: "
        f"label={cfg.curve.label}, nominal={cfg.curve.train_hours:g}h, "
        f"actual_train={train_subset.hours:.3f}h/"
        f"{len(train_subset.episode_indices)} episodes/"
        f"{len(train_subset.clip_indices)} clips, "
        f"validation={manifest['validation_hours']:.3f}h/"
        f"{len(episode_split.val_episode_indices)} episodes/"
        f"{len(episode_split.val_clip_indices)} clips"
    )
    print(
        "Validation episode ids: "
        f"{episode_split.val_episode_indices.tolist()}"
    )
    if cfg.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set = torch.utils.data.Subset(
        dataset, train_subset.clip_indices.tolist()
    )
    val_set = torch.utils.data.Subset(
        dataset, episode_split.val_clip_indices.tolist()
    )
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )
    if cfg.num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=cfg.prefetch_factor,
            multiprocessing_context="spawn",
        )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **loader_kwargs,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, **loader_kwargs, shuffle=False, drop_last=False
    )

    encoder = hydra.utils.instantiate(cfg.model.encoder)
    encoder.eval().requires_grad_(False)
    if not hasattr(encoder.config, "hidden_size"):
        raise ValueError("Expected a ViT backbone with config.hidden_size")
    pixel_dim = int(encoder.config.hidden_size)
    num_patches = (cfg.image_size // cfg.patch_size) ** 2

    with open_dict(cfg):
        cfg.extra_dims = {
            key: (
                int(dataset.get_dim(key)) * cfg.frameskip
                if key == "action"
                else int(dataset.get_dim(key))
            )
            for key in encoding_keys
        }
        cfg.model.predictor.dim = pixel_dim + sum(cfg.wm.encoding.values())
        cfg.model.predictor.num_patches = num_patches
        cfg.model.extra_encoders = {
            "_target_": "torch.nn.ModuleDict",
            "modules": {
                key: {
                    "_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                    "in_chans": cfg.extra_dims[key],
                    "emb_dim": int(cfg.wm.encoding[key]),
                }
                for key in encoding_keys
            },
        }

    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in world_model.parameters()
        if parameter.requires_grad
    )
    predictor_parameters = sum(
        parameter.numel() for parameter in world_model.predictor.parameters()
    )
    manifest["pixel_embedding_dim"] = pixel_dim
    manifest["predictor_input_dim"] = int(cfg.model.predictor.dim)
    manifest["num_patches"] = int(num_patches)
    manifest["predictor_parameters"] = predictor_parameters
    manifest["trainable_parameters"] = trainable_parameters
    print(
        f"Model parameters: predictor={predictor_parameters:,}; "
        f"total trainable={trainable_parameters:,}"
    )
    if not 15_000_000 <= trainable_parameters <= 30_000_000:
        raise ValueError(
            "Expected the moderate DINO-WM predictor scale to have 15M-30M "
            f"trainable parameters, got {trainable_parameters:,}"
        )

    checkpoint_root = Path(
        swm.data.utils.get_cache_dir(sub_folder="checkpoints")
    )
    model_dir = checkpoint_root / cfg.output_model_name
    run_dir = checkpoint_root / cfg.subdir
    _write_metadata(model_dir, cfg, manifest)
    _write_metadata(run_dir, cfg, manifest)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    else:
        logger = CSVLogger(save_dir=str(run_dir / "logs"), name="csv")

    module = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            "model_opt": {
                "modules": "model",
                "optimizer": dict(cfg.optimizer),
            }
        },
    )
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.setdefault("default_root_dir", str(run_dir))
    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name,
                cfg=cfg.model,
                epoch_interval=cfg.checkpoint_interval,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        seed=cfg.seed,
    )
    manager()


if __name__ == "__main__":
    run()
