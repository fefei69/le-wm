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
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from dinowm_utils import select_episode_fraction
from module import SIGReg
from utils import (
    SaveCkptCallback,
    balanced_episode_split,
    column_zscore_stats,
    get_column_normalizer,
    get_img_preprocessor,
    matched_step_epochs,
)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def resolve_dataset_name(name: str) -> str:
    """Resolve repository-local datasets before stable-worldmodel dispatches.

    ``stable_worldmodel.data.load_dataset`` resolves relative names below its
    own dataset cache.  For experiment configs it is convenient to keep local
    files under this repository's ``data/`` directory, so prefer a path
    relative to Hydra's original working directory when that path exists.
    Remote dataset identifiers remain unchanged.
    """
    path = Path(name).expanduser()
    if path.is_absolute():
        return str(path)

    local_path = Path(hydra.utils.get_original_cwd(), path)
    if local_path.exists():
        return str(local_path.resolve())
    return name

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = resolve_dataset_name(dataset_cfg.pop("name"))
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    episode_split = None
    normalizer_episodes = None
    if cfg.data.get("split_by_episode", False):
        episode_split = balanced_episode_split(
            dataset,
            train_fraction=cfg.train_split,
            seed=cfg.seed,
            search_trials=cfg.data.get("split_search_trials", 50_000),
        )
        normalizer_episodes = episode_split.train_episode_indices
        print(
            "Episode-grouped split: "
            f"train={len(episode_split.train_episode_indices)} episodes/"
            f"{len(episode_split.train_clip_indices)} clips, "
            f"validation={len(episode_split.val_episode_indices)} episodes/"
            f"{len(episode_split.val_clip_indices)} clips "
            f"({len(episode_split.val_clip_indices) / len(dataset):.2%}); "
            f"validation episode ids={episode_split.val_episode_indices.tolist()}"
        )

    # Optional data-scaling subset. The selector, seed, and episode split match
    # train_dinowm.py, so a given fraction holds the same episodes for both
    # architectures and the fractions remain nested.
    curve = cfg.get("curve")
    train_fraction = None if curve is None else curve.get("train_fraction")
    train_subset = None
    if train_fraction is not None:
        if episode_split is None:
            raise ValueError(
                "curve.train_fraction requires data.split_by_episode=true"
            )
        train_subset = select_episode_fraction(
            dataset,
            episode_split.train_episode_indices,
            fraction=float(train_fraction),
            sample_hz=curve.sample_hz,
            seed=curve.subset_seed,
        )
        if len(train_subset.clip_indices) < cfg.loader.batch_size:
            raise ValueError(
                f"Training subset has only {len(train_subset.clip_indices)} "
                f"clips, fewer than one batch of {cfg.loader.batch_size}"
            )
        normalizer_episodes = train_subset.episode_indices
        print(
            f"Data-scaling subset: fraction={float(train_fraction):.4g}, "
            f"train={train_subset.hours:.3f}h/"
            f"{len(train_subset.episode_indices)} episodes/"
            f"{len(train_subset.clip_indices)} clips"
        )

    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(
                dataset, col, col, episode_indices=normalizer_episodes
            )
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    if episode_split is None:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
    else:
        train_clip_indices = (
            episode_split.train_clip_indices
            if train_subset is None
            else train_subset.clip_indices
        )
        train_set = torch.utils.data.Subset(dataset, train_clip_indices.tolist())
        val_set = torch.utils.data.Subset(
            dataset, episode_split.val_clip_indices.tolist()
        )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    max_epochs = int(cfg.trainer.max_epochs)
    if train_subset is not None and curve.get("match_optimizer_steps", False):
        max_epochs = matched_step_epochs(
            base_epochs=cfg.trainer.max_epochs,
            subset_clips=len(train_subset.clip_indices),
            full_clips=len(episode_split.train_clip_indices),
            batch_size=cfg.loader.batch_size,
        )
        print(
            f"Matching optimizer steps: {cfg.trainer.max_epochs} epochs on the "
            f"full split -> {max_epochs} epochs on this subset"
        )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    if train_subset is not None:
        validation_frames = int(
            np.asarray(dataset.lengths)[
                episode_split.val_episode_indices
            ].sum()
        )
        normalization = {}
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            mean, std, num_rows = column_zscore_stats(
                dataset, col, normalizer_episodes
            )
            normalization[col] = {
                "mean": mean.squeeze(0).tolist(),
                "std": std.squeeze(0).tolist(),
                "num_rows": num_rows,
            }
        manifest = {
            "dataset": str(dataset_name),
            "sample_hz": float(curve.sample_hz),
            "train_fraction": float(train_fraction),
            "actual_train_hours": train_subset.hours,
            "train_frames": train_subset.num_frames,
            "train_clips": int(len(train_subset.clip_indices)),
            "train_episode_indices": train_subset.episode_indices.tolist(),
            "validation_frames": validation_frames,
            "validation_clips": int(len(episode_split.val_clip_indices)),
            "validation_episode_indices": (
                episode_split.val_episode_indices.tolist()
            ),
            "split_seed": int(cfg.seed),
            "subset_seed": int(curve.subset_seed),
            "base_max_epochs": int(cfg.trainer.max_epochs),
            "max_epochs": max_epochs,
            "match_optimizer_steps": bool(
                curve.get("match_optimizer_steps", False)
            ),
            "optimizer_steps": max_epochs
            * (len(train_subset.clip_indices) // cfg.loader.batch_size),
            "normalization": normalization,
        }
        for directory in (
            run_dir,
            Path(
                swm.data.utils.get_cache_dir(sub_folder="checkpoints"),
                cfg.output_model_name,
            ),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            with open(directory / "split_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=cfg.get("checkpoint_interval", 1),
    )

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg["max_epochs"] = max_epochs
    # Anchor the trainer at this run's own directory. WandbCheckpoint writes its
    # wandb_resume.json sidecar under default_root_dir; left unset that resolves
    # to the CWD, so concurrent runs sharing a checkout all write and read a
    # single file and can resume into each other's W&B run. subdir is stable
    # across a requeue of the same run, so per-run resume still works.
    trainer_cfg.setdefault("default_root_dir", str(run_dir))
    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
