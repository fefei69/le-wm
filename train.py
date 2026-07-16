import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import (
    SaveCkptCallback,
    balanced_episode_split,
    get_column_normalizer,
    get_img_preprocessor,
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
        train_set = torch.utils.data.Subset(
            dataset, episode_split.train_clip_indices.tolist()
        )
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

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=cfg.get("checkpoint_interval", 1),
    )

    trainer = pl.Trainer(
        **cfg.trainer,
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
