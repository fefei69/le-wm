"""Train a CLSDecoder (image_decoder.py) to reconstruct frames from frozen LeWM latents.

Post-hoc pixel probe, as in the LeWM paper (App. D, Fig. 7/10): the world model
is frozen, the decoder regresses ImageNet-normalized pixels from the 192-d
per-frame latent with plain MSE. No gradients ever reach the encoder.

Validation is episode-atomic. A frame-level split leaks: consecutive frames of a
trajectory are near-duplicates, so a held-out frame almost always has a
neighbouring frame in the training set and validation scores an interpolation
rather than generalization. By default the split is recomputed with the same
balanced-episode search, seed and train fraction the world model itself used, so
the decoder is scored on the world model's own held-out episodes; --manifest
pins an explicit episode list when a checkpoint ships one.

Usage:
    python train_decoder.py --checkpoint lewm/weights_epoch_10.pt --dataset pusht_expert_train
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import stable_pretraining as spt
import stable_worldmodel as swm
import wandb
from torchvision.utils import save_image

from utils import balanced_episode_split, get_img_preprocessor
from image_decoder import CLSDecoder


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="lewm/weights_epoch_10.pt",
                   help="world-model ckpt, relative to $STABLEWM_HOME/checkpoints/")
    p.add_argument("--dataset", default="pusht_expert_train",
                   help="HDF5 dataset name under $STABLEWM_HOME/datasets/ (no .h5)")
    p.add_argument("--latent", choices=["cls", "proj"], default="cls",
                   help="cls = pre-projector [CLS] (paper App. D); proj = planning latent")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16, help="decoder output patch size")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--val-batches", type=positive_int, default=32,
                   help="val batches per eval pass")
    p.add_argument("--manifest", default=None,
                   help="split_manifest.json holding validation_episode_indices; "
                        "overrides the recomputed episode split")
    p.add_argument("--split-fraction", type=float, default=0.9,
                   help="train fraction for the episode split (match the WM's train_split)")
    p.add_argument("--split-num-steps", type=positive_int, default=4,
                   help="WM window length (history_size + num_preds); the episode "
                        "search balances clip counts, so this must match WM training "
                        "for the split to reproduce")
    p.add_argument("--split-search-trials", type=positive_int, default=50_000)
    p.add_argument("--viz-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None,
                   help="output dir (default: $STABLEWM_HOME/checkpoints/<ckpt_dir>/decoder)")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing decoder.pt instead of continuing it")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-entity", default="cw5167-nyu")
    p.add_argument("--wandb-project", default="le-wm")
    p.add_argument("--wandb-name", default=None,
                   help="optional explicit W&B run name")
    return p.parse_args()


@torch.no_grad()
def encode(model, pixels, latent):
    out = model.encoder(pixels, interpolate_pos_encoding=True)
    z = out.last_hidden_state[:, 0]
    if latent == "proj":
        z = model.projector(z)
    return z


def denormalize(img):
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats["mean"], device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"], device=img.device).view(1, 3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def indices_for_episodes(dataset, episode_indices):
    selected = {int(index) for index in episode_indices}
    return [
        index
        for index, (episode, _) in enumerate(dataset.clip_indices)
        if int(episode) in selected
    ]


def validation_episodes(args):
    """Episodes held out from decoder training, as a set of episode indices."""
    if args.manifest:
        with Path(args.manifest).expanduser().open() as handle:
            manifest = json.load(handle)
        if "validation_episode_indices" not in manifest:
            raise ValueError(f"{args.manifest} has no validation_episode_indices")
        return {int(index) for index in manifest["validation_episode_indices"]}

    # The balanced search scores candidate splits by usable-clip imbalance, so it
    # only reproduces the world model's partition when the dataset is windowed the
    # same way. Build a WM-shaped view purely to derive the episode assignment.
    wm_view = swm.data.HDF5Dataset(
        args.dataset,
        frameskip=1,
        num_steps=args.split_num_steps,
        keys_to_load=["pixels"],
    )
    split = balanced_episode_split(
        wm_view,
        train_fraction=args.split_fraction,
        seed=args.seed,
        search_trials=args.split_search_trials,
    )
    return {int(index) for index in split.val_episode_indices}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # -- frozen world model
    model = swm.wm.utils.load_pretrained(args.checkpoint)
    model = model.to(device).eval().requires_grad_(False)

    # -- dataset: single frames, no frameskip
    dataset = swm.data.HDF5Dataset(
        args.dataset, frameskip=1, num_steps=1, keys_to_load=["pixels"]
    )
    dataset.transform = get_img_preprocessor(
        source="pixels", target="pixels", img_size=args.img_size
    )

    val_episodes = validation_episodes(args)
    val_indices = indices_for_episodes(dataset, val_episodes)
    train_indices = [
        index
        for index, (episode, _) in enumerate(dataset.clip_indices)
        if int(episode) not in val_episodes
    ]
    if len(train_indices) < args.batch_size or not val_indices:
        raise ValueError(
            f"invalid split sizes: train={len(train_indices)}, val={len(val_indices)}"
        )
    print(f"episode split: {len(train_indices):,} train frames, "
          f"{len(val_indices):,} val frames over {len(val_episodes)} held-out episodes")

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True, generator=gen,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    viz_batch = next(iter(val_loader))["pixels"].squeeze(1)[:8].to(device)

    # -- decoder sized from an actual latent
    z_dim = encode(model, viz_batch[:2], args.latent).shape[-1]
    decoder = CLSDecoder(
        cls_dim=z_dim, img_size=args.img_size, patch_size=args.patch_size,
        dim=args.dim, heads=args.heads, depth=args.depth,
    ).to(device)
    print(f"latent={args.latent} ({z_dim}d) -> decoder "
          f"{sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M params")

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    out_dir = Path(args.out) if args.out else Path(
        swm.data.utils.get_cache_dir(sub_folder="checkpoints"),
        Path(args.checkpoint).parent, "decoder",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"outputs -> {out_dir}")

    latest_path = out_dir / "decoder.pt"
    best_path = out_dir / "decoder_best.pt"
    step, ema, best_val_mse = 0, None, math.inf
    if latest_path.is_file() and not args.no_resume:
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        decoder.load_state_dict(saved["state_dict"])
        if "optimizer" in saved:
            opt.load_state_dict(saved["optimizer"])
        if "scheduler" in saved:
            sched.load_state_dict(saved["scheduler"])
        step = int(saved.get("step", 0))
        ema = saved.get("ema")
        best_val_mse = float(saved.get("best_val_mse", math.inf))
        print(f"resumed decoder at step {step:,} from {latest_path}")

    run_name = args.wandb_name or (
        f"decoder-{args.latent}-{Path(args.checkpoint).stem}-p{args.patch_size}"
    )
    run = None
    if not args.no_wandb:
        # Keying the run on its name lets a requeued job continue the same curves
        # instead of starting a second, truncated run.
        run = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project,
            name=run_name, id=run_name, resume="allow",
            job_type="decoder-probe",
            config={**vars(args), "z_dim": z_dim,
                    "decoder_params": sum(p.numel() for p in decoder.parameters()),
                    "train_frames": len(train_indices),
                    "val_frames": len(val_indices),
                    "val_episodes": sorted(val_episodes)},
        )

    def save_viz(step):
        decoder.eval()
        with torch.no_grad():
            recon = decoder(encode(model, viz_batch, args.latent))
        grid = torch.cat([denormalize(viz_batch), denormalize(recon)], dim=0)
        path = out_dir / f"recon_step{step:06d}.png"
        save_image(grid, path, nrow=viz_batch.size(0))
        decoder.train()
        return path

    @torch.no_grad()
    def evaluate():
        decoder.eval()
        # Sum-reduce over elements rather than averaging per-batch means: the last
        # val batch is short, and a batch-count average would silently overweight it.
        normalized_sse, pixel_sse, count = 0.0, 0.0, 0
        for i, batch in enumerate(val_loader):
            if i == args.val_batches:
                break
            px = batch["pixels"].squeeze(1).to(device)
            recon = decoder(encode(model, px, args.latent))
            normalized_sse += F.mse_loss(recon, px, reduction="sum").item()
            pixel_sse += F.mse_loss(
                denormalize(recon), denormalize(px), reduction="sum"
            ).item()
            count += px.numel()
        decoder.train()
        return normalized_sse / count, -10 * math.log10(max(pixel_sse / count, 1e-12))

    if step == 0:
        save_viz(0)
    while step < args.steps:
        for batch in train_loader:
            pixels = batch["pixels"].squeeze(1).to(device, non_blocking=True)
            z = encode(model, pixels, args.latent)
            loss = F.mse_loss(decoder(z), pixels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            step += 1
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                raise RuntimeError(f"non-finite loss at step {step}")
            ema = loss_val if ema is None else 0.99 * ema + 0.01 * loss_val
            if step % args.log_every == 0:
                print(f"step {step:6d}/{args.steps}  mse {ema:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
                if run:
                    run.log({"train/mse": loss_val, "train/mse_ema": ema,
                             "lr": sched.get_last_lr()[0]}, step=step)
            if step % args.viz_every == 0 or step == args.steps:
                grid_path = save_viz(step)
                val_mse, val_psnr = evaluate()
                is_best = val_mse < best_val_mse
                best_val_mse = min(best_val_mse, val_mse)
                print(f"step {step:6d}/{args.steps}  val mse {val_mse:.4f}  "
                      f"psnr {val_psnr:.1f} dB", flush=True)
                if run:
                    run.log({"val/mse": val_mse, "val/psnr_db": val_psnr,
                             "val/best_mse": best_val_mse,
                             "viz/recon": wandb.Image(str(grid_path))}, step=step)
                payload = {
                    "state_dict": decoder.state_dict(), "args": vars(args),
                    "step": step, "val_mse": val_mse, "val_psnr_db": val_psnr,
                    "z_dim": z_dim, "latent_type": args.latent,
                    "source_checkpoint": args.checkpoint,
                }
                torch.save(
                    {**payload, "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(), "ema": ema,
                     "best_val_mse": best_val_mse},
                    latest_path,
                )
                if is_best:
                    torch.save(payload, best_path)
            if step >= args.steps:
                break

    if run:
        run.summary.update({"val/best_mse": best_val_mse})
        run.finish()
    print(f"done. weights + recon grids in {out_dir}")


if __name__ == "__main__":
    main()
