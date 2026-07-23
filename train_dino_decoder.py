"""Train a post-hoc pixel decoder for a frozen DINO-WM checkpoint.

Only ground-truth frozen DINO patch tokens are decoded during training.  This
matches the gradient separation in the original DINO-WM implementation and
keeps the world model unchanged.  At evaluation time the same decoder can be
applied to the first 384 dimensions of predicted DINO-WM embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import stable_pretraining as spt
import stable_worldmodel as swm
import wandb
from hydra.utils import instantiate
from torchvision.utils import save_image

from dino_decoder import DinoPatchDecoder
from utils import get_img_preprocessor


def positive_int(value: str) -> int:
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="DINO-WM checkpoint, relative to $STABLEWM_HOME/checkpoints",
    )
    parser.add_argument(
        "--dataset",
        default="pushbox_pilot_train",
        help="dataset name or path",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="split_manifest.json (default: beside the checkpoint)",
    )
    parser.add_argument("--steps", type=positive_int, default=20_000)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=positive_int, default=224)
    parser.add_argument("--channels", type=positive_int, default=384)
    parser.add_argument("--residual-blocks", type=positive_int, default=4)
    parser.add_argument("--residual-channels", type=positive_int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--val-batches", type=positive_int, default=8)
    parser.add_argument("--viz-every", type=positive_int, default=1_000)
    parser.add_argument("--log-every", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="cw5167-nyu")
    parser.add_argument("--wandb-project", default="le-wm")
    parser.add_argument("--wandb-group", default="pushbox-dinowm-decoders")
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def resolve_checkpoint(name: str) -> Path:
    path = Path(name).expanduser()
    if not path.is_absolute():
        checkpoint_root = Path(
            swm.data.utils.get_cache_dir(sub_folder="checkpoints")
        )
        path = checkpoint_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if not (path.parent / "config.json").is_file():
        raise FileNotFoundError(f"config.json not found beside {path}")
    return path


def load_frozen_backbone(checkpoint: Path, device: torch.device):
    """Load only the checkpoint's DINO backbone, not its unused predictor."""

    with (checkpoint.parent / "config.json").open() as handle:
        model_config = json.load(handle)
    if "encoder" not in model_config:
        raise ValueError(f"checkpoint config has no encoder: {checkpoint.parent}")

    backbone = instantiate(model_config["encoder"])
    try:
        state_dict = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    prefix = "backbone."
    backbone_state = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError(f"no {prefix!r} weights found in {checkpoint}")
    backbone.load_state_dict(backbone_state, strict=True)
    del backbone_state, state_dict
    return backbone.to(device).eval().requires_grad_(False), model_config


def indices_for_episodes(dataset, episode_indices) -> list[int]:
    selected = {int(index) for index in episode_indices}
    return [
        index
        for index, (episode, _) in enumerate(dataset.clip_indices)
        if int(episode) in selected
    ]


@torch.no_grad()
def encode(backbone, pixels: torch.Tensor) -> torch.Tensor:
    output = backbone(pixels.float(), interpolate_pos_encoding=True)
    if not hasattr(output, "last_hidden_state"):
        raise TypeError("DINO backbone did not return last_hidden_state")
    return output.last_hidden_state[:, 1:].detach()


def denormalize(images: torch.Tensor) -> torch.Tensor:
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.as_tensor(
        stats["mean"], device=images.device, dtype=images.dtype
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        stats["std"], device=images.device, dtype=images.dtype
    ).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    checkpoint = resolve_checkpoint(args.checkpoint)
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else checkpoint.parent / "split_manifest.json"
    )
    with manifest_path.open() as handle:
        manifest = json.load(handle)

    backbone, model_config = load_frozen_backbone(checkpoint, device)

    dataset_name = str(Path(args.dataset).expanduser())
    dataset = swm.data.load_dataset(
        dataset_name,
        frameskip=1,
        num_steps=1,
        keys_to_load=["pixels"],
    )
    dataset.transform = get_img_preprocessor(
        source="pixels", target="pixels", img_size=args.image_size
    )
    train_indices = indices_for_episodes(
        dataset, manifest["train_episode_indices"]
    )
    val_indices = indices_for_episodes(
        dataset, manifest["validation_episode_indices"]
    )
    if len(train_indices) < args.batch_size or not val_indices:
        raise ValueError(
            f"invalid split sizes: train={len(train_indices)}, val={len(val_indices)}"
        )

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": device.type == "cuda",
    }
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        **loader_kwargs,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        **loader_kwargs,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        persistent_workers=args.num_workers > 0,
    )
    viz_batch = next(iter(val_loader))["pixels"].squeeze(1)[:8].to(device)

    use_bf16 = device.type == "cuda" and not args.no_bf16

    def autocast_context():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    with torch.no_grad(), autocast_context():
        sample_tokens = encode(backbone, viz_batch[:2])
    embedding_dim = int(sample_tokens.shape[-1])
    num_patches = int(sample_tokens.shape[-2])
    decoder = DinoPatchDecoder(
        embedding_dim=embedding_dim,
        image_size=args.image_size,
        channels=args.channels,
        num_residual_blocks=args.residual_blocks,
        residual_channels=args.residual_channels,
    ).to(device)
    decoder_parameters = sum(parameter.numel() for parameter in decoder.parameters())
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    latest_path = out_dir / "decoder.pt"
    best_path = out_dir / "decoder_best.pt"
    step, ema, best_val_mse = 0, None, math.inf
    if latest_path.is_file() and not args.no_resume:
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        decoder.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        step = int(saved["step"])
        ema = saved.get("ema")
        best_val_mse = float(saved.get("best_val_mse", math.inf))
        print(f"resumed decoder at step {step:,} from {latest_path}")

    curve_label = checkpoint.parent.name.rsplit("_", 1)[-1]
    run_name = args.wandb_name or f"decoder-dinowm-dinov2s-prop-{curve_label}"
    run = None
    if not args.no_wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            id=run_name,
            resume="allow",
            job_type="decoder-probe",
            tags=["pushbox", "dinowm", "decoder", curve_label],
            config={
                **vars(args),
                "checkpoint_path": str(checkpoint),
                "manifest_path": str(manifest_path),
                "embedding_dim": embedding_dim,
                "num_patches": num_patches,
                "decoder_parameters": decoder_parameters,
                "train_frames": len(train_indices),
                "validation_frames": len(val_indices),
                "model_config": model_config,
            },
        )

    print(
        f"DINO tokens: {num_patches} x {embedding_dim}; decoder: "
        f"{decoder_parameters / 1e6:.2f}M parameters; bf16={use_bf16}"
    )
    print(
        f"split: {len(train_indices):,} train frames, "
        f"{len(val_indices):,} validation frames; outputs -> {out_dir}"
    )

    @torch.no_grad()
    def reconstruct(pixels: torch.Tensor) -> torch.Tensor:
        with autocast_context():
            return decoder(encode(backbone, pixels))

    def save_viz(current_step: int) -> Path:
        decoder.eval()
        with torch.no_grad():
            reconstruction = reconstruct(viz_batch).float()
        grid = torch.cat(
            [denormalize(viz_batch.float()), denormalize(reconstruction)], dim=0
        )
        path = out_dir / f"recon_step{current_step:06d}.png"
        save_image(grid, path, nrow=viz_batch.shape[0])
        decoder.train()
        return path

    @torch.no_grad()
    def evaluate() -> tuple[float, float]:
        decoder.eval()
        normalized_sse = 0.0
        pixel_sse = 0.0
        num_values = 0
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= args.val_batches:
                break
            pixels = batch["pixels"].squeeze(1).to(device, non_blocking=True)
            reconstruction = reconstruct(pixels).float()
            pixels = pixels.float()
            normalized_sse += F.mse_loss(
                reconstruction, pixels, reduction="sum"
            ).item()
            pixel_sse += F.mse_loss(
                denormalize(reconstruction),
                denormalize(pixels),
                reduction="sum",
            ).item()
            num_values += pixels.numel()
        decoder.train()
        normalized_mse = normalized_sse / num_values
        pixel_mse = pixel_sse / num_values
        return normalized_mse, -10.0 * math.log10(max(pixel_mse, 1e-12))

    def save_checkpoint(current_step: int) -> None:
        torch.save(
            {
                "state_dict": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "step": current_step,
                "ema": ema,
                "best_val_mse": best_val_mse,
                "embedding_dim": embedding_dim,
                "num_patches": num_patches,
                "source_checkpoint": str(checkpoint),
            },
            latest_path,
        )

    if step == 0:
        save_viz(0)
    decoder.train()
    while step < args.steps:
        for batch in train_loader:
            pixels = batch["pixels"].squeeze(1).to(device, non_blocking=True)
            with torch.no_grad(), autocast_context():
                tokens = encode(backbone, pixels)
            with autocast_context():
                reconstruction = decoder(tokens)
                loss = F.mse_loss(reconstruction.float(), pixels.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"non-finite loss at step {step}")
            ema = loss_value if ema is None else 0.99 * ema + 0.01 * loss_value

            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "train/mse": loss_value,
                    "train/mse_ema": ema,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    f"step {step:6d}/{args.steps}  mse {ema:.4f}  "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
                if run:
                    run.log(record, step=step)

            if step % args.viz_every == 0 or step == args.steps:
                grid_path = save_viz(step)
                val_mse, val_psnr = evaluate()
                is_best = val_mse < best_val_mse
                best_val_mse = min(best_val_mse, val_mse)
                save_checkpoint(step)
                if is_best:
                    torch.save(
                        {
                            "state_dict": decoder.state_dict(),
                            "args": vars(args),
                            "step": step,
                            "val_mse": val_mse,
                            "embedding_dim": embedding_dim,
                            "num_patches": num_patches,
                            "source_checkpoint": str(checkpoint),
                        },
                        best_path,
                    )
                record = {
                    "step": step,
                    "val/mse": val_mse,
                    "val/psnr_db": val_psnr,
                    "val/best_mse": best_val_mse,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    f"step {step:6d}/{args.steps}  val mse {val_mse:.4f}  "
                    f"psnr {val_psnr:.1f} dB",
                    flush=True,
                )
                if run:
                    run.log(
                        {**record, "viz/reconstruction": wandb.Image(str(grid_path))},
                        step=step,
                    )

            if step >= args.steps:
                break

    if run:
        run.finish()
    print(f"done. decoder weights and reconstruction grids are in {out_dir}")


if __name__ == "__main__":
    main()
