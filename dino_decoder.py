"""Pixel decoder for the spatial patch tokens used by DINO-WM.

The architecture follows the non-quantized VQ-VAE decoder in the original
DINO-WM implementation: two residual convolutional stages, each followed by
4x upsampling.  The original model decodes a 14x14 token grid directly to
224x224.  This repository's DINO-WM uses a 16x16 grid, so the native 256x256
output is resized once at the end to the requested image size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, residual_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, residual_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(residual_channels, channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class UpsamplingStage(nn.Module):
    """Residual processing followed by the 4x upsampling used by DINO-WM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_residual_blocks: int,
        residual_channels: int,
    ):
        super().__init__()
        blocks: list[nn.Module] = [
            nn.Conv2d(in_channels, channels, 3, padding=1)
        ]
        blocks.extend(
            ResidualBlock(channels, residual_channels)
            for _ in range(num_residual_blocks)
        )
        blocks.extend(
            [
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    channels, channels // 2, 4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    channels // 2, out_channels, 4, stride=2, padding=1
                ),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class DinoPatchDecoder(nn.Module):
    """Decode a square grid of DINO patch tokens into RGB pixels.

    Inputs can be ``(B, P, D)`` or ``(B, T, P, D)``.  The corresponding
    outputs are ``(B, 3, H, W)`` or ``(B, T, 3, H, W)``.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        image_size: int = 224,
        channels: int = 384,
        num_residual_blocks: int = 4,
        residual_channels: int = 128,
    ):
        super().__init__()
        if min(
            embedding_dim,
            image_size,
            channels,
            num_residual_blocks,
            residual_channels,
        ) < 1:
            raise ValueError("decoder dimensions and block count must be positive")
        if channels % 2:
            raise ValueError("channels must be even for 2x transposed convolutions")

        self.embedding_dim = int(embedding_dim)
        self.image_size = int(image_size)
        self.upsample_tokens = UpsamplingStage(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            channels=channels,
            num_residual_blocks=num_residual_blocks,
            residual_channels=residual_channels,
        )
        self.to_pixels = UpsamplingStage(
            in_channels=self.embedding_dim,
            out_channels=3,
            channels=channels,
            num_residual_blocks=num_residual_blocks,
            residual_channels=residual_channels,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim not in (3, 4):
            raise ValueError(
                "tokens must have shape (B,P,D) or (B,T,P,D), got "
                f"{tuple(tokens.shape)}"
            )
        if tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected token dim {self.embedding_dim}, got {tokens.shape[-1]}"
            )

        temporal_shape = tokens.shape[:2] if tokens.ndim == 4 else None
        if temporal_shape is not None:
            tokens = tokens.flatten(0, 1)

        batch_size, num_patches, _ = tokens.shape
        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError(f"number of patches must be square, got {num_patches}")

        features = tokens.transpose(1, 2).reshape(
            batch_size, self.embedding_dim, side, side
        )
        pixels = self.to_pixels(self.upsample_tokens(features))
        if pixels.shape[-2:] != (self.image_size, self.image_size):
            pixels = F.interpolate(
                pixels,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        if temporal_shape is not None:
            pixels = pixels.unflatten(0, temporal_shape)
        return pixels
