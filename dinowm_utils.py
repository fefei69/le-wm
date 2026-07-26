"""Deterministic data helpers for PushBox DINO-WM learning curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EpisodeSubset:
    """A duration-targeted, episode-atomic training subset."""

    episode_indices: np.ndarray
    clip_indices: np.ndarray
    num_frames: int
    hours: float


@dataclass(frozen=True)
class ColumnStats:
    """Train-only normalization statistics for one vector-valued column."""

    mean: torch.Tensor
    std: torch.Tensor
    num_rows: int


def episode_clip_indices(dataset, episode_indices) -> np.ndarray:
    """Return clip indices belonging to the requested episodes."""

    episode_indices = np.asarray(episode_indices, dtype=np.int64)
    clip_episodes = np.fromiter(
        (episode for episode, _ in dataset.clip_indices),
        dtype=np.int64,
        count=len(dataset.clip_indices),
    )
    return np.flatnonzero(np.isin(clip_episodes, episode_indices)).astype(
        np.int64
    )


def _nested_prefix(
    dataset,
    candidate_episode_indices,
    target_frames: float,
    sample_hz: float,
    seed: int,
) -> EpisodeSubset:
    """Select the deterministic shuffled-prefix closest to ``target_frames``.

    Episodes stay atomic and the shuffle depends only on ``seed``, so a larger
    ``target_frames`` always yields a superset of a smaller one.
    """

    candidates = np.unique(
        np.asarray(candidate_episode_indices, dtype=np.int64)
    )
    if candidates.size == 0:
        raise ValueError("candidate_episode_indices cannot be empty")
    if candidates.min() < 0 or candidates.max() >= len(dataset.lengths):
        raise IndexError("candidate episode index is outside dataset.lengths")

    ordered = np.random.default_rng(seed).permutation(candidates)
    ordered_lengths = np.asarray(dataset.lengths, dtype=np.int64)[ordered]
    cumulative_frames = np.cumsum(ordered_lengths)

    if target_frames >= cumulative_frames[-1]:
        count = len(ordered)
    else:
        # The closest prefix is deterministic and remains monotonic as the
        # requested duration increases.
        count = int(np.argmin(np.abs(cumulative_frames - target_frames))) + 1

    selected = np.sort(ordered[:count])
    num_frames = int(np.asarray(dataset.lengths)[selected].sum())
    return EpisodeSubset(
        episode_indices=selected,
        clip_indices=episode_clip_indices(dataset, selected),
        num_frames=num_frames,
        hours=num_frames / float(sample_hz) / 3600.0,
    )


def select_nested_episode_subset(
    dataset,
    candidate_episode_indices,
    target_hours: float,
    sample_hz: float,
    seed: int,
) -> EpisodeSubset:
    """Select a reproducible prefix of a shuffled episode list by duration.

    Calling this function with the same candidates and seed but increasing
    ``target_hours`` produces nested subsets. Episodes remain atomic. When the
    requested duration is at least the available duration, all candidates are
    selected; this is how the nominal four-hour curve maps to the full training
    split after holding validation episodes out.
    """

    if target_hours <= 0:
        raise ValueError(f"target_hours must be positive, got {target_hours}")
    if sample_hz <= 0:
        raise ValueError(f"sample_hz must be positive, got {sample_hz}")

    return _nested_prefix(
        dataset,
        candidate_episode_indices,
        target_frames=float(target_hours) * 3600.0 * float(sample_hz),
        sample_hz=sample_hz,
        seed=seed,
    )


def select_episode_fraction(
    dataset,
    candidate_episode_indices,
    fraction: float,
    sample_hz: float,
    seed: int,
) -> EpisodeSubset:
    """Select a nested subset holding ``fraction`` of the candidate frames.

    This is the duration-targeted selection expressed relative to whatever is
    available, so 1.0 always means "every training episode" without needing to
    know the recording length in advance. Sharing ``seed`` and ``candidates``
    across fractions keeps the subsets nested and keeps them identical across
    model architectures, which is what makes a data-scaling comparison fair.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if sample_hz <= 0:
        raise ValueError(f"sample_hz must be positive, got {sample_hz}")

    candidates = np.unique(
        np.asarray(candidate_episode_indices, dtype=np.int64)
    )
    if candidates.size == 0:
        raise ValueError("candidate_episode_indices cannot be empty")
    available_frames = float(
        np.asarray(dataset.lengths, dtype=np.int64)[candidates].sum()
    )
    return _nested_prefix(
        dataset,
        candidates,
        target_frames=float(fraction) * available_frames,
        sample_hz=sample_hz,
        seed=seed,
    )


def compute_column_stats(
    dataset,
    column: str,
    episode_indices,
    minimum_std: float = 1e-6,
) -> ColumnStats:
    """Compute finite-row mean/std using only the selected episodes."""

    episodes = np.unique(np.asarray(episode_indices, dtype=np.int64))
    if episodes.size == 0:
        raise ValueError("episode_indices cannot be empty")

    row_indices = np.concatenate(
        [
            np.arange(
                dataset.offsets[episode],
                dataset.offsets[episode] + dataset.lengths[episode],
                dtype=np.int64,
            )
            for episode in episodes
        ]
    )
    values = np.asarray(dataset.get_col_data(column))[row_indices]
    values = values.reshape(values.shape[0], -1)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] < 2:
        raise ValueError(
            f"Need at least two finite rows for {column!r}, got {values.shape[0]}"
        )

    tensor = torch.from_numpy(values).float()
    mean = tensor.mean(dim=0, keepdim=True)
    raw_std = tensor.std(dim=0, keepdim=True)
    std = torch.where(raw_std >= minimum_std, raw_std, torch.ones_like(raw_std))
    return ColumnStats(mean=mean, std=std, num_rows=values.shape[0])
