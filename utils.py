from dataclasses import dataclass

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from stable_pretraining import data as dt


@dataclass(frozen=True)
class EpisodeSplit:
    train_clip_indices: np.ndarray
    val_clip_indices: np.ndarray
    train_episode_indices: np.ndarray
    val_episode_indices: np.ndarray


def balanced_episode_split(dataset, train_fraction, seed, search_trials=50_000):
    """Keep episodes atomic while matching the requested split by clip count.

    Candidate validation sets contain approximately the requested fraction of
    episodes. The search primarily minimizes usable-clip imbalance, with a
    small tie-breaker favoring coverage across chronology and duration deciles.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if search_trials < 1:
        raise ValueError(f"search_trials must be positive, got {search_trials}")

    clip_episodes = np.asarray([episode for episode, _ in dataset.clip_indices])
    eligible_episodes, clip_counts = np.unique(clip_episodes, return_counts=True)
    if eligible_episodes.size < 2:
        raise ValueError("episode split requires at least two eligible episodes")

    val_fraction = 1.0 - float(train_fraction)
    val_episode_count = int(round(eligible_episodes.size * val_fraction))
    val_episode_count = min(max(val_episode_count, 1), eligible_episodes.size - 1)
    target_val_clips = clip_episodes.size * val_fraction

    # Episode order captures collection chronology. Ranked clip counts provide
    # stable duration strata even when many episodes have the same length.
    num_buckets = min(10, eligible_episodes.size)
    chronology_bucket = (
        np.arange(eligible_episodes.size) * num_buckets // eligible_episodes.size
    )
    duration_rank = np.empty(eligible_episodes.size, dtype=np.int64)
    duration_rank[np.argsort(clip_counts, kind="stable")] = np.arange(
        eligible_episodes.size
    )
    duration_bucket = duration_rank * num_buckets // eligible_episodes.size
    expected_chronology = (
        np.bincount(chronology_bucket, minlength=num_buckets) / eligible_episodes.size
    )
    expected_duration = (
        np.bincount(duration_bucket, minlength=num_buckets) / eligible_episodes.size
    )

    rng = np.random.default_rng(seed)
    best_score = None
    best_positions = None
    for _ in range(search_trials):
        positions = np.sort(
            rng.choice(eligible_episodes.size, val_episode_count, replace=False)
        )
        clip_error = (
            abs(clip_counts[positions].sum() - target_val_clips)
            / clip_episodes.size
        )
        chronology_error = np.abs(
            np.bincount(chronology_bucket[positions], minlength=num_buckets)
            / val_episode_count
            - expected_chronology
        ).sum()
        duration_error = np.abs(
            np.bincount(duration_bucket[positions], minlength=num_buckets)
            / val_episode_count
            - expected_duration
        ).sum()
        score = clip_error + 0.001 * (chronology_error + duration_error)
        if best_score is None or score < best_score:
            best_score = score
            best_positions = positions

    val_episodes = eligible_episodes[best_positions]
    val_mask = np.isin(clip_episodes, val_episodes)
    all_clip_indices = np.arange(clip_episodes.size, dtype=np.int64)
    train_episodes = eligible_episodes[~np.isin(eligible_episodes, val_episodes)]
    return EpisodeSplit(
        train_clip_indices=all_clip_indices[~val_mask],
        val_clip_indices=all_clip_indices[val_mask],
        train_episode_indices=train_episodes,
        val_episode_indices=val_episodes,
    )


def matched_step_epochs(base_epochs, subset_clips, full_clips, batch_size):
    """Epochs giving a subset the same optimizer-step budget as the full split.

    A data-scaling curve trained for a fixed number of epochs confounds two
    variables: the small budgets see less data *and* take proportionally fewer
    optimizer steps, so they are undertrained rather than data-limited. Holding
    steps constant isolates the effect of data volume. Checkpoints are still
    written every interval, so an overfitting subset can be read off the
    validation curve instead of being hidden by an early stop.
    """
    for name, value in (
        ("base_epochs", base_epochs),
        ("subset_clips", subset_clips),
        ("full_clips", full_clips),
        ("batch_size", batch_size),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    if subset_clips > full_clips:
        raise ValueError(
            f"subset_clips ({subset_clips}) exceeds full_clips ({full_clips})"
        )

    # drop_last=True on the training loader, so a partial batch is not a step.
    subset_steps_per_epoch = subset_clips // batch_size
    full_steps_per_epoch = full_clips // batch_size
    if subset_steps_per_epoch < 1:
        raise ValueError(
            f"subset has {subset_clips} clips, fewer than one batch of "
            f"{batch_size}"
        )
    target_steps = int(base_epochs) * full_steps_per_epoch
    return -(-target_steps // subset_steps_per_epoch)  # ceil division


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def column_zscore_stats(dataset, source: str, episode_indices=None):
    """Mean/std/row-count for one column, restricted to selected episodes.

    Exposed separately from :func:`get_column_normalizer` so a run can record
    the exact statistics it trained with; deploying a checkpoint on the robot
    requires reproducing them.
    """
    col_data = dataset.get_col_data(source)
    if episode_indices is not None:
        row_indices = np.concatenate(
            [
                np.arange(
                    dataset.offsets[episode],
                    dataset.offsets[episode] + dataset.lengths[episode],
                )
                for episode in episode_indices
            ]
        )
        col_data = col_data[row_indices]
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return mean, std, int(data.shape[0])


def get_column_normalizer(dataset, source: str, target: str, episode_indices=None):
    """Get normalizer for a specific column in the dataset."""
    mean, std, _ = column_zscore_stats(dataset, source, episode_indices)
    return dt.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std), source=source, target=target
    )


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1
            if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
                self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained

        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
