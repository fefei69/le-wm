"""Headless goal-conditioned planner for the real PushBox robot.

This module runs in the le-wm Python environment.  The ROS/Trossen controller
uses a different Python ABI and communicates with this process over a local
authenticated Unix socket; no robot APIs are imported here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2

from utils import balanced_episode_split


ACTION_DIM = 2
ACTION_CAP_M = 0.010
IMAGE_SHAPE = (224, 224, 3)
KEYBOARD_ACTION_MAGNITUDES_M = (0.0025, 0.005, 0.010)
DEFAULT_LEWM_CHECKPOINT = "pushbox/lewm/weights_epoch_146.pt"
DEFAULT_DINOWM_CHECKPOINT = (
    "stable-wm/checkpoints/dinowm_dinov2s_prop_4h/weights_epoch_10.pt"
)


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 8
    num_samples: int = 256
    iterations: int = 5
    elite_count: int = 32
    initial_std_m: float = 0.005
    min_std_m: float = 0.00025
    action_cap_m: float = ACTION_CAP_M
    action_l2_weight: float = 0.001
    smoothness_weight: float = 0.001
    goal_tolerance: float = 0.0
    action_mode: str = "continuous"
    solver: str = "cem"
    categorical_alpha: float = 0.7
    categorical_min_prob: float = 0.01
    evaluation_batch_size: int = 0
    seed: int = 3072

    def validate(self) -> None:
        scalar_values = (
            self.initial_std_m,
            self.min_std_m,
            self.action_cap_m,
            self.action_l2_weight,
            self.smoothness_weight,
            self.goal_tolerance,
            self.categorical_alpha,
            self.categorical_min_prob,
        )
        if not np.isfinite(np.asarray(scalar_values, dtype=np.float64)).all():
            raise ValueError("CEM scalar configuration must be finite")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if not 1 <= self.elite_count <= self.num_samples:
            raise ValueError("elite_count must be in [1, num_samples]")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.evaluation_batch_size < 0:
            raise ValueError("evaluation_batch_size must be non-negative")
        if not 0.0 < self.min_std_m <= self.initial_std_m:
            raise ValueError("require 0 < min_std_m <= initial_std_m")
        if not 0.0 < self.action_cap_m <= ACTION_CAP_M:
            raise ValueError(f"action_cap_m must be in (0, {ACTION_CAP_M}]")
        if self.action_l2_weight < 0.0 or self.smoothness_weight < 0.0:
            raise ValueError("regularization weights must be non-negative")
        if self.goal_tolerance < 0.0:
            raise ValueError("goal_tolerance must be non-negative")
        if self.solver not in {"cem", "categorical-cem"}:
            raise ValueError("solver must be 'cem' or 'categorical-cem'")
        if self.action_mode not in {"continuous", "keyboard"}:
            raise ValueError("action_mode must be 'continuous' or 'keyboard'")
        if self.solver == "categorical-cem" and self.action_mode != "keyboard":
            raise ValueError("categorical-cem requires action_mode='keyboard'")
        if not 0.0 < self.categorical_alpha <= 1.0:
            raise ValueError("categorical_alpha must be in (0, 1]")
        if not 0.0 <= self.categorical_min_prob < 1.0:
            raise ValueError("categorical_min_prob must be in [0, 1)")


def clamp_action_norm(actions: torch.Tensor, cap_m: float) -> torch.Tensor:
    """Project every final-dimension XY action onto the configured L2 ball."""
    norms = torch.linalg.vector_norm(actions, dim=-1, keepdim=True)
    scale = torch.clamp(float(cap_m) / norms.clamp_min(1e-12), max=1.0)
    return actions * scale


def keyboard_action_vocabulary(
    cap_m: float,
    magnitudes_m: Sequence[float] = KEYBOARD_ACTION_MAGNITUDES_M,
) -> np.ndarray:
    """Return zero plus the exact 8-way held-key actions seen in collection."""
    cap = float(cap_m)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("keyboard action cap must be finite and positive")
    directions = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    actions = [np.zeros(ACTION_DIM, dtype=np.float64)]
    for magnitude in magnitudes_m:
        value = float(magnitude)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("keyboard magnitudes must be finite and positive")
        if value <= cap + 1e-12:
            actions.extend(directions * value)
    if len(actions) == 1:
        raise ValueError("action cap excludes every keyboard motion magnitude")
    return np.asarray(actions, dtype=np.float32)


def quantize_actions(
    actions: torch.Tensor, vocabulary: torch.Tensor
) -> torch.Tensor:
    """Map each XY vector to its nearest exact dataset keyboard action."""
    if actions.shape[-1] != ACTION_DIM:
        raise ValueError("actions must have final dimension 2")
    if vocabulary.ndim != 2 or vocabulary.shape[1] != ACTION_DIM:
        raise ValueError("action vocabulary must have shape (N, 2)")
    if vocabulary.shape[0] < 1 or not torch.isfinite(vocabulary).all():
        raise ValueError("action vocabulary must be non-empty and finite")
    distances = (
        actions.unsqueeze(-2) - vocabulary.to(actions.device, actions.dtype)
    ).square().sum(dim=-1)
    indices = distances.argmin(dim=-1)
    return vocabulary.to(actions.device, actions.dtype)[indices]


def sample_categorical_indices(
    probabilities: torch.Tensor,
    *,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample ``(sample, horizon)`` token indices from per-step probabilities."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("categorical probabilities must have shape (horizon, tokens)")
    if num_samples < 1:
        raise ValueError("categorical sample count must be positive")
    if not torch.isfinite(probabilities).all() or torch.any(probabilities < 0):
        raise ValueError("categorical probabilities must be finite and non-negative")
    row_sums = probabilities.sum(dim=1)
    if torch.any(row_sums <= 0):
        raise ValueError("each categorical distribution must have positive mass")
    normalized = probabilities / row_sums.unsqueeze(1)
    return torch.multinomial(
        normalized,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    ).transpose(0, 1).contiguous()


def categorical_elite_update(
    probabilities: torch.Tensor,
    elite_indices: torch.Tensor,
    *,
    alpha: float,
    min_prob: float,
) -> torch.Tensor:
    """Update per-step token probabilities from elite categorical samples."""
    if probabilities.ndim != 2:
        raise ValueError("categorical probabilities must be two-dimensional")
    horizon, num_actions = probabilities.shape
    if elite_indices.ndim != 2 or elite_indices.shape[1] != horizon:
        raise ValueError("elite indices must have shape (elites, horizon)")
    if elite_indices.shape[0] < 1:
        raise ValueError("categorical update requires at least one elite")
    if torch.any(elite_indices < 0) or torch.any(elite_indices >= num_actions):
        raise ValueError("elite indices contain an invalid action token")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("categorical alpha must be in (0, 1]")
    if not 0.0 <= float(min_prob) < 1.0 / num_actions:
        raise ValueError(
            f"categorical min_prob must be in [0, {1.0 / num_actions:.6g}) "
            f"for {num_actions} tokens"
        )

    elite_frequencies = torch.nn.functional.one_hot(
        elite_indices, num_classes=num_actions
    ).to(probabilities.dtype).mean(dim=0)
    updated = (1.0 - float(alpha)) * probabilities + float(alpha) * elite_frequencies
    updated = updated / updated.sum(dim=1, keepdim=True)
    if min_prob:
        # This mixture preserves a strict probability floor while keeping every
        # row normalized: p' = floor + (1 - K*floor) p.
        updated = float(min_prob) + (
            1.0 - num_actions * float(min_prob)
        ) * updated
    return updated


def compute_training_action_stats(
    dataset: Any,
    *,
    train_fraction: float = 0.9,
    seed: int = 3072,
    search_trials: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the exact train-only action mean/std used by ``train.py``."""
    split = balanced_episode_split(
        dataset,
        train_fraction=train_fraction,
        seed=seed,
        search_trials=search_trials,
    )
    row_indices = np.concatenate(
        [
            np.arange(
                dataset.offsets[episode],
                dataset.offsets[episode] + dataset.lengths[episode],
            )
            for episode in split.train_episode_indices
        ]
    )
    actions = torch.from_numpy(np.asarray(dataset.get_col_data("action"))[row_indices])
    actions = actions[~torch.isnan(actions).any(dim=1)]
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected action data with shape (N, 2), got {tuple(actions.shape)}")
    mean = actions.mean(dim=0)
    std = actions.std(dim=0)  # sample std: identical to training
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("action statistics are not finite")
    if torch.any(std <= 0):
        raise ValueError("action standard deviations must be positive")
    return mean.numpy(), std.numpy(), split.val_episode_indices


def compute_episode_column_stats(
    dataset: Any,
    column: str,
    episode_indices: Sequence[int],
    *,
    minimum_std: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reproduce DINO-WM's episode-subset normalization for one column."""
    episodes = np.unique(np.asarray(episode_indices, dtype=np.int64))
    if episodes.size == 0:
        raise ValueError("episode_indices cannot be empty")
    if episodes.min() < 0 or episodes.max() >= len(dataset.offsets):
        raise ValueError("model manifest contains an out-of-range episode")
    rows = np.concatenate(
        [
            np.arange(
                dataset.offsets[episode],
                dataset.offsets[episode] + dataset.lengths[episode],
                dtype=np.int64,
            )
            for episode in episodes
        ]
    )
    values = np.asarray(dataset.get_col_data(column))[rows]
    values = values.reshape(values.shape[0], -1)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] < 2:
        raise ValueError(f"not enough finite rows for {column!r}")
    tensor = torch.from_numpy(values).float()
    mean = tensor.mean(dim=0)
    raw_std = tensor.std(dim=0)
    std = torch.where(
        raw_std >= float(minimum_std), raw_std, torch.ones_like(raw_std)
    )
    return mean.numpy(), std.numpy(), int(values.shape[0])


def resolve_local_checkpoint(checkpoint: str | Path) -> Path:
    """Resolve an explicit checkpoint or a stable-worldmodel-relative path."""
    path = Path(checkpoint).expanduser()
    if not path.is_absolute() and not path.is_file():
        root = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"world-model checkpoint not found: {path}")
    if not (path.parent / "config.json").is_file():
        raise FileNotFoundError(f"config.json not found beside checkpoint: {path}")
    return path


def create_dinov2_small_checkpoint_backbone(
    name: str | None = None, **_: Any
) -> torch.nn.Module:
    """Instantiate DINOv2-small without a network/cache lookup.

    The world-model checkpoint already contains every frozen backbone weight;
    only the architecture is needed before loading its state dict.
    """
    if name not in (None, "dinov2_small", "facebook/dinov2-small"):
        raise ValueError(f"unsupported DINO-WM backbone: {name}")
    from transformers import Dinov2Config, Dinov2Model

    return Dinov2Model(
        Dinov2Config(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            mlp_ratio=4,
            # facebook/dinov2-small stores its learned positional table at the
            # 518px pretraining resolution (37x37 patches plus CLS). Runtime
            # observations remain 224px and use interpolate_pos_encoding=True.
            image_size=518,
            patch_size=14,
        )
    )


class PushBoxPlanner:
    """CEM planner operating in physical XY-delta units (meters)."""

    def __init__(
        self,
        model: torch.nn.Module,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        *,
        config: CEMConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        action_input_dim = int(self.model.action_encoder.patch_embed.in_channels)
        if action_input_dim != ACTION_DIM:
            raise ValueError(
                f"checkpoint action dimension is {action_input_dim}, expected {ACTION_DIM}"
            )
        self.history_size = int(self.model.predictor.pos_embedding.shape[1])
        self.action_mean = torch.as_tensor(
            action_mean, dtype=torch.float32, device=self.device
        ).reshape(1, 1, ACTION_DIM)
        self.action_std = torch.as_tensor(
            action_std, dtype=torch.float32, device=self.device
        ).reshape(1, 1, ACTION_DIM)
        if not torch.isfinite(self.action_mean).all() or not torch.isfinite(
            self.action_std
        ).all():
            raise ValueError("action statistics must be finite")
        if torch.any(self.action_std <= 0):
            raise ValueError("action standard deviations must be positive")

        stats = spt.data.dataset_stats.ImageNet
        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=stats["mean"], std=stats["std"]),
                v2.Resize((224, 224)),
            ]
        )
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device).manual_seed(config.seed)
        self.action_vocabulary = (
            torch.as_tensor(
                keyboard_action_vocabulary(config.action_cap_m),
                dtype=torch.float32,
                device=self.device,
            )
            if config.action_mode == "keyboard"
            else None
        )
        if (
            config.solver == "categorical-cem"
            and self.action_vocabulary is not None
            and config.categorical_min_prob
            >= 1.0 / int(self.action_vocabulary.shape[0])
        ):
            raise ValueError(
                "categorical_min_prob must be smaller than "
                f"{1.0 / int(self.action_vocabulary.shape[0]):.6g} for "
                f"{int(self.action_vocabulary.shape[0])} action tokens"
            )
        self._previous_plan: torch.Tensor | None = None
        self._previous_action_probabilities: torch.Tensor | None = None
        self._latent_context: torch.Tensor | None = None
        self._executed_actions: torch.Tensor | None = None

    def describe(self) -> dict[str, Any]:
        """Return the model/runtime contract used by the controller process."""
        latent_dim = int(self.model.predictor.pos_embedding.shape[-1])
        return {
            "world_model": "lewm",
            "history_size": self.history_size,
            "latent_dim": latent_dim,
            "latent_storage_shape": [latent_dim],
            "latent_semantic_shape": [latent_dim],
            "latent_representation": "projected_global",
            "requires_proprio": False,
        }

    @classmethod
    def from_artifacts(
        cls,
        *,
        checkpoint: str,
        dataset_path: str | Path,
        config: CEMConfig,
        device: str = "cuda",
        artifact_manifest: str | Path | None = None,
    ) -> "PushBoxPlanner":
        path = Path(dataset_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PushBox dataset not found: {path}")
        split_kwargs: dict[str, Any] = {}
        manifest: dict[str, Any] | None = None
        if artifact_manifest is not None:
            manifest_path = Path(artifact_manifest).expanduser().resolve()
            manifest = json.loads(manifest_path.read_text())
            split = manifest["split"]
            split_kwargs = {
                "train_fraction": float(split["train_fraction"]),
                "seed": int(split["seed"]),
                "search_trials": int(split["search_trials"]),
            }
        dataset = swm.data.load_dataset(
            str(path),
            transform=None,
            keys_to_load=["action"],
            keys_to_cache=["action"],
            num_steps=4,
            frameskip=1,
        )
        mean, std, val_episodes = compute_training_action_stats(
            dataset, **split_kwargs
        )
        if manifest is not None:
            expected = manifest["normalization"]["action"]
            if not np.allclose(mean, expected["mean"], rtol=1e-6, atol=1e-9):
                raise ValueError("dataset action mean does not match artifact manifest")
            if not np.allclose(std, expected["std"], rtol=1e-6, atol=1e-9):
                raise ValueError("dataset action std does not match artifact manifest")
        print(
            "PushBox action normalization: "
            f"mean={mean.tolist()}, std={std.tolist()}, "
            f"validation_episodes={val_episodes.tolist()}"
        )
        model = swm.wm.utils.load_pretrained(checkpoint)
        planner = cls(model, mean, std, config=config, device=device)
        if planner.action_vocabulary is not None:
            actions = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
            val_set = set(int(index) for index in val_episodes)
            train_rows = np.concatenate(
                [
                    np.arange(
                        dataset.offsets[episode],
                        dataset.offsets[episode] + dataset.lengths[episode],
                    )
                    for episode in range(len(dataset.offsets))
                    if episode not in val_set
                ]
            )
            train_actions = actions[train_rows]
            vocabulary = planner.action_vocabulary.detach().cpu().numpy()
            counts = np.asarray(
                [
                    np.count_nonzero(
                        np.all(
                            np.isclose(
                                train_actions,
                                token,
                                rtol=0.0,
                                atol=1e-7,
                            ),
                            axis=1,
                        )
                    )
                    for token in vocabulary
                ],
                dtype=np.int64,
            )
            if np.any(counts == 0):
                missing = vocabulary[counts == 0].tolist()
                raise ValueError(
                    f"keyboard action vocabulary contains unseen training actions: {missing}"
                )
            print(
                "CEM keyboard action vocabulary: "
                f"{len(vocabulary)} exact actions, training counts "
                f"min={int(counts.min())}, max={int(counts.max())}"
            )
        if manifest is not None:
            runtime = manifest["runtime"]
            if planner.history_size != int(runtime["history_size"]):
                raise ValueError("checkpoint history size does not match artifact manifest")
            latent_dim = int(planner.model.predictor.pos_embedding.shape[-1])
            if latent_dim != int(runtime["latent_dim"]):
                raise ValueError("checkpoint latent size does not match artifact manifest")
        return planner

    def reset(self) -> None:
        self._previous_plan = None
        self._previous_action_probabilities = None
        self._latent_context = None
        self._executed_actions = None

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.shape != IMAGE_SHAPE or array.dtype != np.uint8:
            raise ValueError(f"image must be RGB uint8 with shape {IMAGE_SHAPE}")
        tensor = self.image_transform(array).to(self.device)
        return tensor.unsqueeze(0).unsqueeze(0)

    def _encode(self, image: np.ndarray) -> torch.Tensor:
        output = self.model.encode({"pixels": self._prepare_image(image)})
        embedding = output["emb"]
        if not torch.isfinite(embedding).all():
            raise RuntimeError("image encoder produced a non-finite embedding")
        return embedding

    def _constrain_actions(self, actions: torch.Tensor) -> torch.Tensor:
        constrained = clamp_action_norm(actions, self.config.action_cap_m)
        if self.action_vocabulary is not None:
            constrained = quantize_actions(constrained, self.action_vocabulary)
        return constrained

    def _rollout_latents(
        self,
        latent_context: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return current plus every recursively predicted latent for a plan."""
        if raw_actions.ndim != 3 or raw_actions.shape[-1] != ACTION_DIM:
            raise ValueError("raw_actions must have shape (samples, horizon, 2)")
        normalized = (raw_actions - self.action_mean) / self.action_std
        num_samples = raw_actions.shape[0]
        embeddings = latent_context.expand(num_samples, -1, -1)
        if self._executed_actions is None:
            action_prefix = torch.empty(
                num_samples, 0, ACTION_DIM, device=self.device
            )
        else:
            action_prefix = self._executed_actions.expand(num_samples, -1, -1)
        predicted: list[torch.Tensor] = []

        for step in range(raw_actions.shape[1]):
            all_actions = torch.cat(
                [action_prefix, normalized[:, : step + 1]], dim=1
            )
            history_len = min(self.history_size, embeddings.shape[1])
            latent_history = embeddings[:, -history_len:]
            action_history = all_actions[:, -history_len:]
            if latent_history.shape[1] != action_history.shape[1]:
                raise RuntimeError("latent/action history alignment drifted")
            action_embeddings = self.model.action_encoder(action_history)
            prediction = self.model.predict(latent_history, action_embeddings)[:, -1:]
            if not torch.isfinite(prediction).all():
                raise RuntimeError("world model produced a non-finite prediction")
            embeddings = torch.cat([embeddings, prediction], dim=1)
            predicted.append(prediction)

        current = latent_context[:, -1:].expand(num_samples, -1, -1)
        return torch.cat([current, *predicted], dim=1)

    def _selected_imagined_rollout(
        self, selected_plan: torch.Tensor
    ) -> np.ndarray:
        """Return flattened z_hat[t+1:t+H | t] for the selected MPC plan."""
        if self._latent_context is None:
            raise RuntimeError("cannot serialize a rollout without latent context")
        if tuple(selected_plan.shape) != (self.config.horizon, ACTION_DIM):
            raise ValueError("selected plan shape does not match the horizon")
        trajectory = self._rollout_latents(
            self._latent_context, selected_plan.unsqueeze(0)
        )[0, 1:]
        rollout = (
            trajectory.detach()
            .float()
            .cpu()
            .numpy()
            .reshape(self.config.horizon, -1)
            .astype(np.float32, copy=False)
        )
        if rollout.shape[1] < 1 or not np.isfinite(rollout).all():
            raise RuntimeError("selected imagined rollout is empty or non-finite")
        return rollout.copy()

    def _rollout_cost(
        self,
        latent_context: torch.Tensor,
        goal_embedding: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        rollout = self._rollout_latents(latent_context, raw_actions)
        terminal_error = (rollout[:, -1] - goal_embedding[:, -1]).square()
        cost = terminal_error.flatten(start_dim=1).mean(dim=1)
        if self.config.action_l2_weight:
            scaled = raw_actions / self.config.action_cap_m
            cost = cost + self.config.action_l2_weight * scaled.square().mean(
                dim=(1, 2)
            )
        if self.config.smoothness_weight and self.config.horizon > 1:
            differences = raw_actions[:, 1:] - raw_actions[:, :-1]
            cost = cost + self.config.smoothness_weight * (
                differences / self.config.action_cap_m
            ).square().mean(dim=(1, 2))
        if not torch.isfinite(cost).all():
            raise RuntimeError("CEM objective produced a non-finite cost")
        return cost

    def _evaluate_candidate_costs(
        self,
        latent_context: torch.Tensor,
        goal_embedding: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate candidates in bounded batches when spatial latents are large."""
        batch_size = int(self.config.evaluation_batch_size)
        if batch_size <= 0 or candidates.shape[0] <= batch_size:
            return self._rollout_cost(latent_context, goal_embedding, candidates)
        return torch.cat(
            [
                self._rollout_cost(
                    latent_context,
                    goal_embedding,
                    candidates[start : start + batch_size],
                )
                for start in range(0, candidates.shape[0], batch_size)
            ],
            dim=0,
        )

    def _categorical_cem_search(
        self,
        latent_context: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Search exact keyboard tokens with per-horizon categorical CEM."""
        if self.action_vocabulary is None:
            raise RuntimeError("categorical-cem requires an action vocabulary")
        vocabulary = self.action_vocabulary
        num_actions = int(vocabulary.shape[0])
        if self.config.categorical_min_prob >= 1.0 / num_actions:
            raise ValueError(
                "--categorical-min-prob must be smaller than "
                f"{1.0 / num_actions:.6g} for {num_actions} action tokens"
            )
        uniform = torch.full(
            (self.config.horizon, num_actions),
            1.0 / num_actions,
            dtype=torch.float32,
            device=self.device,
        )
        if self._previous_action_probabilities is None:
            probabilities = uniform
        else:
            previous = self._previous_action_probabilities
            if previous.shape != uniform.shape:
                raise RuntimeError("categorical warm-start distribution shape drifted")
            probabilities = torch.cat([previous[1:], uniform[-1:]], dim=0)

        zero_index = int(
            torch.argmin(torch.linalg.vector_norm(vocabulary, dim=1)).item()
        )
        sampled_best_cost = float("inf")
        sampled_best_plan: torch.Tensor | None = None
        for _ in range(self.config.iterations):
            candidate_indices = sample_categorical_indices(
                probabilities,
                num_samples=self.config.num_samples,
                generator=self.generator,
            )
            candidate_indices[0] = probabilities.argmax(dim=1)
            candidate_indices[1].fill_(zero_index)
            candidates = vocabulary[candidate_indices]
            costs = self._evaluate_candidate_costs(
                latent_context, goal_embedding, candidates
            )
            iteration_best = int(torch.argmin(costs).item())
            iteration_best_cost = float(costs[iteration_best].item())
            if iteration_best_cost < sampled_best_cost:
                sampled_best_cost = iteration_best_cost
                sampled_best_plan = candidates[iteration_best].detach().clone()
            elite_rows = torch.topk(
                costs, k=self.config.elite_count, largest=False
            ).indices
            probabilities = categorical_elite_update(
                probabilities,
                candidate_indices[elite_rows],
                alpha=self.config.categorical_alpha,
                min_prob=self.config.categorical_min_prob,
            )

        mode_plan = vocabulary[probabilities.argmax(dim=1)]
        mode_cost = float(
            self._rollout_cost(
                latent_context,
                goal_embedding,
                mode_plan.unsqueeze(0),
            )[0].item()
        )
        if sampled_best_plan is not None and sampled_best_cost < mode_cost:
            selected_plan = sampled_best_plan
            selected_cost = sampled_best_cost
        else:
            selected_plan = mode_plan
            selected_cost = mode_cost
        return selected_plan, selected_cost, probabilities

    def _search_plan(
        self,
        latent_context: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, float, np.ndarray | None]:
        """Run the configured optimizer against a backend-specific rollout."""
        if self.config.solver == "categorical-cem":
            (
                selected_plan,
                selected_cost,
                final_probabilities,
            ) = self._categorical_cem_search(latent_context, goal_embedding)
            self._previous_action_probabilities = (
                final_probabilities.detach().clone()
            )
            return (
                selected_plan,
                selected_cost,
                final_probabilities.detach().cpu().numpy().astype(np.float32),
            )

        shape = (self.config.num_samples, self.config.horizon, ACTION_DIM)
        if self._previous_plan is None:
            mean = torch.zeros(
                self.config.horizon,
                ACTION_DIM,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            mean = torch.cat(
                [
                    self._previous_plan[1:],
                    torch.zeros(1, ACTION_DIM, device=self.device),
                ],
                dim=0,
            )
        std = torch.full_like(mean, self.config.initial_std_m)

        sampled_best_cost = float("inf")
        sampled_best_plan: torch.Tensor | None = None
        for _ in range(self.config.iterations):
            candidates = torch.randn(
                shape,
                generator=self.generator,
                dtype=torch.float32,
                device=self.device,
            )
            candidates = candidates * std.unsqueeze(0) + mean.unsqueeze(0)
            candidates[0] = mean
            candidates[1].zero_()
            candidates = self._constrain_actions(candidates)
            costs = self._evaluate_candidate_costs(
                latent_context, goal_embedding, candidates
            )
            iteration_best = int(torch.argmin(costs).item())
            iteration_best_cost = float(costs[iteration_best].item())
            if iteration_best_cost < sampled_best_cost:
                sampled_best_cost = iteration_best_cost
                sampled_best_plan = candidates[iteration_best].detach().clone()
            elite_indices = torch.topk(
                costs, k=self.config.elite_count, largest=False
            ).indices
            elites = candidates[elite_indices]
            mean = clamp_action_norm(
                elites.mean(dim=0), self.config.action_cap_m
            )
            std = elites.std(dim=0, unbiased=False).clamp_min(
                self.config.min_std_m
            )
        mean_plan = self._constrain_actions(mean.unsqueeze(0))[0]
        mean_cost = float(
            self._rollout_cost(
                latent_context,
                goal_embedding,
                mean_plan.unsqueeze(0),
            )[0].item()
        )
        if sampled_best_plan is not None and sampled_best_cost < mean_cost:
            return sampled_best_plan, sampled_best_cost, None
        return mean_plan, mean_cost, None

    @torch.inference_mode()
    def predict_next_latent(self, action: Any) -> dict[str, np.ndarray]:
        """Predict one transition from the current MPC context without mutation."""
        if self._latent_context is None:
            raise RuntimeError("cannot predict a transition before planning")
        raw_action = np.asarray(action, dtype=np.float32)
        if raw_action.shape != (ACTION_DIM,) or not np.isfinite(raw_action).all():
            raise ValueError("transition action must be a finite XY delta")
        if float(np.linalg.norm(raw_action)) > self.config.action_cap_m + 1e-7:
            raise ValueError("transition action exceeds the configured action cap")
        action_tensor = torch.as_tensor(
            raw_action, dtype=torch.float32, device=self.device
        ).reshape(1, 1, ACTION_DIM)
        trajectory = self._rollout_latents(self._latent_context, action_tensor)[0]
        encoded = trajectory[0].detach().cpu().numpy().astype(np.float32)
        predicted = trajectory[1].detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(encoded).all() or not np.isfinite(predicted).all():
            raise RuntimeError("transition prediction produced a non-finite latent")
        return {
            "encoded_latent": encoded,
            "predicted_next_latent": predicted,
            "prediction_action": raw_action.copy(),
        }

    @torch.inference_mode()
    def plan(
        self,
        current: np.ndarray,
        goal: np.ndarray,
        previous_action: np.ndarray | None = None,
        current_proprio: np.ndarray | None = None,
    ) -> dict[str, Any]:
        del current_proprio  # LeWM was trained without proprioception.
        started = time.monotonic()
        current_embedding = self._encode(current)
        if self._latent_context is None:
            if previous_action is not None:
                raise ValueError("previous_action is invalid for a fresh context")
            self._latent_context = current_embedding
        else:
            action = np.asarray(previous_action, dtype=np.float32)
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError("previous_action must be a finite XY delta")
            if float(np.linalg.norm(action)) > self.config.action_cap_m + 1e-7:
                raise ValueError("previous_action exceeds the configured action cap")
            normalized_action = (
                torch.as_tensor(action, device=self.device).reshape(1, 1, ACTION_DIM)
                - self.action_mean
            ) / self.action_std
            if self._executed_actions is None:
                self._executed_actions = normalized_action
            else:
                self._executed_actions = torch.cat(
                    [self._executed_actions, normalized_action], dim=1
                )
            self._latent_context = torch.cat(
                [self._latent_context, current_embedding], dim=1
            )
            self._latent_context = self._latent_context[:, -self.history_size :]
            required_actions = self._latent_context.shape[1] - 1
            self._executed_actions = (
                self._executed_actions[:, -required_actions:]
                if required_actions
                else None
            )

        goal_embedding = self._encode(goal)
        goal_distance = float(
            (current_embedding[:, -1] - goal_embedding[:, -1])
            .square()
            .mean()
            .item()
        )
        if not np.isfinite(goal_distance) or goal_distance < 0.0:
            raise RuntimeError("goal distance is not finite and non-negative")
        if goal_distance <= self.config.goal_tolerance:
            self.reset()
            return {
                "action": np.zeros(ACTION_DIM, dtype=np.float32),
                "plan": np.zeros((self.config.horizon, ACTION_DIM), dtype=np.float32),
                "cost": goal_distance,
                "goal_distance": goal_distance,
                "solve_time_s": time.monotonic() - started,
                "at_goal": True,
                "solver": self.config.solver,
                "imagined_latent_rollout": None,
            }

        (
            selected_plan,
            selected_cost,
            action_probabilities,
        ) = self._search_plan(self._latent_context, goal_embedding)

        self._previous_plan = selected_plan.detach().clone()
        imagined_latent_rollout = self._selected_imagined_rollout(selected_plan)
        plan = selected_plan.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(plan).all() or not np.isfinite(selected_cost):
            raise RuntimeError("CEM produced a non-finite plan or cost")
        result = {
            "action": plan[0].copy(),
            "plan": plan,
            "cost": selected_cost,
            "goal_distance": goal_distance,
            "solve_time_s": time.monotonic() - started,
            "at_goal": False,
            "solver": self.config.solver,
            "imagined_latent_rollout": imagined_latent_rollout,
        }
        if action_probabilities is not None:
            result["action_probabilities"] = action_probabilities
            result["action_vocabulary"] = (
                self.action_vocabulary.detach().cpu().numpy().astype(np.float32)
            )
        return result


class DinoPushBoxPlanner(PushBoxPlanner):
    """PushBox planner for spatial DINO-WM image/proprio/action embeddings."""

    PROPRIO_DIM = 4

    def __init__(
        self,
        model: torch.nn.Module,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        proprio_mean: np.ndarray,
        proprio_std: np.ndarray,
        *,
        config: CEMConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        extra_encoders = self.model.extra_encoders
        if "action" not in extra_encoders or "proprio" not in extra_encoders:
            raise ValueError(
                "DINO-WM checkpoint must contain action and proprio encoders"
            )
        action_input_dim = int(
            extra_encoders["action"].patch_embed.in_channels
        )
        proprio_input_dim = int(
            extra_encoders["proprio"].patch_embed.in_channels
        )
        if action_input_dim != ACTION_DIM:
            raise ValueError(
                f"checkpoint action dimension is {action_input_dim}, "
                f"expected {ACTION_DIM}"
            )
        if proprio_input_dim != self.PROPRIO_DIM:
            raise ValueError(
                f"checkpoint proprio dimension is {proprio_input_dim}, "
                f"expected {self.PROPRIO_DIM}"
            )

        self.history_size = int(self.model.history_size)
        predictor_history = int(self.model.predictor.num_frames)
        if predictor_history != self.history_size:
            raise ValueError("DINO-WM model and predictor history sizes differ")
        self.num_patches = int(self.model.predictor.num_patches)
        self.pixel_dim = int(self.model.backbone.config.hidden_size)
        self.embedding_dim = int(self.model.predictor.pos_embedding.shape[-1])

        feature_ranges: dict[str, tuple[int, int]] = {}
        start = self.pixel_dim
        for key, encoder in extra_encoders.items():
            width = int(encoder.emb_dim)
            feature_ranges[str(key)] = (start, start + width)
            start += width
        if start != self.embedding_dim:
            raise ValueError(
                "DINO-WM predictor dimension does not match pixel and extra "
                f"encoders: {self.embedding_dim} != {start}"
            )
        self.action_feature_range = feature_ranges["action"]

        self.action_mean = torch.as_tensor(
            action_mean, dtype=torch.float32, device=self.device
        ).reshape(1, 1, ACTION_DIM)
        self.action_std = torch.as_tensor(
            action_std, dtype=torch.float32, device=self.device
        ).reshape(1, 1, ACTION_DIM)
        self.proprio_mean = torch.as_tensor(
            proprio_mean, dtype=torch.float32, device=self.device
        ).reshape(1, 1, self.PROPRIO_DIM)
        self.proprio_std = torch.as_tensor(
            proprio_std, dtype=torch.float32, device=self.device
        ).reshape(1, 1, self.PROPRIO_DIM)
        for name, mean, std in (
            ("action", self.action_mean, self.action_std),
            ("proprio", self.proprio_mean, self.proprio_std),
        ):
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError(f"{name} statistics must be finite")
            if torch.any(std <= 0):
                raise ValueError(f"{name} standard deviations must be positive")

        stats = spt.data.dataset_stats.ImageNet
        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=stats["mean"], std=stats["std"]),
                v2.Resize((224, 224)),
            ]
        )
        generator_device = (
            self.device.type if self.device.type == "cuda" else "cpu"
        )
        self.generator = torch.Generator(device=generator_device).manual_seed(
            config.seed
        )
        self.action_vocabulary = (
            torch.as_tensor(
                keyboard_action_vocabulary(config.action_cap_m),
                dtype=torch.float32,
                device=self.device,
            )
            if config.action_mode == "keyboard"
            else None
        )
        if (
            config.solver == "categorical-cem"
            and self.action_vocabulary is not None
            and config.categorical_min_prob
            >= 1.0 / int(self.action_vocabulary.shape[0])
        ):
            raise ValueError(
                "categorical_min_prob is too large for the action vocabulary"
            )
        self._previous_plan: torch.Tensor | None = None
        self._previous_action_probabilities: torch.Tensor | None = None
        self._latent_context: torch.Tensor | None = None
        self._executed_actions: torch.Tensor | None = None
        self._goal_cache_image: np.ndarray | None = None
        self._goal_cache_embedding: torch.Tensor | None = None

    @classmethod
    def from_artifacts(
        cls,
        *,
        checkpoint: str,
        dataset_path: str | Path,
        config: CEMConfig,
        device: str = "cuda",
        model_manifest: str | Path | None = None,
    ) -> "DinoPushBoxPlanner":
        path = Path(dataset_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PushBox dataset not found: {path}")
        checkpoint_path = resolve_local_checkpoint(checkpoint)
        manifest_path = (
            Path(model_manifest).expanduser().resolve()
            if model_manifest is not None
            else checkpoint_path.parent / "split_manifest.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "DINO-WM split_manifest.json is required for exact action and "
                f"proprio normalization: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        if float(manifest.get("sample_hz", 0.0)) != 5.0:
            raise ValueError("DINO-WM manifest must describe 5 Hz training")
        train_episodes = manifest.get("train_episode_indices")
        normalization = manifest.get("normalization", {})
        if not isinstance(train_episodes, list):
            raise ValueError("DINO-WM manifest has no train_episode_indices")
        if not all(key in normalization for key in ("action", "proprio")):
            raise ValueError(
                "DINO-WM manifest must contain action and proprio normalization"
            )

        dataset = swm.data.load_dataset(
            str(path),
            transform=None,
            keys_to_load=["action", "proprio"],
            keys_to_cache=["action", "proprio"],
            num_steps=4,
            frameskip=1,
        )
        statistics: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for column in ("action", "proprio"):
            mean, std, num_rows = compute_episode_column_stats(
                dataset, column, train_episodes
            )
            expected = normalization[column]
            if not np.allclose(
                mean, expected["mean"], rtol=1e-6, atol=1e-8
            ):
                raise ValueError(
                    f"dataset {column} mean does not match DINO-WM manifest"
                )
            if not np.allclose(
                std, expected["std"], rtol=1e-6, atol=1e-8
            ):
                raise ValueError(
                    f"dataset {column} std does not match DINO-WM manifest"
                )
            if int(expected.get("num_rows", num_rows)) != num_rows:
                raise ValueError(
                    f"dataset {column} row count does not match DINO-WM manifest"
                )
            statistics[column] = (mean, std, num_rows)

        action_mean, action_std, _ = statistics["action"]
        proprio_mean, proprio_std, _ = statistics["proprio"]
        print(
            "DINO-WM normalization: "
            f"action_mean={action_mean.tolist()}, "
            f"action_std={action_std.tolist()}, "
            f"proprio_mean={proprio_mean.tolist()}, "
            f"proprio_std={proprio_std.tolist()}, "
            f"train_episodes={len(train_episodes)}"
        )
        model_config = json.loads(
            (checkpoint_path.parent / "config.json").read_text()
        )
        encoder_config = model_config.get("encoder", {})
        if encoder_config.get("name") not in (
            "dinov2_small",
            "facebook/dinov2-small",
        ):
            raise ValueError(
                "real-robot DINO-WM currently supports the trained "
                "DINOv2-small backbone only"
            )
        model = swm.wm.utils.load_pretrained(
            str(checkpoint_path),
            extra_args={
                "encoder._target_": (
                    "real_robot_planner."
                    "create_dinov2_small_checkpoint_backbone"
                )
            },
        )
        planner = cls(
            model,
            action_mean,
            action_std,
            proprio_mean,
            proprio_std,
            config=config,
            device=device,
        )
        if planner.action_vocabulary is not None:
            rows = np.concatenate(
                [
                    np.arange(
                        dataset.offsets[episode],
                        dataset.offsets[episode] + dataset.lengths[episode],
                    )
                    for episode in train_episodes
                ]
            )
            train_actions = np.asarray(
                dataset.get_col_data("action"), dtype=np.float32
            )[rows]
            vocabulary = planner.action_vocabulary.detach().cpu().numpy()
            counts = np.asarray(
                [
                    np.count_nonzero(
                        np.all(
                            np.isclose(
                                train_actions, token, rtol=0.0, atol=1e-7
                            ),
                            axis=1,
                        )
                    )
                    for token in vocabulary
                ],
                dtype=np.int64,
            )
            if np.any(counts == 0):
                raise ValueError(
                    "DINO-WM action vocabulary contains actions absent from "
                    "its training subset"
                )
            print(
                "DINO-WM keyboard action vocabulary: "
                f"{len(vocabulary)} exact actions, training counts "
                f"min={int(counts.min())}, max={int(counts.max())}"
            )
        return planner

    def describe(self) -> dict[str, Any]:
        semantic_shape = [self.num_patches, self.pixel_dim]
        latent_dim = int(np.prod(semantic_shape))
        return {
            "world_model": "dinowm",
            "history_size": self.history_size,
            "latent_dim": latent_dim,
            "latent_storage_shape": [latent_dim],
            "latent_semantic_shape": semantic_shape,
            "latent_representation": "flattened_dinov2_patch_tokens",
            "requires_proprio": True,
            "proprio_order": ["x", "y", "vx", "vy"],
        }

    def reset(self) -> None:
        super().reset()
        self._goal_cache_image = None
        self._goal_cache_embedding = None

    def _normalize_action(self, action: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).reshape(1, 1, ACTION_DIM)
        return (tensor - self.action_mean) / self.action_std

    def _encode_observation(
        self, image: np.ndarray, proprio: np.ndarray
    ) -> torch.Tensor:
        raw_proprio = np.asarray(proprio, dtype=np.float32)
        if (
            raw_proprio.shape != (self.PROPRIO_DIM,)
            or not np.isfinite(raw_proprio).all()
        ):
            raise ValueError("current_proprio must be finite [x, y, vx, vy]")
        normalized_proprio = (
            torch.as_tensor(
                raw_proprio, dtype=torch.float32, device=self.device
            ).reshape(1, 1, self.PROPRIO_DIM)
            - self.proprio_mean
        ) / self.proprio_std
        normalized_zero_action = (
            torch.zeros(
                1, 1, ACTION_DIM, dtype=torch.float32, device=self.device
            )
            - self.action_mean
        ) / self.action_std
        output = self.model.encode(
            {
                "pixels": self._prepare_image(image),
                "proprio": normalized_proprio,
                "action": normalized_zero_action,
            },
            target="emb",
            is_video=False,
        )
        embedding = output["emb"]
        expected = (1, 1, self.num_patches, self.embedding_dim)
        if tuple(embedding.shape) != expected:
            raise RuntimeError(
                f"DINO-WM observation embedding has shape "
                f"{tuple(embedding.shape)}, expected {expected}"
            )
        if not torch.isfinite(embedding).all():
            raise RuntimeError("DINO-WM observation embedding is non-finite")
        return embedding

    def _encode_goal(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if (
            self._goal_cache_image is not None
            and np.array_equal(array, self._goal_cache_image)
        ):
            assert self._goal_cache_embedding is not None
            return self._goal_cache_embedding
        output = self.model.encode(
            {"pixels": self._prepare_image(array)},
            emb_keys=[],
            target="emb",
            is_video=False,
        )
        embedding = output["pixels_emb"]
        expected = (1, 1, self.num_patches, self.pixel_dim)
        if tuple(embedding.shape) != expected:
            raise RuntimeError(
                f"DINO-WM goal embedding has shape {tuple(embedding.shape)}, "
                f"expected {expected}"
            )
        if not torch.isfinite(embedding).all():
            raise RuntimeError("DINO-WM goal embedding is non-finite")
        self._goal_cache_image = array.copy()
        self._goal_cache_embedding = embedding
        return embedding

    def _replace_action_features(
        self, embedding: torch.Tensor, normalized_actions: torch.Tensor
    ) -> torch.Tensor:
        if embedding.shape[:2] != normalized_actions.shape[:2]:
            raise RuntimeError("DINO-WM latent/action history alignment drifted")
        action_embedding = self.model.extra_encoders["action"](
            normalized_actions
        )
        tiled = action_embedding.unsqueeze(2).expand(
            -1, -1, embedding.shape[2], -1
        )
        start, end = self.action_feature_range
        return torch.cat(
            [embedding[..., :start], tiled, embedding[..., end:]], dim=-1
        )

    def _rollout_latents(
        self,
        latent_context: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return current and recursively predicted DINO image-token states."""
        if raw_actions.ndim != 3 or raw_actions.shape[-1] != ACTION_DIM:
            raise ValueError("raw_actions must have shape (samples, horizon, 2)")
        normalized = (raw_actions - self.action_mean) / self.action_std
        num_samples = raw_actions.shape[0]
        embeddings = latent_context.expand(num_samples, -1, -1, -1)
        if self._executed_actions is None:
            action_prefix = torch.empty(
                num_samples, 0, ACTION_DIM, device=self.device
            )
        else:
            action_prefix = self._executed_actions.expand(
                num_samples, -1, -1
            )
        predicted_pixels: list[torch.Tensor] = []

        for step in range(raw_actions.shape[1]):
            all_actions = torch.cat(
                [action_prefix, normalized[:, : step + 1]], dim=1
            )
            history_len = min(self.history_size, embeddings.shape[1])
            latent_history = embeddings[:, -history_len:]
            action_history = all_actions[:, -history_len:]
            conditioned_history = self._replace_action_features(
                latent_history, action_history
            )
            prediction = self.model.predict(conditioned_history)[:, -1:]
            if not torch.isfinite(prediction).all():
                raise RuntimeError("DINO-WM produced a non-finite prediction")
            embeddings = torch.cat([embeddings, prediction], dim=1)
            predicted_pixels.append(prediction[..., : self.pixel_dim])

        current = latent_context[:, -1:, :, : self.pixel_dim].expand(
            num_samples, -1, -1, -1
        )
        return torch.cat([current, *predicted_pixels], dim=1)

    @torch.inference_mode()
    def predict_next_latent(self, action: Any) -> dict[str, np.ndarray]:
        if self._latent_context is None:
            raise RuntimeError("cannot predict a transition before planning")
        raw_action = np.asarray(action, dtype=np.float32)
        if raw_action.shape != (ACTION_DIM,) or not np.isfinite(raw_action).all():
            raise ValueError("transition action must be a finite XY delta")
        if float(np.linalg.norm(raw_action)) > self.config.action_cap_m + 1e-7:
            raise ValueError("transition action exceeds the configured action cap")
        action_tensor = torch.as_tensor(
            raw_action, dtype=torch.float32, device=self.device
        ).reshape(1, 1, ACTION_DIM)
        trajectory = self._rollout_latents(
            self._latent_context, action_tensor
        )[0]
        encoded = (
            trajectory[0].detach().float().cpu().numpy().reshape(-1)
        ).astype(np.float32)
        predicted = (
            trajectory[1].detach().float().cpu().numpy().reshape(-1)
        ).astype(np.float32)
        if not np.isfinite(encoded).all() or not np.isfinite(predicted).all():
            raise RuntimeError("transition prediction produced a non-finite latent")
        return {
            "encoded_latent": encoded,
            "predicted_next_latent": predicted,
            "prediction_action": raw_action.copy(),
        }

    @torch.inference_mode()
    def plan(
        self,
        current: np.ndarray,
        goal: np.ndarray,
        previous_action: np.ndarray | None = None,
        current_proprio: np.ndarray | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if current_proprio is None:
            raise ValueError("DINO-WM planning requires current_proprio")
        current_embedding = self._encode_observation(current, current_proprio)
        if self._latent_context is None:
            if previous_action is not None:
                raise ValueError("previous_action is invalid for a fresh context")
            self._latent_context = current_embedding
        else:
            action = np.asarray(previous_action, dtype=np.float32)
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError("previous_action must be a finite XY delta")
            if float(np.linalg.norm(action)) > self.config.action_cap_m + 1e-7:
                raise ValueError("previous_action exceeds the configured action cap")
            normalized_action = self._normalize_action(action)
            if self._executed_actions is None:
                self._executed_actions = normalized_action
            else:
                self._executed_actions = torch.cat(
                    [self._executed_actions, normalized_action], dim=1
                )
            self._latent_context = torch.cat(
                [self._latent_context, current_embedding], dim=1
            )
            self._latent_context = self._latent_context[
                :, -self.history_size :
            ]
            required_actions = self._latent_context.shape[1] - 1
            self._executed_actions = (
                self._executed_actions[:, -required_actions:]
                if required_actions
                else None
            )

        goal_embedding = self._encode_goal(goal)
        goal_distance = float(
            (
                current_embedding[:, -1, :, : self.pixel_dim]
                - goal_embedding[:, -1]
            )
            .square()
            .mean()
            .item()
        )
        if not np.isfinite(goal_distance) or goal_distance < 0.0:
            raise RuntimeError("goal distance is not finite and non-negative")
        if goal_distance <= self.config.goal_tolerance:
            self.reset()
            return {
                "action": np.zeros(ACTION_DIM, dtype=np.float32),
                "plan": np.zeros(
                    (self.config.horizon, ACTION_DIM), dtype=np.float32
                ),
                "cost": goal_distance,
                "goal_distance": goal_distance,
                "solve_time_s": time.monotonic() - started,
                "at_goal": True,
                "solver": self.config.solver,
                "imagined_latent_rollout": None,
            }

        (
            selected_plan,
            selected_cost,
            action_probabilities,
        ) = self._search_plan(self._latent_context, goal_embedding)
        self._previous_plan = selected_plan.detach().clone()
        imagined_latent_rollout = self._selected_imagined_rollout(selected_plan)
        plan = selected_plan.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(plan).all() or not np.isfinite(selected_cost):
            raise RuntimeError("CEM produced a non-finite plan or cost")
        result = {
            "action": plan[0].copy(),
            "plan": plan,
            "cost": selected_cost,
            "goal_distance": goal_distance,
            "solve_time_s": time.monotonic() - started,
            "at_goal": False,
            "solver": self.config.solver,
            "imagined_latent_rollout": imagined_latent_rollout,
        }
        if action_probabilities is not None:
            result["action_probabilities"] = action_probabilities
            result["action_vocabulary"] = (
                self.action_vocabulary.detach().cpu().numpy().astype(np.float32)
            )
        return result


def serve(args: argparse.Namespace) -> int:
    config = CEMConfig(
        solver=args.solver,
        horizon=args.horizon,
        num_samples=args.num_samples,
        iterations=args.iterations,
        elite_count=args.elite_count,
        initial_std_m=args.initial_std,
        min_std_m=args.min_std,
        action_cap_m=args.action_cap,
        action_l2_weight=args.action_l2_weight,
        smoothness_weight=args.smoothness_weight,
        goal_tolerance=args.goal_tolerance,
        action_mode=args.action_mode,
        categorical_alpha=args.categorical_alpha,
        categorical_min_prob=args.categorical_min_prob,
        evaluation_batch_size=args.cem_batch_size,
        seed=args.seed,
    )
    checkpoint = args.checkpoint or (
        DEFAULT_DINOWM_CHECKPOINT
        if args.world_model == "dinowm"
        else DEFAULT_LEWM_CHECKPOINT
    )
    if args.world_model == "dinowm":
        planner = DinoPushBoxPlanner.from_artifacts(
            checkpoint=checkpoint,
            dataset_path=args.dataset,
            config=config,
            device=args.device,
            model_manifest=args.model_manifest,
        )
    else:
        planner = PushBoxPlanner.from_artifacts(
            checkpoint=checkpoint,
            dataset_path=args.dataset,
            config=config,
            device=args.device,
            artifact_manifest=args.artifact_manifest,
        )
    socket_path = Path(args.socket)
    socket_path.unlink(missing_ok=True)
    listener = Listener(
        str(socket_path), family="AF_UNIX", authkey=bytes.fromhex(args.authkey)
    )
    print(f"PushBox planner ready on {socket_path}", flush=True)
    try:
        connection = listener.accept()
        try:
            while True:
                request = connection.recv()
                operation = request.get("op")
                if operation == "shutdown":
                    connection.send({"ok": True})
                    break
                if operation == "reset":
                    planner.reset()
                    connection.send({"ok": True})
                    continue
                if operation == "describe":
                    connection.send({"ok": True, **planner.describe()})
                    continue
                if operation == "predict_next":
                    try:
                        result = planner.predict_next_latent(request["action"])
                        connection.send({"ok": True, **result})
                    except Exception as exc:
                        connection.send(
                            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        )
                    continue
                if operation != "plan":
                    connection.send(
                        {"ok": False, "error": f"unknown operation: {operation!r}"}
                    )
                    continue
                try:
                    if request.get("reset", False):
                        planner.reset()
                    result = planner.plan(
                        request["current"],
                        request["goal"],
                        request.get("previous_action"),
                        request.get("current_proprio"),
                    )
                    connection.send({"ok": True, **result})
                except Exception as exc:
                    connection.send(
                        {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    )
        finally:
            connection.close()
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "checkpoint override; defaults to epoch-146 LeWM or the local "
            "epoch-10 4-hour DINO-WM checkpoint"
        ),
    )
    parser.add_argument(
        "--world-model",
        choices=("lewm", "dinowm"),
        default="lewm",
        help="world-model backend used for encoding and recursive prediction",
    )
    parser.add_argument(
        "--model-manifest",
        help=(
            "DINO-WM split_manifest.json; defaults to the checkpoint directory"
        ),
    )
    parser.add_argument(
        "--dataset", default="stable-wm/datasets/pushbox_pilot_train.h5"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--solver",
        choices=("cem", "categorical-cem"),
        default="cem",
        help="Gaussian CEM or direct categorical search over keyboard tokens",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elite-count", type=int, default=32)
    parser.add_argument("--initial-std", type=float, default=0.005)
    parser.add_argument("--min-std", type=float, default=0.00025)
    parser.add_argument("--action-cap", type=float, default=ACTION_CAP_M)
    parser.add_argument(
        "--action-mode",
        choices=("continuous", "keyboard"),
        default="keyboard",
        help="continuous CEM or exact collected keyboard-action vocabulary",
    )
    parser.add_argument("--action-l2-weight", type=float, default=0.001)
    parser.add_argument("--smoothness-weight", type=float, default=0.001)
    parser.add_argument("--goal-tolerance", type=float, default=0.0)
    parser.add_argument("--categorical-alpha", type=float, default=0.7)
    parser.add_argument("--categorical-min-prob", type=float, default=0.01)
    parser.add_argument(
        "--cem-batch-size",
        type=int,
        default=0,
        help="candidate rollout batch size; zero evaluates all candidates together",
    )
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--artifact-manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return serve(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
