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
    seed: int = 3072

    def validate(self) -> None:
        scalar_values = (
            self.initial_std_m,
            self.min_std_m,
            self.action_cap_m,
            self.action_l2_weight,
            self.smoothness_weight,
            self.goal_tolerance,
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
        if not 0.0 < self.min_std_m <= self.initial_std_m:
            raise ValueError("require 0 < min_std_m <= initial_std_m")
        if not 0.0 < self.action_cap_m <= ACTION_CAP_M:
            raise ValueError(f"action_cap_m must be in (0, {ACTION_CAP_M}]")
        if self.action_l2_weight < 0.0 or self.smoothness_weight < 0.0:
            raise ValueError("regularization weights must be non-negative")
        if self.goal_tolerance < 0.0:
            raise ValueError("goal_tolerance must be non-negative")
        if self.action_mode not in {"continuous", "keyboard"}:
            raise ValueError("action_mode must be 'continuous' or 'keyboard'")


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
        self._previous_plan: torch.Tensor | None = None
        self._latent_context: torch.Tensor | None = None
        self._executed_actions: torch.Tensor | None = None

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

    def _rollout_cost(
        self,
        latent_context: torch.Tensor,
        goal_embedding: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        rollout = self._rollout_latents(latent_context, raw_actions)
        cost = (rollout[:, -1] - goal_embedding[:, -1]).square().mean(dim=-1)
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
    ) -> dict[str, Any]:
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
            }

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
            costs = self._rollout_cost(
                self._latent_context, goal_embedding, candidates
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
                self._latent_context,
                goal_embedding,
                mean_plan.unsqueeze(0),
            )[0].item()
        )
        if sampled_best_plan is not None and sampled_best_cost < mean_cost:
            selected_plan = sampled_best_plan
            selected_cost = sampled_best_cost
        else:
            selected_plan = mean_plan
            selected_cost = mean_cost

        self._previous_plan = selected_plan.detach().clone()
        plan = selected_plan.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(plan).all() or not np.isfinite(selected_cost):
            raise RuntimeError("CEM produced a non-finite plan or cost")
        return {
            "action": plan[0].copy(),
            "plan": plan,
            "cost": selected_cost,
            "goal_distance": goal_distance,
            "solve_time_s": time.monotonic() - started,
            "at_goal": False,
        }


def serve(args: argparse.Namespace) -> int:
    config = CEMConfig(
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
        seed=args.seed,
    )
    planner = PushBoxPlanner.from_artifacts(
        checkpoint=args.checkpoint,
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
        "--checkpoint", default="pushbox/lewm/weights_epoch_146.pt"
    )
    parser.add_argument(
        "--dataset", default="stable-wm/datasets/pushbox_pilot_train.h5"
    )
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--artifact-manifest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return serve(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
