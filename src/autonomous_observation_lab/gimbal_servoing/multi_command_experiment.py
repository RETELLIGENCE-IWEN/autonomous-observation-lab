"""V10 development protocol for recurrent multi-command gimbal control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .adaptive_curriculum_objective import (
    selected_adaptive_position_v21_config,
)
from .adaptive_position_supervision import (
    AdaptivePositionSupervision,
    compute_adaptive_position_supervision,
)
from .config import ObservationProfile
from .control_criticality import (
    ControlCriticality,
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from .critical_curriculum import (
    CriticalEpisodeCurriculumConfig,
    compute_critical_episode_curriculum,
    critical_episode_curriculum_report,
)
from .dataset import FEATURE_NAMES, GimbalTargetStateDataset, load_gimbal_dataset
from .gru import GRUAdaptivePositionLossContext, load_gru_checkpoint
from .gru_training import set_gru_seed
from .multi_command_policy import (
    CausalRecurrentPositionResidualPolicy,
    CounterfactualWindowBatch,
    RecurrentPositionResidualPolicyConfig,
    counterfactual_capture_source_indices,
    recurrent_policy_input_dim,
    rollout_counterfactual_window,
)


MULTI_COMMAND_EXPERIMENT_SCHEMA_VERSION = (
    "gimbal_multi_command_counterfactual_v10_development_v1"
)


@dataclass(frozen=True)
class MultiCommandExperimentConfig:
    base_seed: int = 29
    optimization_seed: int = 29
    epochs: int = 6
    batches_per_epoch: int = 12
    batch_size: int = 24
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    window_steps: int = 8
    earliest_training_start_index: int = 8
    validation_start_indices: tuple[int, ...] = (48, 120)
    exact_epoch_candidate_count: int = 3
    training_integration_period_s: float = 0.010
    tracking_weight: float = 7.5
    visibility_weight: float = 2.0
    smoothness_weight: float = 30.0
    saturation_weight: float = 0.05
    residual_magnitude_weight: float = 0.25
    teacher_action_weight: float = 0.0
    terminal_tracking_weight: float = 2.0
    critical_step_weight: float = 1.0
    visibility_margin_fraction: float = 0.85
    maximum_state_regression_fraction: float = 0.0
    maximum_action_regression_fraction: float = 0.02
    minimum_tracking_improvement_fraction: float = 0.005
    maximum_saturation_regression_fraction: float = 0.05
    minimum_training_episodes: int = 1000
    minimum_validation_episodes: int = 200
    device: str = "cpu"
    residual_policy_hidden_dim: int = 32
    residual_policy_embedding_dim: int = 32
    maximum_policy_residual_magnitude: float = 0.25
    residual_application: str = "target_half_fov"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()
    curriculum: CriticalEpisodeCurriculumConfig = (
        CriticalEpisodeCurriculumConfig(concentration_strength=1.0)
    )

    def __post_init__(self) -> None:
        if self.base_seed < 0 or self.optimization_seed < 0:
            raise ValueError("multi-command seeds must be non-negative")
        for name in (
            "epochs",
            "batches_per_epoch",
            "batch_size",
            "window_steps",
            "exact_epoch_candidate_count",
            "minimum_training_episodes",
            "minimum_validation_episodes",
            "residual_policy_hidden_dim",
            "residual_policy_embedding_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.earliest_training_start_index < 0:
            raise ValueError("earliest training start index must be non-negative")
        if not self.validation_start_indices or any(
            value < 0 for value in self.validation_start_indices
        ):
            raise ValueError("validation start indices must be non-negative")
        if len(set(self.validation_start_indices)) != len(
            self.validation_start_indices
        ):
            raise ValueError("validation start indices must be unique")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "training_integration_period_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight decay must be finite and non-negative")
        for name in (
            "tracking_weight",
            "visibility_weight",
            "smoothness_weight",
            "saturation_weight",
            "residual_magnitude_weight",
            "teacher_action_weight",
            "critical_step_weight",
            "maximum_state_regression_fraction",
            "maximum_action_regression_fraction",
            "minimum_tracking_improvement_fraction",
            "maximum_saturation_regression_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.terminal_tracking_weight) or (
            self.terminal_tracking_weight < 1.0
        ):
            raise ValueError("terminal tracking weight must be at least one")
        if not 0.0 < self.visibility_margin_fraction <= 1.0:
            raise ValueError("visibility margin fraction must be in (0, 1]")
        if not 0.0 < self.maximum_policy_residual_magnitude <= 1.0:
            raise ValueError("maximum residual fraction must be in (0, 1]")
        if self.residual_application not in {
            "target_half_fov",
            "command_normalized",
        }:
            raise ValueError("unsupported residual application")


@dataclass
class _MetricAccumulator:
    tracking_squared_sum: float = 0.0
    visibility_squared_sum: float = 0.0
    smoothness_squared_sum: float = 0.0
    saturation_sum: float = 0.0
    action_squared_sum: float = 0.0
    residual_squared_sum: float = 0.0
    residual_absolute_maximum: float = 0.0
    count: int = 0
    smoothness_count: int = 0
    action_count: int = 0

    def report(self) -> dict[str, float | int]:
        return {
            "sample_count": self.count,
            "tracking_rmse_normalized": math.sqrt(
                self.tracking_squared_sum / max(1, self.count)
            ),
            "visibility_rmse_normalized": math.sqrt(
                self.visibility_squared_sum / max(1, self.count)
            ),
            "smoothness_rmse_normalized": math.sqrt(
                self.smoothness_squared_sum
                / max(1, self.smoothness_count)
            ),
            "saturation_rmse_normalized": math.sqrt(
                self.saturation_sum / max(1, self.count)
            ),
            "teacher_action_rmse_normalized": math.sqrt(
                self.action_squared_sum / max(1, self.action_count)
            ),
            "policy_residual_rms": math.sqrt(
                self.residual_squared_sum / max(1, self.count)
            ),
            "policy_residual_maximum_absolute": (
                self.residual_absolute_maximum
            ),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_index(dataset: GimbalTargetStateDataset) -> int:
    return dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )


def _context_window(
    supervision: AdaptivePositionSupervision,
    episode_indices: np.ndarray,
    start: int,
    end: int,
    device: torch.device,
) -> GRUAdaptivePositionLossContext:
    values: dict[str, torch.Tensor] = {}
    for field in fields(GRUAdaptivePositionLossContext):
        array = getattr(supervision, field.name)[episode_indices, start:end]
        tensor = torch.from_numpy(array).to(device)
        values[field.name] = tensor.bool() if field.name == "mask" else tensor.float()
    return GRUAdaptivePositionLossContext(**values)


def _window_batch(
    dataset: GimbalTargetStateDataset,
    supervision: AdaptivePositionSupervision,
    episode_indices: np.ndarray,
    start: int,
    step_count: int,
    device: torch.device,
) -> CounterfactualWindowBatch:
    end = start + step_count
    profile_index = _profile_index(dataset)
    logged = torch.from_numpy(
        dataset.features[episode_indices, profile_index, start:end]
    ).float().to(device)
    warmup = torch.from_numpy(
        dataset.features[episode_indices, profile_index, :start]
    ).float().to(device)
    target_bearing = torch.from_numpy(
        dataset.targets[episode_indices, start : end + 1, 0, 0]
    ).float().to(device)
    time_s = torch.from_numpy(
        dataset.time_s[episode_indices, start : end + 1]
    ).float().to(device)
    capture_source = counterfactual_capture_source_indices(time_s, logged)
    sequence_mask = torch.from_numpy(
        dataset.sequence_mask[episode_indices, start:end]
    ).bool().to(device)
    return CounterfactualWindowBatch(
        logged_features=logged,
        warmup_features=warmup,
        target_bearing_rad=target_bearing,
        time_s=time_s,
        capture_source_index=capture_source,
        context=_context_window(
            supervision,
            episode_indices,
            start,
            end,
            device,
        ),
        sequence_mask=sequence_mask,
    )


def _window_critical_mask(
    criticality: ControlCriticality,
    episode_indices: np.ndarray,
    start: int,
    step_count: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(
        criticality.critical_mask[
            episode_indices,
            start + 1 : start + step_count + 1,
            0,
        ]
    ).bool().to(device)


def _training_loss(
    rollout,
    batch: CounterfactualWindowBatch,
    critical_mask: torch.Tensor,
    config: MultiCommandExperimentConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch.sequence_mask & batch.context.mask
    weights = valid.to(rollout.tracking_error_normalized.dtype) * (
        1.0
        + config.critical_step_weight
        * critical_mask.to(rollout.tracking_error_normalized.dtype)
    )
    weights[:, -1] = weights[:, -1] * config.terminal_tracking_weight
    weight_sum = weights.sum().clamp_min(1.0)
    tracking = (
        rollout.tracking_error_normalized.square() * weights
    ).sum() / weight_sum
    visibility = (
        rollout.visibility_violation_normalized.square() * weights
    ).sum() / weight_sum
    saturation = (rollout.saturation_fraction * weights).sum() / weight_sum
    previous = batch.logged_features[
        :, 0, FEATURE_NAMES.index("previous_action_normalized")
    ]
    command_with_previous = torch.cat(
        (previous[:, None], rollout.command_normalized),
        dim=1,
    )
    command_difference = torch.diff(command_with_previous, dim=1)
    smoothness = (command_difference.square() * weights).sum() / weight_sum
    residual_magnitude = (
        rollout.policy_residual_normalized.square() * weights
    ).sum() / weight_sum
    teacher_action = (
        (
            rollout.command_normalized
            - batch.context.teacher_action_normalized
        ).square()
        * weights
    ).sum() / weight_sum
    total = (
        config.tracking_weight * tracking
        + config.visibility_weight * visibility
        + config.smoothness_weight * smoothness
        + config.saturation_weight * saturation
        + config.residual_magnitude_weight * residual_magnitude
        + config.teacher_action_weight * teacher_action
    )
    return total, {
        "tracking_mse": float(tracking.detach()),
        "visibility_mse": float(visibility.detach()),
        "smoothness_mse": float(smoothness.detach()),
        "saturation_mean": float(saturation.detach()),
        "residual_magnitude_mse": float(residual_magnitude.detach()),
        "teacher_action_mse": float(teacher_action.detach()),
    }


def _accumulate(
    accumulator: _MetricAccumulator,
    rollout,
    batch: CounterfactualWindowBatch,
    teacher_action: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    selected = mask & batch.sequence_mask
    count = int(selected.sum().item())
    if not count:
        return
    tracking = rollout.tracking_error_normalized[selected]
    visibility = rollout.visibility_violation_normalized[selected]
    saturation = rollout.saturation_fraction[selected]
    residual = rollout.policy_residual_normalized[selected]
    action_error = (
        rollout.command_normalized - teacher_action
    )[selected]
    accumulator.tracking_squared_sum += float(tracking.square().sum())
    accumulator.visibility_squared_sum += float(visibility.square().sum())
    accumulator.saturation_sum += float(saturation.sum())
    accumulator.residual_squared_sum += float(residual.square().sum())
    accumulator.residual_absolute_maximum = max(
        accumulator.residual_absolute_maximum,
        float(torch.max(torch.abs(residual))),
    )
    accumulator.action_squared_sum += float(action_error.square().sum())
    accumulator.action_count += count
    accumulator.count += count

    previous = batch.logged_features[
        :, 0, FEATURE_NAMES.index("previous_action_normalized")
    ]
    command_with_previous = torch.cat(
        (previous[:, None], rollout.command_normalized),
        dim=1,
    )
    difference = torch.diff(command_with_previous, dim=1)
    accumulator.smoothness_squared_sum += float(
        difference[selected].square().sum()
    )
    accumulator.smoothness_count += count


@torch.no_grad()
def _evaluate_policy(
    base_model,
    policy,
    dataset: GimbalTargetStateDataset,
    supervision: AdaptivePositionSupervision,
    criticality: ControlCriticality,
    config: MultiCommandExperimentConfig,
    *,
    integration_period_override_s: float | None,
) -> dict[str, Any]:
    device = torch.device(config.device)
    policy.eval()
    base_model.eval()
    global_accumulator = _MetricAccumulator()
    critical_accumulator = _MetricAccumulator()
    episode_count = dataset.episode_count
    all_indices = np.arange(episode_count)
    for start in config.validation_start_indices:
        if np.any(
            np.sum(dataset.sequence_mask, axis=1)
            < start + config.window_steps + 1
        ):
            raise ValueError("validation window exceeds an episode")
        for offset in range(0, episode_count, config.batch_size):
            episode_indices = all_indices[offset : offset + config.batch_size]
            batch = _window_batch(
                dataset,
                supervision,
                episode_indices,
                start,
                config.window_steps,
                device,
            )
            rollout = rollout_counterfactual_window(
                base_model,
                policy,
                batch,
                prediction_horizons_s=(
                    dataset.manifest.prediction_horizons_s
                ),
                adapter=selected_adaptive_position_v21_config(),
                integration_period_override_s=(
                    integration_period_override_s
                ),
                visibility_margin_fraction=(
                    config.visibility_margin_fraction
                ),
                residual_application=config.residual_application,
            )
            teacher_action = torch.from_numpy(
                supervision.teacher_action_normalized[
                    episode_indices,
                    start : start + config.window_steps,
                ]
            ).float().to(device)
            critical_mask = _window_critical_mask(
                criticality,
                episode_indices,
                start,
                config.window_steps,
                device,
            )
            _accumulate(
                global_accumulator,
                rollout,
                batch,
                teacher_action,
                torch.ones_like(critical_mask),
            )
            _accumulate(
                critical_accumulator,
                rollout,
                batch,
                teacher_action,
                critical_mask,
            )
    return {
        "global": global_accumulator.report(),
        "critical": critical_accumulator.report(),
        "integration_period_override_s": integration_period_override_s,
        "window_start_indices": list(config.validation_start_indices),
    }


def _relative_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return candidate / reference - 1.0


def _gate(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    config: MultiCommandExperimentConfig,
) -> dict[str, Any]:
    changes = {}
    for scope in ("global", "critical"):
        for metric, value in reference[scope].items():
            if metric == "sample_count" or metric.startswith(
                "policy_residual_"
            ):
                continue
            changes[f"{scope}_{metric}"] = _relative_change(
                float(candidate[scope][metric]),
                float(value),
            )
    checks = {
        "state_invariant_by_construction": True,
        "global_tracking": changes["global_tracking_rmse_normalized"]
        <= -config.minimum_tracking_improvement_fraction,
        "critical_tracking": changes["critical_tracking_rmse_normalized"]
        <= -config.minimum_tracking_improvement_fraction,
        "global_visibility": changes["global_visibility_rmse_normalized"]
        <= 0.0,
        "critical_visibility": changes[
            "critical_visibility_rmse_normalized"
        ]
        <= 0.0,
        "global_smoothness": changes["global_smoothness_rmse_normalized"]
        <= 0.0,
        "critical_smoothness": changes[
            "critical_smoothness_rmse_normalized"
        ]
        <= 0.0,
        "global_action": changes[
            "global_teacher_action_rmse_normalized"
        ]
        <= config.maximum_action_regression_fraction,
        "critical_action": changes[
            "critical_teacher_action_rmse_normalized"
        ]
        <= config.maximum_action_regression_fraction,
        "saturation": changes["global_saturation_rmse_normalized"]
        <= config.maximum_saturation_regression_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def _approximate_rank(record: dict[str, Any]) -> tuple[float, ...]:
    gate = record["approximate_gate"]
    changes = gate["relative_changes"]
    failed = sum(not value for value in gate["checks"].values())
    return (
        float(failed),
        changes["global_tracking_rmse_normalized"]
        + changes["critical_tracking_rmse_normalized"],
        changes["global_smoothness_rmse_normalized"],
    )


def evaluate_multi_command_experiment(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    base_checkpoint: str | Path,
    checkpoint_directory: str | Path,
    config: MultiCommandExperimentConfig | None = None,
) -> dict[str, Any]:
    """Train V10 without opening any fresh test or closed-loop block."""

    config = config or MultiCommandExperimentConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    base_checkpoint = Path(base_checkpoint)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("V10 training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("V10 validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("V10 dataset seeds overlap")
    if train.manifest.prediction_horizons_s != (
        validation.manifest.prediction_horizons_s
    ):
        raise ValueError("V10 prediction horizons differ")
    if min(np.sum(train.sequence_mask, axis=1)) <= (
        config.earliest_training_start_index + config.window_steps
    ):
        raise ValueError("V10 training episodes are too short")

    base_model, base_metadata = load_gru_checkpoint(
        base_checkpoint,
        device=config.device,
    )
    if base_model.config.mean_parameterization != "integrated_midpoint":
        raise ValueError("V10 requires the hard midpoint state predictor")
    if (
        base_model.config.prediction_horizons_s
        != train.manifest.prediction_horizons_s
    ):
        raise ValueError("V10 base checkpoint horizons differ from the dataset")
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
    train_criticality = compute_control_criticality(
        train,
        config=config.criticality,
    )
    validation_criticality = compute_control_criticality(
        validation,
        config=config.criticality,
    )
    curriculum = compute_critical_episode_curriculum(
        train,
        train_criticality,
        config=config.curriculum,
    )
    adapter = selected_adaptive_position_v21_config()
    train_supervision = compute_adaptive_position_supervision(
        train,
        adapter=adapter,
        profile=ObservationProfile.DISTURBANCE_AWARE,
    )
    validation_supervision = compute_adaptive_position_supervision(
        validation,
        adapter=adapter,
        profile=ObservationProfile.DISTURBANCE_AWARE,
    )

    set_gru_seed(config.optimization_seed)
    device = torch.device(config.device)
    policy_config = RecurrentPositionResidualPolicyConfig(
        input_dim=recurrent_policy_input_dim(base_model.horizon_count),
        hidden_dim=config.residual_policy_hidden_dim,
        embedding_dim=config.residual_policy_embedding_dim,
        maximum_residual_magnitude=(
            config.maximum_policy_residual_magnitude
        ),
    )
    policy = CausalRecurrentPositionResidualPolicy(policy_config).to(device)
    initial_policy_state = copy.deepcopy(policy.state_dict())
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    episode_probabilities = torch.from_numpy(
        curriculum.episode_weights.astype(np.float64)
    )
    episode_probabilities /= episode_probabilities.sum()
    generator = torch.Generator().manual_seed(config.optimization_seed)
    minimum_length = int(np.min(np.sum(train.sequence_mask, axis=1)))
    maximum_start = minimum_length - config.window_steps - 1
    epoch_states = []
    training_history = []

    for epoch in range(1, config.epochs + 1):
        policy.train()
        totals = {
            "total": 0.0,
            "tracking_mse": 0.0,
            "visibility_mse": 0.0,
            "smoothness_mse": 0.0,
            "saturation_mean": 0.0,
            "residual_magnitude_mse": 0.0,
            "teacher_action_mse": 0.0,
        }
        for _ in range(config.batches_per_epoch):
            episode_indices = torch.multinomial(
                episode_probabilities,
                config.batch_size,
                replacement=True,
                generator=generator,
            ).numpy()
            start = int(
                torch.randint(
                    config.earliest_training_start_index,
                    maximum_start + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            batch = _window_batch(
                train,
                train_supervision,
                episode_indices,
                start,
                config.window_steps,
                device,
            )
            critical_mask = _window_critical_mask(
                train_criticality,
                episode_indices,
                start,
                config.window_steps,
                device,
            )
            rollout = rollout_counterfactual_window(
                base_model,
                policy,
                batch,
                prediction_horizons_s=train.manifest.prediction_horizons_s,
                adapter=adapter,
                integration_period_override_s=(
                    config.training_integration_period_s
                ),
                visibility_margin_fraction=(
                    config.visibility_margin_fraction
                ),
                residual_application=config.residual_application,
            )
            loss, components = _training_loss(
                rollout,
                batch,
                critical_mask,
                config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            totals["total"] += float(loss.detach())
            for name, value in components.items():
                totals[name] += value
        training_history.append(
            {
                "epoch": epoch,
                **{
                    name: value / config.batches_per_epoch
                    for name, value in totals.items()
                },
            }
        )
        epoch_states.append(copy.deepcopy(policy.state_dict()))

    policy.load_state_dict(initial_policy_state)
    approximate_reference = _evaluate_policy(
        base_model,
        policy,
        validation,
        validation_supervision,
        validation_criticality,
        config,
        integration_period_override_s=(
            config.training_integration_period_s
        ),
    )
    approximate_records = []
    for epoch, state in enumerate(epoch_states, start=1):
        policy.load_state_dict(state)
        evaluation = _evaluate_policy(
            base_model,
            policy,
            validation,
            validation_supervision,
            validation_criticality,
            config,
            integration_period_override_s=(
                config.training_integration_period_s
            ),
        )
        approximate_records.append(
            {
                "epoch": epoch,
                "training_record": training_history[epoch - 1],
                "approximate_evaluation": evaluation,
                "approximate_gate": _gate(
                    evaluation,
                    approximate_reference,
                    config,
                ),
            }
        )
    exact_epoch_count = min(
        config.exact_epoch_candidate_count,
        len(approximate_records),
    )
    exact_epochs = {
        record["epoch"]
        for record in sorted(
            approximate_records,
            key=_approximate_rank,
        )[:exact_epoch_count]
    }
    policy.load_state_dict(initial_policy_state)
    exact_reference = _evaluate_policy(
        base_model,
        policy,
        validation,
        validation_supervision,
        validation_criticality,
        config,
        integration_period_override_s=None,
    )
    exact_records = []
    passing_records = []
    for epoch in sorted(exact_epochs):
        policy.load_state_dict(epoch_states[epoch - 1])
        evaluation = _evaluate_policy(
            base_model,
            policy,
            validation,
            validation_supervision,
            validation_criticality,
            config,
            integration_period_override_s=None,
        )
        gate = _gate(evaluation, exact_reference, config)
        record = {
            "epoch": epoch,
            "exact_evaluation": evaluation,
            "exact_gate": gate,
        }
        exact_records.append(record)
        if gate["passed"]:
            passing_records.append(record)

    selected = (
        min(
            passing_records,
            key=lambda item: item["exact_evaluation"]["global"][
                "tracking_rmse_normalized"
            ],
        )
        if passing_records
        else None
    )
    checkpoint = None
    if selected is not None:
        policy.load_state_dict(epoch_states[selected["epoch"] - 1])
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / (
            f"gimbal_v10_multi_command_seed_{config.base_seed}.pt"
        )
        torch.save(
            {
                "schema_version": MULTI_COMMAND_EXPERIMENT_SCHEMA_VERSION,
                "policy_config": asdict(policy_config),
                "policy_state_dict": policy.state_dict(),
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "adapter_config": asdict(adapter),
                "metadata": {
                    "epoch": selected["epoch"],
                    "base_seed": config.base_seed,
                    "optimization_seed": config.optimization_seed,
                    "fresh_test_opened": False,
                },
            },
            checkpoint,
        )

    return {
        "experiment": MULTI_COMMAND_EXPERIMENT_SCHEMA_VERSION,
        "config": asdict(config),
        "policy_config": asdict(policy_config),
        "architecture": {
            "base_state_predictor_frozen": True,
            "trainable_policy_parameters": sum(
                parameter.numel() for parameter in policy.parameters()
            ),
            "counterfactual_observation_fields": [
                "image_error_normalized",
                "image_error_rad",
                "image_error_valid",
                "gimbal_position_normalized",
                "gimbal_angle_rad",
                "gimbal_rate_normalized",
                "gimbal_rate_rad_s",
                "previous_action_normalized",
                "previous_position_command_rad",
            ],
            "exogenous_observation_fields": [
                "frame_updated",
                "measurement_age_s",
                "detector_dropout_schedule",
                "bbox_size_when_visible",
                "confidence_when_visible",
                "body_rate",
            ],
            "persistent_command_latency_queue": True,
            "residual_application": config.residual_application,
        },
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "sha256": _sha256(base_checkpoint),
            "metadata": base_metadata,
        },
        "datasets": {
            "train": {
                "path": str(train_path),
                "sha256": _sha256(train_path),
                "episodes": train.episode_count,
                "criticality": control_criticality_report(
                    train,
                    train_criticality,
                    config=config.criticality,
                ),
                "curriculum": critical_episode_curriculum_report(
                    train,
                    curriculum,
                    config=config.curriculum,
                ),
            },
            "validation": {
                "path": str(validation_path),
                "sha256": _sha256(validation_path),
                "episodes": validation.episode_count,
                "criticality": control_criticality_report(
                    validation,
                    validation_criticality,
                    config=config.criticality,
                ),
            },
            "fresh_test": {"opened": False},
        },
        "training_history": training_history,
        "approximate_reference": approximate_reference,
        "approximate_epochs": approximate_records,
        "exact_reference": exact_reference,
        "exact_epochs": exact_records,
        "selected": (
            {
                "epoch": selected["epoch"],
                "checkpoint": str(checkpoint),
                "evaluation": selected["exact_evaluation"],
                "gate": selected["exact_gate"],
            }
            if selected is not None and checkpoint is not None
            else None
        ),
        "recommendation": (
            "replicate_multi_command_policy_on_independent_base_seeds"
            if selected is not None
            else "revise_multi_command_policy_before_replication"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop the V10 recurrent multi-command gimbal policy."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_train.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_validation.npz"),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path(
            "artifacts/gimbal_midpoint_adapter_replication_checkpoints/"
            "gimbal_v7_midpoint_state_reference_seed_29.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_multi_command_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_multi_command_v10.json"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--residual-application",
        choices=("target_half_fov", "command_normalized"),
        default="target_half_fov",
    )
    parser.add_argument("--maximum-residual", type=float, default=0.25)
    parser.add_argument("--teacher-action-weight", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_multi_command_experiment(
        train_path=args.train_data,
        validation_path=args.validation_data,
        base_checkpoint=args.base_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        config=MultiCommandExperimentConfig(
            epochs=args.epochs,
            device=args.device,
            residual_application=args.residual_application,
            maximum_policy_residual_magnitude=args.maximum_residual,
            teacher_action_weight=args.teacher_action_weight,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"selected={result['selected']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
