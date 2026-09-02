"""V16 state-consistent residual-authority oracle distillation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .adaptive_curriculum_objective import selected_adaptive_position_v21_config
from .adaptive_position_supervision import compute_adaptive_position_supervision
from .config import ObservationProfile
from .control_criticality import ControlCriticalityConfig, compute_control_criticality
from .dataset import FEATURE_NAMES, load_gimbal_dataset
from .deployable_failure_gated_rollout import (
    DeployableFailureGatedRollout,
    rollout_deployable_failure_gated_policy,
)
from .deployable_residual_distillation_experiment import (
    DeployableResidualDistillationConfig,
    _concatenate_rollouts,
    _evaluate_policy,
    _generate_state_consistent_records,
    _train_policy,
)
from .failure_gated_policy import (
    FailureGatedCommandResidualPolicy,
    FailureGatedPositionPolicyConfig,
    HardwareConditionedResidualAuthorityCalibrator,
    ResidualAuthorityCalibratorConfig,
)
from .gru import GRUAdaptivePositionLossContext, load_gru_checkpoint
from .gru_training import set_gru_seed
from .multi_command_experiment import _context_window
from .multi_command_policy import (
    CausalRecurrentPositionResidualPolicy,
    RecurrentPositionResidualPolicyConfig,
    counterfactual_capture_source_indices,
    recurrent_policy_input_dim,
)
from .on_policy_distillation_experiment import _rollout_metrics, _selection_score
from .sequence_distillation import normalized_hardware_features
from .sequence_distillation_experiment import _episode_indices
from .sequence_oracle import PrivilegedSequenceOracleConfig
from .sequence_oracle_experiment import _gate, _sha256


DEPLOYABLE_AUTHORITY_DISTILLATION_SCHEMA_VERSION = (
    "gimbal_deployable_authority_distillation_v16_development_v1"
)
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class DeployableAuthorityDistillationConfig:
    behavior_name: str = "privileged_oracle_position"
    sequence_steps: int = 16
    initial_distillation_episode_count: int = 192
    authority_training_episode_count: int = 288
    validation_episode_count: int = 48
    oracle_batch_size: int = 8
    rollout_batch_size: int = 16
    initial_distillation_batch_size: int = 16
    initial_distillation_epochs: int = 30
    initial_distillation_trust_region_weight: float = 0.25
    initial_distillation_learning_rate: float = 1e-3
    authority_scales: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )
    maximum_tracking_regression_fraction: float = 0.0
    maximum_visibility_regression_fraction: float = 0.0
    maximum_smoothness_regression_fraction: float = 0.0
    maximum_saturation_regression_fraction: float = 0.05
    feasibility_absolute_tolerance: float = 1e-8
    near_optimal_tracking_fraction: float = 0.0025
    router_epochs: int = 80
    selection_epoch_interval: int = 10
    router_batch_size: int = 32
    router_learning_rates: tuple[float, ...] = (3e-4, 1e-3)
    unsafe_episode_weights: tuple[float, ...] = (1.0, 4.0)
    critical_step_weight: float = 2.0
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    minimum_tracking_improvement_fraction: float = 0.005
    optimization_seed: int = 29
    device: str = "cpu"
    correction_policy: FailureGatedPositionPolicyConfig = (
        FailureGatedPositionPolicyConfig(
            hidden_dim=48,
            embedding_dim=48,
            maximum_residual_magnitude=0.40,
        )
    )
    authority_router: ResidualAuthorityCalibratorConfig = (
        ResidualAuthorityCalibratorConfig(
            hidden_dim=16,
            initial_authority=0.99,
            maximum_authority=1.0,
        )
    )
    oracle: PrivilegedSequenceOracleConfig = PrivilegedSequenceOracleConfig(
        focus_start_index=0,
        focus_steps=16,
        optimization_iterations=24,
    )
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        for name in (
            "sequence_steps",
            "initial_distillation_episode_count",
            "authority_training_episode_count",
            "validation_episode_count",
            "oracle_batch_size",
            "rollout_batch_size",
            "initial_distillation_batch_size",
            "initial_distillation_epochs",
            "router_epochs",
            "selection_epoch_interval",
            "router_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"V16 {name} must be positive")
        for name in (
            "initial_distillation_learning_rate",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"V16 {name} must be finite and positive")
        for name in (
            "initial_distillation_trust_region_weight",
            "maximum_tracking_regression_fraction",
            "maximum_visibility_regression_fraction",
            "maximum_smoothness_regression_fraction",
            "maximum_saturation_regression_fraction",
            "feasibility_absolute_tolerance",
            "near_optimal_tracking_fraction",
            "critical_step_weight",
            "weight_decay",
            "minimum_tracking_improvement_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V16 {name} must be finite and non-negative")
        for values, name in (
            (self.router_learning_rates, "router learning rates"),
            (self.unsafe_episode_weights, "unsafe episode weights"),
        ):
            if not values or any(
                not math.isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(f"V16 {name} must be finite and positive")
            if len(set(values)) != len(values):
                raise ValueError(f"V16 {name} must be unique")
        if not self.authority_scales or self.authority_scales[0] != 0.0:
            raise ValueError("V16 authority scales must begin with zero")
        if self.authority_scales[-1] != 1.0 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.authority_scales
        ):
            raise ValueError("V16 authority scales must end at one and be bounded")
        if any(
            right <= left
            for left, right in zip(
                self.authority_scales,
                self.authority_scales[1:],
            )
        ):
            raise ValueError("V16 authority scales must be strictly increasing")
        if self.oracle.focus_start_index != 0 or (
            self.oracle.focus_steps != self.sequence_steps
        ):
            raise ValueError("V16 requires an episode-start sequence oracle")


@dataclass
class AuthorityOracleResult:
    selected_authority: torch.Tensor
    selected_rollout: DeployableFailureGatedRollout
    reference_rollout: DeployableFailureGatedRollout
    full_authority_rollout: DeployableFailureGatedRollout
    candidate_metrics: dict[str, torch.Tensor]
    selected_metrics: dict[str, torch.Tensor]
    feasible: torch.Tensor


@dataclass
class _AuthorityRecords:
    hardware: torch.Tensor
    evidence: torch.Tensor
    target_authority: torch.Tensor
    critical: torch.Tensor
    sequence_mask: torch.Tensor


def _distillation_config(
    config: DeployableAuthorityDistillationConfig,
) -> DeployableResidualDistillationConfig:
    return DeployableResidualDistillationConfig(
        behavior_name=config.behavior_name,
        sequence_steps=config.sequence_steps,
        training_episode_count=config.initial_distillation_episode_count,
        validation_episode_count=config.validation_episode_count,
        oracle_batch_size=config.oracle_batch_size,
        rollout_batch_size=config.rollout_batch_size,
        epochs=config.initial_distillation_epochs,
        selection_epoch_interval=config.initial_distillation_epochs,
        batch_size=config.initial_distillation_batch_size,
        learning_rate=config.initial_distillation_learning_rate,
        optimization_seed=config.optimization_seed,
        ordinary_trust_region_weights=(
            config.initial_distillation_trust_region_weight,
        ),
        deployment_residual_scales=(1.0,),
        visibility_shield_strengths=(0.0,),
        minimum_tracking_improvement_fraction=(
            config.minimum_tracking_improvement_fraction
        ),
        maximum_saturation_regression_fraction=(
            config.maximum_saturation_regression_fraction
        ),
        correction_policy=config.correction_policy,
        oracle=config.oracle,
        criticality=config.criticality,
        device=config.device,
    )


def _policy_rollout_batch(
    base_model,
    policy,
    dataset,
    supervision,
    indices: np.ndarray,
    config: DeployableAuthorityDistillationConfig,
    device: torch.device,
    *,
    residual_scale: float | torch.Tensor = 1.0,
) -> DeployableFailureGatedRollout:
    profile = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    features = torch.from_numpy(
        dataset.features[indices, profile, : config.sequence_steps]
    ).float().to(device)
    time_s = torch.from_numpy(
        dataset.time_s[indices, : config.sequence_steps + 1]
    ).float().to(device)
    target = torch.from_numpy(
        dataset.targets[indices, : config.sequence_steps + 1, 0, 0]
    ).float().to(device)
    mask = torch.from_numpy(
        dataset.sequence_mask[indices, : config.sequence_steps]
    ).bool().to(device)
    context = _context_window(
        supervision,
        indices,
        0,
        config.sequence_steps,
        device,
    )
    return rollout_deployable_failure_gated_policy(
        base_model,
        policy,
        features,
        target,
        time_s,
        counterfactual_capture_source_indices(time_s, features),
        context,
        mask,
        prediction_horizons_s=dataset.manifest.prediction_horizons_s,
        adapter=selected_adaptive_position_v21_config(),
        residual_scale=residual_scale,
        visibility_margin_fraction=config.oracle.visibility_margin_fraction,
    )


def _episode_metrics(
    rollout: DeployableFailureGatedRollout,
    dataset,
    indices: np.ndarray,
    criticality,
    config: DeployableAuthorityDistillationConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    mask = torch.from_numpy(
        dataset.sequence_mask[indices, : config.sequence_steps]
    ).bool().to(device)
    critical = torch.from_numpy(
        criticality.critical_mask[
            indices,
            1 : config.sequence_steps + 1,
            0,
        ]
    ).bool().to(device)
    profile = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    previous = torch.from_numpy(
        dataset.features[
            indices,
            profile,
            0,
            _FEATURE_INDEX["previous_action_normalized"],
        ]
    ).float().to(device)
    difference = torch.diff(
        torch.cat((previous[:, None], rollout.command_normalized), dim=1),
        dim=1,
    )
    result = {}
    for scope, selected in (("global", mask), ("critical", mask & critical)):
        selected_float = selected.float()
        denominator = selected_float.sum(dim=1).clamp_min(1.0)
        result.update(
            {
                f"{scope}_tracking_mse": (
                    rollout.tracking_error_normalized.square()
                    * selected_float
                ).sum(dim=1)
                / denominator,
                f"{scope}_visibility_mse": (
                    rollout.visibility_violation_normalized.square()
                    * selected_float
                ).sum(dim=1)
                / denominator,
                f"{scope}_smoothness_mse": (
                    difference.square() * selected_float
                ).sum(dim=1)
                / denominator,
                f"{scope}_saturation_mean": (
                    rollout.saturation_fraction * selected_float
                ).sum(dim=1)
                / denominator,
            }
        )
    return result


def _gather_candidate(
    value: torch.Tensor,
    selected_index: torch.Tensor,
) -> torch.Tensor:
    return value[
        selected_index,
        torch.arange(value.shape[1], device=value.device),
    ]


def _select_rollout_candidates(
    rollouts: list[DeployableFailureGatedRollout],
    selected_index: torch.Tensor,
) -> DeployableFailureGatedRollout:
    return DeployableFailureGatedRollout(
        **{
            field.name: _gather_candidate(
                torch.stack(
                    [getattr(rollout, field.name) for rollout in rollouts],
                    dim=0,
                ),
                selected_index,
            )
            for field in fields(rollouts[0])
        }
    )


def optimize_privileged_episode_authority(
    base_model,
    policy: FailureGatedCommandResidualPolicy,
    reference_policy: FailureGatedCommandResidualPolicy,
    dataset,
    supervision,
    criticality,
    indices: np.ndarray,
    config: DeployableAuthorityDistillationConfig,
    device: torch.device,
) -> AuthorityOracleResult:
    """Choose the highest near-optimal feasible authority per episode."""

    policy.eval()
    reference_policy.eval()
    base_model.eval()
    with torch.no_grad():
        reference = _policy_rollout_batch(
            base_model,
            reference_policy,
            dataset,
            supervision,
            indices,
            config,
            device,
            residual_scale=0.0,
        )
        candidates = [
            _policy_rollout_batch(
                base_model,
                policy,
                dataset,
                supervision,
                indices,
                config,
                device,
                residual_scale=scale,
            )
            for scale in config.authority_scales
        ]
    reference_metrics = _episode_metrics(
        reference,
        dataset,
        indices,
        criticality,
        config,
        device,
    )
    candidate_metrics = {
        name: torch.stack(
            [
                _episode_metrics(
                    candidate,
                    dataset,
                    indices,
                    criticality,
                    config,
                    device,
                )[name]
                for candidate in candidates
            ],
            dim=0,
        )
        for name in reference_metrics
    }
    tolerance = config.feasibility_absolute_tolerance
    feasible = torch.ones_like(
        candidate_metrics["global_tracking_mse"],
        dtype=torch.bool,
    )
    for scope in ("global", "critical"):
        for metric, regression in (
            ("tracking_mse", config.maximum_tracking_regression_fraction),
            ("visibility_mse", config.maximum_visibility_regression_fraction),
            ("smoothness_mse", config.maximum_smoothness_regression_fraction),
            ("saturation_mean", config.maximum_saturation_regression_fraction),
        ):
            name = f"{scope}_{metric}"
            feasible &= candidate_metrics[name] <= (
                reference_metrics[name].unsqueeze(0)
                * (1.0 + regression) ** 2
                + tolerance
            )
    if torch.any(~feasible[0]):
        raise RuntimeError("zero authority must remain an exactly feasible fallback")
    tracking_objective = (
        candidate_metrics["global_tracking_mse"]
        + candidate_metrics["critical_tracking_mse"]
    )
    infinity = torch.full_like(tracking_objective, math.inf)
    feasible_tracking = torch.where(
        feasible,
        tracking_objective,
        infinity,
    )
    best_tracking = feasible_tracking.min(dim=0).values
    near_best = tracking_objective <= (
        best_tracking.unsqueeze(0)
        * (1.0 + config.near_optimal_tracking_fraction) ** 2
        + tolerance
    )
    qualified = feasible & near_best
    scale = torch.tensor(
        config.authority_scales,
        device=device,
        dtype=tracking_objective.dtype,
    ).unsqueeze(1)
    selected_index = torch.argmax(
        torch.where(qualified, scale, torch.full_like(scale, -1.0)),
        dim=0,
    )
    selected_metrics = {
        name: _gather_candidate(value, selected_index)
        for name, value in candidate_metrics.items()
    }
    selected_rollout = _select_rollout_candidates(candidates, selected_index)
    return AuthorityOracleResult(
        selected_authority=scale[:, 0][selected_index],
        selected_rollout=selected_rollout,
        reference_rollout=reference,
        full_authority_rollout=candidates[-1],
        candidate_metrics=candidate_metrics,
        selected_metrics=selected_metrics,
        feasible=feasible,
    )


def _authority_summary(
    authority: torch.Tensor,
    dataset,
    indices: np.ndarray,
    config: DeployableAuthorityDistillationConfig,
) -> dict[str, Any]:
    values = authority.detach().cpu().numpy()
    scenarios = dataset.manifest.generation.get("scenarios", [])

    def histogram(selected: np.ndarray) -> dict[str, int]:
        return {
            f"{scale:.2f}": int(np.sum(np.isclose(selected, scale)))
            for scale in config.authority_scales
        }

    by_scenario = []
    for scenario_index in np.unique(dataset.scenario_index[indices]):
        selected = dataset.scenario_index[indices] == scenario_index
        by_scenario.append(
            {
                "scenario_index": int(scenario_index),
                "scenario_name": (
                    scenarios[int(scenario_index)].get("name")
                    if int(scenario_index) < len(scenarios)
                    else None
                ),
                "episode_count": int(selected.sum()),
                "mean_authority": float(values[selected].mean()),
                "suppressed_fraction": float(
                    np.mean(values[selected] < 1.0 - 1e-6)
                ),
                "histogram": histogram(values[selected]),
            }
        )
    return {
        "episode_count": len(values),
        "mean_authority": float(values.mean()),
        "zero_authority_fraction": float(np.mean(values <= 1e-6)),
        "full_authority_fraction": float(np.mean(values >= 1.0 - 1e-6)),
        "suppressed_fraction": float(np.mean(values < 1.0 - 1e-6)),
        "histogram": histogram(values),
        "by_scenario": by_scenario,
    }


def _generate_authority_records(
    base_model,
    policy,
    reference_policy,
    dataset,
    supervision,
    criticality,
    indices: np.ndarray,
    config: DeployableAuthorityDistillationConfig,
    device: torch.device,
) -> tuple[_AuthorityRecords, AuthorityOracleResult, dict[str, Any]]:
    results = []
    contexts: list[GRUAdaptivePositionLossContext] = []
    for offset in range(0, len(indices), config.oracle_batch_size):
        selected = indices[offset : offset + config.oracle_batch_size]
        results.append(
            optimize_privileged_episode_authority(
                base_model,
                policy,
                reference_policy,
                dataset,
                supervision,
                criticality,
                selected,
                config,
                device,
            )
        )
        contexts.append(
            _context_window(
                supervision,
                selected,
                0,
                config.sequence_steps,
                device,
            )
        )
    authority = torch.cat([result.selected_authority for result in results])
    selected_rollout = _concatenate_rollouts(
        [result.selected_rollout for result in results]
    )
    reference_rollout = _concatenate_rollouts(
        [result.reference_rollout for result in results]
    )
    full_rollout = _concatenate_rollouts(
        [result.full_authority_rollout for result in results]
    )
    replay_rollouts = []
    for offset in range(0, len(indices), config.oracle_batch_size):
        selected = indices[offset : offset + config.oracle_batch_size]
        replay_rollouts.append(
            _policy_rollout_batch(
                base_model,
                policy,
                dataset,
                supervision,
                selected,
                config,
                device,
                residual_scale=authority[offset : offset + len(selected)],
            )
        )
    replay_rollout = _concatenate_rollouts(replay_rollouts)
    replay_errors = {
        field.name: float(
            torch.max(
                torch.abs(
                    getattr(replay_rollout, field.name)
                    - getattr(selected_rollout, field.name)
                )
            )
        )
        for field in fields(selected_rollout)
    }
    merged_result = AuthorityOracleResult(
        selected_authority=authority,
        selected_rollout=replay_rollout,
        reference_rollout=reference_rollout,
        full_authority_rollout=full_rollout,
        candidate_metrics={
            name: torch.cat(
                [result.candidate_metrics[name] for result in results],
                dim=1,
            )
            for name in results[0].candidate_metrics
        },
        selected_metrics={
            name: torch.cat(
                [result.selected_metrics[name] for result in results]
            )
            for name in results[0].selected_metrics
        },
        feasible=torch.cat([result.feasible for result in results], dim=1),
    )
    hardware = torch.cat(
        [normalized_hardware_features(context) for context in contexts],
        dim=0,
    )
    mask = torch.from_numpy(
        dataset.sequence_mask[indices, : config.sequence_steps]
    ).float().to(device)
    critical = torch.from_numpy(
        criticality.critical_mask[
            indices,
            1 : config.sequence_steps + 1,
            0,
        ]
    ).float().to(device)
    records = _AuthorityRecords(
        hardware=hardware,
        evidence=replay_rollout.failure_evidence.detach(),
        target_authority=authority[:, None].expand_as(mask).detach(),
        critical=critical,
        sequence_mask=mask,
    )
    diagnostics = {
        **_authority_summary(authority, dataset, indices, config),
        "state_consistent_replay": True,
        "maximum_tensor_replay_error": max(replay_errors.values()),
        "tensor_replay_errors": replay_errors,
        "feasible_candidate_fraction": float(
            merged_result.feasible.float().mean()
        ),
    }
    return records, merged_result, diagnostics


def _train_router(
    router: HardwareConditionedResidualAuthorityCalibrator,
    records: _AuthorityRecords,
    config: DeployableAuthorityDistillationConfig,
    *,
    learning_rate: float,
    unsafe_episode_weight: float,
) -> tuple[list[dict[str, float | int]], list[tuple[int, dict[str, Any]]]]:
    optimizer = torch.optim.AdamW(
        router.authority_head.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.optimization_seed)
    history = []
    states = [(0, copy.deepcopy(router.state_dict()))]
    episode_weight = 1.0 + unsafe_episode_weight * (
        1.0 - records.target_authority
    )
    weights = (
        episode_weight + config.critical_step_weight * records.critical
    ) * records.sequence_mask
    for epoch in range(1, config.router_epochs + 1):
        order = torch.randperm(len(records.hardware), generator=generator)
        total_loss = 0.0
        total_mae = 0.0
        router.train()
        for offset in range(0, len(order), config.router_batch_size):
            selected = order[offset : offset + config.router_batch_size].to(
                records.hardware.device
            )
            prediction = router.authority(
                records.hardware[selected],
                records.evidence[selected],
            )
            selected_weights = weights[selected]
            error = prediction - records.target_authority[selected]
            loss = (
                error.square() * selected_weights
            ).sum() / selected_weights.sum().clamp_min(1.0)
            mae = (
                error.abs() * selected_weights
            ).sum() / selected_weights.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                router.authority_head.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
            total_mae += float(mae.detach()) * len(selected)
        history.append(
            {
                "epoch": epoch,
                "weighted_authority_mse": total_loss / len(order),
                "weighted_authority_mae": total_mae / len(order),
            }
        )
        if epoch % config.selection_epoch_interval == 0 or (
            epoch == config.router_epochs
        ):
            states.append((epoch, copy.deepcopy(router.state_dict())))
    return history, states


def _select_router_state(
    base_model,
    router,
    states,
    validation,
    validation_supervision,
    validation_indices,
    validation_criticality,
    reference,
    config,
    device,
):
    distillation_config = _distillation_config(config)
    evaluations = []
    state_by_epoch = dict(states)
    for epoch, state in states:
        router.load_state_dict(state)
        rollout = _evaluate_policy(
            base_model,
            router,
            validation,
            validation_supervision,
            validation_indices,
            distillation_config,
            device,
        )
        metrics = _rollout_metrics(
            rollout,
            validation,
            validation_indices,
            validation_criticality,
            config,
            device,
        )
        evaluations.append(
            {
                "epoch": epoch,
                "selection_score": _selection_score(metrics),
                "metrics": metrics,
                "gate": _gate(metrics, reference, config),
                "mean_absolute_residual_normalized": float(
                    rollout.residual_normalized.abs().mean()
                ),
            }
        )
    passing = [record for record in evaluations if record["gate"]["passed"]]
    selected = min(
        passing or evaluations,
        key=lambda record: (
            float(record["metrics"]["global"]["tracking_rmse_normalized"])
            if passing
            else float(record["selection_score"])
        ),
    )
    selected_state = state_by_epoch[int(selected["epoch"])]
    router.load_state_dict(selected_state)
    return selected, evaluations, selected_state


def evaluate_deployable_authority_distillation_experiment(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    base_checkpoint: str | Path,
    checkpoint_directory: str | Path,
    config: DeployableAuthorityDistillationConfig | None = None,
) -> dict[str, Any]:
    """Run the V16 authority ceiling and deployable router distillation."""

    config = config or DeployableAuthorityDistillationConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    base_checkpoint = Path(base_checkpoint)
    checkpoint_directory = Path(checkpoint_directory)
    device = torch.device(config.device)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    initial_indices = _episode_indices(
        train,
        config,
        config.initial_distillation_episode_count,
    )
    authority_indices = _episode_indices(
        train,
        config,
        config.authority_training_episode_count,
    )
    validation_indices = _episode_indices(
        validation,
        config,
        config.validation_episode_count,
    )
    base_model, base_metadata = load_gru_checkpoint(
        base_checkpoint,
        device=config.device,
    )
    if base_model.config.mean_parameterization != "integrated_midpoint":
        raise ValueError("V16 requires the hard-midpoint predictor")
    if base_model.config.prediction_horizons_s != (
        train.manifest.prediction_horizons_s
    ) or base_model.config.prediction_horizons_s != (
        validation.manifest.prediction_horizons_s
    ):
        raise ValueError("V16 checkpoint horizons differ from a dataset")
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
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
    train_criticality = compute_control_criticality(
        train,
        config=config.criticality,
    )
    validation_criticality = compute_control_criticality(
        validation,
        config=config.criticality,
    )
    distillation_config = _distillation_config(config)
    zero_actor = CausalRecurrentPositionResidualPolicy(
        RecurrentPositionResidualPolicyConfig(
            input_dim=recurrent_policy_input_dim(base_model.horizon_count),
            hidden_dim=8,
            embedding_dim=8,
        )
    ).to(device)
    correction_records, correction_diagnostics = (
        _generate_state_consistent_records(
            base_model,
            zero_actor,
            train,
            train_supervision,
            train_criticality,
            initial_indices,
            distillation_config,
            device,
        )
    )
    set_gru_seed(config.optimization_seed)
    initialized_policy = FailureGatedCommandResidualPolicy(
        config.correction_policy
    ).to(device)
    initialization_history, initialization_states = _train_policy(
        initialized_policy,
        correction_records,
        trust_region_weight=(
            config.initial_distillation_trust_region_weight
        ),
        config=distillation_config,
        generator=torch.Generator().manual_seed(config.optimization_seed),
    )
    initialized_policy.load_state_dict(initialization_states[-1][1])
    for parameter in initialized_policy.parameters():
        parameter.requires_grad_(False)
    initialized_policy.eval()

    set_gru_seed(config.optimization_seed)
    reference_policy = FailureGatedCommandResidualPolicy(
        config.correction_policy
    ).to(device)
    for parameter in reference_policy.parameters():
        parameter.requires_grad_(False)
    reference_policy.eval()
    reference_rollout = _evaluate_policy(
        base_model,
        reference_policy,
        validation,
        validation_supervision,
        validation_indices,
        distillation_config,
        device,
    )
    initialized_rollout = _evaluate_policy(
        base_model,
        initialized_policy,
        validation,
        validation_supervision,
        validation_indices,
        distillation_config,
        device,
    )
    reference = _rollout_metrics(
        reference_rollout,
        validation,
        validation_indices,
        validation_criticality,
        config,
        device,
    )
    initialized_metrics = _rollout_metrics(
        initialized_rollout,
        validation,
        validation_indices,
        validation_criticality,
        config,
        device,
    )

    authority_records, training_oracle, training_oracle_diagnostics = (
        _generate_authority_records(
            base_model,
            initialized_policy,
            reference_policy,
            train,
            train_supervision,
            train_criticality,
            authority_indices,
            config,
            device,
        )
    )
    validation_oracle = optimize_privileged_episode_authority(
        base_model,
        initialized_policy,
        reference_policy,
        validation,
        validation_supervision,
        validation_criticality,
        validation_indices,
        config,
        device,
    )
    validation_oracle_metrics = _rollout_metrics(
        validation_oracle.selected_rollout,
        validation,
        validation_indices,
        validation_criticality,
        config,
        device,
    )
    validation_oracle_gate = _gate(
        validation_oracle_metrics,
        reference,
        config,
    )

    initialized_state = copy.deepcopy(initialized_policy.state_dict())
    arms = []
    selected_states = []
    for learning_rate in config.router_learning_rates:
        for unsafe_weight in config.unsafe_episode_weights:
            base_policy = FailureGatedCommandResidualPolicy(
                config.correction_policy
            ).to(device)
            base_policy.load_state_dict(initialized_state)
            router = HardwareConditionedResidualAuthorityCalibrator(
                base_policy,
                config.authority_router,
            ).to(device)
            history, states = _train_router(
                router,
                authority_records,
                config,
                learning_rate=learning_rate,
                unsafe_episode_weight=unsafe_weight,
            )
            selected, evaluations, selected_state = _select_router_state(
                base_model,
                router,
                states,
                validation,
                validation_supervision,
                validation_indices,
                validation_criticality,
                reference,
                config,
                device,
            )
            arms.append(
                {
                    "learning_rate": learning_rate,
                    "unsafe_episode_weight": unsafe_weight,
                    "selected": selected,
                    "training_history": history,
                    "checkpoint_candidates": evaluations,
                }
            )
            selected_states.append(selected_state)
    passing_indices = [
        index
        for index, arm in enumerate(arms)
        if arm["selected"]["gate"]["passed"]
    ]
    selected_arm_index = min(
        passing_indices or range(len(arms)),
        key=lambda index: (
            float(
                arms[index]["selected"]["metrics"]["global"][
                    "tracking_rmse_normalized"
                ]
            )
            if passing_indices
            else float(arms[index]["selected"]["selection_score"])
        ),
    )
    selected_arm = arms[selected_arm_index]
    checkpoint = None
    if selected_arm["selected"]["gate"]["passed"]:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / (
            "gimbal_v16_deployable_authority_router_seed_29.pt"
        )
        torch.save(
            {
                "schema_version": DEPLOYABLE_AUTHORITY_DISTILLATION_SCHEMA_VERSION,
                "correction_policy_config": asdict(config.correction_policy),
                "authority_router_config": asdict(config.authority_router),
                "router_state_dict": selected_states[selected_arm_index],
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "adapter_config": asdict(adapter),
                "metadata": {
                    "selected_epoch": selected_arm["selected"]["epoch"],
                    "learning_rate": selected_arm["learning_rate"],
                    "unsafe_episode_weight": selected_arm[
                        "unsafe_episode_weight"
                    ],
                    "fresh_test_opened": False,
                },
            },
            checkpoint,
        )
    return {
        "experiment": DEPLOYABLE_AUTHORITY_DISTILLATION_SCHEMA_VERSION,
        "config": asdict(config),
        "datasets": {
            "train": {"path": str(train_path), "sha256": _sha256(train_path)},
            "validation": {
                "path": str(validation_path),
                "sha256": _sha256(validation_path),
            },
            "fresh_test": {"opened": False},
        },
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "sha256": _sha256(base_checkpoint),
            "metadata": base_metadata,
        },
        "architecture": {
            "base_midpoint_gru_frozen": True,
            "v2_1_position_adapter_frozen": True,
            "v14_1_recurrent_residual_frozen": True,
            "authority_router_inputs_are_deployable": True,
            "scenario_identity_is_not_a_router_input": True,
            "hardware_conditioned": True,
            "state_consistent_authority_replay": True,
            "exact_multi_command_oracle_and_selection": True,
            "fresh_test_sealed": True,
        },
        "initial_distillation": {
            "metrics": initialized_metrics,
            "gate": _gate(initialized_metrics, reference, config),
            "training_history": initialization_history,
            "oracle_training_data": correction_diagnostics,
        },
        "reference": reference,
        "authority_oracle": {
            "training": training_oracle_diagnostics,
            "validation": {
                **_authority_summary(
                    validation_oracle.selected_authority,
                    validation,
                    validation_indices,
                    config,
                ),
                "metrics": validation_oracle_metrics,
                "gate": validation_oracle_gate,
            },
            "privileged_future_targets_are_training_only": True,
            "scenario_identity_is_diagnostic_only": True,
        },
        "arms": arms,
        "selected_arm_index": selected_arm_index,
        "selected": selected_arm,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "recommendation": (
            "replicate_v16_before_fresh_test"
            if selected_arm["selected"]["gate"]["passed"]
            else "do_not_promote_v16"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run V16 deployable residual-authority distillation."
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
        default=Path("artifacts/gimbal_deployable_residual_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_deployable_authority_v16.json"),
    )
    parser.add_argument("--router-epochs", type=int, default=80)
    parser.add_argument("--selection-interval", type=int, default=10)
    parser.add_argument("--router-learning-rates", type=float, nargs="+")
    parser.add_argument("--unsafe-episode-weights", type=float, nargs="+")
    args = parser.parse_args(argv)
    defaults = DeployableAuthorityDistillationConfig()
    result = evaluate_deployable_authority_distillation_experiment(
        train_path=args.train_data,
        validation_path=args.validation_data,
        base_checkpoint=args.base_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        config=replace(
            defaults,
            router_epochs=args.router_epochs,
            selection_epoch_interval=args.selection_interval,
            router_learning_rates=(
                tuple(args.router_learning_rates)
                if args.router_learning_rates is not None
                else defaults.router_learning_rates
            ),
            unsafe_episode_weights=(
                tuple(args.unsafe_episode_weights)
                if args.unsafe_episode_weights is not None
                else defaults.unsafe_episode_weights
            ),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selected = result["selected"]["selected"]
    print(f"wrote {args.output}")
    print(
        f"oracle_passed={result['authority_oracle']['validation']['gate']['passed']}; "
        f"student_passed={selected['gate']['passed']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
