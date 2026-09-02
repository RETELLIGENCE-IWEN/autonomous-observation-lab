"""V15 constrained sequence fine-tuning of the deployable gated residual."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
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
from .gru import load_gru_checkpoint
from .gru_training import set_gru_seed
from .multi_command_experiment import _context_window
from .multi_command_policy import (
    CausalRecurrentPositionResidualPolicy,
    RecurrentPositionResidualPolicyConfig,
    counterfactual_capture_source_indices,
    recurrent_policy_input_dim,
)
from .on_policy_distillation_experiment import _rollout_metrics, _selection_score
from .sequence_distillation_experiment import _episode_indices
from .sequence_oracle import PrivilegedSequenceOracleConfig
from .sequence_oracle_experiment import _gate, _sha256


DEPLOYABLE_CONSTRAINED_FINETUNING_SCHEMA_VERSION = (
    "gimbal_deployable_constrained_finetuning_v15_development_v1"
)
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class DeployableConstrainedFinetuningConfig:
    """Development-only primal-dual sequence optimization protocol."""

    behavior_name: str = "privileged_oracle_position"
    sequence_steps: int = 16
    training_episode_count: int = 192
    finetuning_episode_count: int = 288
    validation_episode_count: int = 48
    oracle_batch_size: int = 8
    rollout_batch_size: int = 16
    initial_distillation_epochs: int = 30
    initial_distillation_trust_region_weight: float = 0.25
    initial_distillation_learning_rate: float = 1e-3
    finetuning_epochs: int = 12
    selection_epoch_interval: int = 1
    batch_size: int = 16
    critical_episode_fraction: float = 0.50
    trainable_policy_scopes: tuple[str, ...] = ("full_residual",)
    learning_rates: tuple[float, ...] = (1e-4, 3e-4)
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    training_integration_period_s: float = 0.010
    ordinary_tracking_weight: float = 1.0
    critical_tracking_weights: tuple[float, ...] = (1.0,)
    smoothness_weight: float = 0.10
    saturation_weight: float = 0.01
    visibility_penalties: tuple[float, ...] = (1.0, 10.0)
    training_visibility_improvement_fractions: tuple[float, ...] = (0.0,)
    visibility_cvar_fractions: tuple[float, ...] = (1.0,)
    visibility_constraint_scopes: tuple[str, ...] = ("episode_cvar",)
    privileged_scenario_authority_scales: tuple[float, ...] = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    )
    trust_region_penalty: float = 1.0
    trust_region_radius_multipliers: tuple[float, ...] = (1.0,)
    dual_learning_rate: float = 0.25
    maximum_dual_value: float = 100.0
    constraint_scale_epsilon: float = 1e-6
    minimum_tracking_improvement_fraction: float = 0.005
    maximum_saturation_regression_fraction: float = 0.05
    optimization_seed: int = 29
    device: str = "cpu"
    correction_policy: FailureGatedPositionPolicyConfig = (
        FailureGatedPositionPolicyConfig(
            hidden_dim=48,
            embedding_dim=48,
            maximum_residual_magnitude=0.40,
        )
    )
    authority_calibrator: ResidualAuthorityCalibratorConfig = (
        ResidualAuthorityCalibratorConfig()
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
            "training_episode_count",
            "finetuning_episode_count",
            "validation_episode_count",
            "oracle_batch_size",
            "rollout_batch_size",
            "initial_distillation_epochs",
            "finetuning_epochs",
            "selection_epoch_interval",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"V15 {name} must be positive")
        for name in (
            "initial_distillation_trust_region_weight",
            "weight_decay",
            "ordinary_tracking_weight",
            "smoothness_weight",
            "saturation_weight",
            "trust_region_penalty",
            "minimum_tracking_improvement_fraction",
            "maximum_saturation_regression_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V15 {name} must be finite and non-negative")
        for name in (
            "initial_distillation_learning_rate",
            "gradient_clip_norm",
            "training_integration_period_s",
            "dual_learning_rate",
            "maximum_dual_value",
            "constraint_scale_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"V15 {name} must be finite and positive")
        if not 0.0 < self.critical_episode_fraction < 1.0:
            raise ValueError("V15 critical episode fraction must be in (0, 1)")
        for values, name, positive in (
            (self.learning_rates, "learning rates", True),
            (
                self.critical_tracking_weights,
                "critical tracking weights",
                False,
            ),
            (self.visibility_penalties, "visibility penalties", False),
            (
                self.trust_region_radius_multipliers,
                "trust-region radius multipliers",
                True,
            ),
        ):
            if not values or any(
                not math.isfinite(value)
                or (value <= 0.0 if positive else value < 0.0)
                for value in values
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"V15 {name} must be finite and {qualifier}")
            if len(set(values)) != len(values):
                raise ValueError(f"V15 {name} must be unique")
        if not self.training_visibility_improvement_fractions or any(
            not math.isfinite(value) or not 0.0 <= value < 1.0
            for value in self.training_visibility_improvement_fractions
        ):
            raise ValueError(
                "V15 training visibility improvements must be in [0, 1)"
            )
        if len(set(self.training_visibility_improvement_fractions)) != len(
            self.training_visibility_improvement_fractions
        ):
            raise ValueError(
                "V15 training visibility improvements must be unique"
            )
        if not self.visibility_cvar_fractions or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.visibility_cvar_fractions
        ):
            raise ValueError("V15 visibility CVaR fractions must be in (0, 1]")
        if len(set(self.visibility_cvar_fractions)) != len(
            self.visibility_cvar_fractions
        ):
            raise ValueError("V15 visibility CVaR fractions must be unique")
        allowed_scopes = {"episode_cvar", "scenario_max"}
        if not self.visibility_constraint_scopes or any(
            value not in allowed_scopes
            for value in self.visibility_constraint_scopes
        ):
            raise ValueError("V15 visibility constraint scope is invalid")
        if len(set(self.visibility_constraint_scopes)) != len(
            self.visibility_constraint_scopes
        ):
            raise ValueError("V15 visibility constraint scopes must be unique")
        if not self.privileged_scenario_authority_scales or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.privileged_scenario_authority_scales
        ):
            raise ValueError(
                "V15 privileged scenario authority scales must be in [0, 1]"
            )
        if len(set(self.privileged_scenario_authority_scales)) != len(
            self.privileged_scenario_authority_scales
        ):
            raise ValueError(
                "V15 privileged scenario authority scales must be unique"
            )
        allowed_policy_scopes = {"full_residual", "authority_calibrator"}
        if not self.trainable_policy_scopes or any(
            value not in allowed_policy_scopes
            for value in self.trainable_policy_scopes
        ):
            raise ValueError("V15 trainable policy scope is invalid")
        if len(set(self.trainable_policy_scopes)) != len(
            self.trainable_policy_scopes
        ):
            raise ValueError("V15 trainable policy scopes must be unique")
        if self.oracle.focus_start_index != 0 or (
            self.oracle.focus_steps != self.sequence_steps
        ):
            raise ValueError("V15 requires an episode-start sequence oracle")


def _distillation_config(
    config: DeployableConstrainedFinetuningConfig,
) -> DeployableResidualDistillationConfig:
    return DeployableResidualDistillationConfig(
        behavior_name=config.behavior_name,
        sequence_steps=config.sequence_steps,
        training_episode_count=config.training_episode_count,
        validation_episode_count=config.validation_episode_count,
        oracle_batch_size=config.oracle_batch_size,
        rollout_batch_size=config.rollout_batch_size,
        epochs=config.initial_distillation_epochs,
        selection_epoch_interval=config.initial_distillation_epochs,
        batch_size=config.batch_size,
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
    policy: FailureGatedCommandResidualPolicy,
    dataset,
    supervision,
    indices: np.ndarray,
    config: DeployableConstrainedFinetuningConfig,
    device: torch.device,
    *,
    integration_period_override_s: float | None,
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
        integration_period_override_s=integration_period_override_s,
        visibility_margin_fraction=config.oracle.visibility_margin_fraction,
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _critical_scenario_diagnostics(
    reference: DeployableFailureGatedRollout,
    candidate: DeployableFailureGatedRollout,
    dataset,
    indices: np.ndarray,
    criticality,
    config: DeployableConstrainedFinetuningConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    critical = torch.from_numpy(
        criticality.critical_mask[
            indices,
            1 : config.sequence_steps + 1,
            0,
        ]
        & dataset.sequence_mask[indices, : config.sequence_steps]
    ).bool().to(device)
    scenario_values = dataset.scenario_index[indices]
    scenarios = dataset.manifest.generation.get("scenarios", [])
    result = []
    for scenario_index in np.unique(scenario_values):
        selected = critical & torch.from_numpy(
            scenario_values == scenario_index
        ).to(device).unsqueeze(1)
        count = int(selected.sum())
        if count == 0:
            continue
        reference_tracking = math.sqrt(
            float(_masked_mean(reference.tracking_error_normalized.square(), selected))
        )
        candidate_tracking = math.sqrt(
            float(_masked_mean(candidate.tracking_error_normalized.square(), selected))
        )
        reference_visibility = math.sqrt(
            float(
                _masked_mean(
                    reference.visibility_violation_normalized.square(),
                    selected,
                )
            )
        )
        candidate_visibility = math.sqrt(
            float(
                _masked_mean(
                    candidate.visibility_violation_normalized.square(),
                    selected,
                )
            )
        )
        result.append(
            {
                "scenario_index": int(scenario_index),
                "scenario_name": (
                    scenarios[int(scenario_index)].get("name")
                    if int(scenario_index) < len(scenarios)
                    else None
                ),
                "sample_count": count,
                "reference_tracking_rmse_normalized": reference_tracking,
                "candidate_tracking_rmse_normalized": candidate_tracking,
                "reference_visibility_rmse_normalized": reference_visibility,
                "candidate_visibility_rmse_normalized": candidate_visibility,
            }
        )
    return result


def _privileged_scenario_authority_ceiling(
    base_model,
    initialized_policy: FailureGatedCommandResidualPolicy,
    validation,
    validation_supervision,
    validation_indices: np.ndarray,
    validation_criticality,
    reference: dict[str, Any],
    scenario_diagnostics: list[dict[str, Any]],
    config: DeployableConstrainedFinetuningConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Screen scenario-specific authority as a non-deployable diagnostic."""

    failing = max(
        scenario_diagnostics,
        key=lambda record: (
            float(record["candidate_visibility_rmse_normalized"])
            - float(record["reference_visibility_rmse_normalized"])
        ),
    )
    failing_scenario = int(failing["scenario_index"])
    distillation_config = _distillation_config(config)
    scenario_values = validation.scenario_index[validation_indices]
    evaluations = []
    for authority_scale in config.privileged_scenario_authority_scales:
        rollouts = []
        ordered_indices = []
        for scenario_index in sorted(np.unique(scenario_values)):
            indices = validation_indices[scenario_values == scenario_index]
            ordered_indices.extend(indices.tolist())
            rollouts.append(
                _evaluate_policy(
                    base_model,
                    initialized_policy,
                    validation,
                    validation_supervision,
                    indices,
                    distillation_config,
                    device,
                    residual_scale=(
                        authority_scale
                        if int(scenario_index) == failing_scenario
                        else 1.0
                    ),
                )
            )
        rollout = _concatenate_rollouts(rollouts)
        metrics = _rollout_metrics(
            rollout,
            validation,
            np.asarray(ordered_indices, dtype=np.int64),
            validation_criticality,
            config,
            device,
        )
        evaluations.append(
            {
                "failing_scenario_authority_scale": authority_scale,
                "metrics": metrics,
                "gate": _gate(metrics, reference, config),
            }
        )
    selected = min(
        evaluations,
        key=lambda record: float(record["metrics"]["critical"][
            "visibility_rmse_normalized"
        ]),
    )
    return {
        "deployable": False,
        "uses_privileged_scenario_identity": True,
        "purpose": "conditional_authority_feasibility_screen",
        "failing_scenario_index": failing_scenario,
        "failing_scenario_name": failing["scenario_name"],
        "evaluations": evaluations,
        "minimum_visibility_candidate": selected,
    }


def _cached_training_rollout(
    base_model,
    policy: FailureGatedCommandResidualPolicy,
    dataset,
    supervision,
    indices: np.ndarray,
    config: DeployableConstrainedFinetuningConfig,
    device: torch.device,
) -> DeployableFailureGatedRollout:
    """Evaluate one frozen policy once for stochastic constraint batches."""

    rollouts = []
    policy.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), config.rollout_batch_size):
            selected = indices[offset : offset + config.rollout_batch_size]
            rollouts.append(
                _policy_rollout_batch(
                    base_model,
                    policy,
                    dataset,
                    supervision,
                    selected,
                    config,
                    device,
                    integration_period_override_s=(
                        config.training_integration_period_s
                    ),
                )
            )
    return _concatenate_rollouts(rollouts)


def _select_cached_rollout(
    rollout: DeployableFailureGatedRollout,
    positions: np.ndarray,
) -> DeployableFailureGatedRollout:
    return DeployableFailureGatedRollout(
        **{
            field.name: getattr(rollout, field.name)[positions]
            for field in fields(rollout)
        }
    )


def _loss_terms(
    rollout: DeployableFailureGatedRollout,
    reference: DeployableFailureGatedRollout,
    initialized: DeployableFailureGatedRollout,
    dataset,
    criticality,
    indices: np.ndarray,
    config: DeployableConstrainedFinetuningConfig,
    device: torch.device,
    *,
    critical_tracking_weight: float,
    training_visibility_improvement_fraction: float,
    visibility_cvar_fraction: float,
    visibility_constraint_scope: str,
    trust_region_radius_multiplier: float,
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
    critical_mask = mask & critical
    ordinary_mask = mask & ~critical
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
    ordinary_tracking = _masked_mean(
        rollout.tracking_error_normalized.square(),
        ordinary_mask,
    )
    critical_tracking = _masked_mean(
        rollout.tracking_error_normalized.square(),
        critical_mask,
    )
    smoothness = _masked_mean(difference.square(), mask)
    saturation = _masked_mean(rollout.saturation_fraction, mask)
    critical_visibility = _masked_mean(
        rollout.visibility_violation_normalized.square(),
        critical_mask,
    )
    reference_critical_visibility = _masked_mean(
        reference.visibility_violation_normalized.square(),
        critical_mask,
    )
    critical_weights = critical_mask.to(
        rollout.visibility_violation_normalized.dtype
    )
    critical_count_per_episode = critical_weights.sum(dim=1)
    valid_critical_episode = critical_count_per_episode > 0.0
    denominator = critical_count_per_episode.clamp_min(1.0)
    candidate_visibility_per_episode = (
        rollout.visibility_violation_normalized.square() * critical_weights
    ).sum(dim=1) / denominator
    reference_visibility_per_episode = (
        reference.visibility_violation_normalized.square() * critical_weights
    ).sum(dim=1) / denominator
    visibility_limit_per_episode = reference_visibility_per_episode * (
        1.0 - training_visibility_improvement_fraction
    ) ** 2
    visibility_scale = torch.clamp(
        reference_critical_visibility.detach(),
        min=config.constraint_scale_epsilon,
    )
    visibility_excess_per_episode = (
        candidate_visibility_per_episode
        - visibility_limit_per_episode.detach()
    ) / visibility_scale
    if visibility_constraint_scope == "episode_cvar":
        selected_visibility_excess = visibility_excess_per_episode[
            valid_critical_episode
        ]
        tail_count = max(
            1,
            math.ceil(
                len(selected_visibility_excess) * visibility_cvar_fraction
            ),
        )
        visibility_violation = torch.topk(
            selected_visibility_excess,
            tail_count,
        ).values.mean()
    elif visibility_constraint_scope == "scenario_max":
        scenario = torch.from_numpy(dataset.scenario_index[indices]).to(device)
        group_excess = []
        for scenario_index in torch.unique(scenario):
            group_mask = critical_mask & (
                scenario == scenario_index
            ).unsqueeze(1)
            if bool(group_mask.any()):
                candidate_group = _masked_mean(
                    rollout.visibility_violation_normalized.square(),
                    group_mask,
                )
                reference_group = _masked_mean(
                    reference.visibility_violation_normalized.square(),
                    group_mask,
                )
                group_limit = reference_group * (
                    1.0 - training_visibility_improvement_fraction
                ) ** 2
                group_excess.append(
                    (candidate_group - group_limit.detach())
                    / visibility_scale
                )
        visibility_violation = torch.stack(group_excess).max()
    else:
        raise ValueError("V15 visibility constraint scope is invalid")
    critical_visibility_limit = _masked_mean(
        reference.visibility_violation_normalized.square(),
        critical_mask,
    ) * (1.0 - training_visibility_improvement_fraction) ** 2

    ordinary_reference_distance = _masked_mean(
        rollout.residual_normalized.square(),
        ordinary_mask,
    )
    baseline_ordinary_reference_distance = _masked_mean(
        initialized.residual_normalized.square(),
        ordinary_mask,
    )
    trust_region_limit = baseline_ordinary_reference_distance * (
        trust_region_radius_multiplier**2
    )
    trust_scale = torch.clamp(
        baseline_ordinary_reference_distance.detach(),
        min=config.constraint_scale_epsilon,
    )
    trust_region_violation = (
        ordinary_reference_distance - trust_region_limit.detach()
    ) / trust_scale
    objective = (
        config.ordinary_tracking_weight * ordinary_tracking
        + critical_tracking_weight * critical_tracking
        + config.smoothness_weight * smoothness
        + config.saturation_weight * saturation
    )
    return {
        "objective": objective,
        "ordinary_tracking_mse": ordinary_tracking,
        "critical_tracking_mse": critical_tracking,
        "smoothness_mse": smoothness,
        "saturation_mean": saturation,
        "critical_visibility_mse": critical_visibility,
        "critical_visibility_limit_mse": critical_visibility_limit,
        "visibility_constraint": visibility_violation,
        "ordinary_reference_distance_mse": ordinary_reference_distance,
        "ordinary_reference_limit_mse": trust_region_limit,
        "trust_region_constraint": trust_region_violation,
    }


def _critical_focused_batches(
    dataset_indices: np.ndarray,
    criticality,
    dataset,
    config: DeployableConstrainedFinetuningConfig,
    generator: torch.Generator,
) -> list[np.ndarray]:
    critical_mask = criticality.critical_mask[
        dataset_indices,
        1 : config.sequence_steps + 1,
        0,
    ] & dataset.sequence_mask[dataset_indices, : config.sequence_steps]
    critical_positions = np.flatnonzero(critical_mask.any(axis=1))
    ordinary_positions = np.flatnonzero(~critical_mask.any(axis=1))
    if len(critical_positions) == 0 or len(ordinary_positions) == 0:
        raise ValueError("V15 requires both critical and ordinary episodes")
    critical_per_batch = max(
        1,
        min(
            config.batch_size - 1,
            round(config.batch_size * config.critical_episode_fraction),
        ),
    )
    ordinary_per_batch = config.batch_size - critical_per_batch
    critical_groups = {
        int(scenario_index): critical_positions[
            dataset.scenario_index[dataset_indices[critical_positions]]
            == scenario_index
        ]
        for scenario_index in np.unique(
            dataset.scenario_index[dataset_indices[critical_positions]]
        )
    }
    group_names = sorted(critical_groups)
    batch_count = math.ceil(len(dataset_indices) / config.batch_size)
    result = []
    for _ in range(batch_count):
        selected_critical_positions = []
        group_order = torch.randperm(
            len(group_names),
            generator=generator,
        ).tolist()
        for slot in range(critical_per_batch):
            group_name = group_names[group_order[slot % len(group_names)]]
            group = critical_groups[group_name]
            selected_critical_positions.append(
                group[
                    int(
                        torch.randint(
                            len(group),
                            (1,),
                            generator=generator,
                        )
                    )
                ]
            )
        ordinary_selected = torch.randint(
            len(ordinary_positions),
            (ordinary_per_batch,),
            generator=generator,
        ).numpy()
        positions = np.concatenate(
            (
                np.asarray(selected_critical_positions, dtype=np.int64),
                ordinary_positions[ordinary_selected],
            )
        )
        permutation = torch.randperm(
            len(positions),
            generator=generator,
        ).numpy()
        result.append(dataset_indices[positions[permutation]])
    return result


def _train_constrained_arm(
    base_model,
    policy: FailureGatedCommandResidualPolicy,
    reference_training_rollout: DeployableFailureGatedRollout,
    initialized_training_rollout: DeployableFailureGatedRollout,
    dataset,
    supervision,
    dataset_indices: np.ndarray,
    criticality,
    config: DeployableConstrainedFinetuningConfig,
    device: torch.device,
    *,
    learning_rate: float,
    critical_tracking_weight: float,
    visibility_penalty: float,
    training_visibility_improvement_fraction: float,
    visibility_cvar_fraction: float,
    visibility_constraint_scope: str,
    trust_region_radius_multiplier: float,
) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.optimization_seed)
    visibility_dual = 0.0
    trust_region_dual = 0.0
    history = []
    states = [(0, copy.deepcopy(policy.state_dict()))]
    base_model.eval()
    position_by_index = {
        int(dataset_index): position
        for position, dataset_index in enumerate(dataset_indices)
    }
    for epoch in range(1, config.finetuning_epochs + 1):
        totals: dict[str, float] = {}
        batches = _critical_focused_batches(
            dataset_indices,
            criticality,
            dataset,
            config,
            generator,
        )
        policy.train()
        for indices in batches:
            positions = np.asarray(
                [position_by_index[int(index)] for index in indices],
                dtype=np.int64,
            )
            reference = _select_cached_rollout(
                reference_training_rollout,
                positions,
            )
            initialized = _select_cached_rollout(
                initialized_training_rollout,
                positions,
            )
            rollout = _policy_rollout_batch(
                base_model,
                policy,
                dataset,
                supervision,
                indices,
                config,
                device,
                integration_period_override_s=(
                    config.training_integration_period_s
                ),
            )
            terms = _loss_terms(
                rollout,
                reference,
                initialized,
                dataset,
                criticality,
                indices,
                config,
                device,
                critical_tracking_weight=critical_tracking_weight,
                training_visibility_improvement_fraction=(
                    training_visibility_improvement_fraction
                ),
                visibility_cvar_fraction=visibility_cvar_fraction,
                visibility_constraint_scope=visibility_constraint_scope,
                trust_region_radius_multiplier=(
                    trust_region_radius_multiplier
                ),
            )
            visibility_constraint = terms["visibility_constraint"]
            trust_constraint = terms["trust_region_constraint"]
            loss = (
                terms["objective"]
                + visibility_dual * visibility_constraint
                + 0.5
                * visibility_penalty
                * torch.relu(visibility_constraint).square()
                + trust_region_dual * trust_constraint
                + 0.5
                * config.trust_region_penalty
                * torch.relu(trust_constraint).square()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            visibility_dual = min(
                config.maximum_dual_value,
                max(
                    0.0,
                    visibility_dual
                    + config.dual_learning_rate
                    * float(visibility_constraint.detach()),
                ),
            )
            trust_region_dual = min(
                config.maximum_dual_value,
                max(
                    0.0,
                    trust_region_dual
                    + config.dual_learning_rate
                    * float(trust_constraint.detach()),
                ),
            )
            batch_values = {
                "loss": float(loss.detach()),
                **{
                    name: float(value.detach())
                    for name, value in terms.items()
                },
            }
            for name, value in batch_values.items():
                totals[name] = totals.get(name, 0.0) + value
        history.append(
            {
                "epoch": epoch,
                **{
                    name: value / len(batches)
                    for name, value in totals.items()
                },
                "visibility_dual": visibility_dual,
                "trust_region_dual": trust_region_dual,
            }
        )
        if epoch % config.selection_epoch_interval == 0 or (
            epoch == config.finetuning_epochs
        ):
            states.append((epoch, copy.deepcopy(policy.state_dict())))
    return history, states


def _select_state(
    base_model,
    policy: FailureGatedCommandResidualPolicy,
    states: list[tuple[int, dict[str, Any]]],
    validation,
    validation_supervision,
    validation_indices: np.ndarray,
    validation_criticality,
    reference,
    config: DeployableConstrainedFinetuningConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    evaluations = []
    state_by_epoch = dict(states)
    distillation_config = _distillation_config(config)
    for epoch, state in states:
        policy.load_state_dict(state)
        rollout = _evaluate_policy(
            base_model,
            policy,
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
                "mean_gate_probability": float(
                    rollout.gate_probability.mean()
                ),
                "mean_absolute_residual_normalized": float(
                    rollout.residual_normalized.abs().mean()
                ),
                "maximum_absolute_residual_normalized": float(
                    rollout.residual_normalized.abs().max()
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
    policy.load_state_dict(selected_state)
    return selected, evaluations, selected_state


def evaluate_deployable_constrained_finetuning_experiment(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    base_checkpoint: str | Path,
    checkpoint_directory: str | Path,
    config: DeployableConstrainedFinetuningConfig | None = None,
) -> dict[str, Any]:
    """Fine-tune V14.1 with explicit sequence-level safety constraints."""

    config = config or DeployableConstrainedFinetuningConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    base_checkpoint = Path(base_checkpoint)
    checkpoint_directory = Path(checkpoint_directory)
    device = torch.device(config.device)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    distillation_train_indices = _episode_indices(
        train,
        config,
        config.training_episode_count,
    )
    finetuning_train_indices = _episode_indices(
        train,
        config,
        config.finetuning_episode_count,
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
        raise ValueError("V15 requires the hard-midpoint predictor")
    if base_model.config.prediction_horizons_s != (
        train.manifest.prediction_horizons_s
    ) or base_model.config.prediction_horizons_s != (
        validation.manifest.prediction_horizons_s
    ):
        raise ValueError("V15 checkpoint horizons differ from a dataset")
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
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
    zero_policy = CausalRecurrentPositionResidualPolicy(
        RecurrentPositionResidualPolicyConfig(
            input_dim=recurrent_policy_input_dim(base_model.horizon_count),
            hidden_dim=8,
            embedding_dim=8,
        )
    ).to(device)
    records, oracle_diagnostics = _generate_state_consistent_records(
        base_model,
        zero_policy,
        train,
        train_supervision,
        train_criticality,
        distillation_train_indices,
        distillation_config,
        device,
    )

    set_gru_seed(config.optimization_seed)
    reference_policy = FailureGatedCommandResidualPolicy(
        config.correction_policy
    ).to(device)
    reference_rollout = _evaluate_policy(
        base_model,
        reference_policy,
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

    set_gru_seed(config.optimization_seed)
    initialized_policy = FailureGatedCommandResidualPolicy(
        config.correction_policy
    ).to(device)
    initialization_history, initialization_states = _train_policy(
        initialized_policy,
        records,
        trust_region_weight=(
            config.initial_distillation_trust_region_weight
        ),
        config=distillation_config,
        generator=torch.Generator().manual_seed(config.optimization_seed),
    )
    initialized_policy.load_state_dict(initialization_states[-1][1])
    initialized_rollout = _evaluate_policy(
        base_model,
        initialized_policy,
        validation,
        validation_supervision,
        validation_indices,
        distillation_config,
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
    initialized = {
        "epoch": config.initial_distillation_epochs,
        "metrics": initialized_metrics,
        "gate": _gate(initialized_metrics, reference, config),
        "mean_gate_probability": float(
            initialized_rollout.gate_probability.mean()
        ),
        "mean_absolute_residual_normalized": float(
            initialized_rollout.residual_normalized.abs().mean()
        ),
        "maximum_absolute_residual_normalized": float(
            initialized_rollout.residual_normalized.abs().max()
        ),
    }
    scenario_diagnostics = _critical_scenario_diagnostics(
        reference_rollout,
        initialized_rollout,
        validation,
        validation_indices,
        validation_criticality,
        config,
        device,
    )
    privileged_authority_ceiling = _privileged_scenario_authority_ceiling(
        base_model,
        initialized_policy,
        validation,
        validation_supervision,
        validation_indices,
        validation_criticality,
        reference,
        scenario_diagnostics,
        config,
        device,
    )
    initialized_state = copy.deepcopy(initialized_policy.state_dict())
    baseline_policy = FailureGatedCommandResidualPolicy(
        config.correction_policy
    ).to(device)
    baseline_policy.load_state_dict(initialized_state)
    for parameter in baseline_policy.parameters():
        parameter.requires_grad_(False)
    baseline_policy.eval()
    reference_training_rollout = _cached_training_rollout(
        base_model,
        reference_policy,
        train,
        train_supervision,
        finetuning_train_indices,
        config,
        device,
    )
    initialized_training_rollout = _cached_training_rollout(
        base_model,
        baseline_policy,
        train,
        train_supervision,
        finetuning_train_indices,
        config,
        device,
    )

    arms = []
    selected_states = []
    arm_grid = product(
        config.learning_rates,
        config.critical_tracking_weights,
        config.visibility_penalties,
        config.training_visibility_improvement_fractions,
        config.visibility_cvar_fractions,
        config.visibility_constraint_scopes,
        config.trainable_policy_scopes,
        config.trust_region_radius_multipliers,
    )
    for (
        learning_rate,
        critical_tracking_weight,
        visibility_penalty,
        visibility_improvement,
        visibility_cvar_fraction,
        visibility_constraint_scope,
        trainable_policy_scope,
        radius_multiplier,
    ) in arm_grid:
        if trainable_policy_scope == "full_residual":
            policy = FailureGatedCommandResidualPolicy(
                config.correction_policy
            ).to(device)
            policy.load_state_dict(initialized_state)
        else:
            calibrated_base = FailureGatedCommandResidualPolicy(
                config.correction_policy
            ).to(device)
            calibrated_base.load_state_dict(initialized_state)
            policy = HardwareConditionedResidualAuthorityCalibrator(
                calibrated_base,
                config.authority_calibrator,
            ).to(device)
        history, states = _train_constrained_arm(
            base_model,
            policy,
            reference_training_rollout,
            initialized_training_rollout,
            train,
            train_supervision,
            finetuning_train_indices,
            train_criticality,
            config,
            device,
            learning_rate=learning_rate,
            critical_tracking_weight=critical_tracking_weight,
            visibility_penalty=visibility_penalty,
            training_visibility_improvement_fraction=(
                visibility_improvement
            ),
            visibility_cvar_fraction=visibility_cvar_fraction,
            visibility_constraint_scope=visibility_constraint_scope,
            trust_region_radius_multiplier=radius_multiplier,
        )
        selected, evaluations, selected_state = _select_state(
            base_model,
            policy,
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
                "critical_tracking_weight": critical_tracking_weight,
                "visibility_penalty": visibility_penalty,
                "training_visibility_improvement_fraction": (
                    visibility_improvement
                ),
                "visibility_cvar_fraction": visibility_cvar_fraction,
                "visibility_constraint_scope": visibility_constraint_scope,
                "trainable_policy_scope": trainable_policy_scope,
                "trust_region_radius_multiplier": radius_multiplier,
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
            "gimbal_v15_deployable_constrained_residual_seed_29.pt"
        )
        torch.save(
            {
                "schema_version": (
                    DEPLOYABLE_CONSTRAINED_FINETUNING_SCHEMA_VERSION
                ),
                "policy_config": asdict(config.correction_policy),
                "authority_calibrator_config": asdict(
                    config.authority_calibrator
                ),
                "policy_state_dict": selected_states[selected_arm_index],
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "adapter_config": asdict(adapter),
                "metadata": {
                    "selected_epoch": selected_arm["selected"]["epoch"],
                    "learning_rate": selected_arm["learning_rate"],
                    "critical_tracking_weight": selected_arm[
                        "critical_tracking_weight"
                    ],
                    "visibility_penalty": selected_arm[
                        "visibility_penalty"
                    ],
                    "training_visibility_improvement_fraction": selected_arm[
                        "training_visibility_improvement_fraction"
                    ],
                    "visibility_cvar_fraction": selected_arm[
                        "visibility_cvar_fraction"
                    ],
                    "visibility_constraint_scope": selected_arm[
                        "visibility_constraint_scope"
                    ],
                    "trainable_policy_scope": selected_arm[
                        "trainable_policy_scope"
                    ],
                    "trust_region_radius_multiplier": selected_arm[
                        "trust_region_radius_multiplier"
                    ],
                    "fresh_test_opened": False,
                },
            },
            checkpoint,
        )
    return {
        "experiment": DEPLOYABLE_CONSTRAINED_FINETUNING_SCHEMA_VERSION,
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
            "bounded_recurrent_residual_finetuned": True,
            "authority_calibrator_is_deployable": True,
            "policy_scope_candidates": list(config.trainable_policy_scopes),
            "counterfactual_observations_are_recurrent": True,
            "multi_command_servo_state_is_persistent": True,
            "training_plant_is_differentiable": True,
            "exact_serialized_plant_used_for_selection": True,
            "critical_visibility_is_an_explicit_constraint": True,
            "ordinary_reference_distance_is_a_trust_region": True,
            "critical_episode_sampling_is_concentrated": True,
            "hardware_conditioned": True,
        },
        "oracle_training_data": oracle_diagnostics,
        "initial_distillation": {
            **initialized,
            "training_history": initialization_history,
            "critical_scenario_diagnostics": scenario_diagnostics,
        },
        "privileged_scenario_authority_ceiling": privileged_authority_ceiling,
        "reference": reference,
        "arms": arms,
        "selected_arm_index": selected_arm_index,
        "selected": selected_arm,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "recommendation": (
            "replicate_v15_before_fresh_test"
            if selected_arm["selected"]["gate"]["passed"]
            else "do_not_promote_v15"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run V15 constrained deployable residual fine-tuning."
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
        default=Path("artifacts/gimbal_deployable_constrained_v15.json"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--finetuning-episodes", type=int)
    parser.add_argument("--selection-interval", type=int, default=1)
    parser.add_argument("--learning-rates", type=float, nargs="+")
    parser.add_argument("--trainable-policy-scopes", nargs="+")
    parser.add_argument("--critical-tracking-weights", type=float, nargs="+")
    parser.add_argument("--visibility-penalties", type=float, nargs="+")
    parser.add_argument("--training-visibility-improvements", type=float, nargs="+")
    parser.add_argument("--visibility-cvar-fractions", type=float, nargs="+")
    parser.add_argument("--visibility-constraint-scopes", nargs="+")
    parser.add_argument("--trust-radius-multipliers", type=float, nargs="+")
    parser.add_argument("--dual-learning-rate", type=float)
    parser.add_argument("--maximum-dual-value", type=float)
    args = parser.parse_args(argv)
    defaults = DeployableConstrainedFinetuningConfig()
    result = evaluate_deployable_constrained_finetuning_experiment(
        train_path=args.train_data,
        validation_path=args.validation_data,
        base_checkpoint=args.base_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        config=replace(
            defaults,
            finetuning_epochs=args.epochs,
            finetuning_episode_count=(
                args.finetuning_episodes
                if args.finetuning_episodes is not None
                else defaults.finetuning_episode_count
            ),
            selection_epoch_interval=args.selection_interval,
            dual_learning_rate=(
                args.dual_learning_rate
                if args.dual_learning_rate is not None
                else defaults.dual_learning_rate
            ),
            maximum_dual_value=(
                args.maximum_dual_value
                if args.maximum_dual_value is not None
                else defaults.maximum_dual_value
            ),
            learning_rates=(
                tuple(args.learning_rates)
                if args.learning_rates is not None
                else defaults.learning_rates
            ),
            trainable_policy_scopes=(
                tuple(args.trainable_policy_scopes)
                if args.trainable_policy_scopes is not None
                else defaults.trainable_policy_scopes
            ),
            critical_tracking_weights=(
                tuple(args.critical_tracking_weights)
                if args.critical_tracking_weights is not None
                else defaults.critical_tracking_weights
            ),
            visibility_penalties=(
                tuple(args.visibility_penalties)
                if args.visibility_penalties is not None
                else defaults.visibility_penalties
            ),
            training_visibility_improvement_fractions=(
                tuple(args.training_visibility_improvements)
                if args.training_visibility_improvements is not None
                else defaults.training_visibility_improvement_fractions
            ),
            visibility_cvar_fractions=(
                tuple(args.visibility_cvar_fractions)
                if args.visibility_cvar_fractions is not None
                else defaults.visibility_cvar_fractions
            ),
            visibility_constraint_scopes=(
                tuple(args.visibility_constraint_scopes)
                if args.visibility_constraint_scopes is not None
                else defaults.visibility_constraint_scopes
            ),
            trust_region_radius_multipliers=(
                tuple(args.trust_radius_multipliers)
                if args.trust_radius_multipliers is not None
                else defaults.trust_region_radius_multipliers
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
        f"passed={selected['gate']['passed']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
