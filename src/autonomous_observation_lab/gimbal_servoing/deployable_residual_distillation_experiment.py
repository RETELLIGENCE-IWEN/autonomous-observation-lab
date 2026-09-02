"""V14.1 gated distillation of the deployable-reference sequence oracle."""

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
from .deployable_reference import deployable_position_commands_from_features
from .failure_gated_policy import (
    FailureGatedCommandResidualPolicy,
    FailureGatedPositionPolicyConfig,
)
from .gru import GRUAdaptivePositionLossContext, load_gru_checkpoint
from .gru_training import set_gru_seed
from .multi_command_experiment import _context_window, _window_batch
from .multi_command_policy import (
    CausalRecurrentPositionResidualPolicy,
    RecurrentPositionResidualPolicyConfig,
    counterfactual_capture_source_indices,
    recurrent_policy_input_dim,
    rollout_counterfactual_window,
)
from .on_policy_distillation import rollout_counterfactual_position_commands
from .on_policy_distillation_experiment import (
    _concatenate_context,
    _rollout_metrics,
    _selection_score,
)
from .sequence_distillation_experiment import _episode_indices, _select_context
from .sequence_oracle import (
    PrivilegedSequenceOracleConfig,
    optimize_privileged_command_sequence,
)
from .sequence_oracle_experiment import _gate, _sha256


DEPLOYABLE_RESIDUAL_DISTILLATION_SCHEMA_VERSION = (
    "gimbal_deployable_residual_distillation_v14_1_development_v1"
)
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class DeployableResidualDistillationConfig:
    behavior_name: str = "privileged_oracle_position"
    sequence_steps: int = 16
    training_episode_count: int = 192
    validation_episode_count: int = 48
    oracle_batch_size: int = 8
    rollout_batch_size: int = 16
    epochs: int = 60
    selection_epoch_interval: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    optimization_seed: int = 29
    disagreement_weight: float = 4.0
    retention_weight: float = 0.25
    critical_weight: float = 1.0
    gate_classification_weight: float = 0.25
    residual_magnitude_weight: float = 0.01
    ordinary_trust_region_weights: tuple[float, ...] = (0.25, 1.0, 5.0)
    deployment_residual_scales: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    visibility_shield_strengths: tuple[float, ...] = (0.0,)
    correction_label_threshold: float = 0.005
    minimum_tracking_improvement_fraction: float = 0.005
    maximum_saturation_regression_fraction: float = 0.05
    device: str = "cpu"
    correction_policy: FailureGatedPositionPolicyConfig = (
        FailureGatedPositionPolicyConfig(
            hidden_dim=48,
            embedding_dim=48,
            maximum_residual_magnitude=0.40,
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
            "training_episode_count",
            "validation_episode_count",
            "oracle_batch_size",
            "rollout_batch_size",
            "epochs",
            "selection_epoch_interval",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"deployable distillation {name} must be positive")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "disagreement_weight",
            "retention_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"deployable distillation {name} must be positive")
        for name in (
            "weight_decay",
            "critical_weight",
            "gate_classification_weight",
            "residual_magnitude_weight",
            "correction_label_threshold",
            "minimum_tracking_improvement_fraction",
            "maximum_saturation_regression_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"deployable distillation {name} must be non-negative"
                )
        if not self.ordinary_trust_region_weights or any(
            not math.isfinite(value) or value < 0.0
            for value in self.ordinary_trust_region_weights
        ):
            raise ValueError("deployable trust weights must be non-negative")
        if len(set(self.ordinary_trust_region_weights)) != len(
            self.ordinary_trust_region_weights
        ):
            raise ValueError("deployable trust weights must be unique")
        if not self.deployment_residual_scales or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.deployment_residual_scales
        ):
            raise ValueError("deployment residual scales must be in (0, 1]")
        if len(set(self.deployment_residual_scales)) != len(
            self.deployment_residual_scales
        ):
            raise ValueError("deployment residual scales must be unique")
        if not self.visibility_shield_strengths or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.visibility_shield_strengths
        ):
            raise ValueError("visibility shield strengths must be in [0, 1]")
        if len(set(self.visibility_shield_strengths)) != len(
            self.visibility_shield_strengths
        ):
            raise ValueError("visibility shield strengths must be unique")
        if self.oracle.focus_start_index != 0 or (
            self.sequence_steps != self.oracle.focus_steps
        ):
            raise ValueError(
                "deployable distillation requires an episode-start oracle"
            )


@dataclass
class _CorrectionRecords:
    features: torch.Tensor
    context: GRUAdaptivePositionLossContext
    teacher_command: torch.Tensor
    base_command: torch.Tensor
    gate_target: torch.Tensor
    critical: torch.Tensor
    trust_mask: torch.Tensor
    sequence_mask: torch.Tensor


def _concatenate_rollouts(
    rollouts: Sequence[DeployableFailureGatedRollout],
) -> DeployableFailureGatedRollout:
    return DeployableFailureGatedRollout(
        **{
            field.name: torch.cat(
                [getattr(rollout, field.name) for rollout in rollouts],
                dim=0,
            )
            for field in fields(rollouts[0])
        }
    )


def _reference_rollout_batch(
    base_model,
    zero_policy,
    dataset,
    supervision,
    indices: np.ndarray,
    config: DeployableResidualDistillationConfig,
    device: torch.device,
):
    batch = _window_batch(
        dataset,
        supervision,
        indices,
        0,
        config.sequence_steps,
        device,
    )
    with torch.no_grad():
        rollout = rollout_counterfactual_window(
            base_model,
            zero_policy,
            batch,
            prediction_horizons_s=dataset.manifest.prediction_horizons_s,
            adapter=selected_adaptive_position_v21_config(),
            visibility_margin_fraction=config.oracle.visibility_margin_fraction,
        )
    return batch, rollout


def _generate_state_consistent_records(
    base_model,
    zero_policy,
    dataset,
    supervision,
    criticality,
    indices: np.ndarray,
    config: DeployableResidualDistillationConfig,
    device: torch.device,
) -> tuple[_CorrectionRecords, dict[str, float | int]]:
    profile = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    reference_features = []
    oracle_features = []
    reference_commands = []
    oracle_commands = []
    oracle_base_commands = []
    selected_blends = []
    base_command_parity_maximum = 0.0
    for offset in range(0, len(indices), config.oracle_batch_size):
        selected = indices[offset : offset + config.oracle_batch_size]
        batch, reference = _reference_rollout_batch(
            base_model,
            zero_policy,
            dataset,
            supervision,
            selected,
            config,
            device,
        )
        with torch.no_grad():
            reconstructed = deployable_position_commands_from_features(
                base_model,
                reference.synthetic_features,
                batch.context,
                batch.sequence_mask,
                prediction_horizons_s=dataset.manifest.prediction_horizons_s,
                adapter=selected_adaptive_position_v21_config(),
            )
        base_command_parity_maximum = max(
            base_command_parity_maximum,
            float(
                torch.max(
                    torch.abs(
                        reconstructed.command_normalized
                        - reference.command_normalized
                    )
                )
            ),
        )
        previous = reference.synthetic_features[
            :, 0, _FEATURE_INDEX["previous_action_normalized"]
        ]
        oracle = optimize_privileged_command_sequence(
            reference.command_normalized,
            batch.target_bearing_rad[:, 1:],
            batch.context,
            batch.sequence_mask,
            batch.time_s[:, 0],
            previous,
            config=config.oracle,
        )
        logged = torch.from_numpy(
            dataset.features[
                selected,
                profile,
                : config.sequence_steps,
            ]
        ).float().to(device)
        time_s = torch.from_numpy(
            dataset.time_s[selected, : config.sequence_steps + 1]
        ).float().to(device)
        target = torch.from_numpy(
            dataset.targets[
                selected,
                : config.sequence_steps + 1,
                0,
                0,
            ]
        ).float().to(device)
        oracle_rollout = rollout_counterfactual_position_commands(
            oracle.selected_command_normalized,
            logged,
            target,
            time_s,
            counterfactual_capture_source_indices(time_s, logged),
            batch.context,
            batch.sequence_mask,
            visibility_margin_fraction=config.oracle.visibility_margin_fraction,
        )
        with torch.no_grad():
            oracle_base = deployable_position_commands_from_features(
                base_model,
                oracle_rollout.synthetic_features,
                batch.context,
                batch.sequence_mask,
                prediction_horizons_s=dataset.manifest.prediction_horizons_s,
                adapter=selected_adaptive_position_v21_config(),
            )
        reference_features.append(reference.synthetic_features.detach())
        oracle_features.append(oracle_rollout.synthetic_features.detach())
        reference_commands.append(reference.command_normalized.detach())
        oracle_commands.append(oracle.selected_command_normalized.detach())
        oracle_base_commands.append(oracle_base.command_normalized.detach())
        selected_blends.append(oracle.selected_blend_fraction.detach())

    reference_feature = torch.cat(reference_features, dim=0)
    oracle_feature = torch.cat(oracle_features, dim=0)
    reference_command = torch.cat(reference_commands, dim=0)
    oracle_command = torch.cat(oracle_commands, dim=0)
    oracle_base = torch.cat(oracle_base_commands, dim=0)
    correction_delta = torch.abs(oracle_command - oracle_base)
    gate_target = (
        correction_delta > config.correction_label_threshold
    ).float()
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
    context = _context_window(
        supervision,
        indices,
        0,
        config.sequence_steps,
        device,
    )
    records = _CorrectionRecords(
        features=torch.cat((reference_feature, oracle_feature), dim=0),
        context=_concatenate_context((context, context)),
        teacher_command=torch.cat(
            (reference_command, oracle_command),
            dim=0,
        ),
        base_command=torch.cat((reference_command, oracle_base), dim=0),
        gate_target=torch.cat(
            (torch.zeros_like(gate_target), gate_target),
            dim=0,
        ),
        critical=torch.cat((critical, critical), dim=0),
        trust_mask=torch.cat(
            (torch.ones_like(gate_target), 1.0 - gate_target),
            dim=0,
        ),
        sequence_mask=torch.cat((mask, mask), dim=0),
    )
    selected_delta = torch.abs(oracle_command - reference_command)
    blend = torch.cat(selected_blends, dim=0)
    return records, {
        "episode_count": len(indices),
        "training_record_count": len(records.features),
        "nonzero_blend_fraction": float((blend > 0.0).float().mean()),
        "correction_gate_positive_fraction": float(
            (gate_target * mask).sum() / mask.sum().clamp_min(1.0)
        ),
        "selected_command_delta_mae_normalized": float(
            (selected_delta * mask).sum() / mask.sum().clamp_min(1.0)
        ),
        "selected_command_delta_max_normalized": float(selected_delta.max()),
        "reference_command_reconstruction_maximum_error": (
            base_command_parity_maximum
        ),
    }


def _train_policy(
    policy: FailureGatedCommandResidualPolicy,
    records: _CorrectionRecords,
    *,
    trust_region_weight: float,
    config: DeployableResidualDistillationConfig,
    generator: torch.Generator,
) -> tuple[list[dict[str, float | int]], list[tuple[int, dict[str, Any]]]]:
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    disagreement = torch.abs(
        records.teacher_command - records.base_command
    ) / config.oracle.maximum_command_residual
    action_weight = (
        config.retention_weight
        + config.disagreement_weight * disagreement
        + config.critical_weight * records.critical
    ) * records.sequence_mask
    history = []
    states = [(0, copy.deepcopy(policy.state_dict()))]
    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(len(records.features), generator=generator)
        totals = {
            "loss": 0.0,
            "action": 0.0,
            "trust": 0.0,
            "gate": 0.0,
            "residual": 0.0,
        }
        policy.train()
        for offset in range(0, len(order), config.batch_size):
            selected = order[offset : offset + config.batch_size].to(
                records.features.device
            )
            sequence = policy(
                records.features[selected],
                _select_context(records.context, selected),
                records.base_command[selected],
            )
            selected_action_weight = action_weight[selected]
            action_loss = (
                (sequence.command_normalized - records.teacher_command[selected])
                .square()
                .mul(selected_action_weight)
                .sum()
                / selected_action_weight.sum().clamp_min(1.0)
            )
            trust_weight = (
                records.trust_mask[selected] * records.sequence_mask[selected]
            )
            trust_loss = (
                sequence.residual_normalized.square().mul(trust_weight).sum()
                / trust_weight.sum().clamp_min(1.0)
            )
            gate_target = records.gate_target[selected]
            gate = sequence.gate_probability.clamp(1e-6, 1.0 - 1e-6)
            gate_loss_values = -(
                gate_target * torch.log(gate)
                + (1.0 - gate_target) * torch.log(1.0 - gate)
            )
            gate_loss = (
                gate_loss_values.mul(records.sequence_mask[selected]).sum()
                / records.sequence_mask[selected].sum().clamp_min(1.0)
            )
            residual_loss = (
                sequence.residual_normalized.square()
                .mul(records.sequence_mask[selected])
                .sum()
                / records.sequence_mask[selected].sum().clamp_min(1.0)
            )
            loss = (
                action_loss
                + trust_region_weight * trust_loss
                + config.gate_classification_weight * gate_loss
                + config.residual_magnitude_weight * residual_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            count = len(selected)
            for name, value in (
                ("loss", loss),
                ("action", action_loss),
                ("trust", trust_loss),
                ("gate", gate_loss),
                ("residual", residual_loss),
            ):
                totals[name] += float(value.detach()) * count
        history.append(
            {
                "epoch": epoch,
                **{name: value / len(order) for name, value in totals.items()},
            }
        )
        if epoch % config.selection_epoch_interval == 0 or epoch == config.epochs:
            states.append((epoch, copy.deepcopy(policy.state_dict())))
    return history, states


def _evaluate_policy(
    base_model,
    policy: FailureGatedCommandResidualPolicy,
    dataset,
    supervision,
    indices: np.ndarray,
    config: DeployableResidualDistillationConfig,
    device: torch.device,
    *,
    residual_scale: float = 1.0,
    visibility_shield_strength: float = 0.0,
) -> DeployableFailureGatedRollout:
    profile = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    rollouts = []
    policy.eval()
    base_model.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), config.rollout_batch_size):
            selected = indices[offset : offset + config.rollout_batch_size]
            features = torch.from_numpy(
                dataset.features[
                    selected,
                    profile,
                    : config.sequence_steps,
                ]
            ).float().to(device)
            time_s = torch.from_numpy(
                dataset.time_s[selected, : config.sequence_steps + 1]
            ).float().to(device)
            target = torch.from_numpy(
                dataset.targets[
                    selected,
                    : config.sequence_steps + 1,
                    0,
                    0,
                ]
            ).float().to(device)
            mask = torch.from_numpy(
                dataset.sequence_mask[selected, : config.sequence_steps]
            ).bool().to(device)
            context = _context_window(
                supervision,
                selected,
                0,
                config.sequence_steps,
                device,
            )
            rollouts.append(
                rollout_deployable_failure_gated_policy(
                    base_model,
                    policy,
                    features,
                    target,
                    time_s,
                    counterfactual_capture_source_indices(time_s, features),
                    context,
                    mask,
                    prediction_horizons_s=(
                        dataset.manifest.prediction_horizons_s
                    ),
                    adapter=selected_adaptive_position_v21_config(),
                    residual_scale=residual_scale,
                    visibility_shield_strength=visibility_shield_strength,
                    visibility_margin_fraction=(
                        config.oracle.visibility_margin_fraction
                    ),
                )
            )
    return _concatenate_rollouts(rollouts)


def _select_state(
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
):
    evaluations = []
    passing = []
    for epoch, state in states:
        policy.load_state_dict(state)
        for residual_scale in config.deployment_residual_scales:
            for shield_strength in config.visibility_shield_strengths:
                rollout = _evaluate_policy(
                    base_model,
                    policy,
                    validation,
                    validation_supervision,
                    validation_indices,
                    config,
                    device,
                    residual_scale=residual_scale,
                    visibility_shield_strength=shield_strength,
                )
                metrics = _rollout_metrics(
                    rollout,
                    validation,
                    validation_indices,
                    validation_criticality,
                    config,
                    device,
                )
                gate = _gate(metrics, reference, config)
                record = {
                    "epoch": epoch,
                    "residual_scale": residual_scale,
                    "visibility_shield_strength": shield_strength,
                    "selection_score": _selection_score(metrics),
                    "metrics": metrics,
                    "gate": gate,
                    "mean_gate_probability": float(
                        rollout.gate_probability.mean()
                    ),
                    "mean_absolute_residual_normalized": float(
                        torch.abs(rollout.residual_normalized).mean()
                    ),
                    "maximum_absolute_residual_normalized": float(
                        torch.abs(rollout.residual_normalized).max()
                    ),
                }
                evaluations.append(record)
                if gate["passed"]:
                    passing.append((record, state))
    pool = passing or [
        (
            record,
            next(state for epoch, state in states if epoch == record["epoch"]),
        )
        for record in evaluations
    ]
    selected = min(
        pool,
        key=lambda item: (
            float(item[0]["metrics"]["global"]["tracking_rmse_normalized"])
            if passing
            else float(item[0]["selection_score"])
        ),
    )
    policy.load_state_dict(selected[1])
    return selected[0], evaluations


def evaluate_deployable_residual_distillation_experiment(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    base_checkpoint: str | Path,
    checkpoint_directory: str | Path,
    config: DeployableResidualDistillationConfig | None = None,
) -> dict[str, Any]:
    """Distill V14 only after its deployable-reference ceiling has passed."""

    config = config or DeployableResidualDistillationConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    base_checkpoint = Path(base_checkpoint)
    checkpoint_directory = Path(checkpoint_directory)
    device = torch.device(config.device)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    train_indices = _episode_indices(
        train,
        config,
        config.training_episode_count,
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
        raise ValueError("V14.1 requires the hard-midpoint predictor")
    if base_model.config.prediction_horizons_s != (
        train.manifest.prediction_horizons_s
    ) or base_model.config.prediction_horizons_s != (
        validation.manifest.prediction_horizons_s
    ):
        raise ValueError("V14.1 checkpoint horizons differ from a dataset")
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
    train_criticality = compute_control_criticality(train, config=config.criticality)
    validation_criticality = compute_control_criticality(
        validation,
        config=config.criticality,
    )
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
        train_indices,
        config,
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
        config,
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
    arms = []
    selected_states = []
    for trust_weight in config.ordinary_trust_region_weights:
        set_gru_seed(config.optimization_seed)
        policy = FailureGatedCommandResidualPolicy(
            config.correction_policy
        ).to(device)
        history, states = _train_policy(
            policy,
            records,
            trust_region_weight=trust_weight,
            config=config,
            generator=torch.Generator().manual_seed(config.optimization_seed),
        )
        selected, evaluations = _select_state(
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
                "ordinary_trust_region_weight": trust_weight,
                "selected_epoch": selected["epoch"],
                "selected_residual_scale": selected["residual_scale"],
                "selected_visibility_shield_strength": selected[
                    "visibility_shield_strength"
                ],
                "selection_score": selected["selection_score"],
                "metrics": selected["metrics"],
                "gate": selected["gate"],
                "mean_gate_probability": selected["mean_gate_probability"],
                "mean_absolute_residual_normalized": selected[
                    "mean_absolute_residual_normalized"
                ],
                "maximum_absolute_residual_normalized": selected[
                    "maximum_absolute_residual_normalized"
                ],
                "training_history": history,
                "checkpoint_candidates": evaluations,
            }
        )
        selected_states.append(copy.deepcopy(policy.state_dict()))
    passing_indices = [
        index for index, arm in enumerate(arms) if arm["gate"]["passed"]
    ]
    selected_arm_index = min(
        passing_indices or range(len(arms)),
        key=lambda index: (
            float(arms[index]["metrics"]["global"]["tracking_rmse_normalized"])
            if passing_indices
            else float(arms[index]["selection_score"])
        ),
    )
    selected_arm = arms[selected_arm_index]
    checkpoint = None
    if selected_arm["gate"]["passed"]:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / (
            "gimbal_v14_1_deployable_gated_residual_seed_29.pt"
        )
        torch.save(
            {
                "schema_version": DEPLOYABLE_RESIDUAL_DISTILLATION_SCHEMA_VERSION,
                "policy_config": asdict(config.correction_policy),
                "policy_state_dict": selected_states[selected_arm_index],
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "adapter_config": asdict(adapter),
                "metadata": {
                    "selected_epoch": selected_arm["selected_epoch"],
                    "residual_scale": selected_arm["selected_residual_scale"],
                    "visibility_shield_strength": selected_arm[
                        "selected_visibility_shield_strength"
                    ],
                    "ordinary_trust_region_weight": selected_arm[
                        "ordinary_trust_region_weight"
                    ],
                    "fresh_test_opened": False,
                },
            },
            checkpoint,
        )
    return {
        "experiment": DEPLOYABLE_RESIDUAL_DISTILLATION_SCHEMA_VERSION,
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
            "direct_command_residual_is_bounded": True,
            "gate_inputs_are_deployable": True,
            "hardware_conditioned": True,
            "reference_retention_records": True,
            "oracle_labels_are_state_consistent": True,
            "exact_counterfactual_selection": True,
        },
        "oracle_training_data": oracle_diagnostics,
        "reference": reference,
        "arms": arms,
        "selected_arm_index": selected_arm_index,
        "selected": selected_arm,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "recommendation": (
            "replicate_deployable_gated_residual_before_fresh_test"
            if selected_arm["gate"]["passed"]
            else "do_not_promote_deployable_gated_residual"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run V14.1 deployable-reference gated distillation."
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
        default=Path("artifacts/gimbal_deployable_residual_v14_1.json"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--selection-interval", type=int, default=10)
    parser.add_argument("--trust-weights", type=float, nargs="+")
    parser.add_argument("--residual-scales", type=float, nargs="+")
    parser.add_argument("--visibility-shields", type=float, nargs="+")
    args = parser.parse_args(argv)
    defaults = DeployableResidualDistillationConfig()
    result = evaluate_deployable_residual_distillation_experiment(
        train_path=args.train_data,
        validation_path=args.validation_data,
        base_checkpoint=args.base_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        config=replace(
            defaults,
            epochs=args.epochs,
            selection_epoch_interval=args.selection_interval,
            ordinary_trust_region_weights=(
                tuple(args.trust_weights)
                if args.trust_weights is not None
                else defaults.ordinary_trust_region_weights
            ),
            deployment_residual_scales=(
                tuple(args.residual_scales)
                if args.residual_scales is not None
                else defaults.deployment_residual_scales
            ),
            visibility_shield_strengths=(
                tuple(args.visibility_shields)
                if args.visibility_shields is not None
                else defaults.visibility_shield_strengths
            ),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"passed={result['selected']['gate']['passed']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
