"""V13 state-consistent failure-gated sequence-oracle correction."""

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
from .failure_gated_policy import (
    FailureGatedPositionCorrectionPolicy,
    FailureGatedPositionPolicyConfig,
)
from .failure_gated_rollout import (
    FailureGatedCounterfactualRollout,
    rollout_failure_gated_position_policy,
)
from .gru import GRUAdaptivePositionLossContext
from .gru_training import set_gru_seed
from .multi_command_experiment import _context_window
from .multi_command_policy import counterfactual_capture_source_indices
from .on_policy_distillation import (
    CounterfactualPositionPolicyRollout,
    rollout_counterfactual_position_commands,
)
from .on_policy_distillation_experiment import (
    _TrainingRecords,
    _concatenate_context,
    _counterfactual_rollout,
    _gate,
    _logged_training_records,
    _rollout_metrics,
    _selection_score,
    _select_stage_state,
    _train_stage,
)
from .sequence_distillation import (
    CausalHardwareConditionedPositionPolicy,
    SequenceDistillationPolicyConfig,
    normalized_hardware_features,
)
from .sequence_distillation_experiment import (
    _episode_indices,
    _generate_targets,
    _metrics,
    _select_context,
)
from .sequence_oracle import (
    PrivilegedSequenceOracleConfig,
    optimize_privileged_command_sequence,
)


FAILURE_GATED_EXPERIMENT_SCHEMA_VERSION = (
    "gimbal_state_consistent_failure_gated_v13_development_v1"
)
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class FailureGatedExperimentConfig:
    behavior_name: str = "privileged_oracle_position"
    sequence_steps: int = 16
    training_oracle_episode_count: int = 192
    validation_oracle_episode_count: int = 48
    oracle_batch_size: int = 8
    rollout_batch_size: int = 16
    base_epochs: int = 30
    correction_epochs: int = 40
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
    ordinary_trust_region_weights: tuple[float, ...] = (1.0, 5.0, 20.0)
    correction_label_threshold: float = 0.005
    minimum_tracking_improvement_fraction: float = 0.005
    maximum_saturation_regression_fraction: float = 0.05
    device: str = "cpu"
    policy: SequenceDistillationPolicyConfig = SequenceDistillationPolicyConfig(
        hidden_dim=64,
        embedding_dim=64,
    )
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
        optimization_iterations=12,
    )
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        for name in (
            "sequence_steps",
            "training_oracle_episode_count",
            "validation_oracle_episode_count",
            "oracle_batch_size",
            "rollout_batch_size",
            "base_epochs",
            "correction_epochs",
            "selection_epoch_interval",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"failure-gated {name} must be positive")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "disagreement_weight",
            "retention_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"failure-gated {name} must be positive")
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
                raise ValueError(f"failure-gated {name} must be non-negative")
        if not self.ordinary_trust_region_weights or any(
            not math.isfinite(value) or value < 0.0
            for value in self.ordinary_trust_region_weights
        ):
            raise ValueError("ordinary trust-region weights must be non-negative")
        if len(set(self.ordinary_trust_region_weights)) != len(
            self.ordinary_trust_region_weights
        ):
            raise ValueError("ordinary trust-region weights must be unique")
        if self.sequence_steps != self.oracle.focus_steps:
            raise ValueError("failure-gated steps must match oracle focus steps")


@dataclass
class _CorrectionRecords:
    features: torch.Tensor
    context: GRUAdaptivePositionLossContext
    teacher_command: torch.Tensor
    behavior_command: torch.Tensor
    gate_target: torch.Tensor
    critical: torch.Tensor
    trust_mask: torch.Tensor
    sequence_mask: torch.Tensor


@dataclass
class _StateConsistentOracleQuery:
    records: _CorrectionRecords
    base_rollout: CounterfactualPositionPolicyRollout
    oracle_rollout: CounterfactualPositionPolicyRollout
    diagnostics: dict[str, float | int]


def _concatenate_position_rollouts(
    rollouts: Sequence[CounterfactualPositionPolicyRollout],
) -> CounterfactualPositionPolicyRollout:
    return CounterfactualPositionPolicyRollout(
        **{
            field.name: torch.cat(
                [getattr(rollout, field.name) for rollout in rollouts],
                dim=0,
            )
            for field in fields(rollouts[0])
        }
    )


def _recorded_base_commands(
    base_policy: CausalHardwareConditionedPositionPolicy,
    features: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
) -> torch.Tensor:
    hardware = normalized_hardware_features(context)
    hidden = None
    commands = []
    for time_index in range(features.shape[1]):
        previous = features[
            :, time_index, _FEATURE_INDEX["previous_action_normalized"]
        ]
        command, hidden = base_policy.forward_step(
            features[:, time_index],
            hardware[:, time_index],
            hidden,
            previous_command_normalized=previous,
            minimum_angle_rad=context.servo_min_angle_rad[:, time_index],
            maximum_angle_rad=context.servo_max_angle_rad[:, time_index],
        )
        commands.append(command)
    return torch.stack(commands, dim=1)


def _state_consistent_oracle_query(
    base_policy: CausalHardwareConditionedPositionPolicy,
    dataset,
    supervision,
    criticality,
    indices: np.ndarray,
    config: FailureGatedExperimentConfig,
    device: torch.device,
) -> _StateConsistentOracleQuery:
    profile = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    base_rollouts = []
    oracle_rollouts = []
    correction_base_commands = []
    selected_blends = []
    for offset in range(0, len(indices), config.oracle_batch_size):
        selected = indices[offset : offset + config.oracle_batch_size]
        base_rollout = _counterfactual_rollout(
            base_policy,
            dataset,
            supervision,
            selected,
            config,
            device,
        )
        context = _context_window(
            supervision,
            selected,
            0,
            config.sequence_steps,
            device,
        )
        target_after = torch.from_numpy(
            dataset.targets[
                selected,
                1 : config.sequence_steps + 1,
                0,
                0,
            ]
        ).float().to(device)
        mask = torch.from_numpy(
            dataset.sequence_mask[selected, : config.sequence_steps]
        ).bool().to(device)
        previous = base_rollout.synthetic_features[
            :, 0, _FEATURE_INDEX["previous_action_normalized"]
        ]
        oracle = optimize_privileged_command_sequence(
            base_rollout.command_normalized,
            target_after,
            context,
            mask,
            torch.from_numpy(dataset.time_s[selected, 0]).float().to(device),
            previous,
            config=replace(
                config.oracle,
                focus_start_index=0,
                focus_steps=config.sequence_steps,
            ),
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
        capture_source = counterfactual_capture_source_indices(time_s, logged)
        oracle_rollout = rollout_counterfactual_position_commands(
            oracle.selected_command_normalized,
            logged,
            target,
            time_s,
            capture_source,
            context,
            mask,
            visibility_margin_fraction=config.oracle.visibility_margin_fraction,
        )
        with torch.no_grad():
            correction_base_commands.append(
                _recorded_base_commands(
                    base_policy,
                    oracle_rollout.synthetic_features,
                    context,
                )
            )
        base_rollouts.append(base_rollout)
        oracle_rollouts.append(oracle_rollout)
        selected_blends.append(oracle.selected_blend_fraction.detach())

    base_rollout = _concatenate_position_rollouts(base_rollouts)
    oracle_rollout = _concatenate_position_rollouts(oracle_rollouts)
    context = _context_window(
        supervision,
        indices,
        0,
        config.sequence_steps,
        device,
    )
    correction_base = torch.cat(correction_base_commands, dim=0)
    correction_delta = torch.abs(
        oracle_rollout.command_normalized - correction_base
    )
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
    records = _CorrectionRecords(
        features=torch.cat(
            (
                base_rollout.synthetic_features.detach(),
                oracle_rollout.synthetic_features.detach(),
            ),
            dim=0,
        ),
        context=_concatenate_context((context, context)),
        teacher_command=torch.cat(
            (
                base_rollout.command_normalized.detach(),
                oracle_rollout.command_normalized.detach(),
            ),
            dim=0,
        ),
        behavior_command=torch.cat(
            (
                base_rollout.command_normalized.detach(),
                correction_base.detach(),
            ),
            dim=0,
        ),
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
    blend = torch.cat(selected_blends, dim=0)
    selected_delta = torch.abs(
        oracle_rollout.command_normalized - base_rollout.command_normalized
    )
    return _StateConsistentOracleQuery(
        records=records,
        base_rollout=base_rollout,
        oracle_rollout=oracle_rollout,
        diagnostics={
            "episode_count": len(indices),
            "training_record_count": len(records.features),
            "nonzero_blend_fraction": float((blend > 0.0).float().mean()),
            "correction_gate_positive_fraction": float(
                (gate_target * mask).sum() / mask.sum().clamp_min(1.0)
            ),
            "selected_command_delta_mae_normalized": float(
                (selected_delta * mask).sum() / mask.sum().clamp_min(1.0)
            ),
            "selected_command_delta_max_normalized": float(
                selected_delta.max()
            ),
        },
    )


def _train_correction_stage(
    policy: FailureGatedPositionCorrectionPolicy,
    records: _CorrectionRecords,
    *,
    trust_region_weight: float,
    config: FailureGatedExperimentConfig,
    generator: torch.Generator,
) -> tuple[list[dict[str, float | int]], list[tuple[int, dict[str, Any]]]]:
    parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    disagreement = torch.abs(
        records.teacher_command - records.behavior_command
    ) / config.oracle.maximum_command_residual
    action_weight = (
        config.retention_weight
        + config.disagreement_weight * disagreement
        + config.critical_weight * records.critical
    ) * records.sequence_mask
    history = []
    states = [(0, copy.deepcopy(policy.state_dict()))]
    for epoch in range(1, config.correction_epochs + 1):
        order = torch.randperm(len(records.features), generator=generator)
        totals = {
            "loss": 0.0,
            "action": 0.0,
            "trust": 0.0,
            "gate": 0.0,
            "residual": 0.0,
        }
        for offset in range(0, len(order), config.batch_size):
            selected = order[offset : offset + config.batch_size].to(
                records.features.device
            )
            sequence = policy(
                records.features[selected],
                _select_context(records.context, selected),
                use_recorded_previous_command=True,
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
                parameters,
                config.gradient_clip_norm,
            )
            optimizer.step()
            batch_count = len(selected)
            totals["loss"] += float(loss.detach()) * batch_count
            totals["action"] += float(action_loss.detach()) * batch_count
            totals["trust"] += float(trust_loss.detach()) * batch_count
            totals["gate"] += float(gate_loss.detach()) * batch_count
            totals["residual"] += float(residual_loss.detach()) * batch_count
        history.append(
            {
                "epoch": epoch,
                "loss": totals["loss"] / len(order),
                "action_loss": totals["action"] / len(order),
                "ordinary_trust_loss": totals["trust"] / len(order),
                "gate_loss": totals["gate"] / len(order),
                "residual_loss": totals["residual"] / len(order),
            }
        )
        if epoch % config.selection_epoch_interval == 0 or (
            epoch == config.correction_epochs
        ):
            states.append((epoch, copy.deepcopy(policy.state_dict())))
    return history, states


def _failure_gated_rollout(
    policy: FailureGatedPositionCorrectionPolicy,
    dataset,
    supervision,
    indices: np.ndarray,
    config: FailureGatedExperimentConfig,
    device: torch.device,
) -> FailureGatedCounterfactualRollout:
    profile = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    results = []
    policy.eval()
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
            results.append(
                rollout_failure_gated_position_policy(
                    policy,
                    features,
                    target,
                    time_s,
                    counterfactual_capture_source_indices(time_s, features),
                    context,
                    mask,
                    visibility_margin_fraction=(
                        config.oracle.visibility_margin_fraction
                    ),
                )
            )
    return FailureGatedCounterfactualRollout(
        **{
            field.name: torch.cat(
                [getattr(result, field.name) for result in results],
                dim=0,
            )
            for field in fields(results[0])
        }
    )


def _select_correction_state(
    policy: FailureGatedPositionCorrectionPolicy,
    states: Sequence[tuple[int, dict[str, Any]]],
    validation,
    validation_supervision,
    validation_indices: np.ndarray,
    validation_criticality,
    config: FailureGatedExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluations = []
    best = None
    for epoch, state in states:
        policy.load_state_dict(state)
        rollout = _failure_gated_rollout(
            policy,
            validation,
            validation_supervision,
            validation_indices,
            config,
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
        score = _selection_score(metrics)
        record = {
            "epoch": epoch,
            "selection_score": score,
            "metrics": metrics,
            "mean_gate_probability": float(rollout.gate_probability.mean()),
            "mean_absolute_residual_normalized": float(
                torch.abs(rollout.residual_normalized).mean()
            ),
            "maximum_absolute_residual_normalized": float(
                torch.abs(rollout.residual_normalized).max()
            ),
        }
        evaluations.append(record)
        if best is None or score < best[0]:
            best = (score, state, record)
    assert best is not None
    policy.load_state_dict(best[1])
    return best[2], evaluations


def evaluate_failure_gated_experiment(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    config: FailureGatedExperimentConfig | None = None,
) -> dict[str, Any]:
    """Train V13 correction arms and preserve the sealed fresh-test boundary."""

    config = config or FailureGatedExperimentConfig()
    device = torch.device(config.device)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    train_indices = _episode_indices(
        train,
        config,
        config.training_oracle_episode_count,
    )
    validation_indices = _episode_indices(
        validation,
        config,
        config.validation_oracle_episode_count,
    )
    adapter = selected_adaptive_position_v21_config()
    train_supervision = compute_adaptive_position_supervision(train, adapter=adapter)
    validation_supervision = compute_adaptive_position_supervision(
        validation,
        adapter=adapter,
    )
    train_criticality = compute_control_criticality(train, config=config.criticality)
    validation_criticality = compute_control_criticality(
        validation,
        config=config.criticality,
    )
    train_teacher, train_blends = _generate_targets(
        train,
        train_supervision,
        train_indices,
        config,
        device,
    )
    logged_records: _TrainingRecords = _logged_training_records(
        train,
        train_supervision,
        train_criticality,
        train_indices,
        train_teacher,
        config,
        device,
    )
    reference_commands = validation.oracle_actions[
        validation_indices,
        : config.sequence_steps,
        1,
    ]
    reference = _metrics(
        reference_commands,
        validation,
        validation_supervision,
        validation_indices,
        validation_criticality,
        config,
        device,
    )

    set_gru_seed(config.optimization_seed)
    generator = torch.Generator().manual_seed(config.optimization_seed)
    base_policy = CausalHardwareConditionedPositionPolicy(config.policy).to(device)
    base_history, base_states = _train_stage(
        base_policy,
        logged_records,
        epochs=config.base_epochs,
        stage="base_actor",
        config=config,
        generator=generator,
    )
    selected_base, base_evaluations = _select_stage_state(
        base_policy,
        base_states,
        validation,
        validation_supervision,
        validation_indices,
        validation_criticality,
        config,
        device,
    )

    train_query = _state_consistent_oracle_query(
        base_policy,
        train,
        train_supervision,
        train_criticality,
        train_indices,
        config,
        device,
    )
    validation_query = _state_consistent_oracle_query(
        base_policy,
        validation,
        validation_supervision,
        validation_criticality,
        validation_indices,
        config,
        device,
    )
    base_metrics = _rollout_metrics(
        validation_query.base_rollout,
        validation,
        validation_indices,
        validation_criticality,
        config,
        device,
    )
    state_consistent_oracle_metrics = _rollout_metrics(
        validation_query.oracle_rollout,
        validation,
        validation_indices,
        validation_criticality,
        config,
        device,
    )

    arms = []
    selected_policies = []
    for trust_region_weight in config.ordinary_trust_region_weights:
        set_gru_seed(config.optimization_seed)
        policy = FailureGatedPositionCorrectionPolicy(
            copy.deepcopy(base_policy),
            config.correction_policy,
        ).to(device)
        arm_generator = torch.Generator().manual_seed(config.optimization_seed)
        history, states = _train_correction_stage(
            policy,
            train_query.records,
            trust_region_weight=trust_region_weight,
            config=config,
            generator=arm_generator,
        )
        selected, evaluations = _select_correction_state(
            policy,
            states,
            validation,
            validation_supervision,
            validation_indices,
            validation_criticality,
            config,
            device,
        )
        arms.append(
            {
                "ordinary_trust_region_weight": trust_region_weight,
                "selected_epoch": selected["epoch"],
                "selection_score": selected["selection_score"],
                "metrics": selected["metrics"],
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
        selected_policies.append(copy.deepcopy(policy.state_dict()))
    selected_arm_index = min(
        range(len(arms)),
        key=lambda index: arms[index]["selection_score"],
    )
    final_policy = FailureGatedPositionCorrectionPolicy(
        copy.deepcopy(base_policy),
        config.correction_policy,
    ).to(device)
    final_policy.load_state_dict(selected_policies[selected_arm_index])
    final_rollout = _failure_gated_rollout(
        final_policy,
        validation,
        validation_supervision,
        validation_indices,
        config,
        device,
    )
    candidate = _rollout_metrics(
        final_rollout,
        validation,
        validation_indices,
        validation_criticality,
        config,
        device,
    )
    gate = _gate(candidate, reference, config)
    return {
        "experiment": FAILURE_GATED_EXPERIMENT_SCHEMA_VERSION,
        "config": asdict(config),
        "datasets": {
            "train": str(train_path),
            "validation": str(validation_path),
            "fresh_test": {"opened": False},
        },
        "architecture": {
            "base_actor_frozen_during_correction": True,
            "actor_inputs_are_deployable_o2_only": True,
            "gate_evidence_is_deployable_and_hardware_relative": True,
            "privileged_criticality_is_training_only": True,
            "correction_is_bounded": True,
            "ordinary_base_states_are_explicit_retention_records": True,
            "oracle_correction_states_replayed_from_selected_commands": True,
            "sequence_labels_and_observations_are_state_consistent": True,
            "exact_counterfactual_validation": True,
        },
        "logged_teacher": {
            "nonzero_blend_fraction": float(
                np.mean(np.asarray(train_blends) > 0.0)
            )
        },
        "base_actor": {
            "selected_epoch": selected_base["epoch"],
            "selection_score": selected_base["selection_score"],
            "metrics": base_metrics,
            "training_history": base_history,
            "checkpoint_candidates": base_evaluations,
        },
        "state_consistent_oracle": {
            "train": train_query.diagnostics,
            "validation": validation_query.diagnostics,
            "validation_ceiling": state_consistent_oracle_metrics,
        },
        "arms": arms,
        "selected_arm_index": selected_arm_index,
        "selected_ordinary_trust_region_weight": arms[selected_arm_index][
            "ordinary_trust_region_weight"
        ],
        "reference": reference,
        "candidate": candidate,
        "gate": gate,
        "recommendation": (
            "replicate_failure_gated_policy_before_fresh_test"
            if gate["passed"]
            else "do_not_promote_failure_gated_policy"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run V13 state-consistent failure-gated correction."
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
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_failure_gated_v13.json"),
    )
    parser.add_argument("--base-epochs", type=int, default=30)
    parser.add_argument("--correction-epochs", type=int, default=40)
    args = parser.parse_args(argv)
    result = evaluate_failure_gated_experiment(
        train_path=args.train_data,
        validation_path=args.validation_data,
        config=replace(
            FailureGatedExperimentConfig(),
            base_epochs=args.base_epochs,
            correction_epochs=args.correction_epochs,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"passed={result['gate']['passed']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
