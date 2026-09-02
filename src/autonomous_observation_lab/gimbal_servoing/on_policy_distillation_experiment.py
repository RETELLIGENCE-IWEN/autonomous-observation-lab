"""V12 on-policy aggregation for constrained sequence-oracle distillation."""

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
from .gru import GRUAdaptivePositionLossContext
from .gru_training import set_gru_seed
from .multi_command_experiment import _context_window
from .multi_command_policy import counterfactual_capture_source_indices
from .on_policy_distillation import (
    CounterfactualPositionPolicyRollout,
    rollout_counterfactual_position_policy,
)
from .sequence_distillation import (
    CausalHardwareConditionedPositionPolicy,
    SequenceDistillationPolicyConfig,
)
from .sequence_distillation_experiment import (
    _episode_indices,
    _generate_targets,
    _metrics,
    _relative,
    _select_context,
)
from .sequence_oracle import (
    PrivilegedSequenceOracleConfig,
    optimize_privileged_command_sequence,
)


ON_POLICY_DISTILLATION_SCHEMA_VERSION = (
    "gimbal_on_policy_sequence_distillation_v12_development_v1"
)
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class OnPolicyDistillationExperimentConfig:
    behavior_name: str = "privileged_oracle_position"
    sequence_steps: int = 16
    training_oracle_episode_count: int = 192
    validation_oracle_episode_count: int = 48
    oracle_batch_size: int = 8
    rollout_batch_size: int = 16
    initial_epochs: int = 60
    aggregation_rounds: int = 2
    aggregation_epochs: int = 40
    selection_epoch_interval: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    optimization_seed: int = 29
    disagreement_weight: float = 4.0
    retention_weight: float = 0.25
    critical_weight: float = 1.0
    minimum_tracking_improvement_fraction: float = 0.005
    maximum_saturation_regression_fraction: float = 0.05
    device: str = "cpu"
    policy: SequenceDistillationPolicyConfig = SequenceDistillationPolicyConfig(
        hidden_dim=64,
        embedding_dim=64,
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
            "initial_epochs",
            "aggregation_epochs",
            "selection_epoch_interval",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"on-policy distillation {name} must be positive")
        if self.aggregation_rounds < 0:
            raise ValueError("aggregation rounds must be non-negative")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "disagreement_weight",
            "retention_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"on-policy distillation {name} must be positive")
        for name in (
            "weight_decay",
            "critical_weight",
            "minimum_tracking_improvement_fraction",
            "maximum_saturation_regression_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"on-policy distillation {name} must be non-negative"
                )
        if self.sequence_steps != self.oracle.focus_steps:
            raise ValueError(
                "on-policy sequence steps must match oracle focus steps"
            )


@dataclass
class _TrainingRecords:
    features: torch.Tensor
    context: GRUAdaptivePositionLossContext
    teacher: torch.Tensor
    behavior_command: torch.Tensor
    critical: torch.Tensor
    sequence_mask: torch.Tensor


def _concatenate_context(
    contexts: Sequence[GRUAdaptivePositionLossContext],
) -> GRUAdaptivePositionLossContext:
    return GRUAdaptivePositionLossContext(
        **{
            field.name: torch.cat(
                [getattr(context, field.name) for context in contexts],
                dim=0,
            )
            for field in fields(contexts[0])
        }
    )


def _concatenate_records(records: Sequence[_TrainingRecords]) -> _TrainingRecords:
    return _TrainingRecords(
        features=torch.cat([record.features for record in records], dim=0),
        context=_concatenate_context([record.context for record in records]),
        teacher=torch.cat([record.teacher for record in records], dim=0),
        behavior_command=torch.cat(
            [record.behavior_command for record in records], dim=0
        ),
        critical=torch.cat([record.critical for record in records], dim=0),
        sequence_mask=torch.cat(
            [record.sequence_mask for record in records], dim=0
        ),
    )


def _counterfactual_rollout(
    policy: CausalHardwareConditionedPositionPolicy,
    dataset,
    supervision,
    indices: np.ndarray,
    config: OnPolicyDistillationExperimentConfig,
    device: torch.device,
) -> CounterfactualPositionPolicyRollout:
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
                rollout_counterfactual_position_policy(
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
    return CounterfactualPositionPolicyRollout(
        **{
            field.name: torch.cat(
                [getattr(result, field.name) for result in results],
                dim=0,
            )
            for field in fields(results[0])
        }
    )


def _rollout_metrics(
    rollout: CounterfactualPositionPolicyRollout,
    dataset,
    indices: np.ndarray,
    criticality,
    config: OnPolicyDistillationExperimentConfig,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
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
    result: dict[str, dict[str, float | int]] = {}
    for name, selected in (("global", mask), ("critical", mask & critical)):
        count = int(selected.sum())
        denominator = max(1, count)
        result[name] = {
            "sample_count": count,
            "tracking_rmse_normalized": math.sqrt(
                float(
                    rollout.tracking_error_normalized[selected].square().sum()
                )
                / denominator
            ),
            "visibility_rmse_normalized": math.sqrt(
                float(
                    rollout.visibility_violation_normalized[selected]
                    .square()
                    .sum()
                )
                / denominator
            ),
            "smoothness_rmse_normalized": math.sqrt(
                float(difference[selected].square().sum()) / denominator
            ),
            "saturation_rmse_normalized": math.sqrt(
                float(rollout.saturation_fraction[selected].sum()) / denominator
            ),
        }
    return result


def _selection_score(metrics: dict[str, dict[str, float | int]]) -> float:
    """Rank checkpoints by exact closed-loop tracking and constraint costs."""

    return (
        float(metrics["global"]["tracking_rmse_normalized"])
        + float(metrics["critical"]["tracking_rmse_normalized"])
        + 2.0
        * (
            float(metrics["global"]["visibility_rmse_normalized"])
            + float(metrics["critical"]["visibility_rmse_normalized"])
        )
        + 0.1
        * (
            float(metrics["global"]["smoothness_rmse_normalized"])
            + float(metrics["critical"]["smoothness_rmse_normalized"])
        )
        + 0.01 * float(metrics["global"]["saturation_rmse_normalized"])
    )


def _train_stage(
    policy: CausalHardwareConditionedPositionPolicy,
    records: _TrainingRecords,
    *,
    epochs: int,
    stage: str,
    config: OnPolicyDistillationExperimentConfig,
    generator: torch.Generator,
) -> tuple[list[dict[str, float | int | str]], list[tuple[int, dict[str, Any]]]]:
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    disagreement = torch.abs(
        records.teacher - records.behavior_command
    ) / config.oracle.maximum_command_residual
    weights = (
        config.retention_weight
        + config.disagreement_weight * disagreement
        + config.critical_weight * records.critical
    ) * records.sequence_mask
    history: list[dict[str, float | int | str]] = []
    candidate_states: list[tuple[int, dict[str, Any]]] = []
    for epoch in range(1, epochs + 1):
        policy.train()
        order = torch.randperm(len(records.features), generator=generator)
        total = 0.0
        for offset in range(0, len(order), config.batch_size):
            selected = order[offset : offset + config.batch_size].to(
                records.features.device
            )
            prediction = policy(
                records.features[selected],
                _select_context(records.context, selected),
            )
            selected_weights = weights[selected]
            loss = (
                (prediction - records.teacher[selected]).square()
                * selected_weights
            ).sum() / selected_weights.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            total += float(loss.detach()) * len(selected)
        history.append(
            {
                "stage": stage,
                "epoch": epoch,
                "weighted_command_mse": total / len(order),
            }
        )
        if epoch % config.selection_epoch_interval == 0 or epoch == epochs:
            candidate_states.append((epoch, copy.deepcopy(policy.state_dict())))
    return history, candidate_states


def _select_stage_state(
    policy: CausalHardwareConditionedPositionPolicy,
    states: Sequence[tuple[int, dict[str, Any]]],
    validation,
    validation_supervision,
    validation_indices: np.ndarray,
    validation_criticality,
    config: OnPolicyDistillationExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluations = []
    best = None
    for epoch, state in states:
        policy.load_state_dict(state)
        rollout = _counterfactual_rollout(
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
        record = {"epoch": epoch, "selection_score": score, "metrics": metrics}
        evaluations.append(record)
        if best is None or score < best[0]:
            best = (score, state, record)
    assert best is not None
    policy.load_state_dict(best[1])
    return best[2], evaluations


def _logged_training_records(
    dataset,
    supervision,
    criticality,
    indices: np.ndarray,
    teacher: np.ndarray,
    config: OnPolicyDistillationExperimentConfig,
    device: torch.device,
) -> _TrainingRecords:
    profile = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    return _TrainingRecords(
        features=torch.from_numpy(
            dataset.features[
                indices,
                profile,
                : config.sequence_steps,
            ]
        ).float().to(device),
        context=_context_window(
            supervision,
            indices,
            0,
            config.sequence_steps,
            device,
        ),
        teacher=torch.from_numpy(teacher).float().to(device),
        behavior_command=torch.from_numpy(
            dataset.oracle_actions[
                indices,
                : config.sequence_steps,
                1,
            ]
        ).float().to(device),
        critical=torch.from_numpy(
            criticality.critical_mask[
                indices,
                1 : config.sequence_steps + 1,
                0,
            ]
        ).float().to(device),
        sequence_mask=torch.from_numpy(
            dataset.sequence_mask[indices, : config.sequence_steps]
        ).float().to(device),
    )


def _aggregate_student_records(
    policy: CausalHardwareConditionedPositionPolicy,
    dataset,
    supervision,
    criticality,
    indices: np.ndarray,
    config: OnPolicyDistillationExperimentConfig,
    device: torch.device,
) -> tuple[_TrainingRecords, dict[str, float]]:
    target_commands = []
    behavior_commands = []
    synthetic_features = []
    blends = []
    policy.eval()
    for offset in range(0, len(indices), config.oracle_batch_size):
        selected = indices[offset : offset + config.oracle_batch_size]
        rollout = _counterfactual_rollout(
            policy,
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
        target = torch.from_numpy(
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
        previous = rollout.synthetic_features[
            :, 0, _FEATURE_INDEX["previous_action_normalized"]
        ]
        oracle = optimize_privileged_command_sequence(
            rollout.command_normalized,
            target,
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
        synthetic_features.append(rollout.synthetic_features.detach())
        behavior_commands.append(rollout.command_normalized.detach())
        target_commands.append(oracle.selected_command_normalized.detach())
        blends.append(oracle.selected_blend_fraction.detach())
    teacher = torch.cat(target_commands, dim=0)
    behavior = torch.cat(behavior_commands, dim=0)
    blend = torch.cat(blends, dim=0)
    context = _context_window(
        supervision,
        indices,
        0,
        config.sequence_steps,
        device,
    )
    records = _TrainingRecords(
        features=torch.cat(synthetic_features, dim=0),
        context=context,
        teacher=teacher,
        behavior_command=behavior,
        critical=torch.from_numpy(
            criticality.critical_mask[
                indices,
                1 : config.sequence_steps + 1,
                0,
            ]
        ).float().to(device),
        sequence_mask=torch.from_numpy(
            dataset.sequence_mask[indices, : config.sequence_steps]
        ).float().to(device),
    )
    disagreement = torch.abs(teacher - behavior)
    return records, {
        "episode_count": len(indices),
        "nonzero_blend_fraction": float((blend > 0.0).float().mean()),
        "nonzero_command_label_fraction": float(
            (disagreement > 1e-7).float().mean()
        ),
        "command_label_mae_normalized": float(disagreement.mean()),
        "command_label_max_normalized": float(disagreement.max()),
    }


def _gate(
    candidate: dict[str, dict[str, float | int]],
    reference: dict[str, dict[str, float | int]],
    config: OnPolicyDistillationExperimentConfig,
) -> dict[str, Any]:
    changes = {
        f"{scope}_{metric}": _relative(candidate[scope], reference[scope], metric)
        for scope in ("global", "critical")
        for metric in reference[scope]
        if metric != "sample_count"
    }
    checks = {
        "global_tracking": changes["global_tracking_rmse_normalized"]
        <= -config.minimum_tracking_improvement_fraction,
        "critical_tracking": changes["critical_tracking_rmse_normalized"]
        <= -config.minimum_tracking_improvement_fraction,
        "global_visibility": changes["global_visibility_rmse_normalized"] <= 0.0,
        "critical_visibility": changes["critical_visibility_rmse_normalized"]
        <= 0.0,
        "global_smoothness": changes["global_smoothness_rmse_normalized"] <= 0.0,
        "critical_smoothness": changes["critical_smoothness_rmse_normalized"]
        <= 0.0,
        "global_saturation": changes["global_saturation_rmse_normalized"]
        <= config.maximum_saturation_regression_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def evaluate_on_policy_distillation_experiment(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    config: OnPolicyDistillationExperimentConfig | None = None,
) -> dict[str, Any]:
    """Train and validate iterative student-state oracle aggregation."""

    config = config or OnPolicyDistillationExperimentConfig()
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
    validation_teacher, validation_blends = _generate_targets(
        validation,
        validation_supervision,
        validation_indices,
        config,
        device,
    )
    logged_records = _logged_training_records(
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
    oracle_ceiling = _metrics(
        validation_teacher,
        validation,
        validation_supervision,
        validation_indices,
        validation_criticality,
        config,
        device,
    )

    set_gru_seed(config.optimization_seed)
    generator = torch.Generator().manual_seed(config.optimization_seed)
    policy = CausalHardwareConditionedPositionPolicy(config.policy).to(device)
    all_records = [logged_records]
    training_history = []
    stage_results = []
    aggregation_results = []

    history, states = _train_stage(
        policy,
        logged_records,
        epochs=config.initial_epochs,
        stage="teacher_forced",
        config=config,
        generator=generator,
    )
    training_history.extend(history)
    selected, evaluations = _select_stage_state(
        policy,
        states,
        validation,
        validation_supervision,
        validation_indices,
        validation_criticality,
        config,
        device,
    )
    stage_results.append(
        {
            "stage": "teacher_forced",
            "training_record_count": len(logged_records.features),
            "selected_epoch": selected["epoch"],
            "selection_score": selected["selection_score"],
            "metrics": selected["metrics"],
            "checkpoint_candidates": evaluations,
        }
    )

    for round_index in range(1, config.aggregation_rounds + 1):
        records, aggregation = _aggregate_student_records(
            policy,
            train,
            train_supervision,
            train_criticality,
            train_indices,
            config,
            device,
        )
        aggregation["round"] = round_index
        aggregation_results.append(aggregation)
        all_records.append(records)
        combined = _concatenate_records(all_records)
        prior_state = copy.deepcopy(policy.state_dict())
        history, states = _train_stage(
            policy,
            combined,
            epochs=config.aggregation_epochs,
            stage=f"aggregation_{round_index}",
            config=config,
            generator=generator,
        )
        training_history.extend(history)
        states.insert(0, (0, prior_state))
        selected, evaluations = _select_stage_state(
            policy,
            states,
            validation,
            validation_supervision,
            validation_indices,
            validation_criticality,
            config,
            device,
        )
        stage_results.append(
            {
                "stage": f"aggregation_{round_index}",
                "training_record_count": len(combined.features),
                "selected_epoch": selected["epoch"],
                "selection_score": selected["selection_score"],
                "metrics": selected["metrics"],
                "checkpoint_candidates": evaluations,
            }
        )

    final_rollout = _counterfactual_rollout(
        policy,
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
    teacher_forced_gate = _gate(stage_results[0]["metrics"], reference, config)
    return {
        "experiment": ON_POLICY_DISTILLATION_SCHEMA_VERSION,
        "config": asdict(config),
        "datasets": {
            "train": str(train_path),
            "validation": str(validation_path),
            "fresh_test": {"opened": False},
        },
        "architecture": {
            "actor_input_is_deployable_o2_only": True,
            "hardware_conditioned": True,
            "recurrent_policy_state_persists": True,
            "student_previous_action_is_counterfactual": True,
            "geometry_dependent_observations_are_counterfactual": True,
            "detector_timing_is_logged_exogenous": True,
            "exact_persistent_latency_plant_for_validation": True,
            "privileged_target_used_only_by_simulator_and_oracle": True,
        },
        "oracle": {
            "train_nonzero_blend_fraction": float(
                np.mean(np.asarray(train_blends) > 0.0)
            ),
            "validation_nonzero_blend_fraction": float(
                np.mean(np.asarray(validation_blends) > 0.0)
            ),
            "validation_ceiling": oracle_ceiling,
        },
        "aggregation": aggregation_results,
        "training_history": training_history,
        "stage_results": stage_results,
        "reference": reference,
        "teacher_forced_gate": teacher_forced_gate,
        "candidate": candidate,
        "gate": gate,
        "recommendation": (
            "replicate_on_policy_student_before_fresh_test"
            if gate["passed"]
            else "do_not_promote_on_policy_student"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run V12 on-policy sequence-oracle distillation."
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
        default=Path("artifacts/gimbal_on_policy_distillation_v12.json"),
    )
    parser.add_argument("--initial-epochs", type=int, default=60)
    parser.add_argument("--aggregation-rounds", type=int, default=2)
    parser.add_argument("--aggregation-epochs", type=int, default=40)
    args = parser.parse_args(argv)
    result = evaluate_on_policy_distillation_experiment(
        train_path=args.train_data,
        validation_path=args.validation_data,
        config=replace(
            OnPolicyDistillationExperimentConfig(),
            initial_epochs=args.initial_epochs,
            aggregation_rounds=args.aggregation_rounds,
            aggregation_epochs=args.aggregation_epochs,
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
