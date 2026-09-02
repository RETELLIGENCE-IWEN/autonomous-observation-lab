"""V11.1 failure-focused distillation of the constrained sequence oracle."""

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
from .gru import (
    GRUAdaptivePositionLossContext,
    angular_residual_rad,
    differentiable_position_servo_sequence,
)
from .gru_training import set_gru_seed
from .multi_command_experiment import _context_window
from .sequence_distillation import (
    CausalHardwareConditionedPositionPolicy,
    SequenceDistillationPolicyConfig,
)
from .sequence_oracle import (
    PrivilegedSequenceOracleConfig,
    optimize_privileged_command_sequence,
)


SEQUENCE_DISTILLATION_SCHEMA_VERSION = (
    "gimbal_sequence_oracle_distillation_v11_1_development_v1"
)


@dataclass(frozen=True)
class SequenceDistillationExperimentConfig:
    behavior_name: str = "privileged_oracle_position"
    sequence_steps: int = 16
    training_oracle_episode_count: int = 192
    validation_oracle_episode_count: int = 48
    oracle_batch_size: int = 8
    epochs: int = 80
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimization_seed: int = 29
    disagreement_weight: float = 4.0
    retention_weight: float = 0.25
    critical_weight: float = 1.0
    minimum_tracking_improvement_fraction: float = 0.005
    device: str = "cpu"
    policy: SequenceDistillationPolicyConfig = SequenceDistillationPolicyConfig(
        hidden_dim=64,
        embedding_dim=64,
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
            "training_oracle_episode_count",
            "validation_oracle_episode_count",
            "oracle_batch_size",
            "epochs",
            "batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"distillation {name} must be positive")
        for name in (
            "learning_rate",
            "disagreement_weight",
            "retention_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"distillation {name} must be positive")


def _select_context(context, indices):
    return GRUAdaptivePositionLossContext(
        **{f.name: getattr(context, f.name)[indices] for f in fields(context)}
    )


def _episode_indices(dataset, config, episode_count):
    behavior = dataset.manifest.behavior_names.index(config.behavior_name)
    result = np.flatnonzero(dataset.behavior_index == behavior)
    if len(result) < episode_count:
        raise ValueError("distillation dataset has too few oracle episodes")
    return result[:episode_count]


def _generate_targets(dataset, supervision, indices, config, device):
    steps = config.sequence_steps
    target_commands = np.zeros((len(indices), steps), dtype=np.float32)
    selected_blends = []
    for offset in range(0, len(indices), config.oracle_batch_size):
        batch_indices = indices[offset : offset + config.oracle_batch_size]
        context = _context_window(supervision, batch_indices, 0, steps, device)
        base = torch.from_numpy(
            dataset.oracle_actions[batch_indices, :steps, 1]
        ).float().to(device)
        target = torch.from_numpy(
            dataset.targets[batch_indices, 1 : steps + 1, 0, 0]
        ).float().to(device)
        mask = torch.from_numpy(
            dataset.sequence_mask[batch_indices, :steps]
        ).bool().to(device)
        previous = torch.from_numpy(
            dataset.features[
                batch_indices,
                dataset.manifest.observation_profiles.index(
                    ObservationProfile.DISTURBANCE_AWARE.value
                ),
                0,
                FEATURE_NAMES.index("previous_action_normalized"),
            ]
        ).float().to(device)
        result = optimize_privileged_command_sequence(
            base,
            target,
            context,
            mask,
            torch.from_numpy(dataset.time_s[batch_indices, 0]).float().to(device),
            previous,
            config=replace(
                config.oracle,
                focus_start_index=0,
                focus_steps=steps,
            ),
        )
        target_commands[offset : offset + len(batch_indices)] = (
            result.selected_command_normalized.cpu().numpy()
        )
        selected_blends.extend(result.selected_blend_fraction.cpu().tolist())
    return target_commands, selected_blends


def _metrics(commands, dataset, supervision, indices, criticality, config, device):
    steps = config.sequence_steps
    context = _context_window(supervision, indices, 0, steps, device)
    command = torch.as_tensor(commands, device=device)
    mask = torch.from_numpy(dataset.sequence_mask[indices, :steps]).bool().to(device)
    rollout = differentiable_position_servo_sequence(
        command,
        context,
        mask,
        initial_time_s=torch.from_numpy(dataset.time_s[indices, 0]).float().to(device),
    )
    target = torch.from_numpy(dataset.targets[indices, 1 : steps + 1, 0, 0]).float().to(device)
    error = angular_residual_rad(target, rollout.angle_rad) / (
        0.5 * context.selected_axis_fov_rad
    )
    visibility = torch.relu(torch.abs(error) - config.oracle.visibility_margin_fraction)
    profile = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    previous = torch.from_numpy(
        dataset.features[indices, profile, 0, FEATURE_NAMES.index("previous_action_normalized")]
    ).float().to(device)
    difference = torch.diff(torch.cat((previous[:, None], command), dim=1), dim=1)
    critical = torch.from_numpy(
        criticality.critical_mask[indices, 1 : steps + 1, 0]
    ).bool().to(device)
    result = {}
    for name, selected in (("global", mask), ("critical", mask & critical)):
        count = int(selected.sum())
        denominator = max(1, count)
        result[name] = {
            "sample_count": count,
            "tracking_rmse_normalized": math.sqrt(float(error[selected].square().sum()) / denominator),
            "visibility_rmse_normalized": math.sqrt(float(visibility[selected].square().sum()) / denominator),
            "smoothness_rmse_normalized": math.sqrt(float(difference[selected].square().sum()) / denominator),
            "saturation_rmse_normalized": math.sqrt(float(rollout.saturation_fraction[selected].sum()) / denominator),
        }
    return result


def _relative(candidate, reference, metric):
    r = reference[metric]
    return 0.0 if r == 0.0 and candidate[metric] == 0.0 else (
        math.inf if r == 0.0 else candidate[metric] / r - 1.0
    )


def evaluate_sequence_distillation_experiment(
    *, train_path: str | Path, validation_path: str | Path,
    config: SequenceDistillationExperimentConfig | None = None,
) -> dict[str, Any]:
    config = config or SequenceDistillationExperimentConfig()
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
    validation_supervision = compute_adaptive_position_supervision(validation, adapter=adapter)
    train_criticality = compute_control_criticality(train, config=config.criticality)
    validation_criticality = compute_control_criticality(validation, config=config.criticality)
    train_target, train_blends = _generate_targets(
        train, train_supervision, train_indices, config, device
    )
    validation_target, validation_blends = _generate_targets(
        validation, validation_supervision, validation_indices, config, device
    )
    profile = train.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    features = torch.from_numpy(
        train.features[train_indices, profile, : config.sequence_steps]
    ).float().to(device)
    context = _context_window(
        train_supervision, train_indices, 0, config.sequence_steps, device
    )
    base = torch.from_numpy(
        train.oracle_actions[train_indices, : config.sequence_steps, 1]
    ).float().to(device)
    teacher = torch.from_numpy(train_target).float().to(device)
    critical = torch.from_numpy(
        train_criticality.critical_mask[
            train_indices, 1 : config.sequence_steps + 1, 0
        ]
    ).float().to(device)
    weights = config.retention_weight + config.disagreement_weight * (
        torch.abs(teacher - base) / config.oracle.maximum_command_residual
    ) + config.critical_weight * critical
    set_gru_seed(config.optimization_seed)
    model = CausalHardwareConditionedPositionPolicy(config.policy).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.optimization_seed)
    history = []
    states = []
    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(len(train_indices), generator=generator)
        total = 0.0
        for offset in range(0, len(order), config.batch_size):
            selected = order[offset : offset + config.batch_size].to(device)
            prediction = model(features[selected], _select_context(context, selected))
            loss = (
                (prediction - teacher[selected]).square() * weights[selected]
            ).sum() / weights[selected].sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(selected)
        history.append({"epoch": epoch, "weighted_command_mse": total / len(order)})
        states.append(copy.deepcopy(model.state_dict()))
    validation_features = torch.from_numpy(
        validation.features[validation_indices, profile, : config.sequence_steps]
    ).float().to(device)
    validation_context = _context_window(
        validation_supervision, validation_indices, 0, config.sequence_steps, device
    )
    validation_teacher = torch.from_numpy(validation_target).float().to(device)
    epoch_errors = []
    with torch.no_grad():
        for epoch, state in enumerate(states, 1):
            model.load_state_dict(state)
            prediction = model(validation_features, validation_context)
            epoch_errors.append((float((prediction - validation_teacher).square().mean()), epoch))
        selected_epoch = min(epoch_errors)[1]
        model.load_state_dict(states[selected_epoch - 1])
        student = model(validation_features, validation_context).cpu().numpy()
    base_commands = validation.oracle_actions[
        validation_indices, : config.sequence_steps, 1
    ]
    reference = _metrics(
        base_commands, validation, validation_supervision, validation_indices,
        validation_criticality, config, device,
    )
    oracle_metrics = _metrics(
        validation_target, validation, validation_supervision, validation_indices,
        validation_criticality, config, device,
    )
    candidate = _metrics(
        student, validation, validation_supervision, validation_indices,
        validation_criticality, config, device,
    )
    changes = {
        f"{scope}_{metric}": _relative(candidate[scope], reference[scope], metric)
        for scope in ("global", "critical")
        for metric in reference[scope]
        if metric != "sample_count"
    }
    checks = {
        "global_tracking": changes["global_tracking_rmse_normalized"] <= -config.minimum_tracking_improvement_fraction,
        "critical_tracking": changes["critical_tracking_rmse_normalized"] <= -config.minimum_tracking_improvement_fraction,
        "global_visibility": changes["global_visibility_rmse_normalized"] <= 0.0,
        "critical_visibility": changes["critical_visibility_rmse_normalized"] <= 0.0,
        "global_smoothness": changes["global_smoothness_rmse_normalized"] <= 0.0,
        "critical_smoothness": changes["critical_smoothness_rmse_normalized"] <= 0.0,
        "global_saturation": changes["global_saturation_rmse_normalized"] <= 0.05,
    }
    return {
        "experiment": SEQUENCE_DISTILLATION_SCHEMA_VERSION,
        "config": asdict(config),
        "datasets": {"train": str(train_path), "validation": str(validation_path), "fresh_test": {"opened": False}},
        "oracle": {"train_nonzero_blend_fraction": float(np.mean(np.asarray(train_blends) > 0)), "validation_nonzero_blend_fraction": float(np.mean(np.asarray(validation_blends) > 0)), "validation_ceiling": oracle_metrics},
        "training_history": history,
        "selected_epoch": selected_epoch,
        "reference": reference,
        "candidate": candidate,
        "gate": {"passed": all(checks.values()), "checks": checks, "relative_changes": changes},
        "recommendation": "advance_to_on_policy_distillation" if all(checks.values()) else "revise_distillation_before_on_policy_rollout",
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run V11.1 sequence distillation.")
    parser.add_argument("--train-data", type=Path, default=Path("artifacts/gimbal_control_aware_train.npz"))
    parser.add_argument("--validation-data", type=Path, default=Path("artifacts/gimbal_midpoint_adapter_validation.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gimbal_sequence_distillation_v11_1.json"))
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args(argv)
    result = evaluate_sequence_distillation_experiment(
        train_path=args.train_data, validation_path=args.validation_data,
        config=replace(SequenceDistillationExperimentConfig(), epochs=args.epochs),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.output}")
    print(f"passed={result['gate']['passed']}; recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
