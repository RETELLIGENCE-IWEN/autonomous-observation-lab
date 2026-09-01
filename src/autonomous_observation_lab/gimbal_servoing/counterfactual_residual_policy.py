"""V8.9 frozen-state, bounded residual position-policy experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .adaptive_position_supervision import (
    compute_adaptive_position_supervision,
)
from .config import ObservationProfile
from .control_criticality import (
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from .counterfactual_plant_objective import (
    CounterfactualPlantObjectiveConfig,
    _gate,
    _record_model,
    selected_counterfactual_plant_v87_candidate,
)
from .counterfactual_plant_replication import (
    CounterfactualPlantReplicationConfig,
    _per_seed_safety,
)
from .critical_curriculum import (
    CriticalEpisodeCurriculumConfig,
    compute_critical_episode_curriculum,
    critical_episode_curriculum_report,
)
from .dataset import load_gimbal_dataset
from .gru import (
    CausalTargetStateGRUWithPositionResidual,
    GRUPositionResidualConfig,
    load_gru_checkpoint,
    save_gru_position_residual_checkpoint,
    target_state_nll,
)
from .gru_training import (
    GimbalTorchSequenceDataset,
    _adaptive_position_context_from_batch,
    set_gru_seed,
)


COUNTERFACTUAL_RESIDUAL_POLICY_SCHEMA_VERSION = (
    "gimbal_counterfactual_residual_policy_v8_9_development_v1"
)


@dataclass(frozen=True)
class CounterfactualResidualPolicyConfig:
    base_seed: int = 29
    optimization_seed: int = 29
    epochs: int = 8
    batch_size: int = 24
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    rollout_horizon_index: int = 3
    training_integration_period_s: float = 0.010
    minimum_training_episodes: int = 1000
    minimum_validation_episodes: int = 200
    maximum_state_regression_fraction: float = 0.02
    maximum_adapter_regression_fraction: float = 0.02
    minimum_tracking_improvement_fraction: float = 0.005
    maximum_saturation_regression_fraction: float = 0.05
    maximum_per_seed_state_regression_fraction: float = 0.03
    maximum_per_seed_adapter_regression_fraction: float = 0.03
    adaptive_action_weight: float = 0.0
    residual_smoothness_weight: float = 10.0
    direct_tracking_weight: float = 0.0
    direct_visibility_weight: float = 0.0
    use_critical_label_weights: bool = True
    device: str = "cpu"
    residual: GRUPositionResidualConfig = GRUPositionResidualConfig(
        hidden_dim=32,
        maximum_half_fov_fraction=0.25,
    )
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()
    curriculum: CriticalEpisodeCurriculumConfig = (
        CriticalEpisodeCurriculumConfig(concentration_strength=1.0)
    )

    def __post_init__(self) -> None:
        if self.base_seed < 0 or self.optimization_seed < 0:
            raise ValueError("residual-policy seeds must be non-negative")
        for name in (
            "epochs",
            "batch_size",
            "minimum_training_episodes",
            "minimum_validation_episodes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
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
            "adaptive_action_weight",
            "residual_smoothness_weight",
            "direct_tracking_weight",
            "direct_visibility_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.rollout_horizon_index <= 0:
            raise ValueError("residual-policy rollout horizon must be positive")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development_config(
    config: CounterfactualResidualPolicyConfig,
) -> CounterfactualPlantObjectiveConfig:
    return CounterfactualPlantObjectiveConfig(
        training_seed=config.optimization_seed,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        rollout_horizon_index=config.rollout_horizon_index,
        training_integration_period_s=config.training_integration_period_s,
        minimum_training_episodes=config.minimum_training_episodes,
        minimum_validation_episodes=config.minimum_validation_episodes,
        maximum_state_regression_fraction=(
            config.maximum_state_regression_fraction
        ),
        maximum_adapter_regression_fraction=(
            config.maximum_adapter_regression_fraction
        ),
        minimum_tracking_improvement_fraction=(
            config.minimum_tracking_improvement_fraction
        ),
        maximum_saturation_regression_fraction=(
            config.maximum_saturation_regression_fraction
        ),
        device=config.device,
        criticality=config.criticality,
        curriculum=config.curriculum,
        candidates=(selected_counterfactual_plant_v87_candidate(),),
    )


def _safety_config(
    config: CounterfactualResidualPolicyConfig,
) -> CounterfactualPlantReplicationConfig:
    return CounterfactualPlantReplicationConfig(
        training_seeds=(config.base_seed,),
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        rollout_horizon_index=config.rollout_horizon_index,
        training_integration_period_s=config.training_integration_period_s,
        minimum_training_episodes=config.minimum_training_episodes,
        minimum_validation_episodes=config.minimum_validation_episodes,
        minimum_passing_seed_count=1,
        maximum_state_regression_fraction=(
            config.maximum_state_regression_fraction
        ),
        maximum_adapter_regression_fraction=(
            config.maximum_adapter_regression_fraction
        ),
        minimum_tracking_improvement_fraction=(
            config.minimum_tracking_improvement_fraction
        ),
        maximum_saturation_regression_fraction=(
            config.maximum_saturation_regression_fraction
        ),
        maximum_per_seed_state_regression_fraction=(
            config.maximum_per_seed_state_regression_fraction
        ),
        maximum_per_seed_adapter_regression_fraction=(
            config.maximum_per_seed_adapter_regression_fraction
        ),
        device=config.device,
        criticality=config.criticality,
        curriculum=config.curriculum,
    )


@torch.no_grad()
def _residual_report(
    model: CausalTargetStateGRUWithPositionResidual,
    dataset,
    *,
    batch_size: int,
    device: str,
) -> dict[str, float | int]:
    loader = DataLoader(
        GimbalTorchSequenceDataset(
            dataset,
            ObservationProfile.DISTURBANCE_AWARE,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    values = []
    model.to(device)
    model.eval()
    for batch in loader:
        output = model(batch["features"].to(device))
        residual = output.position_target_residual_fov_fraction
        assert residual is not None
        mask = batch["sequence_mask"].to(device)
        values.append(residual[mask].cpu().numpy())
    combined = np.concatenate(values) if values else np.empty(0)
    return {
        "sample_count": int(combined.size),
        "mean_half_fov_fraction": (
            float(np.mean(combined)) if combined.size else 0.0
        ),
        "mean_absolute_half_fov_fraction": (
            float(np.mean(np.abs(combined))) if combined.size else 0.0
        ),
        "rms_half_fov_fraction": (
            float(np.sqrt(np.mean(combined**2))) if combined.size else 0.0
        ),
        "maximum_absolute_half_fov_fraction": (
            float(np.max(np.abs(combined))) if combined.size else 0.0
        ),
    }


def evaluate_counterfactual_residual_policy(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    base_checkpoint: str | Path,
    checkpoint_directory: str | Path,
    config: CounterfactualResidualPolicyConfig | None = None,
) -> dict[str, Any]:
    """Train only a bounded control head while keeping state frozen."""

    config = config or CounterfactualResidualPolicyConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    base_checkpoint = Path(base_checkpoint)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("residual-policy training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("residual-policy validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("residual-policy dataset seeds overlap")
    if (
        train.manifest.prediction_horizons_s
        != validation.manifest.prediction_horizons_s
    ):
        raise ValueError("residual-policy horizons differ")
    if config.rollout_horizon_index >= len(
        train.manifest.prediction_horizons_s
    ):
        raise ValueError("residual-policy rollout horizon is absent")

    base_model, base_metadata = load_gru_checkpoint(
        base_checkpoint,
        device=config.device,
    )
    if base_model.config.mean_parameterization != "integrated_midpoint":
        raise ValueError("residual-policy base must use midpoint dynamics")
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
    candidate = selected_counterfactual_plant_v87_candidate()
    loss_config = replace(
        candidate.loss(
            rollout_horizon_index=config.rollout_horizon_index,
            training_integration_period_s=(
                config.training_integration_period_s
            ),
        ),
        adaptive_position_action_weight=config.adaptive_action_weight,
        position_plant_tracking_weight=config.direct_tracking_weight,
        position_plant_visibility_weight=config.direct_visibility_weight,
        position_plant_smoothness_weight=(
            config.residual_smoothness_weight
        ),
    )
    reference = _record_model(
        base_model,
        validation,
        validation_criticality,
        batch_size=config.batch_size,
        device=config.device,
        rollout_horizon_index=config.rollout_horizon_index,
    )

    set_gru_seed(config.optimization_seed)
    device = torch.device(config.device)
    model = CausalTargetStateGRUWithPositionResidual(
        base_model,
        config.residual,
    ).to(device)
    adaptive_supervision = compute_adaptive_position_supervision(
        train,
        adapter=loss_config.adaptive_position_config,
        profile=ObservationProfile.DISTURBANCE_AWARE,
    )
    generator = torch.Generator().manual_seed(config.optimization_seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(curriculum.episode_weights, dtype=torch.double),
        num_samples=train.episode_count,
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        GimbalTorchSequenceDataset(
            train,
            ObservationProfile.DISTURBANCE_AWARE,
            label_weights=(
                train_criticality.weights
                if config.use_critical_label_weights
                else None
            ),
            adaptive_position_supervision=adaptive_supervision,
        ),
        batch_size=config.batch_size,
        sampler=sampler,
    )
    optimizer = torch.optim.AdamW(
        model.residual_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    epoch_states = []
    training_history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_labels = 0
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            target_mask = batch["target_mask"].to(device)
            sequence_mask = batch["sequence_mask"].to(device)
            output = model(features)
            loss = target_state_nll(
                output,
                targets,
                target_mask,
                sequence_mask,
                loss_config,
                label_weights=(
                    batch["label_weights"].to(device)
                    if "label_weights" in batch
                    else None
                ),
                prediction_horizons_s=train.manifest.prediction_horizons_s,
                adaptive_position_context=(
                    _adaptive_position_context_from_batch(batch, device)
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.residual_head.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            label_count = int(
                (target_mask & sequence_mask.unsqueeze(-1)).sum().item()
            )
            total_loss += float(loss.total.detach()) * label_count
            total_labels += label_count
        training_history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / max(1, total_labels),
            }
        )
        epoch_states.append(copy.deepcopy(model.residual_head.state_dict()))

    development_config = _development_config(config)
    safety_config = _safety_config(config)
    epoch_records = []
    passing_epochs = []
    for epoch, state_dict in enumerate(epoch_states, start=1):
        model.residual_head.load_state_dict(state_dict)
        evaluated = _record_model(
            model,
            validation,
            validation_criticality,
            batch_size=config.batch_size,
            device=config.device,
            rollout_horizon_index=config.rollout_horizon_index,
        )
        development_gate = _gate(
            evaluated["selection_view"],
            reference["selection_view"],
            development_config,
        )
        safety = _per_seed_safety(
            evaluated["selection_view"],
            reference["selection_view"],
            safety_config,
        )
        record = {
            "epoch": epoch,
            "training_record": training_history[epoch - 1],
            "evaluation": evaluated,
            "residual": _residual_report(
                model,
                validation,
                batch_size=config.batch_size,
                device=config.device,
            ),
            "development_gate": development_gate,
            "per_seed_safety": safety,
        }
        epoch_records.append(record)
        if development_gate["passed"] and safety["passed"]:
            passing_epochs.append(record)

    selected = (
        min(
            passing_epochs,
            key=lambda item: item["evaluation"]["selection_view"][
                "position_plant_tracking_rmse_normalized"
            ],
        )
        if passing_epochs
        else None
    )
    selected_checkpoint = None
    if selected is not None:
        model.residual_head.load_state_dict(
            epoch_states[selected["epoch"] - 1]
        )
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        selected_checkpoint = checkpoint_directory / (
            f"gimbal_v8_9_residual_policy_seed_{config.base_seed}.pt"
        )
        save_gru_position_residual_checkpoint(
            selected_checkpoint,
            model,
            metadata={
                "experiment": COUNTERFACTUAL_RESIDUAL_POLICY_SCHEMA_VERSION,
                "base_seed": config.base_seed,
                "optimization_seed": config.optimization_seed,
                "epoch": selected["epoch"],
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "fresh_test_opened": False,
            },
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.residual_head.parameters()
        if parameter.requires_grad
    )
    return {
        "experiment": COUNTERFACTUAL_RESIDUAL_POLICY_SCHEMA_VERSION,
        "config": asdict(config),
        "candidate": asdict(candidate),
        "loss_config": asdict(loss_config),
        "architecture": {
            "base_parameters_frozen": True,
            "trainable_residual_parameters": trainable_parameters,
            "state_output_is_base_output": True,
            "residual_bound_is_hardware_normalized": True,
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
        "reference": reference,
        "epochs": epoch_records,
        "passing_epoch_count": len(passing_epochs),
        "selected": (
            {
                "epoch": selected["epoch"],
                "selection_view": selected["evaluation"]["selection_view"],
                "residual": selected["residual"],
                "checkpoint": str(selected_checkpoint),
            }
            if selected is not None and selected_checkpoint is not None
            else None
        ),
        "recommendation": (
            "evaluate_residual_policy_on_independent_base_seeds"
            if selected is not None
            else "residual_policy_did_not_pass_seed_29_development"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V8.9 frozen-state residual position policy."
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
        default=Path("artifacts/gimbal_counterfactual_residual_policy"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_counterfactual_residual_policy.json"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adaptive-action-weight", type=float, default=0.0)
    parser.add_argument("--smoothness-weight", type=float, default=10.0)
    parser.add_argument("--tracking-weight", type=float, default=0.0)
    parser.add_argument("--visibility-weight", type=float, default=0.0)
    parser.add_argument(
        "--uniform-label-weights",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_counterfactual_residual_policy(
        train_path=args.train_data,
        validation_path=args.validation_data,
        base_checkpoint=args.base_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        config=CounterfactualResidualPolicyConfig(
            epochs=args.epochs,
            device=args.device,
            adaptive_action_weight=args.adaptive_action_weight,
            residual_smoothness_weight=args.smoothness_weight,
            direct_tracking_weight=args.tracking_weight,
            direct_visibility_weight=args.visibility_weight,
            use_critical_label_weights=not args.uniform_label_weights,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"passing_epochs={result['passing_epoch_count']}; "
        f"selected={result['selected']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
