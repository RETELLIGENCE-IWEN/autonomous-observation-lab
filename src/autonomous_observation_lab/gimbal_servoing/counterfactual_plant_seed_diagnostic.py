"""Epoch-wise diagnostic for a failed V8.7 replication seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

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
from .gru import CausalTargetStateGRU, load_gru_checkpoint, save_gru_checkpoint
from .gru_training import GRUTrainingConfig, train_gru


COUNTERFACTUAL_PLANT_SEED_DIAGNOSTIC_SCHEMA_VERSION = (
    "gimbal_counterfactual_plant_v8_7_seed_diagnostic_v2"
)


@dataclass(frozen=True)
class CounterfactualPlantSeedDiagnosticConfig:
    training_seed: int = 29
    optimization_seeds: tuple[int, ...] = (17, 43)
    learning_rates: tuple[float, ...] = (1e-4,)
    epochs: int = 6
    batch_size: int = 24
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
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()
    curriculum: CriticalEpisodeCurriculumConfig = (
        CriticalEpisodeCurriculumConfig(concentration_strength=1.0)
    )

    def __post_init__(self) -> None:
        if self.training_seed < 0:
            raise ValueError("diagnostic training seed must be non-negative")
        if not self.optimization_seeds or len(set(self.optimization_seeds)) != len(
            self.optimization_seeds
        ):
            raise ValueError("optimization seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.optimization_seeds):
            raise ValueError("optimization seeds must be non-negative")
        if not self.learning_rates or len(set(self.learning_rates)) != len(
            self.learning_rates
        ):
            raise ValueError("diagnostic learning rates must be unique")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.learning_rates
        ):
            raise ValueError("diagnostic learning rates must be positive")
        for name in (
            "epochs",
            "batch_size",
            "minimum_training_episodes",
            "minimum_validation_episodes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "gradient_clip_norm",
            "training_integration_period_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight decay must be finite and non-negative")
        if self.rollout_horizon_index <= 0:
            raise ValueError("diagnostic rollout horizon must be positive")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_config(
    config: CounterfactualPlantSeedDiagnosticConfig,
    learning_rate: float,
) -> CounterfactualPlantObjectiveConfig:
    return CounterfactualPlantObjectiveConfig(
        training_seed=config.training_seed,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=learning_rate,
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
    config: CounterfactualPlantSeedDiagnosticConfig,
    learning_rate: float,
) -> CounterfactualPlantReplicationConfig:
    return CounterfactualPlantReplicationConfig(
        training_seeds=(config.training_seed,),
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=learning_rate,
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


def evaluate_counterfactual_plant_seed_diagnostic(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    base_checkpoint: str | Path,
    checkpoint_directory: str | Path,
    config: CounterfactualPlantSeedDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Inspect epoch and learning-rate sensitivity without opening a test."""

    config = config or CounterfactualPlantSeedDiagnosticConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    base_checkpoint = Path(base_checkpoint)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("seed diagnostic training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("seed diagnostic validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("seed diagnostic dataset seeds overlap")
    if (
        train.manifest.prediction_horizons_s
        != validation.manifest.prediction_horizons_s
    ):
        raise ValueError("seed diagnostic horizons differ")
    if config.rollout_horizon_index >= len(
        train.manifest.prediction_horizons_s
    ):
        raise ValueError("seed diagnostic rollout horizon is absent")

    base_model, base_metadata = load_gru_checkpoint(
        base_checkpoint,
        device=config.device,
    )
    if base_model.config.mean_parameterization != "integrated_midpoint":
        raise ValueError("seed diagnostic base must use midpoint dynamics")
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
    loss_config = candidate.loss(
        rollout_horizon_index=config.rollout_horizon_index,
        training_integration_period_s=config.training_integration_period_s,
    )
    reference = _record_model(
        base_model,
        validation,
        validation_criticality,
        batch_size=config.batch_size,
        device=config.device,
        rollout_horizon_index=config.rollout_horizon_index,
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    trajectories = []
    passing_epochs = []
    retained_states: dict[tuple[int, float, int], dict[str, Any]] = {}
    for optimization_seed in config.optimization_seeds:
        for learning_rate in config.learning_rates:
            training = train_gru(
                train,
                validation,
                ObservationProfile.DISTURBANCE_AWARE,
                model_config=base_model.config,
                training_config=GRUTrainingConfig(
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=learning_rate,
                    weight_decay=config.weight_decay,
                    gradient_clip_norm=config.gradient_clip_norm,
                    seed=optimization_seed,
                    device=config.device,
                ),
                loss_config=loss_config,
                training_episode_weights=curriculum.episode_weights,
                training_label_weights=train_criticality.weights,
                validation_label_weights=validation_criticality.weights,
                initial_state_dict=base_model.state_dict(),
                retain_epoch_states=True,
            )
            gate_config = _gate_config(config, learning_rate)
            safety_config = _safety_config(config, learning_rate)
            epochs = []
            for epoch, state_dict in enumerate(
                training.epoch_state_dicts,
                start=1,
            ):
                retained_states[
                    (optimization_seed, learning_rate, epoch)
                ] = state_dict
                model = CausalTargetStateGRU(base_model.config)
                model.load_state_dict(state_dict)
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
                    gate_config,
                )
                safety = _per_seed_safety(
                    evaluated["selection_view"],
                    reference["selection_view"],
                    safety_config,
                )
                record = {
                    "optimization_seed": optimization_seed,
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                    "training_record": asdict(training.history[epoch - 1]),
                    "evaluation": evaluated,
                    "development_gate": development_gate,
                    "per_seed_safety": safety,
                }
                epochs.append(record)
                if development_gate["passed"] and safety["passed"]:
                    passing_epochs.append(record)
            trajectories.append(
                {
                    "optimization_seed": optimization_seed,
                    "learning_rate": learning_rate,
                    "composite_best_epoch": training.best_epoch,
                    "epochs": epochs,
                }
            )

    selected = (
        min(
            passing_epochs,
            key=lambda item: (
                item["evaluation"]["selection_view"][
                    "position_plant_tracking_rmse_normalized"
                ],
                item["evaluation"]["selection_view"][
                    "critical_position_plant_tracking_rmse_normalized"
                ],
            ),
        )
        if passing_epochs
        else None
    )
    selected_checkpoint = None
    if selected is not None:
        state_dict = retained_states[
            (
                selected["optimization_seed"],
                selected["learning_rate"],
                selected["epoch"],
            )
        ]
        model = CausalTargetStateGRU(base_model.config)
        model.load_state_dict(state_dict)
        selected_checkpoint = checkpoint_directory / (
            f"gimbal_v8_7_seed_{config.training_seed}_diagnostic.pt"
        )
        save_gru_checkpoint(
            selected_checkpoint,
            model,
            metadata={
                "experiment": (
                    COUNTERFACTUAL_PLANT_SEED_DIAGNOSTIC_SCHEMA_VERSION
                ),
                "training_seed": config.training_seed,
                "optimization_seed": selected["optimization_seed"],
                "learning_rate": selected["learning_rate"],
                "epoch": selected["epoch"],
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "fresh_test_opened": False,
            },
        )

    return {
        "experiment": COUNTERFACTUAL_PLANT_SEED_DIAGNOSTIC_SCHEMA_VERSION,
        "config": asdict(config),
        "candidate": asdict(candidate),
        "loss_config": asdict(loss_config),
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
        "trajectories": trajectories,
        "passing_epoch_count": len(passing_epochs),
        "selected": (
            {
                "learning_rate": selected["learning_rate"],
                "optimization_seed": selected["optimization_seed"],
                "epoch": selected["epoch"],
                "selection_view": selected["evaluation"]["selection_view"],
                "checkpoint": str(selected_checkpoint),
            }
            if selected is not None and selected_checkpoint is not None
            else None
        ),
        "recommendation": (
            "freeze_early_stop_and_rereplicate_all_seeds"
            if selected is not None
            else "seed_29_failure_is_not_simple_overshoot"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect epoch-wise V8.7 behavior for one failed seed."
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
        default=Path("artifacts/gimbal_counterfactual_seed_diagnostic"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_counterfactual_seed_29_diagnostic.json"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_counterfactual_plant_seed_diagnostic(
        train_path=args.train_data,
        validation_path=args.validation_data,
        base_checkpoint=args.base_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        config=CounterfactualPlantSeedDiagnosticConfig(
            epochs=args.epochs,
            device=args.device,
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
