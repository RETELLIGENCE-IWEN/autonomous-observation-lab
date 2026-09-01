"""Seed-matched replication of the selected V8.7 plant-regret objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

from .config import ObservationProfile
from .control_aware_predictor import _relative_change
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
from .critical_curriculum import (
    CriticalEpisodeCurriculumConfig,
    compute_critical_episode_curriculum,
    critical_episode_curriculum_report,
)
from .dataset import load_gimbal_dataset
from .gru import load_gru_checkpoint, save_gru_checkpoint
from .gru_training import GRUTrainingConfig, train_gru


COUNTERFACTUAL_PLANT_REPLICATION_SCHEMA_VERSION = (
    "gimbal_counterfactual_plant_v8_7_replication_v1"
)


@dataclass(frozen=True)
class CounterfactualPlantReplicationConfig:
    training_seeds: tuple[int, ...] = (17, 29, 43)
    epochs: int = 6
    batch_size: int = 24
    learning_rate: float = 1e-4
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
    minimum_passing_seed_count: int = 2
    maximum_per_seed_state_regression_fraction: float = 0.03
    maximum_per_seed_adapter_regression_fraction: float = 0.03
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()
    curriculum: CriticalEpisodeCurriculumConfig = (
        CriticalEpisodeCurriculumConfig(concentration_strength=1.0)
    )

    def __post_init__(self) -> None:
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("replication seeds must be non-empty and unique")
        for name in (
            "epochs",
            "batch_size",
            "minimum_training_episodes",
            "minimum_validation_episodes",
            "minimum_passing_seed_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_passing_seed_count > len(self.training_seeds):
            raise ValueError("passing-seed requirement exceeds seed count")
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
            "maximum_state_regression_fraction",
            "maximum_adapter_regression_fraction",
            "minimum_tracking_improvement_fraction",
            "maximum_saturation_regression_fraction",
            "maximum_per_seed_state_regression_fraction",
            "maximum_per_seed_adapter_regression_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.rollout_horizon_index <= 0:
            raise ValueError("replication rollout horizon must be positive")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _mean_view(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    keys = records[0][field].keys()
    return {
        key: mean(float(record[field][key]) for record in records)
        for key in keys
    }


def _development_gate_config(
    config: CounterfactualPlantReplicationConfig,
) -> CounterfactualPlantObjectiveConfig:
    candidate = selected_counterfactual_plant_v87_candidate()
    return CounterfactualPlantObjectiveConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        rollout_horizon_index=config.rollout_horizon_index,
        training_integration_period_s=(
            config.training_integration_period_s
        ),
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
        candidates=(candidate,),
    )


def _per_seed_safety(
    candidate: dict[str, float],
    reference: dict[str, float],
    config: CounterfactualPlantReplicationConfig,
) -> dict[str, Any]:
    changes = {
        key: _relative_change(candidate[key], value)
        for key, value in reference.items()
        if key != "common_weighted_validation_loss"
    }
    state_keys = (
        "standard_bearing_rmse_deg",
        "standard_rate_rmse_deg_s",
        "future_bearing_rmse_deg",
        "critical_future_bearing_rmse_deg",
        "critical_future_rate_rmse_deg_s",
    )
    adapter_keys = (
        "adaptive_position_action_rmse_normalized",
        "critical_adaptive_position_action_rmse_normalized",
    )
    checks = {
        "state": all(
            changes[key]
            <= config.maximum_per_seed_state_regression_fraction
            for key in state_keys
        ),
        "adapter": all(
            changes[key]
            <= config.maximum_per_seed_adapter_regression_fraction
            for key in adapter_keys
        ),
        "global_tracking_non_regression": changes[
            "position_plant_tracking_rmse_normalized"
        ] <= 0.0,
        "critical_tracking_non_regression": changes[
            "critical_position_plant_tracking_rmse_normalized"
        ] <= 0.0,
        "smoothness_non_regression": changes[
            "position_plant_smoothness_rmse_normalized"
        ] <= 0.0,
        "saturation_non_regression": changes[
            "position_plant_saturation_rmse_normalized"
        ] <= 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def evaluate_counterfactual_plant_replication(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    base_checkpoint_directory: str | Path,
    checkpoint_directory: str | Path,
    config: CounterfactualPlantReplicationConfig | None = None,
) -> dict[str, Any]:
    """Replicate V8.7 without opening closed-loop or fresh-test blocks."""

    config = config or CounterfactualPlantReplicationConfig()
    protocol = _development_gate_config(config)
    candidate = selected_counterfactual_plant_v87_candidate()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    base_checkpoint_directory = Path(base_checkpoint_directory)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("V8.7 replication training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("V8.7 replication validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("V8.7 replication dataset seeds overlap")
    if (
        train.manifest.prediction_horizons_s
        != validation.manifest.prediction_horizons_s
    ):
        raise ValueError("V8.7 replication horizons differ")
    if config.rollout_horizon_index >= len(
        train.manifest.prediction_horizons_s
    ):
        raise ValueError("V8.7 replication rollout horizon is absent")

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
    loss_config = candidate.loss(
        rollout_horizon_index=config.rollout_horizon_index,
        training_integration_period_s=config.training_integration_period_s,
    )
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    seed_records = []
    for seed in config.training_seeds:
        base_checkpoint = base_checkpoint_directory / (
            f"gimbal_v7_midpoint_state_reference_seed_{seed}.pt"
        )
        base_model, base_metadata = load_gru_checkpoint(
            base_checkpoint,
            device=config.device,
        )
        if base_model.config.mean_parameterization != "integrated_midpoint":
            raise ValueError("replication base must use hard midpoint dynamics")
        if (
            base_model.config.prediction_horizons_s
            != train.manifest.prediction_horizons_s
        ):
            raise ValueError("replication checkpoint horizons differ")
        reference = _record_model(
            base_model,
            validation,
            validation_criticality,
            batch_size=config.batch_size,
            device=config.device,
            rollout_horizon_index=config.rollout_horizon_index,
        )
        training = train_gru(
            train,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            model_config=base_model.config,
            training_config=GRUTrainingConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                gradient_clip_norm=config.gradient_clip_norm,
                seed=seed,
                device=config.device,
            ),
            loss_config=loss_config,
            training_episode_weights=curriculum.episode_weights,
            training_label_weights=train_criticality.weights,
            validation_label_weights=validation_criticality.weights,
            initial_state_dict=base_model.state_dict(),
        )
        evaluated = _record_model(
            training.model,
            validation,
            validation_criticality,
            batch_size=config.batch_size,
            device=config.device,
            rollout_horizon_index=config.rollout_horizon_index,
        )
        checkpoint = checkpoint_directory / (
            f"gimbal_v8_7_regret_seed_{seed}.pt"
        )
        save_gru_checkpoint(
            checkpoint,
            training.model,
            metadata={
                "experiment": COUNTERFACTUAL_PLANT_REPLICATION_SCHEMA_VERSION,
                "candidate": candidate.name,
                "training_seed": seed,
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "train_sha256": _sha256(train_path),
                "validation_sha256": _sha256(validation_path),
                "fresh_test_opened": False,
            },
        )
        seed_records.append(
            {
                "training_seed": seed,
                "base_checkpoint": {
                    "path": str(base_checkpoint),
                    "sha256": _sha256(base_checkpoint),
                    "metadata": base_metadata,
                },
                "checkpoint": str(checkpoint),
                "best_epoch": training.best_epoch,
                "training_history": [
                    asdict(item) for item in training.history
                ],
                "reference": reference,
                "candidate": evaluated,
                "development_gate": _gate(
                    evaluated["selection_view"],
                    reference["selection_view"],
                    protocol,
                ),
                "per_seed_safety": _per_seed_safety(
                    evaluated["selection_view"],
                    reference["selection_view"],
                    config,
                ),
            }
        )

    reference_mean = _mean_view(
        [record["reference"] for record in seed_records],
        "selection_view",
    )
    candidate_mean = _mean_view(
        [record["candidate"] for record in seed_records],
        "selection_view",
    )
    mean_gate = _gate(candidate_mean, reference_mean, protocol)
    passing_seed_count = sum(
        record["development_gate"]["passed"] for record in seed_records
    )
    per_seed_safety = all(
        record["per_seed_safety"]["passed"] for record in seed_records
    )
    aggregate_checks = {
        "mean_gate": mean_gate["passed"],
        "passing_seed_count": (
            passing_seed_count >= config.minimum_passing_seed_count
        ),
        "per_seed_safety": per_seed_safety,
    }
    passed = all(aggregate_checks.values())
    metric_distributions = {
        key: {
            "reference": _distribution(
                [
                    float(record["reference"]["selection_view"][key])
                    for record in seed_records
                ]
            ),
            "candidate": _distribution(
                [
                    float(record["candidate"]["selection_view"][key])
                    for record in seed_records
                ]
            ),
        }
        for key in reference_mean
        if key != "common_weighted_validation_loss"
    }
    return {
        "experiment": COUNTERFACTUAL_PLANT_REPLICATION_SCHEMA_VERSION,
        "config": asdict(config),
        "candidate": asdict(candidate),
        "loss_config": asdict(loss_config),
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
        "seed_records": seed_records,
        "mean_reference": reference_mean,
        "mean_candidate": candidate_mean,
        "mean_gate": mean_gate,
        "metric_distributions": metric_distributions,
        "passing_seed_count": passing_seed_count,
        "aggregate_gate": {
            "passed": passed,
            "checks": aggregate_checks,
        },
        "recommendation": (
            "evaluate_v8_7_replication_in_closed_loop_before_fresh_test"
            if passed
            else "do_not_promote_v8_7"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the selected V8.7 plant-regret objective."
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
        "--base-checkpoint-directory",
        type=Path,
        default=Path(
            "artifacts/gimbal_midpoint_adapter_replication_checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_counterfactual_plant_replication"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_counterfactual_plant_replication.json"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_counterfactual_plant_replication(
        train_path=args.train_data,
        validation_path=args.validation_data,
        base_checkpoint_directory=args.base_checkpoint_directory,
        checkpoint_directory=args.checkpoint_directory,
        config=CounterfactualPlantReplicationConfig(
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
        f"passed={result['aggregate_gate']['passed']}; "
        f"passing_seeds={result['passing_seed_count']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
