"""V8 development screen for outcome-aware counterfactual servo rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .adaptive_curriculum_objective import (
    _selection_metrics,
    selected_adaptive_position_v21_config,
)
from .config import ObservationProfile
from .control_aware_predictor import _future_index, _relative_change
from .control_criticality import (
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from .critical_curriculum import (
    CriticalEpisodeCurriculumConfig,
    compute_critical_episode_curriculum,
    critical_episode_curriculum_report,
)
from .dataset import load_gimbal_dataset
from .gru import (
    GRULossConfig,
    GRUPositionPlantRolloutConfig,
    load_gru_checkpoint,
    save_gru_checkpoint,
)
from .gru_training import GRUTrainingConfig, evaluate_gru, train_gru


COUNTERFACTUAL_PLANT_OBJECTIVE_SCHEMA_VERSION = (
    "gimbal_counterfactual_plant_objective_v8_7_development_v1"
)


@dataclass(frozen=True)
class CounterfactualPlantCandidate:
    name: str
    regret_weight: float
    response_weight: float = 0.0
    nll_weight_scale: float = 0.0
    mean_error_weight: float = 1.0
    bearing_mean_error_weight: float = 0.0
    rate_mean_error_weight: float = 0.0
    tracking_weight: float = 0.0
    visibility_weight: float = 0.0
    smoothness_weight: float = 0.0
    saturation_weight: float = 0.0
    use_critical_episode_curriculum: bool = False
    use_critical_label_weights: bool = False

    def __post_init__(self) -> None:
        for name in (
            "regret_weight",
            "response_weight",
            "nll_weight_scale",
            "mean_error_weight",
            "bearing_mean_error_weight",
            "rate_mean_error_weight",
            "tracking_weight",
            "visibility_weight",
            "smoothness_weight",
            "saturation_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.regret_weight <= 0.0:
            raise ValueError("counterfactual regret weight must be positive")

    def loss(
        self,
        *,
        rollout_horizon_index: int,
        training_integration_period_s: float,
    ) -> GRULossConfig:
        return GRULossConfig(
            bearing_weight=self.nll_weight_scale,
            rate_weight=0.75 * self.nll_weight_scale,
            mean_error_weight=self.mean_error_weight,
            bearing_mean_error_weight=self.bearing_mean_error_weight,
            rate_mean_error_weight=self.rate_mean_error_weight,
            adaptive_position_config=selected_adaptive_position_v21_config(),
            position_plant_tracking_weight=self.tracking_weight,
            position_plant_response_weight=self.response_weight,
            position_plant_regret_weight=self.regret_weight,
            position_plant_visibility_weight=self.visibility_weight,
            position_plant_smoothness_weight=self.smoothness_weight,
            position_plant_saturation_weight=self.saturation_weight,
            position_plant_config=GRUPositionPlantRolloutConfig(
                horizon_index=rollout_horizon_index,
                integration_period_override_s=training_integration_period_s,
            ),
            horizon_weights=(0.75, 1.5, 1.0, 1.0),
        )


def default_counterfactual_plant_candidates() -> tuple[
    CounterfactualPlantCandidate, ...
]:
    return (
        CounterfactualPlantCandidate(
            "conservative_regret_balanced_critical_curriculum",
            5.0,
            mean_error_weight=0.0,
            bearing_mean_error_weight=6.0,
            rate_mean_error_weight=2.0,
            smoothness_weight=10.0,
            saturation_weight=0.05,
            use_critical_episode_curriculum=True,
            use_critical_label_weights=True,
        ),
        CounterfactualPlantCandidate(
            "moderate_regret_balanced_critical_curriculum",
            7.5,
            mean_error_weight=0.0,
            bearing_mean_error_weight=6.0,
            rate_mean_error_weight=2.0,
            smoothness_weight=10.0,
            saturation_weight=0.05,
            use_critical_episode_curriculum=True,
            use_critical_label_weights=True,
        ),
    )


@dataclass(frozen=True)
class CounterfactualPlantObjectiveConfig:
    training_seed: int = 17
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
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()
    curriculum: CriticalEpisodeCurriculumConfig = (
        CriticalEpisodeCurriculumConfig(concentration_strength=1.0)
    )
    candidates: tuple[CounterfactualPlantCandidate, ...] = (
        default_counterfactual_plant_candidates()
    )

    def __post_init__(self) -> None:
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
            "maximum_state_regression_fraction",
            "maximum_adapter_regression_fraction",
            "minimum_tracking_improvement_fraction",
            "maximum_saturation_regression_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.candidates:
            raise ValueError("at least one counterfactual candidate is required")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("counterfactual candidate names must be unique")
        if self.rollout_horizon_index <= 0:
            raise ValueError("counterfactual rollout horizon must be positive")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome_evaluation_loss(
    horizon_count: int,
    horizon_index: int,
) -> GRULossConfig:
    if horizon_count <= 1:
        raise ValueError("counterfactual evaluation needs a future horizon")
    if horizon_index >= horizon_count:
        raise ValueError("counterfactual evaluation horizon is out of range")
    return GRULossConfig(
        bearing_weight=0.0,
        rate_weight=0.0,
        mean_error_weight=0.0,
        adaptive_position_config=selected_adaptive_position_v21_config(),
        position_plant_tracking_weight=1.0,
        position_plant_response_weight=1.0,
        position_plant_regret_weight=1.0,
        position_plant_visibility_weight=1.0,
        position_plant_smoothness_weight=1.0,
        position_plant_saturation_weight=1.0,
        position_plant_config=GRUPositionPlantRolloutConfig(
            horizon_index=horizon_index,
        ),
    )


def _record_model(
    model,
    validation,
    criticality,
    *,
    batch_size: int,
    device: str,
    rollout_horizon_index: int,
) -> dict[str, Any]:
    adapter = selected_adaptive_position_v21_config()
    horizon_count = len(validation.manifest.prediction_horizons_s)
    common_adapter_loss = GRULossConfig(
        adaptive_position_action_weight=1.0,
        adaptive_position_config=adapter,
    )
    outcome_loss = _outcome_evaluation_loss(
        horizon_count,
        rollout_horizon_index,
    )
    arguments = dict(
        model=model,
        dataset=validation,
        profile=ObservationProfile.DISTURBANCE_AWARE,
        batch_size=batch_size,
        device=device,
    )
    standard = evaluate_gru(**arguments)
    critical = evaluate_gru(
        **arguments,
        evaluation_mask=criticality.critical_mask,
    )
    adapter_metrics = evaluate_gru(
        **arguments,
        loss_config=common_adapter_loss,
    )
    critical_adapter = evaluate_gru(
        **arguments,
        loss_config=common_adapter_loss,
        evaluation_mask=criticality.critical_mask,
    )
    outcome = evaluate_gru(**arguments, loss_config=outcome_loss)
    critical_outcome = evaluate_gru(
        **arguments,
        loss_config=outcome_loss,
        evaluation_mask=criticality.critical_mask,
    )
    future_index = _future_index(validation.manifest.prediction_horizons_s)
    selection = _selection_metrics(
        standard,
        critical,
        adapter_metrics,
        critical_adapter,
        future_index,
    )
    for prefix, metrics in (
        ("", outcome),
        ("critical_", critical_outcome),
    ):
        for name in (
            "position_plant_tracking_rmse_normalized",
            "position_plant_response_rmse_normalized",
            "position_plant_regret_rmse_normalized",
            "position_plant_visibility_rmse_normalized",
            "position_plant_smoothness_rmse_normalized",
            "position_plant_saturation_rmse_normalized",
        ):
            value = getattr(metrics, name)
            if value is None:
                raise ValueError("counterfactual outcome metric is absent")
            selection[f"{prefix}{name}"] = float(value)
    return {
        "standard_validation": asdict(standard),
        "critical_validation": asdict(critical),
        "common_adapter_validation": asdict(adapter_metrics),
        "critical_common_adapter_validation": asdict(critical_adapter),
        "counterfactual_outcome_validation": asdict(outcome),
        "critical_counterfactual_outcome_validation": asdict(
            critical_outcome
        ),
        "selection_view": selection,
    }


def _gate(
    candidate: dict[str, float],
    reference: dict[str, float],
    config: CounterfactualPlantObjectiveConfig,
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
    checks = {
        "state_safety": all(
            changes[key] <= config.maximum_state_regression_fraction
            for key in state_keys
        ),
        "adapter_safety": all(
            changes[key] <= config.maximum_adapter_regression_fraction
            for key in (
                "adaptive_position_action_rmse_normalized",
                "critical_adaptive_position_action_rmse_normalized",
            )
        ),
        "tracking": changes[
            "position_plant_tracking_rmse_normalized"
        ] <= -config.minimum_tracking_improvement_fraction,
        "critical_tracking": changes[
            "critical_position_plant_tracking_rmse_normalized"
        ] <= -config.minimum_tracking_improvement_fraction,
        "regret": changes[
            "position_plant_regret_rmse_normalized"
        ] <= -config.minimum_tracking_improvement_fraction,
        "critical_regret": changes[
            "critical_position_plant_regret_rmse_normalized"
        ] <= -config.minimum_tracking_improvement_fraction,
        "visibility": changes[
            "position_plant_visibility_rmse_normalized"
        ] <= 0.0,
        "critical_visibility": changes[
            "critical_position_plant_visibility_rmse_normalized"
        ] <= 0.0,
        "smoothness": changes[
            "position_plant_smoothness_rmse_normalized"
        ] <= 0.0,
        "saturation": changes[
            "position_plant_saturation_rmse_normalized"
        ] <= config.maximum_saturation_regression_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def evaluate_counterfactual_plant_objective(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    base_checkpoint: str | Path,
    checkpoint_directory: str | Path,
    config: CounterfactualPlantObjectiveConfig | None = None,
) -> dict[str, Any]:
    """Fine-tune the V7 midpoint model without opening a fresh test block."""

    config = config or CounterfactualPlantObjectiveConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    base_checkpoint = Path(base_checkpoint)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("V8 training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("V8 validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("V8 train and validation seeds overlap")
    if (
        train.manifest.prediction_horizons_s
        != validation.manifest.prediction_horizons_s
    ):
        raise ValueError("V8 train and validation horizons differ")
    if len(train.manifest.prediction_horizons_s) != 4:
        raise ValueError("V8 expects the frozen four-horizon dataset")
    if config.rollout_horizon_index >= len(
        train.manifest.prediction_horizons_s
    ):
        raise ValueError("V8 rollout horizon index is out of range")

    base_model, base_metadata = load_gru_checkpoint(
        base_checkpoint,
        device=config.device,
    )
    if base_model.config.mean_parameterization != "integrated_midpoint":
        raise ValueError("V8 base checkpoint must use hard midpoint dynamics")
    if (
        base_model.config.prediction_horizons_s
        != train.manifest.prediction_horizons_s
    ):
        raise ValueError("V8 base checkpoint horizons differ from the dataset")

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
    reference = {
        "name": "midpoint_state_reference",
        "base_checkpoint": str(base_checkpoint),
        **_record_model(
            base_model,
            validation,
            validation_criticality,
            batch_size=config.batch_size,
            device=config.device,
            rollout_horizon_index=config.rollout_horizon_index,
        ),
        "gate": {
            "passed": False,
            "checks": {"reference": True},
            "relative_changes": {},
        },
    }
    records = [reference]
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    for candidate in config.candidates:
        loss_config = candidate.loss(
            rollout_horizon_index=config.rollout_horizon_index,
            training_integration_period_s=(
                config.training_integration_period_s
            )
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
                seed=config.training_seed,
                device=config.device,
            ),
            loss_config=loss_config,
            training_episode_weights=(
                curriculum.episode_weights
                if candidate.use_critical_episode_curriculum
                else None
            ),
            training_label_weights=(
                train_criticality.weights
                if candidate.use_critical_label_weights
                else None
            ),
            validation_label_weights=(
                validation_criticality.weights
                if candidate.use_critical_label_weights
                else None
            ),
            initial_state_dict=base_model.state_dict(),
        )
        checkpoint = checkpoint_directory / (
            f"gimbal_v8_{candidate.name}_seed_{config.training_seed}.pt"
        )
        save_gru_checkpoint(
            checkpoint,
            training.model,
            metadata={
                "experiment": COUNTERFACTUAL_PLANT_OBJECTIVE_SCHEMA_VERSION,
                "candidate": candidate.name,
                "training_seed": config.training_seed,
                "base_checkpoint_sha256": _sha256(base_checkpoint),
                "train_sha256": _sha256(train_path),
                "validation_sha256": _sha256(validation_path),
                "fresh_test_opened": False,
            },
        )
        record = {
            "name": candidate.name,
            "candidate": asdict(candidate),
            "loss_config": asdict(loss_config),
            "best_epoch": training.best_epoch,
            "training_history": [
                asdict(item) for item in training.history
            ],
            "checkpoint": str(checkpoint),
            **_record_model(
                training.model,
                validation,
                validation_criticality,
                batch_size=config.batch_size,
                device=config.device,
                rollout_horizon_index=config.rollout_horizon_index,
            ),
        }
        record["gate"] = _gate(
            record["selection_view"],
            reference["selection_view"],
            config,
        )
        records.append(record)

    eligible = [record for record in records[1:] if record["gate"]["passed"]]
    selected = (
        min(
            eligible,
            key=lambda item: (
                item["selection_view"][
                    "critical_position_plant_tracking_rmse_normalized"
                ],
                item["selection_view"][
                    "position_plant_tracking_rmse_normalized"
                ],
            ),
        )
        if eligible
        else None
    )
    return {
        "experiment": COUNTERFACTUAL_PLANT_OBJECTIVE_SCHEMA_VERSION,
        "config": asdict(config),
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
        "candidates": records,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected["name"] if selected else None,
        "selected_checkpoint": selected.get("checkpoint") if selected else None,
        "recommendation": (
            "replicate_v8_candidate_before_fresh_test"
            if selected is not None
            else "revise_counterfactual_objective_before_replication"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune hard-midpoint GRUs with counterfactual plant loss."
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
            "artifacts/gimbal_midpoint_adapter_checkpoints/"
            "gimbal_v6_midpoint_state_reference_seed_17.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_counterfactual_plant_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_counterfactual_plant_development.json"),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_counterfactual_plant_objective(
        train_path=args.train_data,
        validation_path=args.validation_data,
        base_checkpoint=args.base_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        config=CounterfactualPlantObjectiveConfig(
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
        f"selected={result['selected_candidate']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
