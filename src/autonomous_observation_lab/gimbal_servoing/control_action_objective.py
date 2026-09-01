"""Development ablation for explicitly action-aware GRU supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .config import ObservationProfile
from .control_aware_predictor import _future_index, _relative_change, _selection_view
from .control_criticality import (
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from .dataset import FEATURE_NAMES, load_gimbal_dataset
from .gru import (
    GRULossConfig,
    GRUTargetStateModelConfig,
    gru_parameter_count,
    save_gru_checkpoint,
)
from .gru_training import GRUTrainingConfig, evaluate_gru, train_gru


CONTROL_ACTION_OBJECTIVE_SCHEMA_VERSION = (
    "gimbal_control_action_objective_v42_development_v1"
)


@dataclass(frozen=True)
class ControlActionCandidate:
    name: str
    rate_action_weight: float
    position_action_weight: float
    dynamic_consistency_weight: float = 25.0
    mean_parameterization: str = "independent"

    @property
    def loss(self) -> GRULossConfig:
        return GRULossConfig(
            dynamic_consistency_weight=self.dynamic_consistency_weight,
            rate_action_weight=self.rate_action_weight,
            position_action_weight=self.position_action_weight,
        )


def default_control_action_candidates() -> tuple[ControlActionCandidate, ...]:
    return (
        ControlActionCandidate("consistent_reference", 0.0, 0.0),
        ControlActionCandidate("rate_action", 0.50, 0.0),
        ControlActionCandidate("position_action", 0.0, 0.50),
        ControlActionCandidate("dual_action_balanced", 0.50, 0.50),
        ControlActionCandidate("dual_action_position_priority", 0.25, 0.75),
    )


@dataclass(frozen=True)
class ControlActionObjectiveConfig:
    training_seed: int = 17
    epochs: int = 20
    batch_size: int = 24
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    hidden_dim: int = 64
    embedding_dim: int = 64
    minimum_training_episodes: int = 1000
    minimum_validation_episodes: int = 200
    maximum_standard_regression_fraction: float = 0.02
    maximum_critical_bearing_regression_fraction: float = 0.02
    maximum_critical_rate_regression_fraction: float = 0.02
    maximum_consistency_regression_fraction: float = 0.02
    minimum_rate_action_improvement_fraction: float = 0.01
    minimum_position_action_improvement_fraction: float = 0.01
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()
    candidates: tuple[ControlActionCandidate, ...] = (
        default_control_action_candidates()
    )

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "hidden_dim",
            "embedding_dim",
            "minimum_training_episodes",
            "minimum_validation_episodes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight decay must be finite and non-negative")
        if not self.candidates:
            raise ValueError("at least one action-objective candidate is required")
        if self.candidates[0].rate_action_weight != 0.0 or (
            self.candidates[0].position_action_weight != 0.0
        ):
            raise ValueError("first action-objective candidate must be a reference")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("action-objective candidate names must be unique")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_selection_view(
    standard: Any,
    critical: Any,
    common: Any,
    future_index: int,
) -> dict[str, float]:
    view = _selection_view(standard, critical, common, future_index)
    if (
        common.rate_action_rmse_normalized is None
        or common.position_action_rmse_normalized is None
    ):
        raise ValueError("action-aware evaluation did not produce action metrics")
    view.update(
        {
            "rate_action_rmse_normalized": float(
                common.rate_action_rmse_normalized
            ),
            "position_action_rmse_normalized": float(
                common.position_action_rmse_normalized
            ),
        }
    )
    return view


def _gate(
    candidate: dict[str, float],
    reference: dict[str, float],
    config: ControlActionObjectiveConfig,
) -> dict[str, Any]:
    changes = {
        key: _relative_change(candidate[key], reference[key])
        for key in reference
        if key != "common_weighted_validation_loss"
    }
    checks = {
        "standard_bearing": changes["standard_bearing_rmse_deg"]
        <= config.maximum_standard_regression_fraction,
        "standard_rate": changes["standard_rate_rmse_deg_s"]
        <= config.maximum_standard_regression_fraction,
        "future_bearing": changes["future_bearing_rmse_deg"]
        <= config.maximum_standard_regression_fraction,
        "critical_future_bearing": changes[
            "critical_future_bearing_rmse_deg"
        ]
        <= config.maximum_critical_bearing_regression_fraction,
        "critical_future_rate": changes["critical_future_rate_rmse_deg_s"]
        <= config.maximum_critical_rate_regression_fraction,
        "dynamic_consistency": changes["dynamic_consistency_rmse_deg"]
        <= config.maximum_consistency_regression_fraction,
        "rate_action": changes["rate_action_rmse_normalized"]
        <= -config.minimum_rate_action_improvement_fraction,
        "position_action": changes["position_action_rmse_normalized"]
        <= -config.minimum_position_action_improvement_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def evaluate_control_action_objective(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_directory: str | Path,
    config: ControlActionObjectiveConfig | None = None,
) -> dict[str, Any]:
    """Ablate action losses on development data without reopening test data."""

    config = config or ControlActionObjectiveConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("action-objective training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("action-objective validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("action-objective dataset seeds overlap")
    horizons_s = train.manifest.prediction_horizons_s
    if horizons_s != validation.manifest.prediction_horizons_s:
        raise ValueError("action-objective horizons differ")

    validation_criticality = compute_control_criticality(
        validation,
        config=config.criticality,
    )
    common_loss = GRULossConfig(
        bearing_weight=1.0,
        rate_weight=0.75,
        mean_error_weight=0.20,
        dynamic_consistency_weight=25.0,
        rate_action_weight=1.0,
        position_action_weight=1.0,
        horizon_weights=tuple(
            1.5 if index == 1 else (0.75 if index == 0 else 1.0)
            for index in range(len(horizons_s))
        ),
    )
    training_config = GRUTrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        seed=config.training_seed,
        device=config.device,
    )
    base_model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=horizons_s,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    )
    future_index = _future_index(horizons_s)
    records = []
    models = {}
    for candidate in config.candidates:
        model_config = replace(
            base_model_config,
            mean_parameterization=candidate.mean_parameterization,
        )
        training = train_gru(
            train,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            model_config=model_config,
            training_config=training_config,
            loss_config=candidate.loss,
        )
        standard = evaluate_gru(
            training.model,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
        )
        critical = evaluate_gru(
            training.model,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
            evaluation_mask=validation_criticality.critical_mask,
        )
        common = evaluate_gru(
            training.model,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
            loss_config=common_loss,
            label_weights=validation_criticality.weights,
        )
        records.append(
            {
                "name": candidate.name,
                "loss_config": asdict(candidate.loss),
                "model_config": asdict(model_config),
                "parameter_count": gru_parameter_count(training.model),
                "best_epoch": training.best_epoch,
                "training_history": [asdict(item) for item in training.history],
                "standard_validation": asdict(standard),
                "critical_validation": asdict(critical),
                "common_control_validation": asdict(common),
                "selection_view": _action_selection_view(
                    standard,
                    critical,
                    common,
                    future_index,
                ),
            }
        )
        models[candidate.name] = training.model

    reference = records[0]
    for record in records:
        record["gate"] = (
            {"passed": False, "checks": {"reference": True}, "relative_changes": {}}
            if record is reference
            else _gate(record["selection_view"], reference["selection_view"], config)
        )
    eligible = [record for record in records if record["gate"]["passed"]]
    selected = (
        min(
            eligible,
            key=lambda item: item["selection_view"][
                "common_weighted_validation_loss"
            ],
        )
        if eligible
        else None
    )
    checkpoint = None
    if selected is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / (
            f"gimbal_{selected['name']}_seed_{config.training_seed}.pt"
        )
        save_gru_checkpoint(
            checkpoint,
            models[selected["name"]],
            metadata={
                "experiment": CONTROL_ACTION_OBJECTIVE_SCHEMA_VERSION,
                "candidate": selected["name"],
                "training_seed": config.training_seed,
                "train_sha256": _sha256(train_path),
                "validation_sha256": _sha256(validation_path),
                "fresh_test_reopened": False,
            },
        )

    return {
        "experiment": CONTROL_ACTION_OBJECTIVE_SCHEMA_VERSION,
        "config": asdict(config),
        "base_model_config": asdict(base_model_config),
        "parameter_count": gru_parameter_count(models[reference["name"]]),
        "datasets": {
            "train": {
                "path": str(train_path),
                "sha256": _sha256(train_path),
                "episodes": train.episode_count,
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
            "fresh_test": {"reopened": False},
        },
        "candidates": records,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected["name"] if selected else None,
        "selected_checkpoint": str(checkpoint) if checkpoint else None,
        "recommendation": (
            "replicate_action_objective_before_closed_loop"
            if selected is not None
            else "retain_consistent_v4"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate hardware-normalized action-aware GRU losses."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_train.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_refinement_validation.npz"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_control_action_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_control_action_development.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_control_action_objective(
        train_path=args.train_data,
        validation_path=args.validation_data,
        checkpoint_directory=args.checkpoint_directory,
        config=ControlActionObjectiveConfig(
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
