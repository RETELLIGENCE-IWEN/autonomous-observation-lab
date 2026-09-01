"""Frozen V6 ablation for critical sampling and adapter-aware supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .adaptive_position_v21 import default_visibility_risk_candidates
from .config import ObservationProfile
from .control_aware_predictor import (
    _future_index,
    _relative_change,
    _selection_view,
)
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
from .dataset import FEATURE_NAMES, load_gimbal_dataset
from .gru import (
    GRULossConfig,
    GRUTargetStateModelConfig,
    gru_parameter_count,
    save_gru_checkpoint,
)
from .gru_training import GRUTrainingConfig, evaluate_gru, train_gru


ADAPTIVE_CURRICULUM_OBJECTIVE_SCHEMA_VERSION = (
    "gimbal_adaptive_curriculum_objective_v6_development_v1"
)


def selected_adaptive_position_v21_config():
    """Return the frozen preview-125 V2.1 controller configuration."""

    return next(
        candidate.controller
        for candidate in default_visibility_risk_candidates()
        if candidate.name == "preview_125"
    )


@dataclass(frozen=True)
class AdaptiveCurriculumCandidate:
    name: str
    adaptive_position_action_weight: float
    use_critical_episode_curriculum: bool
    dynamic_consistency_weight: float = 25.0
    mean_parameterization: str = "independent"
    curriculum_concentration_strength: float | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.adaptive_position_action_weight)
            or self.adaptive_position_action_weight < 0.0
        ):
            raise ValueError(
                "adaptive position action weight must be finite and non-negative"
            )
        if (
            not math.isfinite(self.dynamic_consistency_weight)
            or self.dynamic_consistency_weight < 0.0
        ):
            raise ValueError(
                "dynamic consistency weight must be finite and non-negative"
            )
        if self.mean_parameterization not in {
            "independent",
            "integrated_rate",
            "integrated_midpoint",
        }:
            raise ValueError("unsupported V6 mean parameterization")
        if self.curriculum_concentration_strength is not None:
            if (
                not math.isfinite(self.curriculum_concentration_strength)
                or self.curriculum_concentration_strength < 0.0
            ):
                raise ValueError(
                    "curriculum concentration must be finite and non-negative"
                )
            if not self.use_critical_episode_curriculum:
                raise ValueError(
                    "curriculum concentration requires episode curriculum"
                )

    @property
    def loss(self) -> GRULossConfig:
        return GRULossConfig(
            dynamic_consistency_weight=self.dynamic_consistency_weight,
            adaptive_position_action_weight=(
                self.adaptive_position_action_weight
            ),
            adaptive_position_config=(
                selected_adaptive_position_v21_config()
                if self.adaptive_position_action_weight > 0.0
                else None
            ),
        )


def default_adaptive_curriculum_candidates() -> tuple[
    AdaptiveCurriculumCandidate, ...
]:
    """Predeclare the factorial and one loss-strength sensitivity point."""

    return (
        AdaptiveCurriculumCandidate("v4_reference", 0.0, False),
        AdaptiveCurriculumCandidate("curriculum_only", 0.0, True),
        AdaptiveCurriculumCandidate("adapter_only", 0.10, False),
        AdaptiveCurriculumCandidate("combined_gentle", 0.10, True),
        AdaptiveCurriculumCandidate("combined_moderate", 0.25, True),
    )


@dataclass(frozen=True)
class AdaptiveCurriculumObjectiveConfig:
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
    minimum_adaptive_action_improvement_fraction: float = 0.01
    minimum_critical_adaptive_action_improvement_fraction: float = 0.01
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()
    curriculum: CriticalEpisodeCurriculumConfig = (
        CriticalEpisodeCurriculumConfig()
    )
    candidates: tuple[AdaptiveCurriculumCandidate, ...] = (
        default_adaptive_curriculum_candidates()
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
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight decay must be finite and non-negative")
        for name in (
            "maximum_standard_regression_fraction",
            "maximum_critical_bearing_regression_fraction",
            "maximum_critical_rate_regression_fraction",
            "maximum_consistency_regression_fraction",
            "minimum_adaptive_action_improvement_fraction",
            "minimum_critical_adaptive_action_improvement_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.candidates:
            raise ValueError("at least one V6 candidate is required")
        reference = self.candidates[0]
        if reference.adaptive_position_action_weight != 0.0 or (
            reference.use_critical_episode_curriculum
        ) or reference.dynamic_consistency_weight != 25.0:
            raise ValueError("first V6 candidate must be the V4 reference")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("V6 candidate names must be unique")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_metrics(
    standard: Any,
    critical: Any,
    common: Any,
    critical_common: Any,
    future_index: int,
) -> dict[str, float]:
    view = _selection_view(standard, critical, common, future_index)
    if common.adaptive_position_action_rmse_normalized is None or (
        critical_common.adaptive_position_action_rmse_normalized is None
    ):
        raise ValueError("adapter-aware evaluation did not produce action metrics")
    view.update(
        {
            "adaptive_position_action_rmse_normalized": float(
                common.adaptive_position_action_rmse_normalized
            ),
            "critical_adaptive_position_action_rmse_normalized": float(
                critical_common.adaptive_position_action_rmse_normalized
            ),
        }
    )
    return view


def _gate(
    candidate: dict[str, float],
    reference: dict[str, float],
    config: AdaptiveCurriculumObjectiveConfig,
) -> dict[str, Any]:
    changes = {
        key: _relative_change(candidate[key], value)
        for key, value in reference.items()
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
        "adaptive_position_action": changes[
            "adaptive_position_action_rmse_normalized"
        ]
        <= -config.minimum_adaptive_action_improvement_fraction,
        "critical_adaptive_position_action": changes[
            "critical_adaptive_position_action_rmse_normalized"
        ]
        <= -config.minimum_critical_adaptive_action_improvement_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def evaluate_adaptive_curriculum_objective(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_directory: str | Path,
    config: AdaptiveCurriculumObjectiveConfig | None = None,
) -> dict[str, Any]:
    """Run a frozen development ablation without opening a test block."""

    config = config or AdaptiveCurriculumObjectiveConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("V6 training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("V6 validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("V6 train and validation seeds overlap")
    horizons_s = train.manifest.prediction_horizons_s
    if horizons_s != validation.manifest.prediction_horizons_s:
        raise ValueError("V6 train and validation horizons differ")

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
    adapter = selected_adaptive_position_v21_config()
    common_loss = GRULossConfig(
        bearing_weight=1.0,
        rate_weight=0.75,
        mean_error_weight=0.20,
        dynamic_consistency_weight=25.0,
        adaptive_position_action_weight=1.0,
        adaptive_position_config=adapter,
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
        candidate_curriculum = None
        candidate_curriculum_config = config.curriculum
        if candidate.use_critical_episode_curriculum:
            if candidate.curriculum_concentration_strength is not None:
                candidate_curriculum_config = replace(
                    config.curriculum,
                    concentration_strength=(
                        candidate.curriculum_concentration_strength
                    ),
                )
            candidate_curriculum = compute_critical_episode_curriculum(
                train,
                train_criticality,
                config=candidate_curriculum_config,
            )
        training = train_gru(
            train,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            model_config=model_config,
            training_config=training_config,
            loss_config=candidate.loss,
            training_episode_weights=(
                candidate_curriculum.episode_weights
                if candidate_curriculum is not None
                else None
            ),
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
        )
        critical_common = evaluate_gru(
            training.model,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
            loss_config=common_loss,
            evaluation_mask=validation_criticality.critical_mask,
        )
        records.append(
            {
                "name": candidate.name,
                "loss_config": asdict(candidate.loss),
                "model_config": asdict(model_config),
                "parameter_count": gru_parameter_count(training.model),
                "use_critical_episode_curriculum": (
                    candidate.use_critical_episode_curriculum
                ),
                "curriculum": (
                    critical_episode_curriculum_report(
                        train,
                        candidate_curriculum,
                        config=candidate_curriculum_config,
                    )
                    if candidate_curriculum is not None
                    else None
                ),
                "best_epoch": training.best_epoch,
                "training_history": [
                    asdict(item) for item in training.history
                ],
                "standard_validation": asdict(standard),
                "critical_validation": asdict(critical),
                "common_adapter_validation": asdict(common),
                "critical_common_adapter_validation": asdict(critical_common),
                "selection_view": _selection_metrics(
                    standard,
                    critical,
                    common,
                    critical_common,
                    future_index,
                ),
            }
        )
        models[candidate.name] = training.model

    reference = records[0]
    for record in records:
        record["gate"] = (
            {
                "passed": False,
                "checks": {"reference": True},
                "relative_changes": {},
            }
            if record is reference
            else _gate(
                record["selection_view"],
                reference["selection_view"],
                config,
            )
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

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        checkpoint = checkpoint_directory / (
            f"gimbal_v6_{record['name']}_seed_{config.training_seed}.pt"
        )
        save_gru_checkpoint(
            checkpoint,
            models[record["name"]],
            metadata={
                "experiment": ADAPTIVE_CURRICULUM_OBJECTIVE_SCHEMA_VERSION,
                "candidate": record["name"],
                "training_seed": config.training_seed,
                "train_sha256": _sha256(train_path),
                "validation_sha256": _sha256(validation_path),
                "fresh_test_opened": False,
            },
        )
        record["checkpoint"] = str(checkpoint)

    return {
        "experiment": ADAPTIVE_CURRICULUM_OBJECTIVE_SCHEMA_VERSION,
        "config": asdict(config),
        "model_config": asdict(base_model_config),
        "adapter_config": asdict(adapter),
        "datasets": {
            "train": {
                "path": str(train_path),
                "sha256": _sha256(train_path),
                "episodes": train.episode_count,
                "seeds": list(train.manifest.seeds),
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
                "seeds": list(validation.manifest.seeds),
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
        "selected_checkpoint": (
            selected["checkpoint"] if selected is not None else None
        ),
        "recommendation": (
            "replicate_v6_candidate_before_fresh_test"
            if selected is not None
            else "retain_v4_and_revise_adapter_aware_objective"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate critical sampling and V2.1 adapter-aware GRU loss."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_train.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_curriculum_validation.npz"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_curriculum_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_curriculum_development.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_adaptive_curriculum_objective(
        train_path=args.train_data,
        validation_path=args.validation_data,
        checkpoint_directory=args.checkpoint_directory,
        config=AdaptiveCurriculumObjectiveConfig(
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
