"""Fresh development refinement of control-critical supervision strength."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .config import ObservationProfile
from .control_aware_predictor import _future_index, _selection_view
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


CRITICAL_WEIGHT_REFINEMENT_SCHEMA_VERSION = (
    "gimbal_critical_weight_refinement_v41_development_v1"
)


@dataclass(frozen=True)
class CriticalWeightRefinementConfig:
    weighting_strengths: tuple[float, ...] = (0.0, 0.20, 0.35, 0.50)
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
    minimum_critical_bearing_improvement_fraction: float = 0.01
    maximum_critical_rate_regression_fraction: float = 0.02
    maximum_consistency_regression_fraction: float = 0.02
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        if not self.weighting_strengths or self.weighting_strengths[0] != 0.0:
            raise ValueError("refinement strengths must begin with zero")
        if len(set(self.weighting_strengths)) != len(self.weighting_strengths):
            raise ValueError("refinement strengths must be unique")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.weighting_strengths
        ):
            raise ValueError("refinement strengths must be in [0, 1]")
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(
    candidate: dict[str, float],
    reference: dict[str, float],
    config: CriticalWeightRefinementConfig,
) -> dict[str, Any]:
    changes = {
        key: (candidate[key] - reference[key]) / reference[key]
        for key in reference
        if key != "common_weighted_validation_loss" and reference[key] != 0.0
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
        <= -config.minimum_critical_bearing_improvement_fraction,
        "critical_future_rate": changes["critical_future_rate_rmse_deg_s"]
        <= config.maximum_critical_rate_regression_fraction,
        "dynamic_consistency": changes["dynamic_consistency_rmse_deg"]
        <= config.maximum_consistency_regression_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def evaluate_critical_weight_refinement(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_directory: str | Path,
    config: CriticalWeightRefinementConfig | None = None,
) -> dict[str, Any]:
    """Select supervision strength on a new development seed block."""

    config = config or CriticalWeightRefinementConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("critical-weight refinement training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("critical-weight refinement validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("critical-weight train and validation seeds overlap")
    horizons_s = train.manifest.prediction_horizons_s
    if horizons_s != validation.manifest.prediction_horizons_s:
        raise ValueError("critical-weight dataset horizons differ")

    full_train = compute_control_criticality(train, config=config.criticality)
    full_validation = compute_control_criticality(
        validation,
        config=config.criticality,
    )
    consistent_loss = GRULossConfig(dynamic_consistency_weight=25.0)
    common_loss = GRULossConfig(
        bearing_weight=1.0,
        rate_weight=0.75,
        mean_error_weight=0.20,
        dynamic_consistency_weight=25.0,
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
    model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=horizons_s,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    )
    future_index = _future_index(horizons_s)
    records = []
    models = {}
    for strength in config.weighting_strengths:
        criticality_config = replace(
            config.criticality,
            weighting_strength=strength,
        )
        train_criticality = compute_control_criticality(
            train,
            config=criticality_config,
        )
        validation_criticality = compute_control_criticality(
            validation,
            config=criticality_config,
        )
        training = train_gru(
            train,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            model_config=model_config,
            training_config=training_config,
            loss_config=consistent_loss,
            training_label_weights=(
                train_criticality.weights if strength > 0.0 else None
            ),
            validation_label_weights=(
                validation_criticality.weights if strength > 0.0 else None
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
            evaluation_mask=full_validation.critical_mask,
        )
        weighted = evaluate_gru(
            training.model,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
            loss_config=common_loss,
            label_weights=full_validation.weights,
        )
        name = f"critical_strength_{strength:.2f}".replace(".", "p")
        records.append(
            {
                "name": name,
                "weighting_strength": strength,
                "best_epoch": training.best_epoch,
                "training_history": [asdict(item) for item in training.history],
                "standard_validation": asdict(standard),
                "critical_validation": asdict(critical),
                "common_weighted_validation": asdict(weighted),
                "selection_view": _selection_view(
                    standard,
                    critical,
                    weighted,
                    future_index,
                ),
            }
        )
        models[name] = training.model

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
                "experiment": CRITICAL_WEIGHT_REFINEMENT_SCHEMA_VERSION,
                "candidate": selected["name"],
                "weighting_strength": selected["weighting_strength"],
                "training_seed": config.training_seed,
                "test_opened": False,
            },
        )

    return {
        "experiment": CRITICAL_WEIGHT_REFINEMENT_SCHEMA_VERSION,
        "config": asdict(config),
        "model_config": asdict(model_config),
        "parameter_count": gru_parameter_count(models[reference["name"]]),
        "datasets": {
            "train": {
                "path": str(train_path),
                "sha256": _sha256(train_path),
                "episodes": train.episode_count,
                "criticality": control_criticality_report(
                    train,
                    full_train,
                    config=config.criticality,
                ),
            },
            "validation": {
                "path": str(validation_path),
                "sha256": _sha256(validation_path),
                "episodes": validation.episode_count,
                "seeds": list(validation.manifest.seeds),
                "criticality": control_criticality_report(
                    validation,
                    full_validation,
                    config=config.criticality,
                ),
            },
            "test": {"opened": False},
        },
        "candidates": records,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected["name"] if selected else None,
        "selected_checkpoint": str(checkpoint) if checkpoint else None,
        "recommendation": (
            "replicate_refined_objective_before_test"
            if selected is not None
            else "retain_unweighted_consistency_objective"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine critical-state supervision on fresh development data."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_train.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path(
            "artifacts/gimbal_control_aware_refinement_validation.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_critical_weight_refinement.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_critical_weight_refinement(
        train_path=args.train_data,
        validation_path=args.validation_data,
        checkpoint_directory=args.checkpoint_directory,
        config=CriticalWeightRefinementConfig(
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
        f"eligible={result['eligible_candidate_count']}; "
        f"selected={result['selected_candidate']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
