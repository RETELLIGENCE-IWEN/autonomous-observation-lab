"""Development protocol for control-aware, consistent GRU prediction V4."""

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
from .dataset import FEATURE_NAMES, load_gimbal_dataset
from .gru import (
    GRULossConfig,
    GRUTargetStateModelConfig,
    gru_parameter_count,
    load_gru_checkpoint,
    save_gru_checkpoint,
)
from .gru_training import (
    GRUTrainingConfig,
    evaluate_gru,
    train_gru,
)


CONTROL_AWARE_PREDICTOR_SCHEMA_VERSION = (
    "gimbal_control_aware_predictor_v4_development_v1"
)


@dataclass(frozen=True)
class ControlAwareCandidate:
    name: str
    loss: GRULossConfig
    use_criticality_weights: bool


@dataclass(frozen=True)
class ControlAwarePredictorProtocolConfig:
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
    maximum_standard_error_regression_fraction: float = 0.02
    minimum_critical_bearing_improvement_fraction: float = 0.01
    maximum_critical_rate_regression_fraction: float = 0.02
    minimum_consistency_improvement_fraction: float = 0.02
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

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
            raise ValueError("weight_decay must be finite and non-negative")
        for name in (
            "maximum_standard_error_regression_fraction",
            "minimum_critical_bearing_improvement_fraction",
            "maximum_critical_rate_regression_fraction",
            "minimum_consistency_improvement_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def default_control_aware_candidates(
    horizons_s: tuple[float, ...],
) -> tuple[ControlAwareCandidate, ...]:
    horizon_weights = tuple(1.0 for _ in horizons_s)
    if len(horizons_s) >= 2:
        horizon_weights = tuple(
            1.5 if index == 1 else (0.75 if index == 0 else 1.0)
            for index in range(len(horizons_s))
        )
    baseline = GRULossConfig()
    consistent = GRULossConfig(dynamic_consistency_weight=25.0)
    return (
        ControlAwareCandidate("baseline_expanded", baseline, False),
        ControlAwareCandidate("critical_only", baseline, True),
        ControlAwareCandidate("consistency_only", consistent, False),
        ControlAwareCandidate("critical_consistency", consistent, True),
        ControlAwareCandidate(
            "control_focused",
            GRULossConfig(
                bearing_weight=1.0,
                rate_weight=0.75,
                mean_error_weight=0.20,
                dynamic_consistency_weight=25.0,
                horizon_weights=horizon_weights,
            ),
            True,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics_payload(metrics: Any) -> dict[str, Any]:
    return asdict(metrics)


def _future_index(horizons_s: tuple[float, ...]) -> int:
    positive = [
        (index, horizon)
        for index, horizon in enumerate(horizons_s)
        if horizon > 0.0
    ]
    return min(positive, key=lambda item: abs(item[1] - 0.1))[0] if positive else 0


def _selection_view(
    standard: Any,
    critical: Any,
    weighted: Any,
    future_index: int,
) -> dict[str, float]:
    standard_future = standard.per_horizon[future_index]
    critical_future = critical.per_horizon[future_index]
    values = {
        "standard_bearing_rmse_deg": float(standard.bearing_rmse_deg),
        "standard_rate_rmse_deg_s": float(standard.rate_rmse_deg_s),
        "future_bearing_rmse_deg": float(standard_future.bearing_rmse_deg),
        "future_rate_rmse_deg_s": float(standard_future.rate_rmse_deg_s),
        "critical_future_bearing_rmse_deg": float(
            critical_future.bearing_rmse_deg
        ),
        "critical_future_rate_rmse_deg_s": float(
            critical_future.rate_rmse_deg_s
        ),
        "dynamic_consistency_rmse_deg": float(
            standard.dynamic_consistency_rmse_deg
        ),
        "common_weighted_validation_loss": float(weighted.loss),
    }
    return values


def _relative_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return (candidate - reference) / reference


def _candidate_gate(
    candidate: dict[str, float],
    reference: dict[str, float],
    protocol: ControlAwarePredictorProtocolConfig,
) -> dict[str, Any]:
    changes = {
        key: _relative_change(candidate[key], reference[key])
        for key in candidate
        if key != "common_weighted_validation_loss"
    }
    checks = {
        "standard_bearing": changes["standard_bearing_rmse_deg"]
        <= protocol.maximum_standard_error_regression_fraction,
        "standard_rate": changes["standard_rate_rmse_deg_s"]
        <= protocol.maximum_standard_error_regression_fraction,
        "future_bearing": changes["future_bearing_rmse_deg"]
        <= protocol.maximum_standard_error_regression_fraction,
        "critical_future_bearing": changes[
            "critical_future_bearing_rmse_deg"
        ]
        <= -protocol.minimum_critical_bearing_improvement_fraction,
        "critical_future_rate": changes["critical_future_rate_rmse_deg_s"]
        <= protocol.maximum_critical_rate_regression_fraction,
        "dynamic_consistency": changes["dynamic_consistency_rmse_deg"]
        <= -protocol.minimum_consistency_improvement_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def evaluate_control_aware_predictor_development(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    legacy_checkpoint: str | Path,
    checkpoint_directory: str | Path,
    protocol: ControlAwarePredictorProtocolConfig | None = None,
) -> dict[str, Any]:
    """Screen objective ablations without opening a fresh test block."""

    protocol = protocol or ControlAwarePredictorProtocolConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    legacy_checkpoint = Path(legacy_checkpoint)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < protocol.minimum_training_episodes:
        raise ValueError("control-aware development training set is too small")
    if validation.episode_count < protocol.minimum_validation_episodes:
        raise ValueError("control-aware development validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("control-aware train and validation seeds overlap")
    horizons_s = train.manifest.prediction_horizons_s
    if horizons_s != validation.manifest.prediction_horizons_s:
        raise ValueError("control-aware dataset horizons differ")

    train_criticality = compute_control_criticality(
        train,
        config=protocol.criticality,
    )
    validation_criticality = compute_control_criticality(
        validation,
        config=protocol.criticality,
    )
    common_loss = GRULossConfig(
        bearing_weight=1.0,
        rate_weight=0.75,
        mean_error_weight=0.20,
        dynamic_consistency_weight=25.0,
        horizon_weights=default_control_aware_candidates(horizons_s)[-1]
        .loss.horizon_weights,
    )
    future_index = _future_index(horizons_s)

    legacy_model, legacy_metadata = load_gru_checkpoint(
        legacy_checkpoint,
        device=protocol.device,
    )
    if legacy_model.config.input_dim != len(FEATURE_NAMES):
        raise ValueError("legacy checkpoint feature schema is incompatible")
    legacy_standard = evaluate_gru(
        legacy_model,
        validation,
        ObservationProfile.DISTURBANCE_AWARE,
        batch_size=protocol.batch_size,
        device=protocol.device,
    )
    legacy_critical = evaluate_gru(
        legacy_model,
        validation,
        ObservationProfile.DISTURBANCE_AWARE,
        batch_size=protocol.batch_size,
        device=protocol.device,
        evaluation_mask=validation_criticality.critical_mask,
    )
    legacy_weighted = evaluate_gru(
        legacy_model,
        validation,
        ObservationProfile.DISTURBANCE_AWARE,
        batch_size=protocol.batch_size,
        device=protocol.device,
        loss_config=common_loss,
        label_weights=validation_criticality.weights,
    )

    training_config = GRUTrainingConfig(
        epochs=protocol.epochs,
        batch_size=protocol.batch_size,
        learning_rate=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
        gradient_clip_norm=protocol.gradient_clip_norm,
        seed=protocol.training_seed,
        device=protocol.device,
    )
    model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=horizons_s,
        hidden_dim=protocol.hidden_dim,
        embedding_dim=protocol.embedding_dim,
    )
    candidates = []
    trained_models = {}
    for candidate in default_control_aware_candidates(horizons_s):
        training = train_gru(
            train,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            model_config=model_config,
            training_config=training_config,
            loss_config=candidate.loss,
            training_label_weights=(
                train_criticality.weights
                if candidate.use_criticality_weights
                else None
            ),
            validation_label_weights=(
                validation_criticality.weights
                if candidate.use_criticality_weights
                else None
            ),
        )
        standard = evaluate_gru(
            training.model,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=protocol.batch_size,
            device=protocol.device,
        )
        critical = evaluate_gru(
            training.model,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=protocol.batch_size,
            device=protocol.device,
            evaluation_mask=validation_criticality.critical_mask,
        )
        weighted = evaluate_gru(
            training.model,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=protocol.batch_size,
            device=protocol.device,
            loss_config=common_loss,
            label_weights=validation_criticality.weights,
        )
        view = _selection_view(standard, critical, weighted, future_index)
        candidates.append(
            {
                "name": candidate.name,
                "loss_config": asdict(candidate.loss),
                "use_criticality_weights": candidate.use_criticality_weights,
                "best_epoch": training.best_epoch,
                "training_history": [asdict(item) for item in training.history],
                "standard_validation": _metrics_payload(standard),
                "critical_validation": _metrics_payload(critical),
                "common_weighted_validation": _metrics_payload(weighted),
                "selection_view": view,
            }
        )
        trained_models[candidate.name] = training.model

    baseline = next(
        item for item in candidates if item["name"] == "baseline_expanded"
    )
    for item in candidates:
        item["gate"] = (
            {
                "passed": False,
                "checks": {"reference": True},
                "relative_changes": {},
            }
            if item["name"] == "baseline_expanded"
            else _candidate_gate(
                item["selection_view"],
                baseline["selection_view"],
                protocol,
            )
        )
    eligible = [item for item in candidates if item["gate"]["passed"]]
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
            f"gimbal_control_aware_{selected['name']}_seed_"
            f"{protocol.training_seed}.pt"
        )
        save_gru_checkpoint(
            checkpoint,
            trained_models[selected["name"]],
            metadata={
                "experiment": CONTROL_AWARE_PREDICTOR_SCHEMA_VERSION,
                "candidate": selected["name"],
                "training_seed": protocol.training_seed,
                "train_sha256": _sha256(train_path),
                "validation_sha256": _sha256(validation_path),
                "test_opened": False,
            },
        )

    return {
        "experiment": CONTROL_AWARE_PREDICTOR_SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "model_config": asdict(model_config),
        "parameter_count": gru_parameter_count(
            trained_models["baseline_expanded"]
        ),
        "datasets": {
            "train": {
                "path": str(train_path),
                "sha256": _sha256(train_path),
                "episodes": train.episode_count,
                "seeds": list(train.manifest.seeds),
                "criticality": control_criticality_report(
                    train,
                    train_criticality,
                    config=protocol.criticality,
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
                    config=protocol.criticality,
                ),
            },
            "test": {"opened": False},
        },
        "legacy_reference": {
            "checkpoint": str(legacy_checkpoint),
            "metadata": legacy_metadata,
            "standard_validation": _metrics_payload(legacy_standard),
            "critical_validation": _metrics_payload(legacy_critical),
            "common_weighted_validation": _metrics_payload(legacy_weighted),
            "selection_view": _selection_view(
                legacy_standard,
                legacy_critical,
                legacy_weighted,
                future_index,
            ),
        },
        "candidates": candidates,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected["name"] if selected else None,
        "selected_checkpoint": str(checkpoint) if checkpoint else None,
        "recommendation": (
            "replicate_selected_objective_before_test"
            if selected is not None
            else "revise_control_aware_objective_without_opening_test"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen control-aware consistent GRU objectives."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_train.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_validation.npz"),
    )
    parser.add_argument(
        "--legacy-checkpoint",
        type=Path,
        default=Path(
            "artifacts/gimbal_o2_replication_checkpoints/"
            "gimbal_gru_o2_seed_17.pt"
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
        default=Path("artifacts/gimbal_control_aware_development.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_control_aware_predictor_development(
        train_path=args.train_data,
        validation_path=args.validation_data,
        legacy_checkpoint=args.legacy_checkpoint,
        checkpoint_directory=args.checkpoint_directory,
        protocol=ControlAwarePredictorProtocolConfig(
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
