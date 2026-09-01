"""Seed-matched replication of the V7 hard midpoint candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

from .adaptive_curriculum_objective import (
    _gate,
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
from .dataset import FEATURE_NAMES, load_gimbal_dataset
from .gru import GRULossConfig, GRUTargetStateModelConfig, save_gru_checkpoint
from .gru_training import GRUTrainingConfig, evaluate_gru, train_gru


MIDPOINT_ADAPTER_REPLICATION_SCHEMA_VERSION = (
    "gimbal_midpoint_adapter_v7_replication_v1"
)


@dataclass(frozen=True)
class MidpointAdapterReplicationConfig:
    training_seeds: tuple[int, ...] = (17, 29, 43)
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
    maximum_per_seed_action_regression_fraction: float = 0.02
    minimum_improving_seed_count: int = 2
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("replication seeds must be non-empty and unique")
        for name in (
            "epochs",
            "batch_size",
            "hidden_dim",
            "embedding_dim",
            "minimum_training_episodes",
            "minimum_validation_episodes",
            "minimum_improving_seed_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_improving_seed_count > len(self.training_seeds):
            raise ValueError("improving-seed requirement exceeds seed count")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
        ):
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
            "maximum_per_seed_action_regression_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def evaluate_midpoint_adapter_replication(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_directory: str | Path,
    config: MidpointAdapterReplicationConfig | None = None,
) -> dict[str, Any]:
    """Replicate V7 without opening any test or closed-loop block."""

    config = config or MidpointAdapterReplicationConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("V7 replication training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("V7 replication validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("V7 replication dataset seeds overlap")
    horizons_s = train.manifest.prediction_horizons_s
    if horizons_s != validation.manifest.prediction_horizons_s:
        raise ValueError("V7 replication horizons differ")

    criticality = compute_control_criticality(
        validation,
        config=config.criticality,
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
    base_model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=horizons_s,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    )
    candidates = (
        (
            "v4_reference",
            base_model_config,
            GRULossConfig(dynamic_consistency_weight=25.0),
        ),
        (
            "midpoint_state_reference",
            replace(
                base_model_config,
                mean_parameterization="integrated_midpoint",
            ),
            GRULossConfig(),
        ),
    )
    future_index = _future_index(horizons_s)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    seed_records = []
    for seed in config.training_seeds:
        training_config = GRUTrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=seed,
            device=config.device,
        )
        pair: dict[str, Any] = {}
        for name, model_config, loss_config in candidates:
            training = train_gru(
                train,
                validation,
                ObservationProfile.DISTURBANCE_AWARE,
                model_config=model_config,
                training_config=training_config,
                loss_config=loss_config,
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
                evaluation_mask=criticality.critical_mask,
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
                evaluation_mask=criticality.critical_mask,
            )
            checkpoint = checkpoint_directory / (
                f"gimbal_v7_{name}_seed_{seed}.pt"
            )
            save_gru_checkpoint(
                checkpoint,
                training.model,
                metadata={
                    "experiment": MIDPOINT_ADAPTER_REPLICATION_SCHEMA_VERSION,
                    "candidate": name,
                    "training_seed": seed,
                    "train_sha256": _sha256(train_path),
                    "validation_sha256": _sha256(validation_path),
                    "test_opened": False,
                },
            )
            pair[name] = {
                "model_config": asdict(model_config),
                "loss_config": asdict(loss_config),
                "best_epoch": training.best_epoch,
                "training_history": [
                    asdict(record) for record in training.history
                ],
                "standard_validation": asdict(standard),
                "critical_validation": asdict(critical),
                "common_adapter_validation": asdict(common),
                "critical_common_adapter_validation": asdict(
                    critical_common
                ),
                "selection_view": _selection_metrics(
                    standard,
                    critical,
                    common,
                    critical_common,
                    future_index,
                ),
                "checkpoint": str(checkpoint),
            }
        reference = pair["v4_reference"]["selection_view"]
        candidate = pair["midpoint_state_reference"]["selection_view"]
        pair["gate"] = _gate(candidate, reference, config)
        pair["relative_changes"] = {
            metric: _relative_change(candidate[metric], reference[metric])
            for metric in reference
            if metric != "common_weighted_validation_loss"
        }
        seed_records.append({"training_seed": seed, **pair})

    names = ("v4_reference", "midpoint_state_reference")
    metric_names = tuple(seed_records[0][names[0]]["selection_view"])
    distributions = {
        metric: {
            name: _distribution(
                [
                    float(record[name]["selection_view"][metric])
                    for record in seed_records
                ]
            )
            for name in names
        }
        for metric in metric_names
    }
    mean_reference = {
        metric: distributions[metric]["v4_reference"]["mean"]
        for metric in metric_names
    }
    mean_candidate = {
        metric: distributions[metric]["midpoint_state_reference"]["mean"]
        for metric in metric_names
    }
    mean_gate = _gate(mean_candidate, mean_reference, config)
    action_key = "adaptive_position_action_rmse_normalized"
    critical_action_key = "critical_adaptive_position_action_rmse_normalized"
    action_changes = [
        record["relative_changes"][action_key] for record in seed_records
    ]
    critical_action_changes = [
        record["relative_changes"][critical_action_key]
        for record in seed_records
    ]
    excluded_checks = {"adaptive_position_action", "critical_adaptive_position_action"}
    per_seed_state_safety = {
        str(record["training_seed"]): all(
            passed
            for name, passed in record["gate"]["checks"].items()
            if name not in excluded_checks
        )
        for record in seed_records
    }
    checks = {
        "mean_gate": mean_gate["passed"],
        "per_seed_state_safety": all(per_seed_state_safety.values()),
        "per_seed_action_safety": max(action_changes)
        <= config.maximum_per_seed_action_regression_fraction,
        "per_seed_critical_action_safety": max(critical_action_changes)
        <= config.maximum_per_seed_action_regression_fraction,
        "action_improving_seed_count": sum(
            change < 0.0 for change in action_changes
        )
        >= config.minimum_improving_seed_count,
        "critical_action_improving_seed_count": sum(
            change < 0.0 for change in critical_action_changes
        )
        >= config.minimum_improving_seed_count,
    }
    passed = all(checks.values())
    return {
        "experiment": MIDPOINT_ADAPTER_REPLICATION_SCHEMA_VERSION,
        "config": asdict(config),
        "adapter_config": asdict(adapter),
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
                    criticality,
                    config=config.criticality,
                ),
            },
            "test": {"opened": False},
        },
        "training_seed_results": seed_records,
        "metric_distributions": distributions,
        "mean_gate": mean_gate,
        "gate": {
            "passed": passed,
            "checks": checks,
            "per_seed_state_safety": per_seed_state_safety,
            "action_improving_seed_count": sum(
                change < 0.0 for change in action_changes
            ),
            "critical_action_improving_seed_count": sum(
                change < 0.0 for change in critical_action_changes
            ),
            "per_seed_action_relative_changes": action_changes,
            "per_seed_critical_action_relative_changes": (
                critical_action_changes
            ),
        },
        "recommendation": (
            "freeze_midpoint_v7_for_fresh_test"
            if passed
            else "do_not_open_fresh_test"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the V7 midpoint model against seed-matched V4."
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
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_replication_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_replication.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_midpoint_adapter_replication(
        train_path=args.train_data,
        validation_path=args.validation_data,
        checkpoint_directory=args.checkpoint_directory,
        config=MidpointAdapterReplicationConfig(
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
        f"replication={'PASS' if result['gate']['passed'] else 'FAIL'}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
