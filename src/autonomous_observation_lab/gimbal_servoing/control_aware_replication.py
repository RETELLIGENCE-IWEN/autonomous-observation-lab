"""Matched multi-seed replication of the selected consistent GRU objective."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

from .config import ObservationProfile
from .control_aware_predictor import _candidate_gate, _future_index, _selection_view
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


CONTROL_AWARE_REPLICATION_SCHEMA_VERSION = (
    "gimbal_control_aware_consistency_replication_v4_v1"
)


@dataclass(frozen=True)
class ControlAwareReplicationConfig:
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
    maximum_standard_error_regression_fraction: float = 0.02
    minimum_critical_bearing_improvement_fraction: float = 0.01
    maximum_critical_rate_regression_fraction: float = 0.02
    minimum_consistency_improvement_fraction: float = 0.02
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("replication training seeds must be non-empty and unique")
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


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def evaluate_control_aware_replication(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_directory: str | Path,
    config: ControlAwareReplicationConfig | None = None,
) -> dict[str, Any]:
    """Replicate consistency gains against a seed-matched expanded baseline."""

    config = config or ControlAwareReplicationConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    if train.episode_count < config.minimum_training_episodes:
        raise ValueError("control-aware replication training set is too small")
    if validation.episode_count < config.minimum_validation_episodes:
        raise ValueError("control-aware replication validation set is too small")
    if set(train.manifest.seeds) & set(validation.manifest.seeds):
        raise ValueError("control-aware replication dataset seeds overlap")
    horizons_s = train.manifest.prediction_horizons_s
    if horizons_s != validation.manifest.prediction_horizons_s:
        raise ValueError("control-aware replication horizons differ")

    validation_criticality = compute_control_criticality(
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
    model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=horizons_s,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    )
    future_index = _future_index(horizons_s)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    records = []
    parameter_count = None
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
        pair = {}
        for name, loss_config in (
            ("baseline_expanded", GRULossConfig()),
            ("consistent_v4", consistent_loss),
        ):
            training = train_gru(
                train,
                validation,
                ObservationProfile.DISTURBANCE_AWARE,
                model_config=model_config,
                training_config=training_config,
                loss_config=loss_config,
            )
            if parameter_count is None:
                parameter_count = gru_parameter_count(training.model)
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
            weighted = evaluate_gru(
                training.model,
                validation,
                ObservationProfile.DISTURBANCE_AWARE,
                batch_size=config.batch_size,
                device=config.device,
                loss_config=common_loss,
                label_weights=validation_criticality.weights,
            )
            view = _selection_view(
                standard,
                critical,
                weighted,
                future_index,
            )
            item = {
                "best_epoch": training.best_epoch,
                "training_history": [asdict(value) for value in training.history],
                "standard_validation": asdict(standard),
                "critical_validation": asdict(critical),
                "common_weighted_validation": asdict(weighted),
                "selection_view": view,
            }
            checkpoint = checkpoint_directory / f"gimbal_{name}_seed_{seed}.pt"
            save_gru_checkpoint(
                checkpoint,
                training.model,
                metadata={
                    "experiment": CONTROL_AWARE_REPLICATION_SCHEMA_VERSION,
                    "candidate": name,
                    "training_seed": seed,
                    "train_sha256": _sha256(train_path),
                    "validation_sha256": _sha256(validation_path),
                    "test_opened": False,
                },
            )
            item["checkpoint"] = str(checkpoint)
            pair[name] = item
        pair["gate"] = _candidate_gate(
            pair["consistent_v4"]["selection_view"],
            pair["baseline_expanded"]["selection_view"],
            config,  # structurally matches the gate fields
        )
        records.append({"training_seed": seed, **pair})

    metric_names = tuple(records[0]["consistent_v4"]["selection_view"])
    distributions = {}
    for metric in metric_names:
        distributions[metric] = {
            controller: _distribution(
                [
                    float(record[controller]["selection_view"][metric])
                    for record in records
                ]
            )
            for controller in ("baseline_expanded", "consistent_v4")
        }
    passed = all(record["gate"]["passed"] for record in records)
    return {
        "experiment": CONTROL_AWARE_REPLICATION_SCHEMA_VERSION,
        "config": asdict(config),
        "model_config": asdict(model_config),
        "parameter_count": parameter_count,
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
            "test": {"opened": False},
        },
        "training_seed_results": records,
        "metric_distributions": distributions,
        "gate": {
            "passed": passed,
            "per_training_seed": {
                str(record["training_seed"]): record["gate"]["passed"]
                for record in records
            },
        },
        "recommendation": (
            "freeze_consistent_v4_for_fresh_test"
            if passed
            else "do_not_open_fresh_test"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the consistent V4 GRU against matched baselines."
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
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_replication_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_replication.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_control_aware_replication(
        train_path=args.train_data,
        validation_path=args.validation_data,
        checkpoint_directory=args.checkpoint_directory,
        config=ControlAwareReplicationConfig(
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
