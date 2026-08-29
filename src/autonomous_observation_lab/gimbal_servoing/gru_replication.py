"""Multi-initialization replication for disturbance-aware gimbal GRUs.

This module requires the optional ``learning`` dependency.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import GimbalCommandMode, ObservationProfile
from .dataset import (
    FEATURE_NAMES,
    TARGET_NAMES,
    load_gimbal_dataset,
    validate_disjoint_seed_blocks,
)
from .gru import (
    GRUTargetStateModelConfig,
    gru_parameter_count,
    save_gru_checkpoint,
)
from .gru_control import (
    GRUControlEvaluationConfig,
    _aggregate_runs,
    _baseline_runs,
    _command_modes_from_behaviors,
    _control_cost,
    _learned_run,
    _paired_comparison,
    _scenario_variants,
    _select_horizon,
)
from .gru_training import (
    GRUTrainingConfig,
    evaluate_gru,
    train_gru,
)


GRU_REPLICATION_SCHEMA_VERSION = "gimbal_gru_o2_replication_v1"


@dataclass(frozen=True)
class GRUReplicationConfig:
    training_seeds: tuple[int, ...] = (17, 29, 43)
    epochs: int = 50
    batch_size: int = 24
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    hidden_dim: int = 64
    embedding_dim: int = 64
    device: str = "cpu"
    control: GRUControlEvaluationConfig = GRUControlEvaluationConfig()

    def __post_init__(self) -> None:
        if (
            not self.training_seeds
            or len(self.training_seeds) != len(set(self.training_seeds))
        ):
            raise ValueError("training seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.training_seeds):
            raise ValueError("training seeds must be non-negative")
        GRUTrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip_norm=self.gradient_clip_norm,
            seed=self.training_seeds[0],
            device=self.device,
        )
        if self.hidden_dim <= 0 or self.embedding_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.control.device != self.device:
            raise ValueError("training and control devices must match")
        if not self.control.include_position_diagnostic:
            raise ValueError("replication requires rate and position control")


def _metric_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    metrics = aggregate["mean_metrics"]
    return {
        "mean_absolute_error_deg": metrics["mean_absolute_error_deg"],
        "p95_absolute_error_deg": metrics["p95_absolute_error_deg"],
        "loss_of_view_fraction": metrics["loss_of_view_fraction"],
        "rate_saturation_fraction": metrics["rate_saturation_fraction"],
        "command_rms_normalized": metrics["command_rms_normalized"],
        "command_variation_per_s": metrics["command_variation_per_s"],
        "mean_control_cost": aggregate["mean_control_cost"],
        "total_unrecovered_loss_events": aggregate[
            "total_unrecovered_loss_events"
        ],
        "event_weighted_mean_recovery_time_s": aggregate[
            "event_weighted_mean_recovery_time_s"
        ],
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": (
            float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _replication_aggregate(
    seed_results: list[dict[str, Any]],
    baseline_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for mode in ("rate", "position"):
        controller = f"gru_o2_{mode}"
        reference = f"analytical_{mode}"
        reference_metrics = baseline_summary[reference]
        metric_names = (
            "mean_absolute_error_deg",
            "p95_absolute_error_deg",
            "loss_of_view_fraction",
            "mean_control_cost",
            "rate_saturation_fraction",
            "command_variation_per_s",
        )
        metric_distributions = {}
        delta_distributions = {}
        for metric in metric_names:
            values = [
                float(seed["closed_loop_summary"][controller][metric])
                for seed in seed_results
            ]
            deltas = [value - float(reference_metrics[metric]) for value in values]
            metric_distributions[metric] = _distribution(values)
            delta_distributions[metric] = {
                **_distribution(deltas),
                "all_training_seeds_improve": all(delta < 0.0 for delta in deltas),
            }
        selected_horizons = [
            float(seed["selected_horizons"][mode]["horizon_s"])
            for seed in seed_results
        ]
        result[mode] = {
            "training_seed_count": len(seed_results),
            "analytical_reference": reference_metrics,
            "learned_metric_distribution": metric_distributions,
            "delta_vs_analytical_distribution": delta_distributions,
            "selected_horizon_s_by_training_seed": selected_horizons,
            "selected_horizon_consistent": (
                len(set(selected_horizons)) == 1
            ),
        }
    return result


def run_gru_o2_replication(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    checkpoint_directory: str | Path,
    config: GRUReplicationConfig | None = None,
) -> dict[str, Any]:
    """Train matched O2 initializations and evaluate paired closed-loop tests."""
    config = config or GRUReplicationConfig()
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    test = load_gimbal_dataset(test_path)
    validate_disjoint_seed_blocks(
        (train.manifest, validation.manifest, test.manifest)
    )
    if not (
        train.manifest.feature_names
        == validation.manifest.feature_names
        == test.manifest.feature_names
        == FEATURE_NAMES
    ):
        raise ValueError("dataset feature schemas differ")
    if not (
        train.manifest.prediction_horizons_s
        == validation.manifest.prediction_horizons_s
        == test.manifest.prediction_horizons_s
    ):
        raise ValueError("dataset prediction horizons differ")
    for dataset in (train, validation, test):
        if ObservationProfile.DISTURBANCE_AWARE.value not in (
            dataset.manifest.observation_profiles
        ):
            raise ValueError("dataset is missing the O2 observation profile")
    training_modes = _command_modes_from_behaviors(
        train.manifest.behavior_names
    )
    if set(training_modes) != {
        GimbalCommandMode.RATE.value,
        GimbalCommandMode.POSITION.value,
    }:
        raise ValueError("replication requires rate and position training support")

    validation_variants = _scenario_variants(validation.manifest)
    test_variants = _scenario_variants(test.manifest)
    baseline_runs: dict[str, list[Any]] = {}
    for seed, _scenario_index, scenario in test_variants:
        for run in _baseline_runs(scenario, seed, config.control):
            baseline_runs.setdefault(run.episode.name, []).append(run)
    baseline_aggregates = {
        name: _aggregate_runs(runs) for name, runs in baseline_runs.items()
    }
    baseline_summary = {
        name: _metric_summary(aggregate)
        for name, aggregate in baseline_aggregates.items()
    }

    model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=train.manifest.prediction_horizons_s,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    )
    dataset_hashes = {
        "train": train.manifest.configuration_hash,
        "validation": validation.manifest.configuration_hash,
        "test": test.manifest.configuration_hash,
    }
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for training_seed in config.training_seeds:
        training_config = GRUTrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=training_seed,
            device=config.device,
        )
        training = train_gru(
            train,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            model_config=model_config,
            training_config=training_config,
        )
        open_loop_test = evaluate_gru(
            training.model,
            test,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
        )
        selected_horizons = {}
        selected_indices = {}
        for command_mode in (
            GimbalCommandMode.RATE,
            GimbalCommandMode.POSITION,
        ):
            index, candidates = _select_horizon(
                variants=validation_variants,
                model=training.model,
                profile=ObservationProfile.DISTURBANCE_AWARE,
                command_mode=command_mode,
                evaluation=config.control,
            )
            selected_indices[command_mode] = index
            selected_horizons[
                "rate" if command_mode is GimbalCommandMode.RATE else "position"
            ] = {
                "horizon_index": index,
                "horizon_s": model_config.prediction_horizons_s[index],
                "validation_candidates": candidates,
            }

        learned_runs: dict[str, list[Any]] = {}
        run_records = []
        for seed, scenario_index, scenario in test_variants:
            for command_mode in (
                GimbalCommandMode.RATE,
                GimbalCommandMode.POSITION,
            ):
                run = _learned_run(
                    scenario=scenario,
                    seed=seed,
                    model=training.model,
                    profile=ObservationProfile.DISTURBANCE_AWARE,
                    horizon_index=selected_indices[command_mode],
                    command_mode=command_mode,
                    evaluation=config.control,
                    search_fallback=False,
                )
                learned_runs.setdefault(run.episode.name, []).append(run)
                run_records.append(
                    {
                        "seed": seed,
                        "scenario_index": scenario_index,
                        "scenario_name": scenario.name,
                        "controller": run.episode.name,
                        "tracking_metrics": asdict(run.metrics),
                        "estimator_metrics": (
                            asdict(run.estimator_metrics)
                            if run.estimator_metrics is not None
                            else None
                        ),
                        "control_cost": _control_cost(run),
                    }
                )
        learned_aggregates = {
            name: _aggregate_runs(runs) for name, runs in learned_runs.items()
        }
        closed_loop_summary = {
            name: _metric_summary(aggregate)
            for name, aggregate in learned_aggregates.items()
        }
        paired = {
            "gru_o2_rate": _paired_comparison(
                learned_runs["gru_o2_rate"], baseline_runs["analytical_rate"]
            ),
            "gru_o2_position": _paired_comparison(
                learned_runs["gru_o2_position"],
                baseline_runs["analytical_position"],
            ),
        }
        checkpoint_path = (
            checkpoint_directory / f"gimbal_gru_o2_seed_{training_seed}.pt"
        )
        save_gru_checkpoint(
            checkpoint_path,
            training.model,
            metadata={
                "experiment": GRU_REPLICATION_SCHEMA_VERSION,
                "profile": ObservationProfile.DISTURBANCE_AWARE.value,
                "feature_names": list(FEATURE_NAMES),
                "target_names": list(TARGET_NAMES),
                "dataset_hashes": dataset_hashes,
                "training_config": asdict(training_config),
                "best_epoch": training.best_epoch,
                "best_validation": asdict(training.best_validation),
                "open_loop_test": asdict(open_loop_test),
                "selected_horizons": selected_horizons,
            },
        )
        seed_results.append(
            {
                "training_seed": training_seed,
                "checkpoint": str(checkpoint_path),
                "best_epoch": training.best_epoch,
                "initial_validation": asdict(training.initial_validation),
                "best_validation": asdict(training.best_validation),
                "open_loop_test": asdict(open_loop_test),
                "training_history": [
                    asdict(record) for record in training.history
                ],
                "selected_horizons": selected_horizons,
                "closed_loop_summary": closed_loop_summary,
                "closed_loop_aggregates": learned_aggregates,
                "paired_vs_analytical": paired,
                "runs": run_records,
            }
        )

    return {
        "experiment": GRU_REPLICATION_SCHEMA_VERSION,
        "torch_version": torch.__version__,
        "replication_config": asdict(config),
        "model_config": asdict(model_config),
        "parameter_count_per_model": gru_parameter_count(training.model),
        "dataset_hashes": dataset_hashes,
        "dataset_episodes": {
            "train": train.episode_count,
            "validation": validation.episode_count,
            "test": test.episode_count,
        },
        "validation_variant_count": len(validation_variants),
        "test_variant_count": len(test_variants),
        "baseline_summary": baseline_summary,
        "baseline_aggregates": baseline_aggregates,
        "training_seed_results": seed_results,
        "replication_summary": _replication_aggregate(
            seed_results, baseline_summary
        ),
        "selection_policy": (
            "Each training seed selects horizons independently on validation. "
            "No best training seed is selected; test results are summarized "
            "with training initialization as the replication unit."
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate O2 gimbal GRU training across initializations."
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, action="append", dest="seeds")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    seeds = tuple(args.seeds or GRUReplicationConfig().training_seeds)
    control = replace(GRUControlEvaluationConfig(), device=args.device)
    result = run_gru_o2_replication(
        train_path=args.train_data,
        validation_path=args.validation_data,
        test_path=args.test_data,
        checkpoint_directory=args.checkpoint_directory,
        config=GRUReplicationConfig(
            training_seeds=seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            device=args.device,
            control=control,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["replication_summary"], indent=2))


if __name__ == "__main__":
    main()
