"""Matched O0/O1/O2 experiments for the causal gimbal GRU."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from .config import ObservationProfile
from .dataset import (
    TARGET_NAMES,
    load_gimbal_dataset,
    validate_disjoint_seed_blocks,
)
from .gru import (
    GRUTargetStateModelConfig,
    gru_parameter_count,
    save_gru_checkpoint,
)
from .gru_training import (
    GRUTrainingConfig,
    constant_velocity_predictions,
    evaluate_constant_velocity_baseline,
    evaluate_gru,
    train_gru,
)


PROFILE_SHORT_NAMES = {
    ObservationProfile.VISION_ONLY: "o0",
    ObservationProfile.SERVO_AWARE: "o1",
    ObservationProfile.DISTURBANCE_AWARE: "o2",
}


def run_gru_profile_comparison(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    checkpoint_directory: str | Path,
    profiles: tuple[ObservationProfile, ...] = tuple(ObservationProfile),
    hidden_dim: int = 64,
    embedding_dim: int = 64,
    training_config: GRUTrainingConfig | None = None,
) -> dict[str, object]:
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("profiles must be non-empty and unique")
    training_config = training_config or GRUTrainingConfig()
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    test = load_gimbal_dataset(test_path)
    validate_disjoint_seed_blocks(
        (train.manifest, validation.manifest, test.manifest)
    )
    if not (
        train.manifest.prediction_horizons_s
        == validation.manifest.prediction_horizons_s
        == test.manifest.prediction_horizons_s
    ):
        raise ValueError("dataset prediction horizons differ")
    if not (
        train.manifest.feature_names
        == validation.manifest.feature_names
        == test.manifest.feature_names
    ):
        raise ValueError("dataset feature schemas differ")
    for dataset in (train, validation, test):
        missing = {
            profile.value for profile in profiles
        } - set(dataset.manifest.observation_profiles)
        if missing:
            raise ValueError(f"dataset is missing profiles: {sorted(missing)}")

    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    model_config = GRUTargetStateModelConfig(
        input_dim=len(train.manifest.feature_names),
        prediction_horizons_s=train.manifest.prediction_horizons_s,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    )
    profile_results: dict[str, object] = {}
    for profile in profiles:
        training = train_gru(
            train,
            validation,
            profile,
            model_config=model_config,
            training_config=training_config,
        )
        learned_test = evaluate_gru(
            training.model,
            test,
            profile,
            batch_size=training_config.batch_size,
            device=training_config.device,
        )
        analytical_test = evaluate_constant_velocity_baseline(test, profile)
        _, _, analytical_mask, _ = constant_velocity_predictions(test, profile)
        learned_matched = evaluate_gru(
            training.model,
            test,
            profile,
            batch_size=training_config.batch_size,
            device=training_config.device,
            evaluation_mask=analytical_mask,
        )
        learned_outside_analytical_support = evaluate_gru(
            training.model,
            test,
            profile,
            batch_size=training_config.batch_size,
            device=training_config.device,
            evaluation_mask=~analytical_mask,
        )
        checkpoint_path = checkpoint_directory / (
            f"gimbal_gru_{PROFILE_SHORT_NAMES[profile]}.pt"
        )
        record: dict[str, object] = {
            "best_epoch": training.best_epoch,
            "initial_validation": asdict(training.initial_validation),
            "best_validation": asdict(training.best_validation),
            "learned_test": asdict(learned_test),
            "learned_test_on_analytical_support": asdict(learned_matched),
            "learned_test_outside_analytical_support": asdict(
                learned_outside_analytical_support
            ),
            "constant_velocity_test": asdict(analytical_test),
            "history": [asdict(item) for item in training.history],
            "checkpoint": str(checkpoint_path),
        }
        save_gru_checkpoint(
            checkpoint_path,
            training.model,
            metadata={
                "experiment": "gimbal_gru_observation_profile_comparison",
                "profile": profile.value,
                "feature_names": list(train.manifest.feature_names),
                "target_names": list(TARGET_NAMES),
                "dataset_hashes": {
                    "train": train.manifest.configuration_hash,
                    "validation": validation.manifest.configuration_hash,
                    "test": test.manifest.configuration_hash,
                },
                "training_config": asdict(training_config),
                "best_epoch": training.best_epoch,
                "best_validation": record["best_validation"],
                "learned_test": record["learned_test"],
            },
        )
        profile_results[profile.value] = record

    summary = {}
    for profile in profiles:
        metrics = profile_results[profile.value]
        assert isinstance(metrics, dict)
        test_metrics = metrics["learned_test"]
        assert isinstance(test_metrics, dict)
        summary[profile.value] = {
            "bearing_rmse_deg": test_metrics["bearing_rmse_deg"],
            "rate_rmse_deg_s": test_metrics["rate_rmse_deg_s"],
            "bearing_two_sigma_coverage": test_metrics[
                "bearing_two_sigma_coverage"
            ],
            "rate_two_sigma_coverage": test_metrics[
                "rate_two_sigma_coverage"
            ],
            "availability_fraction": test_metrics["availability_fraction"],
        }

    return {
        "experiment": "gimbal_gru_observation_profile_comparison",
        "torch_version": torch.__version__,
        "profiles": [profile.value for profile in profiles],
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "parameter_count_per_model": gru_parameter_count(training.model),
        "dataset_hashes": {
            "train": train.manifest.configuration_hash,
            "validation": validation.manifest.configuration_hash,
            "test": test.manifest.configuration_hash,
        },
        "dataset_episodes": {
            "train": train.episode_count,
            "validation": validation.episode_count,
            "test": test.episode_count,
        },
        "privileged_target_state_ceiling": {
            "availability_fraction": 1.0,
            "bearing_rmse_deg": 0.0,
            "rate_rmse_deg_s": 0.0,
        },
        "oracle_control_ceilings": [
            asdict(record) for record in test.manifest.oracle_ceilings
        ],
        "summary": summary,
        "profile_results": profile_results,
        "interpretation": (
            "All profiles share trajectories, behavior actions, labels, model "
            "architecture, initialization seed, and optimization settings."
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train matched O0/O1/O2 causal gimbal GRUs."
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.value for profile in ObservationProfile],
        dest="profiles",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    requested_profiles = args.profiles or [
        profile.value for profile in ObservationProfile
    ]
    profiles = tuple(
        ObservationProfile(value)
        for value in requested_profiles
    )
    result = run_gru_profile_comparison(
        train_path=args.train_data,
        validation_path=args.validation_data,
        test_path=args.test_data,
        checkpoint_directory=args.checkpoint_directory,
        profiles=profiles,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        training_config=GRUTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip_norm=args.gradient_clip,
            seed=args.seed,
            device=args.device,
        ),
    )
    text = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
