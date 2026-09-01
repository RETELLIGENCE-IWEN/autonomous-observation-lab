"""Frozen fresh-test protocol for the replicated control-aware GRU V4."""

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
from .control_aware_predictor import _future_index, _selection_view
from .control_criticality import (
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from .dataset import (
    FEATURE_NAMES,
    load_gimbal_dataset,
    validate_disjoint_seed_blocks,
)
from .gru import GRULossConfig, load_gru_checkpoint
from .gru_training import evaluate_gru


CONTROL_AWARE_FRESH_TEST_SCHEMA_VERSION = "gimbal_control_aware_v4_fresh_test_v1"


@dataclass(frozen=True)
class ControlAwareFreshTestConfig:
    """Acceptance criteria frozen before the fresh test block is opened."""

    training_seeds: tuple[int, ...] = (17, 29, 43)
    batch_size: int = 24
    minimum_test_episodes: int = 200
    maximum_mean_standard_regression_fraction: float = 0.02
    minimum_mean_critical_bearing_improvement_fraction: float = 0.03
    maximum_mean_critical_rate_regression_fraction: float = 0.02
    minimum_mean_consistency_improvement_fraction: float = 0.25
    maximum_per_seed_critical_bearing_regression_fraction: float = 0.02
    minimum_improving_seed_count: int = 2
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("fresh-test training seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.training_seeds):
            raise ValueError("fresh-test training seeds must be non-negative")
        if self.batch_size <= 0 or self.minimum_test_episodes <= 0:
            raise ValueError("batch size and minimum test episodes must be positive")
        if not 1 <= self.minimum_improving_seed_count <= len(self.training_seeds):
            raise ValueError("minimum improving seed count is invalid")
        for name in (
            "maximum_mean_standard_regression_fraction",
            "minimum_mean_critical_bearing_improvement_fraction",
            "maximum_mean_critical_rate_regression_fraction",
            "minimum_mean_consistency_improvement_fraction",
            "maximum_per_seed_critical_bearing_regression_fraction",
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


def _relative_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return (candidate - reference) / reference


def _checkpoint_path(
    checkpoint_directory: Path,
    candidate: str,
    training_seed: int,
) -> Path:
    return checkpoint_directory / f"gimbal_{candidate}_seed_{training_seed}.pt"


def _verify_checkpoint_metadata(
    metadata: dict[str, Any],
    *,
    candidate: str,
    training_seed: int,
) -> None:
    if metadata.get("experiment") != (
        "gimbal_control_aware_consistency_replication_v4_v1"
    ):
        raise ValueError("checkpoint was not produced by the frozen replication")
    if metadata.get("candidate") != candidate:
        raise ValueError("checkpoint candidate metadata does not match")
    if metadata.get("training_seed") != training_seed:
        raise ValueError("checkpoint training seed metadata does not match")
    if metadata.get("test_opened") is not False:
        raise ValueError("checkpoint was not frozen before fresh testing")


def _acceptance_gate(
    records: list[dict[str, Any]],
    distributions: dict[str, Any],
    config: ControlAwareFreshTestConfig,
) -> dict[str, Any]:
    mean_changes = {
        metric: _relative_change(
            values["consistent_v4"]["mean"],
            values["baseline_expanded"]["mean"],
        )
        for metric, values in distributions.items()
        if metric != "common_weighted_validation_loss"
    }
    critical_changes = [
        record["relative_changes"]["critical_future_bearing_rmse_deg"]
        for record in records
    ]
    improving_count = sum(change < 0.0 for change in critical_changes)
    checks = {
        "mean_standard_bearing": mean_changes["standard_bearing_rmse_deg"]
        <= config.maximum_mean_standard_regression_fraction,
        "mean_standard_rate": mean_changes["standard_rate_rmse_deg_s"]
        <= config.maximum_mean_standard_regression_fraction,
        "mean_future_bearing": mean_changes["future_bearing_rmse_deg"]
        <= config.maximum_mean_standard_regression_fraction,
        "mean_critical_future_bearing": mean_changes[
            "critical_future_bearing_rmse_deg"
        ]
        <= -config.minimum_mean_critical_bearing_improvement_fraction,
        "mean_critical_future_rate": mean_changes[
            "critical_future_rate_rmse_deg_s"
        ]
        <= config.maximum_mean_critical_rate_regression_fraction,
        "mean_dynamic_consistency": mean_changes[
            "dynamic_consistency_rmse_deg"
        ]
        <= -config.minimum_mean_consistency_improvement_fraction,
        "per_seed_critical_bearing_safety": max(critical_changes)
        <= config.maximum_per_seed_critical_bearing_regression_fraction,
        "critical_bearing_seed_count": improving_count
        >= config.minimum_improving_seed_count,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_relative_changes": mean_changes,
        "critical_bearing_improving_seed_count": improving_count,
    }


def evaluate_control_aware_fresh_test(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    checkpoint_directory: str | Path,
    config: ControlAwareFreshTestConfig | None = None,
) -> dict[str, Any]:
    """Open the fresh test block once for frozen, seed-matched checkpoints."""

    config = config or ControlAwareFreshTestConfig()
    train_path = Path(train_path)
    validation_path = Path(validation_path)
    test_path = Path(test_path)
    checkpoint_directory = Path(checkpoint_directory)
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    test = load_gimbal_dataset(test_path)
    validate_disjoint_seed_blocks(
        (train.manifest, validation.manifest, test.manifest)
    )
    if test.episode_count < config.minimum_test_episodes:
        raise ValueError("control-aware fresh test set is too small")
    if not (
        train.manifest.feature_names
        == validation.manifest.feature_names
        == test.manifest.feature_names
        == FEATURE_NAMES
    ):
        raise ValueError("control-aware fresh-test feature schemas differ")
    horizons_s = train.manifest.prediction_horizons_s
    if not (
        horizons_s
        == validation.manifest.prediction_horizons_s
        == test.manifest.prediction_horizons_s
    ):
        raise ValueError("control-aware fresh-test horizons differ")
    if ObservationProfile.DISTURBANCE_AWARE.value not in (
        test.manifest.observation_profiles
    ):
        raise ValueError("fresh test is missing the disturbance-aware profile")

    criticality = compute_control_criticality(test, config=config.criticality)
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
    future_index = _future_index(horizons_s)
    records = []
    checkpoint_hashes = {}
    for training_seed in config.training_seeds:
        pair = {}
        for candidate in ("baseline_expanded", "consistent_v4"):
            checkpoint = _checkpoint_path(
                checkpoint_directory,
                candidate,
                training_seed,
            )
            model, metadata = load_gru_checkpoint(checkpoint, device=config.device)
            _verify_checkpoint_metadata(
                metadata,
                candidate=candidate,
                training_seed=training_seed,
            )
            if (
                model.config.input_dim != len(FEATURE_NAMES)
                or model.config.prediction_horizons_s != horizons_s
            ):
                raise ValueError("checkpoint model schema does not match fresh test")
            standard = evaluate_gru(
                model,
                test,
                ObservationProfile.DISTURBANCE_AWARE,
                batch_size=config.batch_size,
                device=config.device,
            )
            critical = evaluate_gru(
                model,
                test,
                ObservationProfile.DISTURBANCE_AWARE,
                batch_size=config.batch_size,
                device=config.device,
                evaluation_mask=criticality.critical_mask,
            )
            weighted = evaluate_gru(
                model,
                test,
                ObservationProfile.DISTURBANCE_AWARE,
                batch_size=config.batch_size,
                device=config.device,
                loss_config=common_loss,
                label_weights=criticality.weights,
            )
            pair[candidate] = {
                "checkpoint": str(checkpoint),
                "standard_test": asdict(standard),
                "critical_test": asdict(critical),
                "common_weighted_test": asdict(weighted),
                "selection_view": _selection_view(
                    standard,
                    critical,
                    weighted,
                    future_index,
                ),
            }
            checkpoint_hashes[f"{candidate}_seed_{training_seed}"] = _sha256(
                checkpoint
            )
        changes = {
            metric: _relative_change(
                pair["consistent_v4"]["selection_view"][metric],
                pair["baseline_expanded"]["selection_view"][metric],
            )
            for metric in pair["baseline_expanded"]["selection_view"]
            if metric != "common_weighted_validation_loss"
        }
        records.append(
            {
                "training_seed": training_seed,
                **pair,
                "relative_changes": changes,
            }
        )

    metric_names = tuple(records[0]["consistent_v4"]["selection_view"])
    distributions = {
        metric: {
            candidate: _distribution(
                [
                    float(record[candidate]["selection_view"][metric])
                    for record in records
                ]
            )
            for candidate in ("baseline_expanded", "consistent_v4")
        }
        for metric in metric_names
    }
    gate = _acceptance_gate(records, distributions, config)
    return {
        "experiment": CONTROL_AWARE_FRESH_TEST_SCHEMA_VERSION,
        "config": asdict(config),
        "datasets": {
            "train": {
                "path": str(train_path),
                "sha256": _sha256(train_path),
                "seeds": list(train.manifest.seeds),
            },
            "validation": {
                "path": str(validation_path),
                "sha256": _sha256(validation_path),
                "seeds": list(validation.manifest.seeds),
            },
            "test": {
                "path": str(test_path),
                "sha256": _sha256(test_path),
                "seeds": list(test.manifest.seeds),
                "episodes": test.episode_count,
                "opened": True,
                "criticality": control_criticality_report(
                    test,
                    criticality,
                    config=config.criticality,
                ),
            },
        },
        "checkpoint_sha256": checkpoint_hashes,
        "training_seed_results": records,
        "metric_distributions": distributions,
        "gate": gate,
        "recommendation": (
            "advance_consistent_v4_to_closed_loop_test"
            if gate["passed"]
            else "do_not_advance_consistent_v4"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen V4 checkpoints on a fresh seed block."
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
        "--test-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_test.npz"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_replication_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_fresh_test.json"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_control_aware_fresh_test(
        train_path=args.train_data,
        validation_path=args.validation_data,
        test_path=args.test_data,
        checkpoint_directory=args.checkpoint_directory,
        config=ControlAwareFreshTestConfig(device=args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"fresh_test={'PASS' if result['gate']['passed'] else 'FAIL'}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
