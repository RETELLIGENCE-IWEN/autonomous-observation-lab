"""Development-only ensemble evaluation for the replicated V7 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .adaptive_curriculum_objective import (
    _gate,
    _selection_metrics,
    selected_adaptive_position_v21_config,
)
from .config import ObservationProfile
from .control_aware_predictor import _future_index
from .control_criticality import (
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from .dataset import load_gimbal_dataset
from .gru import (
    CausalTargetStateGRUEnsemble,
    GRULossConfig,
    gru_parameter_count,
    load_gru_checkpoint,
)
from .gru_training import evaluate_gru
from .midpoint_adapter_replication import (
    MIDPOINT_ADAPTER_REPLICATION_SCHEMA_VERSION,
)


MIDPOINT_ADAPTER_ENSEMBLE_SCHEMA_VERSION = (
    "gimbal_midpoint_adapter_v7_ensemble_development_v1"
)


@dataclass(frozen=True)
class MidpointAdapterEnsembleConfig:
    batch_size: int = 24
    minimum_member_count: int = 3
    maximum_standard_regression_fraction: float = 0.02
    maximum_critical_bearing_regression_fraction: float = 0.02
    maximum_critical_rate_regression_fraction: float = 0.02
    maximum_consistency_regression_fraction: float = 0.02
    minimum_adaptive_action_improvement_fraction: float = 0.01
    minimum_critical_adaptive_action_improvement_fraction: float = 0.01
    device: str = "cpu"
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.minimum_member_count <= 0:
            raise ValueError("ensemble batch and member counts must be positive")
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_midpoint_adapter_ensemble(
    *,
    validation_path: str | Path,
    replication_path: str | Path,
    config: MidpointAdapterEnsembleConfig | None = None,
) -> dict[str, Any]:
    """Compare matched V4 and midpoint ensembles on development data."""

    config = config or MidpointAdapterEnsembleConfig()
    validation_path = Path(validation_path)
    replication_path = Path(replication_path)
    replication = json.loads(replication_path.read_text(encoding="utf-8"))
    if (
        replication.get("experiment")
        != MIDPOINT_ADAPTER_REPLICATION_SCHEMA_VERSION
    ):
        raise ValueError("unexpected V7 replication schema")
    if replication.get("datasets", {}).get("test") != {"opened": False}:
        raise ValueError("V7 replication did not keep its test closed")
    validation = load_gimbal_dataset(validation_path)
    expected_hash = replication["datasets"]["validation"]["sha256"]
    if _sha256(validation_path) != expected_hash:
        raise ValueError("ensemble validation data differs from replication")
    records = replication["training_seed_results"]
    if len(records) < config.minimum_member_count:
        raise ValueError("V7 replication has too few ensemble members")

    ensembles = {}
    member_metadata = {}
    for candidate in ("v4_reference", "midpoint_state_reference"):
        models = []
        metadata_records = []
        for record in records:
            checkpoint = Path(record[candidate]["checkpoint"])
            model, metadata = load_gru_checkpoint(
                checkpoint,
                device=config.device,
            )
            if metadata.get("candidate") != candidate or metadata.get(
                "training_seed"
            ) != record["training_seed"]:
                raise ValueError("V7 checkpoint metadata does not match artifact")
            models.append(model)
            metadata_records.append(metadata)
        ensembles[candidate] = CausalTargetStateGRUEnsemble(tuple(models))
        member_metadata[candidate] = metadata_records

    criticality = compute_control_criticality(
        validation,
        config=config.criticality,
    )
    adapter = selected_adaptive_position_v21_config()
    horizons_s = validation.manifest.prediction_horizons_s
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
    future_index = _future_index(horizons_s)
    results = {}
    for candidate, ensemble in ensembles.items():
        standard = evaluate_gru(
            ensemble,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
        )
        critical = evaluate_gru(
            ensemble,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
            evaluation_mask=criticality.critical_mask,
        )
        common = evaluate_gru(
            ensemble,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
            loss_config=common_loss,
        )
        critical_common = evaluate_gru(
            ensemble,
            validation,
            ObservationProfile.DISTURBANCE_AWARE,
            batch_size=config.batch_size,
            device=config.device,
            loss_config=common_loss,
            evaluation_mask=criticality.critical_mask,
        )
        results[candidate] = {
            "member_count": ensemble.member_count,
            "parameter_count": gru_parameter_count(ensemble),
            "member_metadata": member_metadata[candidate],
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
    gate = _gate(
        results["midpoint_state_reference"]["selection_view"],
        results["v4_reference"]["selection_view"],
        config,
    )
    return {
        "experiment": MIDPOINT_ADAPTER_ENSEMBLE_SCHEMA_VERSION,
        "config": asdict(config),
        "adapter_config": asdict(adapter),
        "replication": {
            "path": str(replication_path),
            "sha256": _sha256(replication_path),
            "gate_passed": replication["gate"]["passed"],
        },
        "datasets": {
            "validation": {
                "path": str(validation_path),
                "sha256": expected_hash,
                "episodes": validation.episode_count,
                "criticality": control_criticality_report(
                    validation,
                    criticality,
                    config=config.criticality,
                ),
            },
            "test": {"opened": False},
        },
        "ensembles": results,
        "gate": gate,
        "recommendation": (
            "advance_v7_ensemble_to_fresh_test"
            if gate["passed"]
            else "do_not_open_fresh_test"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate matched V4 and midpoint GRU ensembles."
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_validation.npz"),
    )
    parser.add_argument(
        "--replication-results",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_replication.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_ensemble.json"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_midpoint_adapter_ensemble(
        validation_path=args.validation_data,
        replication_path=args.replication_results,
        config=MidpointAdapterEnsembleConfig(device=args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"ensemble={'PASS' if result['gate']['passed'] else 'FAIL'}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
