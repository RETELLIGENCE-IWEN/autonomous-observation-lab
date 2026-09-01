"""Fresh closed-loop benchmark for the replicated control-aware GRU V4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .adaptive_position import AdaptivePositionProtocolConfig, _adaptive_run
from .adaptive_position_v21 import ADAPTIVE_POSITION_V21_SCHEMA_VERSION
from .closed_loop import closed_loop_scenarios
from .config import GimbalCommandMode, ObservationProfile
from .controller_arena import _selected_candidate_config
from .gru import CausalTargetStateGRU, load_gru_checkpoint
from .gru_control import GRUControlEvaluationConfig, _baseline_runs, _learned_run
from .performance_atlas import (
    DEFAULT_TRACKED_SCENARIOS,
    FailureAtlasConfig,
    PerformanceContract,
    _aggregate_records,
    analyze_controller_run,
    evaluate_contract,
    load_performance_contract,
)
from .predictive_position_v3 import _variants


CONTROL_AWARE_CLOSED_LOOP_SCHEMA_VERSION = (
    "gimbal_control_aware_v4_closed_loop_fresh_v1"
)


@dataclass(frozen=True)
class ControlAwareClosedLoopConfig:
    training_seeds: tuple[int, ...] = (17, 29, 43)
    world_seeds: tuple[int, ...] = tuple(range(87000, 87008))
    scenario_names: tuple[str, ...] = tuple(
        scenario.name for scenario in closed_loop_scenarios()
    )
    maximum_staleness_s: float = 0.50
    maximum_mean_error_regression_fraction: float = 0.01
    maximum_p95_error_regression_fraction: float = 0.01
    maximum_loss_of_view_regression: float = 0.002
    maximum_avoidable_loss_regression: float = 0.002
    maximum_forecast_error_regression_fraction: float = 0.01
    maximum_command_variation_regression_fraction: float = 0.02
    maximum_actuator_acceleration_regression_fraction: float = 0.02
    minimum_mean_error_improving_seed_count: int = 2
    device: str = "cpu"
    analysis: FailureAtlasConfig = FailureAtlasConfig()

    def __post_init__(self) -> None:
        for name in ("training_seeds", "world_seeds", "scenario_names"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be non-empty and unique")
        if any(seed < 0 for seed in self.training_seeds + self.world_seeds):
            raise ValueError("closed-loop seeds must be non-negative")
        if not math.isfinite(self.maximum_staleness_s) or (
            self.maximum_staleness_s < 0.0
        ):
            raise ValueError("maximum staleness must be finite and non-negative")
        if not 1 <= self.minimum_mean_error_improving_seed_count <= len(
            self.training_seeds
        ):
            raise ValueError("minimum improving seed count is invalid")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return (candidate - reference) / reference


def _load_model_pair(
    checkpoint_directory: Path,
    training_seed: int,
    device: str,
) -> tuple[dict[str, CausalTargetStateGRU], dict[str, str]]:
    models = {}
    hashes = {}
    for candidate in ("baseline_expanded", "consistent_v4"):
        path = checkpoint_directory / (
            f"gimbal_{candidate}_seed_{training_seed}.pt"
        )
        model, metadata = load_gru_checkpoint(path, device=device)
        if metadata.get("candidate") != candidate or metadata.get(
            "training_seed"
        ) != training_seed:
            raise ValueError("closed-loop checkpoint metadata mismatch")
        models[candidate] = model
        hashes[f"{candidate}_seed_{training_seed}"] = _sha256(path)
    return models, hashes


def _summaries(
    records: list[dict[str, Any]],
    contract: PerformanceContract,
) -> dict[str, Any]:
    tracked = [
        record
        for record in records
        if record["scenario_name"] in contract.tracked_scenarios
    ]
    by_scenario = {}
    for scenario_name in dict.fromkeys(
        record["scenario_name"] for record in records
    ):
        selected = [
            record
            for record in records
            if record["scenario_name"] == scenario_name
        ]
        by_scenario[scenario_name] = _aggregate_records(selected)
    tracked_summary = _aggregate_records(tracked)
    return {
        "tracked": tracked_summary,
        "all_scenarios": _aggregate_records(records),
        "by_scenario": by_scenario,
        "contract": evaluate_contract(tracked_summary, contract),
    }


def _paired_gate(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    per_seed_candidate: dict[int, dict[str, Any]],
    per_seed_reference: dict[int, dict[str, Any]],
    config: ControlAwareClosedLoopConfig,
) -> dict[str, Any]:
    relative = {
        key: _relative_change(float(candidate[key]), float(reference[key]))
        for key in (
            "mean_absolute_error_fov_fraction",
            "p95_absolute_error_fov_fraction",
            "forecast_error_fov_fraction",
            "command_variation_per_s",
            "actuator_acceleration_rms_normalized",
        )
    }
    absolute = {
        key: float(candidate[key]) - float(reference[key])
        for key in ("loss_of_view_fraction", "avoidable_loss_fraction")
    }
    improving_count = sum(
        per_seed_candidate[seed]["mean_absolute_error_fov_fraction"]
        < per_seed_reference[seed]["mean_absolute_error_fov_fraction"]
        for seed in config.training_seeds
    )
    checks = {
        "mean_error": relative["mean_absolute_error_fov_fraction"]
        <= config.maximum_mean_error_regression_fraction,
        "p95_error": relative["p95_absolute_error_fov_fraction"]
        <= config.maximum_p95_error_regression_fraction,
        "loss_of_view": absolute["loss_of_view_fraction"]
        <= config.maximum_loss_of_view_regression,
        "avoidable_loss": absolute["avoidable_loss_fraction"]
        <= config.maximum_avoidable_loss_regression,
        "forecast_error": relative["forecast_error_fov_fraction"]
        <= config.maximum_forecast_error_regression_fraction,
        "command_variation": relative["command_variation_per_s"]
        <= config.maximum_command_variation_regression_fraction,
        "actuator_acceleration": relative[
            "actuator_acceleration_rms_normalized"
        ]
        <= config.maximum_actuator_acceleration_regression_fraction,
        "training_seed_consistency": improving_count
        >= config.minimum_mean_error_improving_seed_count,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": relative,
        "absolute_changes": absolute,
        "mean_error_improving_training_seed_count": improving_count,
    }


def evaluate_control_aware_closed_loop(
    *,
    checkpoint_directory: str | Path,
    legacy_checkpoint_directory: str | Path,
    visibility_risk_results: str | Path,
    contract: PerformanceContract | None = None,
    config: ControlAwareClosedLoopConfig | None = None,
) -> dict[str, Any]:
    """Evaluate frozen predictor pairs through rate and position controllers."""

    config = config or ControlAwareClosedLoopConfig()
    contract = contract or PerformanceContract()
    checkpoint_directory = Path(checkpoint_directory)
    legacy_checkpoint_directory = Path(legacy_checkpoint_directory)
    visibility_risk_results = Path(visibility_risk_results)
    visibility_result = json.loads(
        visibility_risk_results.read_text(encoding="utf-8")
    )
    if visibility_result.get("experiment") != ADAPTIVE_POSITION_V21_SCHEMA_VERSION:
        raise ValueError("closed-loop benchmark requires a frozen V2.1 result")
    selected_adapter, adapter_config = _selected_candidate_config(
        visibility_result
    )
    variants = _variants(config.world_seeds, config.scenario_names)
    runtime = AdaptivePositionProtocolConfig(
        maximum_staleness_s=config.maximum_staleness_s,
        device=config.device,
    )
    rate_runtime = GRUControlEvaluationConfig(
        maximum_staleness_s=config.maximum_staleness_s,
        device=config.device,
    )
    records: dict[str, list[dict[str, Any]]] = {
        "analytical_position": [],
        "legacy_o2_position_v21": [],
        "baseline_expanded_position_v21": [],
        "consistent_v4_position_v21": [],
        "baseline_expanded_rate_100ms": [],
        "consistent_v4_rate_100ms": [],
    }
    checkpoint_hashes = {}
    loaded_pairs = {}
    legacy_models = {}
    for training_seed in config.training_seeds:
        pair, hashes = _load_model_pair(
            checkpoint_directory,
            training_seed,
            config.device,
        )
        loaded_pairs[training_seed] = pair
        checkpoint_hashes.update(hashes)
        legacy_path = legacy_checkpoint_directory / (
            f"gimbal_gru_o2_seed_{training_seed}.pt"
        )
        legacy_models[training_seed], _ = load_gru_checkpoint(
            legacy_path,
            device=config.device,
        )
        checkpoint_hashes[f"legacy_o2_seed_{training_seed}"] = _sha256(
            legacy_path
        )

    for world_seed, _scenario_index, scenario in variants:
        analytical = next(
            run
            for run in _baseline_runs(scenario, world_seed, rate_runtime)
            if run.episode.name == "analytical_position"
        )
        records["analytical_position"].append(
            analyze_controller_run(
                analytical,
                scenario,
                controller_name="analytical_position",
                world_seed=world_seed,
                training_seed=-1,
                analysis=config.analysis,
            )
        )
        for training_seed in config.training_seeds:
            model_specs = (
                (
                    "legacy_o2_position_v21",
                    legacy_models[training_seed],
                ),
                (
                    "baseline_expanded_position_v21",
                    loaded_pairs[training_seed]["baseline_expanded"],
                ),
                (
                    "consistent_v4_position_v21",
                    loaded_pairs[training_seed]["consistent_v4"],
                ),
            )
            for controller_name, model in model_specs:
                run = _adaptive_run(
                    scenario=scenario,
                    seed=world_seed,
                    model=model,
                    adapter=adapter_config,
                    evaluation=runtime,
                    name=controller_name,
                )
                records[controller_name].append(
                    analyze_controller_run(
                        run,
                        scenario,
                        controller_name=controller_name,
                        world_seed=world_seed,
                        training_seed=training_seed,
                        analysis=config.analysis,
                    )
                )
            for candidate in ("baseline_expanded", "consistent_v4"):
                controller_name = f"{candidate}_rate_100ms"
                model = loaded_pairs[training_seed][candidate]
                horizon_index = min(
                    range(model.horizon_count),
                    key=lambda index: abs(
                        model.config.prediction_horizons_s[index] - 0.1
                    ),
                )
                run = _learned_run(
                    scenario=scenario,
                    seed=world_seed,
                    model=model,
                    profile=ObservationProfile.DISTURBANCE_AWARE,
                    horizon_index=horizon_index,
                    command_mode=GimbalCommandMode.RATE,
                    evaluation=rate_runtime,
                    search_fallback=False,
                )
                records[controller_name].append(
                    analyze_controller_run(
                        run,
                        scenario,
                        controller_name=controller_name,
                        world_seed=world_seed,
                        training_seed=training_seed,
                        analysis=config.analysis,
                    )
                )

    summaries = {
        name: _summaries(values, contract) for name, values in records.items()
    }
    per_seed = {}
    for name in (
        "baseline_expanded_position_v21",
        "consistent_v4_position_v21",
        "baseline_expanded_rate_100ms",
        "consistent_v4_rate_100ms",
    ):
        per_seed[name] = {
            seed: _aggregate_records(
                [
                    record
                    for record in records[name]
                    if record["training_seed"] == seed
                    and record["scenario_name"] in contract.tracked_scenarios
                ]
            )
            for seed in config.training_seeds
        }
    position_gate = _paired_gate(
        candidate=summaries["consistent_v4_position_v21"]["tracked"],
        reference=summaries["baseline_expanded_position_v21"]["tracked"],
        per_seed_candidate=per_seed["consistent_v4_position_v21"],
        per_seed_reference=per_seed["baseline_expanded_position_v21"],
        config=config,
    )
    rate_gate = _paired_gate(
        candidate=summaries["consistent_v4_rate_100ms"]["tracked"],
        reference=summaries["baseline_expanded_rate_100ms"]["tracked"],
        per_seed_candidate=per_seed["consistent_v4_rate_100ms"],
        per_seed_reference=per_seed["baseline_expanded_rate_100ms"],
        config=config,
    )
    return {
        "experiment": CONTROL_AWARE_CLOSED_LOOP_SCHEMA_VERSION,
        "config": asdict(config),
        "contract": asdict(contract),
        "visibility_risk_adapter": {
            "source": str(visibility_risk_results),
            "sha256": _sha256(visibility_risk_results),
            "selected_candidate": selected_adapter,
            "configuration": asdict(adapter_config),
        },
        "checkpoint_sha256": checkpoint_hashes,
        "worlds": {
            "seeds": list(config.world_seeds),
            "scenarios": list(config.scenario_names),
            "variant_count": len(variants),
            "historical_worlds_reused": False,
        },
        "summaries": summaries,
        "per_training_seed": {
            name: {str(seed): value for seed, value in values.items()}
            for name, values in per_seed.items()
        },
        "gates": {
            "position_primary": position_gate,
            "rate_diagnostic": rate_gate,
        },
        "recommendation": (
            "promote_consistent_v4_position_predictor"
            if position_gate["passed"]
            else "retain_expanded_baseline_in_controller"
        ),
        "analysis_records": records,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fresh V4 closed-loop controller benchmark."
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_replication_checkpoints"),
    )
    parser.add_argument(
        "--legacy-checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_o2_replication_checkpoints"),
    )
    parser.add_argument(
        "--visibility-risk-results",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_position_v21.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/gimbal_performance_contract.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_closed_loop.json"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_control_aware_closed_loop(
        checkpoint_directory=args.checkpoint_directory,
        legacy_checkpoint_directory=args.legacy_checkpoint_directory,
        visibility_risk_results=args.visibility_risk_results,
        contract=load_performance_contract(args.contract),
        config=ControlAwareClosedLoopConfig(device=args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        "position="
        f"{'PASS' if result['gates']['position_primary']['passed'] else 'FAIL'}; "
        "rate="
        f"{'PASS' if result['gates']['rate_diagnostic']['passed'] else 'FAIL'}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
