"""Fresh development/confirmation protocol for constrained position V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adaptive_position import (
    AdaptivePositionProtocolConfig,
    _adaptive_run,
    _fixed_run,
)
from .adaptive_position_v21 import ADAPTIVE_POSITION_V21_SCHEMA_VERSION
from .closed_loop import ClosedLoopScenario, ControllerRun, closed_loop_scenarios
from .config import GimbalCommandMode, ObservationProfile
from .controller_arena import _selected_candidate_config
from .controllers import (
    ConstrainedPredictivePositionController,
    PredictivePositionOptimizerConfig,
)
from .gru import (
    CausalTargetStateGRU,
    GRUInferenceConfig,
    GRUTargetStateEstimator,
)
from .performance_atlas import (
    DEFAULT_TRACKED_SCENARIOS,
    FailureAtlasConfig,
    PerformanceContract,
    _aggregate_records,
    _load_models,
    analyze_controller_run,
    evaluate_contract,
    load_performance_contract,
)
from .randomization import (
    GimbalDomainRandomizationConfig,
    randomize_closed_loop_scenario,
)


PREDICTIVE_POSITION_V3_SCHEMA_VERSION = (
    "gimbal_constrained_predictive_position_v3_protocol_v1"
)
DEFAULT_DEVELOPMENT_SEEDS = tuple(range(83000, 83008))
DEFAULT_CONFIRMATION_SEEDS = tuple(range(84000, 84008))
HISTORICAL_WORLD_SEEDS = tuple(range(80000, 80004)) + tuple(
    range(81000, 81008)
) + tuple(range(82000, 82008))


@dataclass(frozen=True)
class PredictivePositionV3Candidate:
    name: str
    controller: PredictivePositionOptimizerConfig

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("V3 candidate name must be identifier-like")


def default_v3_candidates() -> tuple[PredictivePositionV3Candidate, ...]:
    shared = {
        "maximum_optimization_horizon_s": 0.10,
        "terminal_tracking_weight": 1.0,
        "rate_matching_weight": 0.10,
        "visibility_weight": 4.0,
        "command_effect_response_fraction": 0.50,
        "minimum_optimizer_position_gain_s_inv": 4.0,
    }
    return (
        PredictivePositionV3Candidate(
            "capacity_strong",
            PredictivePositionOptimizerConfig(
                **shared,
                activation_rate_onset_fraction=0.45,
                activation_rate_full_fraction=0.65,
                activation_visibility_onset_fraction=0.55,
                activation_visibility_full_fraction=0.75,
            ),
        ),
        PredictivePositionV3Candidate(
            "capacity_moderate",
            PredictivePositionOptimizerConfig(
                **shared,
                activation_rate_onset_fraction=0.55,
                activation_rate_full_fraction=0.75,
                activation_visibility_onset_fraction=0.65,
                activation_visibility_full_fraction=0.85,
            ),
        ),
        PredictivePositionV3Candidate(
            "rate_priority",
            PredictivePositionOptimizerConfig(
                **shared,
                activation_rate_onset_fraction=0.45,
                activation_rate_full_fraction=0.65,
                activation_visibility_onset_fraction=0.70,
                activation_visibility_full_fraction=0.90,
            ),
        ),
        PredictivePositionV3Candidate(
            "capacity_smooth",
            PredictivePositionOptimizerConfig(
                **shared,
                activation_rate_onset_fraction=0.45,
                activation_rate_full_fraction=0.65,
                activation_visibility_onset_fraction=0.55,
                activation_visibility_full_fraction=0.75,
                command_change_weight=0.06,
                command_rate_change_weight=0.010,
            ),
        ),
    )


@dataclass(frozen=True)
class PredictivePositionV3ProtocolConfig:
    maximum_staleness_s: float = 0.50
    development_minimum_avoidable_loss_reduction_fraction: float = 0.02
    development_minimum_variation_reduction_fraction: float = 0.03
    development_max_mean_error_regression_fov_fraction: float = 0.01
    development_max_p95_regression_fov_fraction: float = 0.01
    development_max_scenario_p95_regression_fov_fraction: float = 0.03
    development_max_scenario_avoidable_loss_regression: float = 0.005
    confirmation_minimum_avoidable_loss_reduction_fraction: float = 0.02
    confirmation_minimum_variation_reduction_fraction: float = 0.03
    confirmation_max_mean_error_regression_fov_fraction: float = 0.0
    confirmation_max_p95_regression_fov_fraction: float = 0.0
    confirmation_max_scenario_p95_regression_fov_fraction: float = 0.03
    confirmation_max_scenario_avoidable_loss_regression: float = 0.005
    device: str = "cpu"
    candidates: tuple[PredictivePositionV3Candidate, ...] = (
        default_v3_candidates()
    )

    def __post_init__(self) -> None:
        for name in (
            "maximum_staleness_s",
            "development_minimum_avoidable_loss_reduction_fraction",
            "development_minimum_variation_reduction_fraction",
            "development_max_mean_error_regression_fov_fraction",
            "development_max_p95_regression_fov_fraction",
            "development_max_scenario_p95_regression_fov_fraction",
            "development_max_scenario_avoidable_loss_regression",
            "confirmation_minimum_avoidable_loss_reduction_fraction",
            "confirmation_minimum_variation_reduction_fraction",
            "confirmation_max_mean_error_regression_fov_fraction",
            "confirmation_max_p95_regression_fov_fraction",
            "confirmation_max_scenario_p95_regression_fov_fraction",
            "confirmation_max_scenario_avoidable_loss_regression",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.candidates:
            raise ValueError("at least one V3 candidate is required")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("V3 candidate names must be unique")


def _variants(
    seeds: tuple[int, ...],
    scenario_names: tuple[str, ...],
) -> tuple[tuple[int, int, ClosedLoopScenario], ...]:
    catalog = {
        scenario.name: (index, scenario)
        for index, scenario in enumerate(closed_loop_scenarios())
    }
    if not scenario_names or not set(scenario_names) <= set(catalog):
        raise ValueError("V3 scenario names are unavailable")
    return tuple(
        (
            seed,
            catalog[name][0],
            randomize_closed_loop_scenario(
                catalog[name][1],
                seed=seed,
                config=GimbalDomainRandomizationConfig(),
            ),
        )
        for seed in seeds
        for name in scenario_names
    )


def _v3_run(
    *,
    scenario: ClosedLoopScenario,
    seed: int,
    model: CausalTargetStateGRU,
    controller_config: PredictivePositionOptimizerConfig,
    protocol: PredictivePositionV3ProtocolConfig,
    name: str,
) -> ControllerRun:
    config = replace(
        scenario.config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        command_mode=GimbalCommandMode.POSITION,
    )
    estimator = GRUTargetStateEstimator(
        model,
        config,
        GRUInferenceConfig(
            observation_profile=ObservationProfile.DISTURBANCE_AWARE,
            horizon_index=0,
            maximum_staleness_s=protocol.maximum_staleness_s,
            device=protocol.device,
        ),
    )
    controller = ConstrainedPredictivePositionController(
        estimator=estimator,
        servo=config.servo,
        selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
        config=controller_config,
        name=name,
    )
    from .closed_loop import run_closed_loop_controller

    return run_closed_loop_controller(
        name=name,
        description=(
            "Multi-horizon GRU with constrained servo rollout, capacity-risk "
            "activation, and frozen V2.1-compatible fallback."
        ),
        scenario=scenario,
        config=config,
        controller=controller,
        seed=seed,
    )


def _run_block(
    *,
    variants: tuple[tuple[int, int, ClosedLoopScenario], ...],
    models: dict[int, CausalTargetStateGRU],
    horizon_indices: dict[int, int],
    v21_config: Any,
    protocol: PredictivePositionV3ProtocolConfig,
    candidates: tuple[PredictivePositionV3Candidate, ...],
) -> dict[str, dict[int, list[ControllerRun]]]:
    names = ("fixed_horizon", "visibility_risk_v21") + tuple(
        candidate.name for candidate in candidates
    )
    runs = {name: {seed: [] for seed in models} for name in names}
    adaptive_runtime = AdaptivePositionProtocolConfig(
        maximum_staleness_s=protocol.maximum_staleness_s,
        device=protocol.device,
    )
    for training_seed, model in models.items():
        for world_seed, _scenario_index, scenario in variants:
            runs["fixed_horizon"][training_seed].append(
                _fixed_run(
                    scenario=scenario,
                    seed=world_seed,
                    model=model,
                    horizon_index=horizon_indices[training_seed],
                    evaluation=adaptive_runtime,
                )
            )
            runs["visibility_risk_v21"][training_seed].append(
                _adaptive_run(
                    scenario=scenario,
                    seed=world_seed,
                    model=model,
                    adapter=v21_config,
                    evaluation=adaptive_runtime,
                    name="visibility_risk_v21",
                )
            )
            for candidate in candidates:
                runs[candidate.name][training_seed].append(
                    _v3_run(
                        scenario=scenario,
                        seed=world_seed,
                        model=model,
                        controller_config=candidate.controller,
                        protocol=protocol,
                        name=f"constrained_v3_{candidate.name}",
                    )
                )
    return runs


def _diagnostic_summary(runs: list[ControllerRun]) -> dict[str, float]:
    diagnostics = [
        item
        for run in runs
        for item in run.adapter_diagnostics
        if item.get("valid", False)
    ]
    if not diagnostics:
        return {
            "valid_step_count": 0,
            "optimizer_active_fraction": 0.0,
            "mean_activation_score": 0.0,
            "mean_predicted_terminal_error_fov_fraction": 0.0,
            "mean_predicted_peak_error_fov_fraction": 0.0,
            "mean_evaluated_candidate_count": 0.0,
        }

    def mean(name: str) -> float:
        return float(np.mean([float(item[name]) for item in diagnostics]))

    return {
        "valid_step_count": len(diagnostics),
        "optimizer_active_fraction": mean("optimizer_active"),
        "mean_activation_score": mean("activation_score"),
        "mean_predicted_terminal_error_fov_fraction": mean(
            "predicted_terminal_error_fov_fraction"
        ),
        "mean_predicted_peak_error_fov_fraction": mean(
            "predicted_peak_error_fov_fraction"
        ),
        "mean_evaluated_candidate_count": mean(
            "evaluated_candidate_count"
        ),
    }


def _block_records(
    *,
    variants: tuple[tuple[int, int, ClosedLoopScenario], ...],
    runs: dict[int, list[ControllerRun]],
    controller_name: str,
    analysis: FailureAtlasConfig,
) -> list[dict[str, Any]]:
    records = []
    for training_seed, seed_runs in runs.items():
        for index, (world_seed, _scenario_index, scenario) in enumerate(variants):
            records.append(
                analyze_controller_run(
                    seed_runs[index],
                    scenario,
                    controller_name=controller_name,
                    world_seed=world_seed,
                    training_seed=training_seed,
                    analysis=analysis,
                )
            )
    return records


def _summarize_controller(
    records: list[dict[str, Any]],
    contract: PerformanceContract,
) -> dict[str, Any]:
    tracked = [
        record
        for record in records
        if record["scenario_name"] in contract.tracked_scenarios
    ]
    summary = _aggregate_records(tracked)
    by_scenario = {}
    for scenario_name in dict.fromkeys(
        record["scenario_name"] for record in records
    ):
        values = [
            record for record in records if record["scenario_name"] == scenario_name
        ]
        scenario_summary = _aggregate_records(values)
        applicable = scenario_name in contract.tracked_scenarios
        by_scenario[scenario_name] = {
            "summary": scenario_summary,
            "contract_applicable": applicable,
            "contract": (
                evaluate_contract(scenario_summary, contract)
                if applicable
                else None
            ),
        }
    return {
        "tracked_summary": summary,
        "all_scenarios_summary": _aggregate_records(records),
        "contract": evaluate_contract(summary, contract),
        "by_scenario": by_scenario,
    }


def _relative_reduction(candidate: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0
    return (reference - candidate) / reference


def _comparison(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    keys = (
        "mean_absolute_error_fov_fraction",
        "p95_absolute_error_fov_fraction",
        "loss_of_view_fraction",
        "avoidable_loss_fraction",
        "command_variation_per_s",
        "actuator_acceleration_rms_normalized",
    )
    return {
        "deltas": {
            key: float(candidate[key]) - float(reference[key]) for key in keys
        },
        "avoidable_loss_reduction_fraction": _relative_reduction(
            float(candidate["avoidable_loss_fraction"]),
            float(reference["avoidable_loss_fraction"]),
        ),
        "command_variation_reduction_fraction": _relative_reduction(
            float(candidate["command_variation_per_s"]),
            float(reference["command_variation_per_s"]),
        ),
        "unrecovered_loss_event_delta": (
            int(candidate["unrecovered_loss_events"])
            - int(reference["unrecovered_loss_events"])
        ),
    }


def _gate(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    candidate_by_scenario: dict[str, Any],
    reference_by_scenario: dict[str, Any],
    minimum_avoidable_reduction: float,
    minimum_variation_reduction: float,
    maximum_mean_regression: float,
    maximum_p95_regression: float,
    maximum_scenario_p95_regression: float,
    maximum_scenario_avoidable_regression: float,
) -> dict[str, Any]:
    comparison = _comparison(candidate, reference)
    checks = {
        "mean_error": comparison["deltas"][
            "mean_absolute_error_fov_fraction"
        ]
        <= maximum_mean_regression,
        "p95_error": comparison["deltas"][
            "p95_absolute_error_fov_fraction"
        ]
        <= maximum_p95_regression,
        "total_loss": comparison["deltas"]["loss_of_view_fraction"] <= 0.0,
        "avoidable_loss": comparison["avoidable_loss_reduction_fraction"]
        >= minimum_avoidable_reduction,
        "command_variation": comparison[
            "command_variation_reduction_fraction"
        ]
        >= minimum_variation_reduction,
        "unrecovered_events": comparison["unrecovered_loss_event_delta"] <= 0,
    }
    scenario_checks = {}
    for name, candidate_item in candidate_by_scenario.items():
        if not candidate_item["contract_applicable"]:
            continue
        candidate_summary = candidate_item["summary"]
        reference_summary = reference_by_scenario[name]["summary"]
        scenario_checks[name] = (
            candidate_summary["p95_absolute_error_fov_fraction"]
            - reference_summary["p95_absolute_error_fov_fraction"]
            <= maximum_scenario_p95_regression
            and candidate_summary["avoidable_loss_fraction"]
            - reference_summary["avoidable_loss_fraction"]
            <= maximum_scenario_avoidable_regression
            and candidate_summary["unrecovered_loss_events"]
            <= reference_summary["unrecovered_loss_events"]
        )
    return {
        "passed": all(checks.values()) and all(scenario_checks.values()),
        "checks": checks,
        "per_scenario_checks": scenario_checks,
        "comparison": comparison,
    }


def _representative_trace(
    *,
    variants: tuple[tuple[int, int, ClosedLoopScenario], ...],
    reference_runs: dict[int, list[ControllerRun]],
    candidate_runs: dict[int, list[ControllerRun]],
) -> dict[str, Any]:
    choices = []
    for training_seed in candidate_runs:
        for index, (reference, candidate) in enumerate(
            zip(
                reference_runs[training_seed],
                candidate_runs[training_seed],
                strict=True,
            )
        ):
            improvement = (
                reference.metrics.loss_of_view_fraction
                - candidate.metrics.loss_of_view_fraction
            )
            choices.append((improvement, training_seed, index))
    _improvement, training_seed, index = max(choices)
    world_seed, _scenario_index, scenario = variants[index]
    reference = reference_runs[training_seed][index]
    candidate = candidate_runs[training_seed][index]
    length = min(
        len(reference.episode.frames),
        len(candidate.episode.frames),
        len(candidate.adapter_diagnostics),
    )
    records = []
    for frame_index in range(length):
        reference_frame = reference.episode.frames[frame_index]
        candidate_frame = candidate.episode.frames[frame_index]
        diagnostics = candidate.adapter_diagnostics[frame_index]
        records.append(
            {
                "time_s": candidate_frame.diagnostics.time_s,
                "target_body_bearing_deg": math.degrees(
                    candidate_frame.diagnostics.target_bearing_rad
                    - candidate_frame.diagnostics.body_bearing_rad
                ),
                "v21_gimbal_angle_deg": math.degrees(
                    reference_frame.diagnostics.gimbal_angle_rad
                ),
                "v3_gimbal_angle_deg": math.degrees(
                    candidate_frame.diagnostics.gimbal_angle_rad
                ),
                "v21_command_deg": math.degrees(
                    reference_frame.diagnostics.requested_position_rad or 0.0
                ),
                "v3_command_deg": math.degrees(
                    candidate_frame.diagnostics.requested_position_rad or 0.0
                ),
                "v3_fallback_target_deg": math.degrees(
                    float(diagnostics.get("fallback_target_angle_rad", 0.0))
                ),
                "optimizer_active": bool(
                    diagnostics.get("optimizer_active", False)
                ),
                "activation_score": float(
                    diagnostics.get("activation_score", 0.0)
                ),
                "predicted_terminal_error_fov_fraction": float(
                    diagnostics.get(
                        "predicted_terminal_error_fov_fraction",
                        0.0,
                    )
                ),
                "v21_in_view": reference_frame.diagnostics.target_in_view,
                "v3_in_view": candidate_frame.diagnostics.target_in_view,
            }
        )
    return {
        "world_seed": world_seed,
        "training_seed": training_seed,
        "scenario_name": scenario.name,
        "records": records,
    }


def evaluate_predictive_position_v3(
    *,
    visibility_risk_results: str | Path,
    protocol: PredictivePositionV3ProtocolConfig | None = None,
    contract: PerformanceContract | None = None,
    analysis: FailureAtlasConfig | None = None,
    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS,
    confirmation_seeds: tuple[int, ...] = DEFAULT_CONFIRMATION_SEEDS,
) -> dict[str, Any]:
    """Select V3 on 83000 worlds, then open untouched 84000 confirmation."""

    protocol = protocol or PredictivePositionV3ProtocolConfig()
    contract = contract or PerformanceContract()
    analysis = analysis or FailureAtlasConfig()
    source_path = Path(visibility_risk_results)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("experiment") != ADAPTIVE_POSITION_V21_SCHEMA_VERSION:
        raise ValueError("V3 requires a frozen V2.1 result")
    if not development_seeds or not confirmation_seeds:
        raise ValueError("V3 development and confirmation seeds are required")
    if len(set(development_seeds)) != len(development_seeds) or len(
        set(confirmation_seeds)
    ) != len(confirmation_seeds):
        raise ValueError("V3 seeds must be unique within each block")
    if set(development_seeds) & set(confirmation_seeds):
        raise ValueError("V3 development and confirmation seeds overlap")
    if set(HISTORICAL_WORLD_SEEDS) & (
        set(development_seeds) | set(confirmation_seeds)
    ):
        raise ValueError("V3 cannot reuse historical evaluation seeds")

    training_seeds = tuple(int(seed) for seed in source["training_seeds"])
    models, horizon_indices = _load_models(
        source,
        training_seeds,
        protocol.device,
    )
    selected_v21_name, v21_config = _selected_candidate_config(source)
    development_variants = _variants(
        development_seeds,
        DEFAULT_TRACKED_SCENARIOS,
    )
    development_runs = _run_block(
        variants=development_variants,
        models=models,
        horizon_indices=horizon_indices,
        v21_config=v21_config,
        protocol=protocol,
        candidates=protocol.candidates,
    )
    development_records = {
        name: _block_records(
            variants=development_variants,
            runs=runs,
            controller_name=name,
            analysis=analysis,
        )
        for name, runs in development_runs.items()
    }
    development_summaries = {
        name: _summarize_controller(records, contract)
        for name, records in development_records.items()
    }
    reference = development_summaries["visibility_risk_v21"]
    candidate_results = []
    for candidate in protocol.candidates:
        summary = development_summaries[candidate.name]
        flat_runs = [
            run
            for seed_runs in development_runs[candidate.name].values()
            for run in seed_runs
        ]
        gate = _gate(
            candidate=summary["tracked_summary"],
            reference=reference["tracked_summary"],
            candidate_by_scenario=summary["by_scenario"],
            reference_by_scenario=reference["by_scenario"],
            minimum_avoidable_reduction=(
                protocol.development_minimum_avoidable_loss_reduction_fraction
            ),
            minimum_variation_reduction=(
                protocol.development_minimum_variation_reduction_fraction
            ),
            maximum_mean_regression=(
                protocol.development_max_mean_error_regression_fov_fraction
            ),
            maximum_p95_regression=(
                protocol.development_max_p95_regression_fov_fraction
            ),
            maximum_scenario_p95_regression=(
                protocol.development_max_scenario_p95_regression_fov_fraction
            ),
            maximum_scenario_avoidable_regression=(
                protocol.development_max_scenario_avoidable_loss_regression
            ),
        )
        candidate_results.append(
            {
                "name": candidate.name,
                "controller_config": asdict(candidate.controller),
                "summary": summary,
                "diagnostics": _diagnostic_summary(flat_runs),
                "gate": gate,
            }
        )
    eligible = [item for item in candidate_results if item["gate"]["passed"]]
    selected_record = (
        min(
            eligible,
            key=lambda item: (
                item["summary"]["tracked_summary"][
                    "unrecovered_loss_events"
                ],
                item["summary"]["tracked_summary"][
                    "avoidable_loss_fraction"
                ],
                item["summary"]["tracked_summary"][
                    "p95_absolute_error_fov_fraction"
                ],
                item["summary"]["tracked_summary"][
                    "mean_absolute_error_fov_fraction"
                ],
                item["summary"]["tracked_summary"][
                    "command_variation_per_s"
                ],
            ),
        )
        if eligible
        else None
    )
    development = {
        "world_seeds": list(development_seeds),
        "scenario_names": list(DEFAULT_TRACKED_SCENARIOS),
        "variant_count_per_training_seed": len(development_variants),
        "fixed_horizon": development_summaries["fixed_horizon"],
        "visibility_risk_v21": reference,
        "candidates": candidate_results,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": (
            selected_record["name"] if selected_record is not None else None
        ),
    }
    common = {
        "experiment": PREDICTIVE_POSITION_V3_SCHEMA_VERSION,
        "source_visibility_risk_result": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_v21_candidate": selected_v21_name,
        "training_seeds": list(training_seeds),
        "protocol": asdict(protocol),
        "contract": asdict(contract),
        "analysis_config": asdict(analysis),
        "development": development,
    }
    if selected_record is None:
        return {
            **common,
            "confirmation": {
                "opened": False,
                "reason": "no V3 candidate passed the development gate",
                "recommendation": "retain_visibility_risk_v21",
            },
        }

    selected_candidate = next(
        candidate
        for candidate in protocol.candidates
        if candidate.name == selected_record["name"]
    )
    confirmation_variants = _variants(
        confirmation_seeds,
        tuple(scenario.name for scenario in closed_loop_scenarios()),
    )
    confirmation_runs = _run_block(
        variants=confirmation_variants,
        models=models,
        horizon_indices=horizon_indices,
        v21_config=v21_config,
        protocol=protocol,
        candidates=(selected_candidate,),
    )
    confirmation_records = {
        name: _block_records(
            variants=confirmation_variants,
            runs=runs,
            controller_name=name,
            analysis=analysis,
        )
        for name, runs in confirmation_runs.items()
    }
    confirmation_summaries = {
        name: _summarize_controller(records, contract)
        for name, records in confirmation_records.items()
    }
    selected_summary = confirmation_summaries[selected_candidate.name]
    reference_summary = confirmation_summaries["visibility_risk_v21"]
    confirmation_gate = _gate(
        candidate=selected_summary["tracked_summary"],
        reference=reference_summary["tracked_summary"],
        candidate_by_scenario=selected_summary["by_scenario"],
        reference_by_scenario=reference_summary["by_scenario"],
        minimum_avoidable_reduction=(
            protocol.confirmation_minimum_avoidable_loss_reduction_fraction
        ),
        minimum_variation_reduction=(
            protocol.confirmation_minimum_variation_reduction_fraction
        ),
        maximum_mean_regression=(
            protocol.confirmation_max_mean_error_regression_fov_fraction
        ),
        maximum_p95_regression=(
            protocol.confirmation_max_p95_regression_fov_fraction
        ),
        maximum_scenario_p95_regression=(
            protocol.confirmation_max_scenario_p95_regression_fov_fraction
        ),
        maximum_scenario_avoidable_regression=(
            protocol.confirmation_max_scenario_avoidable_loss_regression
        ),
    )
    flat_selected_runs = [
        run
        for seed_runs in confirmation_runs[selected_candidate.name].values()
        for run in seed_runs
    ]
    confirmation = {
        "opened": True,
        "world_seeds": list(confirmation_seeds),
        "scenario_names": [
            scenario.name for scenario in closed_loop_scenarios()
        ],
        "variant_count_per_training_seed": len(confirmation_variants),
        "fixed_horizon": confirmation_summaries["fixed_horizon"],
        "visibility_risk_v21": reference_summary,
        "predictive_position_v3": selected_summary,
        "selected_candidate": selected_candidate.name,
        "diagnostics": _diagnostic_summary(flat_selected_runs),
        "acceptance_gate": confirmation_gate,
        "recommendation": (
            "constrained_predictive_position_v3"
            if confirmation_gate["passed"]
            else "retain_visibility_risk_v21"
        ),
    }
    return {
        **common,
        "confirmation": confirmation,
        "representative_trace": _representative_trace(
            variants=confirmation_variants,
            reference_runs=confirmation_runs["visibility_risk_v21"],
            candidate_runs=confirmation_runs[selected_candidate.name],
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and confirm constrained predictive position V3."
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
        default=Path("artifacts/gimbal_predictive_position_v3.json"),
    )
    parser.add_argument("--development-seed", type=int, action="append")
    parser.add_argument("--confirmation-seed", type=int, action="append")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_predictive_position_v3(
        visibility_risk_results=args.visibility_risk_results,
        protocol=PredictivePositionV3ProtocolConfig(device=args.device),
        contract=load_performance_contract(args.contract),
        development_seeds=(
            tuple(args.development_seed)
            if args.development_seed
            else DEFAULT_DEVELOPMENT_SEEDS
        ),
        confirmation_seeds=(
            tuple(args.confirmation_seed)
            if args.confirmation_seed
            else DEFAULT_CONFIRMATION_SEEDS
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    confirmation = result["confirmation"]
    if confirmation["opened"]:
        gate = confirmation["acceptance_gate"]
        print(
            "V3 confirmation: "
            f"{'PASS' if gate['passed'] else 'FAIL'}; "
            f"recommendation={confirmation['recommendation']}"
        )
    else:
        print("V3 confirmation remained closed")


if __name__ == "__main__":
    main()
