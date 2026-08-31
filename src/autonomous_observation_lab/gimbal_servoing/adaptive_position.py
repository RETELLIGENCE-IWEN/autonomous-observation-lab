"""Validation-selected evaluation for the adaptive predictive position V2.

This module requires the optional ``learning`` dependency. A single adapter
configuration is selected across all supplied GRU initializations on the
recorded validation worlds, frozen, and evaluated once on the disjoint test
worlds. The existing fixed-horizon O2 position controller is replayed as the
paired reference.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .closed_loop import ControllerRun, run_closed_loop_controller
from .closed_loop import closed_loop_scenarios
from .config import GimbalCommandMode, ObservationProfile
from .controllers import (
    AdaptivePositionControllerConfig,
    AdaptiveTargetStatePositionController,
)
from .dataset import FEATURE_NAMES, GimbalDatasetManifest
from .gru import (
    CausalTargetStateGRU,
    GRUInferenceConfig,
    GRUTargetStateEstimator,
    load_gru_checkpoint,
)
from .gru_control import (
    GRUControlEvaluationConfig,
    _aggregate_runs,
    _control_cost,
    _learned_run,
    _load_manifest,
    _paired_comparison,
    _scenario_variants,
)
from .randomization import (
    GimbalDomainRandomizationConfig,
    randomize_closed_loop_scenario,
)


ADAPTIVE_POSITION_SCHEMA_VERSION = "gimbal_adaptive_position_v2_protocol_v1"


@dataclass(frozen=True)
class AdaptivePositionCandidate:
    name: str
    controller: AdaptivePositionControllerConfig

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("candidate name must be non-empty and identifier-like")


def default_adaptive_position_candidates() -> tuple[
    AdaptivePositionCandidate, ...
]:
    """Small predeclared grid spanning timing/trust/smoothness trade-offs."""

    return (
        AdaptivePositionCandidate(
            "near_transparent_calibrated",
            AdaptivePositionControllerConfig(
                actuator_arrival_time_scale=0.85,
                position_response_fraction=0.0,
                full_trust_std_ratio=1.0,
                zero_trust_std_ratio=4.0,
                minimum_prediction_weight=0.50,
                setpoint_rate_limit_scale=8.0,
                setpoint_acceleration_limit_scale=16.0,
                setpoint_jerk_rise_time_s=0.010,
            ),
        ),
        AdaptivePositionCandidate(
            "near_transparent_full_timing",
            AdaptivePositionControllerConfig(
                actuator_arrival_time_scale=1.0,
                position_response_fraction=0.0,
                full_trust_std_ratio=1.0,
                zero_trust_std_ratio=4.0,
                minimum_prediction_weight=0.50,
                setpoint_rate_limit_scale=8.0,
                setpoint_acceleration_limit_scale=16.0,
                setpoint_jerk_rise_time_s=0.010,
            ),
        ),
        AdaptivePositionCandidate(
            "light_smoothing_calibrated",
            AdaptivePositionControllerConfig(
                actuator_arrival_time_scale=0.85,
                position_response_fraction=0.0,
                full_trust_std_ratio=1.0,
                zero_trust_std_ratio=4.0,
                minimum_prediction_weight=0.50,
                setpoint_rate_limit_scale=6.0,
                setpoint_acceleration_limit_scale=12.0,
                setpoint_jerk_rise_time_s=0.015,
            ),
        ),
        AdaptivePositionCandidate(
            "light_smoothing_full_timing",
            AdaptivePositionControllerConfig(
                actuator_arrival_time_scale=1.0,
                position_response_fraction=0.0,
                full_trust_std_ratio=1.0,
                zero_trust_std_ratio=4.0,
                minimum_prediction_weight=0.50,
                setpoint_rate_limit_scale=6.0,
                setpoint_acceleration_limit_scale=12.0,
                setpoint_jerk_rise_time_s=0.015,
            ),
        ),
    )


@dataclass(frozen=True)
class AdaptivePositionProtocolConfig:
    maximum_staleness_s: float = 0.50
    validation_max_mean_error_regression_deg: float = 0.25
    validation_max_p95_regression_deg: float = 0.50
    validation_max_loss_of_view_regression: float = 0.005
    validation_max_control_cost_regression: float = 0.020
    test_max_mean_error_regression_deg: float = 0.25
    test_max_p95_regression_deg: float = 0.50
    test_max_loss_of_view_regression: float = 0.005
    test_minimum_command_variation_reduction_fraction: float = 0.05
    test_max_scenario_p95_regression_deg: float = 2.0
    test_max_scenario_loss_of_view_regression: float = 0.02
    device: str = "cpu"
    candidates: tuple[AdaptivePositionCandidate, ...] = (
        default_adaptive_position_candidates()
    )

    def __post_init__(self) -> None:
        if self.maximum_staleness_s < 0.0:
            raise ValueError("maximum staleness must be non-negative")
        for field_name in (
            "validation_max_mean_error_regression_deg",
            "validation_max_p95_regression_deg",
            "validation_max_loss_of_view_regression",
            "validation_max_control_cost_regression",
            "test_max_mean_error_regression_deg",
            "test_max_p95_regression_deg",
            "test_max_loss_of_view_regression",
            "test_minimum_command_variation_reduction_fraction",
            "test_max_scenario_p95_regression_deg",
            "test_max_scenario_loss_of_view_regression",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not self.candidates:
            raise ValueError("at least one adaptive-position candidate is required")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("adaptive-position candidate names must be unique")


def _load_checkpoint(
    path: Path,
    training_seed: int,
    manifests: dict[str, GimbalDatasetManifest],
    device: str,
) -> tuple[CausalTargetStateGRU, dict[str, Any], int]:
    model, metadata = load_gru_checkpoint(path, device=device)
    if metadata.get("profile") != ObservationProfile.DISTURBANCE_AWARE.value:
        raise ValueError(f"checkpoint {path} is not an O2 model")
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError(f"checkpoint {path} feature schema mismatch")
    hashes = metadata.get("dataset_hashes")
    if not isinstance(hashes, dict):
        raise ValueError(f"checkpoint {path} is missing dataset hashes")
    for split, manifest in manifests.items():
        if hashes.get(split) != manifest.configuration_hash:
            raise ValueError(f"checkpoint {path} {split} dataset hash mismatch")
    recorded_seed = metadata.get("training_config", {}).get("seed")
    if recorded_seed is not None and int(recorded_seed) != training_seed:
        raise ValueError(f"checkpoint {path} training seed mismatch")
    selected = metadata.get("selected_horizons", {}).get("position")
    if not isinstance(selected, dict):
        raise ValueError(f"checkpoint {path} lacks fixed position selection")
    horizon_index = int(selected["horizon_index"])
    if not 0 <= horizon_index < model.horizon_count:
        raise ValueError(f"checkpoint {path} selected horizon is invalid")
    return model, metadata, horizon_index


def _adaptive_run(
    *,
    scenario: Any,
    seed: int,
    model: CausalTargetStateGRU,
    adapter: AdaptivePositionControllerConfig,
    evaluation: AdaptivePositionProtocolConfig,
    name: str = "gru_o2_position_v2",
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
            maximum_staleness_s=evaluation.maximum_staleness_s,
            device=evaluation.device,
        ),
    )
    controller = AdaptiveTargetStatePositionController(
        estimator=estimator,
        servo=config.servo,
        config=adapter,
        name=name,
    )
    return run_closed_loop_controller(
        name=name,
        description=(
            "O2 multi-horizon GRU with servo-arrival interpolation, relative "
            "uncertainty trust, native hold, and jerk-limited position shaping."
        ),
        scenario=scenario,
        config=config,
        controller=controller,
        seed=seed,
    )


def _fixed_run(
    *,
    scenario: Any,
    seed: int,
    model: CausalTargetStateGRU,
    horizon_index: int,
    evaluation: AdaptivePositionProtocolConfig,
) -> ControllerRun:
    return _learned_run(
        scenario=scenario,
        seed=seed,
        model=model,
        profile=ObservationProfile.DISTURBANCE_AWARE,
        horizon_index=horizon_index,
        command_mode=GimbalCommandMode.POSITION,
        evaluation=GRUControlEvaluationConfig(
            maximum_staleness_s=evaluation.maximum_staleness_s,
            device=evaluation.device,
        ),
        search_fallback=False,
    )


def _summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    metrics = aggregate["mean_metrics"]
    return {
        "mean_absolute_error_deg": metrics["mean_absolute_error_deg"],
        "p95_absolute_error_deg": metrics["p95_absolute_error_deg"],
        "loss_of_view_fraction": metrics["loss_of_view_fraction"],
        "command_variation_per_s": metrics["command_variation_per_s"],
        "command_rms_normalized": metrics["command_rms_normalized"],
        "rate_saturation_fraction": metrics["rate_saturation_fraction"],
        "actuator_acceleration_rms_normalized": metrics[
            "actuator_acceleration_rms_normalized"
        ],
        "mean_control_cost": aggregate["mean_control_cost"],
        "total_unrecovered_loss_events": aggregate[
            "total_unrecovered_loss_events"
        ],
        "event_weighted_mean_recovery_time_s": aggregate[
            "event_weighted_mean_recovery_time_s"
        ],
    }


def _adapter_diagnostic_summary(runs: Sequence[ControllerRun]) -> dict[str, float]:
    diagnostics = [
        item
        for run in runs
        for item in run.adapter_diagnostics
        if item.get("valid", False)
    ]
    if not diagnostics:
        return {
            "valid_step_count": 0,
            "mean_requested_horizon_s": 0.0,
            "mean_effective_horizon_s": 0.0,
            "mean_prediction_weight": 0.0,
            "mean_uncertainty_ratio": 0.0,
            "rate_limited_fraction": 0.0,
            "acceleration_limited_fraction": 0.0,
            "jerk_limited_fraction": 0.0,
        }

    def mean(name: str) -> float:
        return float(np.mean([float(item[name]) for item in diagnostics]))

    return {
        "valid_step_count": len(diagnostics),
        "mean_requested_horizon_s": mean("requested_horizon_s"),
        "mean_effective_horizon_s": mean("effective_horizon_s"),
        "mean_prediction_weight": mean("prediction_weight"),
        "mean_uncertainty_ratio": mean("uncertainty_ratio"),
        "rate_limited_fraction": mean("rate_limited"),
        "acceleration_limited_fraction": mean("acceleration_limited"),
        "jerk_limited_fraction": mean("jerk_limited"),
    }


def _delta(candidate: dict[str, Any], reference: dict[str, Any], name: str) -> float:
    return float(candidate[name]) - float(reference[name])


def _validation_gate(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    protocol: AdaptivePositionProtocolConfig,
) -> dict[str, Any]:
    checks = {
        "mean_error": _delta(candidate, reference, "mean_absolute_error_deg")
        <= protocol.validation_max_mean_error_regression_deg,
        "p95_error": _delta(candidate, reference, "p95_absolute_error_deg")
        <= protocol.validation_max_p95_regression_deg,
        "loss_of_view": _delta(candidate, reference, "loss_of_view_fraction")
        <= protocol.validation_max_loss_of_view_regression,
        "control_cost": _delta(candidate, reference, "mean_control_cost")
        <= protocol.validation_max_control_cost_regression,
        "unrecovered_events": candidate["total_unrecovered_loss_events"]
        <= reference["total_unrecovered_loss_events"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {
            name: _delta(candidate, reference, name)
            for name in (
                "mean_absolute_error_deg",
                "p95_absolute_error_deg",
                "loss_of_view_fraction",
                "command_variation_per_s",
                "mean_control_cost",
            )
        },
        "unrecovered_event_delta": (
            candidate["total_unrecovered_loss_events"]
            - reference["total_unrecovered_loss_events"]
        ),
    }


def _run_grid(
    *,
    variants: Sequence[tuple[int, int, Any]],
    models: dict[int, CausalTargetStateGRU],
    horizon_indices: dict[int, int],
    protocol: AdaptivePositionProtocolConfig,
    candidates: Sequence[AdaptivePositionCandidate],
) -> tuple[
    dict[int, list[ControllerRun]],
    dict[str, dict[int, list[ControllerRun]]],
]:
    fixed = {training_seed: [] for training_seed in models}
    adaptive = {
        candidate.name: {training_seed: [] for training_seed in models}
        for candidate in candidates
    }
    for training_seed, model in models.items():
        for world_seed, _scenario_index, scenario in variants:
            fixed[training_seed].append(
                _fixed_run(
                    scenario=scenario,
                    seed=world_seed,
                    model=model,
                    horizon_index=horizon_indices[training_seed],
                    evaluation=protocol,
                )
            )
            for candidate in candidates:
                adaptive[candidate.name][training_seed].append(
                    _adaptive_run(
                        scenario=scenario,
                        seed=world_seed,
                        model=model,
                        adapter=candidate.controller,
                        evaluation=protocol,
                    )
                )
    return fixed, adaptive


def _fresh_variants(
    seeds: tuple[int, ...],
) -> tuple[tuple[int, int, Any], ...]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("fresh test seeds must be non-empty and unique")
    randomization = GimbalDomainRandomizationConfig()
    return tuple(
        (
            seed,
            scenario_index,
            randomize_closed_loop_scenario(
                scenario,
                seed=seed,
                config=randomization,
            ),
        )
        for seed in seeds
        for scenario_index, scenario in enumerate(closed_loop_scenarios())
    )


def _flatten(runs: dict[int, list[ControllerRun]]) -> list[ControllerRun]:
    return [run for seed_runs in runs.values() for run in seed_runs]


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _seed_summaries(
    candidate_runs: dict[int, list[ControllerRun]],
    reference_runs: dict[int, list[ControllerRun]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    metric_names = (
        "mean_absolute_error_deg",
        "p95_absolute_error_deg",
        "loss_of_view_fraction",
        "command_variation_per_s",
        "mean_control_cost",
    )
    for seed in candidate_runs:
        candidate = _summary(_aggregate_runs(candidate_runs[seed]))
        reference = _summary(_aggregate_runs(reference_runs[seed]))
        summaries[str(seed)] = {
            "fixed": reference,
            "v2": candidate,
            "deltas": {
                name: _delta(candidate, reference, name) for name in metric_names
            },
        }
    summaries["distributions"] = {
        controller: {
            metric: _distribution(
                [
                    summaries[str(seed)][controller][metric]
                    for seed in candidate_runs
                ]
            )
            for metric in metric_names
        }
        for controller in ("fixed", "v2")
    }
    return summaries


def _scenario_summaries_from_variants(
    *,
    variants: Sequence[tuple[int, int, Any]],
    candidate_runs: dict[int, list[ControllerRun]],
    reference_runs: dict[int, list[ControllerRun]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for training_seed in candidate_runs:
        for variant, candidate, reference in zip(
            variants,
            candidate_runs[training_seed],
            reference_runs[training_seed],
            strict=True,
        ):
            scenario_name = variant[2].name
            bucket = result.setdefault(
                scenario_name,
                {"fixed_runs": [], "v2_runs": []},
            )
            bucket["fixed_runs"].append(reference)
            bucket["v2_runs"].append(candidate)
    serialized = {}
    for scenario_name, bucket in result.items():
        fixed = _summary(_aggregate_runs(bucket["fixed_runs"]))
        candidate = _summary(_aggregate_runs(bucket["v2_runs"]))
        serialized[scenario_name] = {
            "fixed": fixed,
            "v2": candidate,
            "deltas": {
                name: _delta(candidate, fixed, name)
                for name in (
                    "mean_absolute_error_deg",
                    "p95_absolute_error_deg",
                    "loss_of_view_fraction",
                    "command_variation_per_s",
                    "mean_control_cost",
                )
            },
        }
    return serialized


def _test_gate(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    seed_summaries: dict[str, Any],
    scenario_summaries: dict[str, Any],
    protocol: AdaptivePositionProtocolConfig,
) -> dict[str, Any]:
    reference_variation = float(reference["command_variation_per_s"])
    variation_reduction = (
        (reference_variation - candidate["command_variation_per_s"])
        / reference_variation
        if reference_variation > 0.0
        else 0.0
    )
    aggregate_checks = {
        "mean_error": _delta(candidate, reference, "mean_absolute_error_deg")
        <= protocol.test_max_mean_error_regression_deg,
        "p95_error": _delta(candidate, reference, "p95_absolute_error_deg")
        <= protocol.test_max_p95_regression_deg,
        "loss_of_view": _delta(candidate, reference, "loss_of_view_fraction")
        <= protocol.test_max_loss_of_view_regression,
        "command_variation": variation_reduction
        >= protocol.test_minimum_command_variation_reduction_fraction,
        "unrecovered_events": candidate["total_unrecovered_loss_events"]
        <= reference["total_unrecovered_loss_events"],
    }
    per_seed_checks = {
        seed: (
            value["deltas"]["mean_absolute_error_deg"]
            <= protocol.test_max_mean_error_regression_deg
            and value["deltas"]["p95_absolute_error_deg"]
            <= protocol.test_max_p95_regression_deg
            and value["deltas"]["loss_of_view_fraction"]
            <= protocol.test_max_loss_of_view_regression
        )
        for seed, value in seed_summaries.items()
        if seed != "distributions"
    }
    scenario_checks = {
        scenario: (
            value["deltas"]["p95_absolute_error_deg"]
            <= protocol.test_max_scenario_p95_regression_deg
            and value["deltas"]["loss_of_view_fraction"]
            <= protocol.test_max_scenario_loss_of_view_regression
        )
        for scenario, value in scenario_summaries.items()
    }
    return {
        "passed": (
            all(aggregate_checks.values())
            and all(per_seed_checks.values())
            and all(scenario_checks.values())
        ),
        "aggregate_checks": aggregate_checks,
        "per_training_seed_core_checks": per_seed_checks,
        "per_scenario_tail_visibility_checks": scenario_checks,
        "command_variation_reduction_fraction": variation_reduction,
        "aggregate_deltas": {
            name: _delta(candidate, reference, name)
            for name in (
                "mean_absolute_error_deg",
                "p95_absolute_error_deg",
                "loss_of_view_fraction",
                "command_variation_per_s",
                "mean_control_cost",
            )
        },
        "unrecovered_event_delta": (
            candidate["total_unrecovered_loss_events"]
            - reference["total_unrecovered_loss_events"]
        ),
    }


def _representative_trace(
    fixed: ControllerRun,
    candidate: ControllerRun,
    *,
    world_seed: int,
    training_seed: int,
    scenario_name: str,
) -> dict[str, Any]:
    length = min(
        len(candidate.adapter_diagnostics),
        len(candidate.episode.frames),
        len(fixed.episode.frames),
    )
    records = []
    for index in range(length):
        item = candidate.adapter_diagnostics[index]
        candidate_frame = candidate.episode.frames[index]
        fixed_frame = fixed.episode.frames[index]
        fixed_command_frame = fixed.episode.frames[index + 1]
        fixed_command_rad = fixed.episode.config.servo.position_from_normalized(
            fixed_command_frame.action.command_normalized
        )
        records.append(
            {
                "time_s": candidate_frame.diagnostics.time_s,
                "target_body_bearing_deg": math.degrees(
                    candidate_frame.diagnostics.target_bearing_rad
                    - candidate_frame.diagnostics.body_bearing_rad
                ),
                "fixed_gimbal_angle_deg": math.degrees(
                    fixed_frame.diagnostics.gimbal_angle_rad
                ),
                "v2_gimbal_angle_deg": math.degrees(
                    candidate_frame.diagnostics.gimbal_angle_rad
                ),
                "fixed_command_deg": math.degrees(fixed_command_rad),
                "v2_shaped_command_deg": math.degrees(
                    float(item.get("shaped_target_angle_rad", 0.0))
                ),
                "v2_raw_target_deg": math.degrees(
                    float(item.get("raw_target_angle_rad", 0.0))
                ),
                "requested_horizon_s": float(
                    item.get("requested_horizon_s", 0.0)
                ),
                "effective_horizon_s": float(
                    item.get("effective_horizon_s", 0.0)
                ),
                "prediction_weight": float(item.get("prediction_weight", 0.0)),
                "uncertainty_ratio": float(item.get("uncertainty_ratio", 0.0)),
                "target_in_view": candidate_frame.diagnostics.target_in_view,
            }
        )
    return {
        "world_seed": world_seed,
        "training_seed": training_seed,
        "scenario_name": scenario_name,
        "records": records,
    }


def evaluate_adaptive_position_v2(
    *,
    validation_data: str | Path,
    test_data: str | Path,
    checkpoints: dict[int, str | Path],
    protocol: AdaptivePositionProtocolConfig | None = None,
    fresh_test_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Select one V2 configuration on validation, then open test once."""

    protocol = protocol or AdaptivePositionProtocolConfig()
    if len(checkpoints) < 2:
        raise ValueError("V2 evaluation requires at least two training seeds")
    manifests = {
        "validation": _load_manifest(validation_data),
        "test": _load_manifest(test_data),
    }
    if set(manifests["validation"].seeds) & set(manifests["test"].seeds):
        raise ValueError("validation and test world seeds must be disjoint")
    if (
        manifests["validation"].prediction_horizons_s
        != manifests["test"].prediction_horizons_s
    ):
        raise ValueError("validation and test prediction horizons differ")
    validation_variants = _scenario_variants(manifests["validation"])
    models: dict[int, CausalTargetStateGRU] = {}
    horizon_indices: dict[int, int] = {}
    checkpoint_paths = {
        int(seed): Path(path) for seed, path in checkpoints.items()
    }
    for training_seed, path in checkpoint_paths.items():
        model, _model_metadata, horizon_index = _load_checkpoint(
            path,
            training_seed,
            manifests,
            protocol.device,
        )
        models[training_seed] = model
        horizon_indices[training_seed] = horizon_index

    fixed_validation, candidate_validation = _run_grid(
        variants=validation_variants,
        models=models,
        horizon_indices=horizon_indices,
        protocol=protocol,
        candidates=protocol.candidates,
    )
    fixed_validation_runs = _flatten(fixed_validation)
    fixed_validation_summary = _summary(
        _aggregate_runs(fixed_validation_runs)
    )
    validation_candidates = []
    for candidate in protocol.candidates:
        runs = _flatten(candidate_validation[candidate.name])
        summary = _summary(_aggregate_runs(runs))
        gate = _validation_gate(summary, fixed_validation_summary, protocol)
        validation_candidates.append(
            {
                "name": candidate.name,
                "controller_config": asdict(candidate.controller),
                "summary": summary,
                "adapter_diagnostics": _adapter_diagnostic_summary(runs),
                "gate": gate,
            }
        )
    eligible = [item for item in validation_candidates if item["gate"]["passed"]]
    common = {
        "experiment": ADAPTIVE_POSITION_SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "dataset_hashes": {
            split: manifest.configuration_hash
            for split, manifest in manifests.items()
        },
        "training_seeds": list(models),
        "checkpoints": {
            str(seed): str(path) for seed, path in checkpoint_paths.items()
        },
        "fixed_horizons": {
            str(seed): {
                "horizon_index": horizon_indices[seed],
                "horizon_s": models[seed].config.prediction_horizons_s[
                    horizon_indices[seed]
                ],
            }
            for seed in models
        },
        "selection_policy": (
            "One candidate is selected across every training initialization "
            "on disjoint validation worlds. Candidates passing the declared "
            "tracking/visibility gate are ranked by command variation, then "
            "control cost. Test remains closed if no candidate passes."
        ),
    }
    validation_result = {
        "variant_count_per_training_seed": len(validation_variants),
        "fixed_summary": fixed_validation_summary,
        "candidates": validation_candidates,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": None,
        "selected_candidate_passed_gate": False,
    }
    if not eligible:
        return {
            **common,
            "validation": validation_result,
            "test": {
                "opened": False,
                "reason": "no adaptive candidate passed the validation gate",
                "recommendation": "retain_fixed_horizon_position",
            },
        }

    selected_record = min(
        eligible,
        key=lambda item: (
            item["summary"]["command_variation_per_s"],
            item["summary"]["mean_control_cost"],
        ),
    )
    validation_result["selected_candidate"] = selected_record["name"]
    validation_result["selected_candidate_passed_gate"] = True
    selected_candidate = next(
        candidate
        for candidate in protocol.candidates
        if candidate.name == selected_record["name"]
    )

    if fresh_test_seeds is None:
        test_variants = _scenario_variants(manifests["test"])
        test_source = "recorded_dataset_test_variants"
        evaluated_test_seeds = tuple(manifests["test"].seeds)
    else:
        occupied = set(manifests["validation"].seeds) | set(
            manifests["test"].seeds
        )
        if occupied & set(fresh_test_seeds):
            raise ValueError("fresh test seeds overlap recorded data splits")
        test_variants = _fresh_variants(fresh_test_seeds)
        test_source = "fresh_domain_randomization"
        evaluated_test_seeds = fresh_test_seeds

    fixed_test, adaptive_test_grid = _run_grid(
        variants=test_variants,
        models=models,
        horizon_indices=horizon_indices,
        protocol=protocol,
        candidates=(selected_candidate,),
    )
    adaptive_test = adaptive_test_grid[selected_candidate.name]
    fixed_test_runs = _flatten(fixed_test)
    adaptive_test_runs = _flatten(adaptive_test)
    fixed_test_summary = _summary(_aggregate_runs(fixed_test_runs))
    adaptive_test_summary = _summary(_aggregate_runs(adaptive_test_runs))
    seed_summaries = _seed_summaries(adaptive_test, fixed_test)
    scenario_summaries = _scenario_summaries_from_variants(
        variants=test_variants,
        candidate_runs=adaptive_test,
        reference_runs=fixed_test,
    )
    test_gate = _test_gate(
        candidate=adaptive_test_summary,
        reference=fixed_test_summary,
        seed_summaries=seed_summaries,
        scenario_summaries=scenario_summaries,
        protocol=protocol,
    )

    first_training_seed = next(iter(models))
    representative_training_seed = first_training_seed
    representative_variant_index = next(
        (
            index
            for index, variant in enumerate(test_variants)
            if variant[2].name == "nominal_combined"
        ),
        0,
    )
    terminal_regressions = [
        (
            candidate.metrics.unrecovered_loss_events
            - reference.metrics.unrecovered_loss_events,
            training_seed,
            index,
        )
        for training_seed in adaptive_test
        for index, (candidate, reference) in enumerate(
            zip(
                adaptive_test[training_seed],
                fixed_test[training_seed],
                strict=True,
            )
        )
    ]
    worst_terminal_delta, worst_training_seed, worst_variant_index = max(
        terminal_regressions
    )
    if worst_terminal_delta > 0:
        representative_training_seed = worst_training_seed
        representative_variant_index = worst_variant_index
    representative_variant = test_variants[representative_variant_index]
    representative = _representative_trace(
        fixed_test[representative_training_seed][representative_variant_index],
        adaptive_test[representative_training_seed][representative_variant_index],
        world_seed=representative_variant[0],
        training_seed=representative_training_seed,
        scenario_name=representative_variant[2].name,
    )

    paired = _paired_comparison(adaptive_test_runs, fixed_test_runs)
    return {
        **common,
        "validation": validation_result,
        "test": {
            "opened": True,
            "source": test_source,
            "world_seeds": list(evaluated_test_seeds),
            "variant_count_per_training_seed": len(test_variants),
            "fixed_summary": fixed_test_summary,
            "v2_summary": adaptive_test_summary,
            "adapter_diagnostics": _adapter_diagnostic_summary(
                adaptive_test_runs
            ),
            "paired_v2_minus_fixed": paired,
            "by_training_seed": seed_summaries,
            "by_scenario": scenario_summaries,
            "acceptance_gate": test_gate,
            "recommendation": (
                "adaptive_position_v2"
                if test_gate["passed"]
                else "retain_fixed_horizon_position"
            ),
        },
        "representative_trace": representative,
    }


def _checkpoint_argument(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("checkpoint must be TRAINING_SEED=PATH")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint seed must be an integer") from error
    return seed, Path(path_text)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and evaluate adaptive predictive position V2."
    )
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=_checkpoint_argument,
        action="append",
        required=True,
        help="repeat TRAINING_SEED=PATH for each O2 initialization",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fresh-test-seed",
        type=int,
        action="append",
        dest="fresh_test_seeds",
        help=(
            "evaluate freshly randomized worlds after validation selection; "
            "repeat for each disjoint seed"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise SystemExit("checkpoint training seeds must be unique")
    result = evaluate_adaptive_position_v2(
        validation_data=args.validation_data,
        test_data=args.test_data,
        checkpoints=checkpoints,
        protocol=AdaptivePositionProtocolConfig(device=args.device),
        fresh_test_seeds=(
            tuple(args.fresh_test_seeds)
            if args.fresh_test_seeds is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
