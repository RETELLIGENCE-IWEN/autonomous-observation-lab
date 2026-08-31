"""Absolute performance contract and failure atlas for predictive position control.

The earlier evaluation protocols answer a relative question: whether a new
controller is no worse than its frozen reference.  This module answers the
separate absolute question: whether any controller is good enough for a
declared mission-style contract, and where its remaining losses originate.

All angular thresholds are normalized by the configured camera half-FOV and
all plant-capacity thresholds are normalized by the configured servo limits.
The analysis therefore remains valid while camera and servo hardware are
still being selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adaptive_position import (
    AdaptivePositionCandidate,
    AdaptivePositionProtocolConfig,
    _run_grid,
)
from .adaptive_position_v21 import (
    ADAPTIVE_POSITION_V21_SCHEMA_VERSION,
    adaptive_position_v2_config,
)
from .closed_loop import ClosedLoopScenario, ControllerRun, closed_loop_scenarios
from .controller_arena import _selected_candidate_config
from .estimators import angle_delta_rad
from .gru import CausalTargetStateGRU, load_gru_checkpoint
from .randomization import (
    GimbalDomainRandomizationConfig,
    randomize_closed_loop_scenario,
)


FAILURE_ATLAS_SCHEMA_VERSION = "gimbal_position_failure_atlas_v1"
DEFAULT_TRACKED_SCENARIOS = (
    "nominal_combined",
    "high_latency",
    "dropout_noise",
    "slow_servo",
    "aggressive_motion",
)


@dataclass(frozen=True)
class PerformanceContract:
    """Provisional absolute targets, independent of a particular camera/servo.

    Error limits are fractions of the selected-axis camera half-FOV.  For a
    60-degree camera, the default mean and P95 limits correspond to 7.5 and
    18 degrees.  They are deliberately stronger than the current controller
    and must be reviewed against the real mission before hardware acceptance.
    """

    name: str = "provisional_research_stretch_v1"
    tracked_scenarios: tuple[str, ...] = DEFAULT_TRACKED_SCENARIOS
    maximum_mean_absolute_error_fov_fraction: float = 0.25
    maximum_p95_absolute_error_fov_fraction: float = 0.60
    maximum_loss_of_view_fraction: float = 0.01
    maximum_avoidable_loss_fraction: float = 0.005
    maximum_unrecovered_loss_events: int = 0
    maximum_recovery_time_s: float = 0.75
    maximum_command_variation_per_s: float = 1.25
    maximum_actuator_acceleration_rms_normalized: float = 0.65
    maximum_rate_saturation_fraction: float = 0.01
    maximum_angle_saturation_fraction: float = 0.005

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("performance contract name must be non-empty")
        if not self.tracked_scenarios:
            raise ValueError("performance contract needs tracked scenarios")
        if len(set(self.tracked_scenarios)) != len(self.tracked_scenarios):
            raise ValueError("tracked scenario names must be unique")
        for name in (
            "maximum_mean_absolute_error_fov_fraction",
            "maximum_p95_absolute_error_fov_fraction",
            "maximum_loss_of_view_fraction",
            "maximum_avoidable_loss_fraction",
            "maximum_recovery_time_s",
            "maximum_command_variation_per_s",
            "maximum_actuator_acceleration_rms_normalized",
            "maximum_rate_saturation_fraction",
            "maximum_angle_saturation_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.maximum_unrecovered_loss_events < 0:
            raise ValueError(
                "maximum_unrecovered_loss_events must be non-negative"
            )


@dataclass(frozen=True)
class FailureAtlasConfig:
    """Configurable, hardware-relative loss-attribution thresholds."""

    onset_lookback_s: float = 0.50
    reversal_window_s: float = 0.40
    reversal_rate_threshold_fraction: float = 0.20
    rate_capacity_fraction: float = 0.90
    acceleration_capacity_fraction: float = 0.90
    minimum_recent_detection_valid_fraction: float = 0.50
    forecast_error_threshold_fov_fraction: float = 0.25
    command_error_threshold_fov_fraction: float = 0.35
    plant_tracking_error_threshold_fov_fraction: float = 0.35
    mechanical_limit_margin_fov_fraction: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "onset_lookback_s",
            "reversal_window_s",
            "reversal_rate_threshold_fraction",
            "rate_capacity_fraction",
            "acceleration_capacity_fraction",
            "minimum_recent_detection_valid_fraction",
            "forecast_error_threshold_fov_fraction",
            "command_error_threshold_fov_fraction",
            "plant_tracking_error_threshold_fov_fraction",
            "mechanical_limit_margin_fov_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "reversal_rate_threshold_fraction",
            "rate_capacity_fraction",
            "acceleration_capacity_fraction",
            "minimum_recent_detection_valid_fraction",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")


def load_performance_contract(path: str | Path) -> PerformanceContract:
    """Load a complete or partial contract override from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("performance contract JSON must contain an object")
    if "tracked_scenarios" in payload:
        payload["tracked_scenarios"] = tuple(payload["tracked_scenarios"])
    return PerformanceContract(**payload)


def _mechanically_reachable(
    scenario: ClosedLoopScenario,
    target_body_bearing_rad: float,
) -> bool:
    servo = scenario.config.servo
    camera = scenario.config.camera
    desired = math.atan2(
        math.sin(target_body_bearing_rad),
        math.cos(target_body_bearing_rad),
    )
    nearest = float(np.clip(desired, servo.min_angle_rad, servo.max_angle_rad))
    residual = abs(angle_delta_rad(desired, nearest))
    visible_limit = 0.5 * camera.selected_axis_fov_rad
    if camera.require_full_bbox_in_view:
        visible_limit -= 0.5 * scenario.config.scenario.target_angular_width_rad
    return residual <= max(0.0, visible_limit)


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else math.inf


def _loss_segments(lost: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    index = 0
    while index < len(lost):
        if not lost[index]:
            index += 1
            continue
        start = index
        while index < len(lost) and lost[index]:
            index += 1
        segments.append((start, index))
    return segments


def _reversal_indices(
    relative_rates: np.ndarray,
    *,
    control_period_s: float,
    window_s: float,
    threshold_rad_s: float,
) -> list[int]:
    signs = np.sign(relative_rates)
    crossings = np.flatnonzero(signs[1:] * signs[:-1] < 0.0) + 1
    radius = max(1, int(round(window_s / control_period_s)))
    return [
        int(index)
        for index in crossings
        if np.max(
            np.abs(
                relative_rates[
                    max(0, index - radius) : min(len(relative_rates), index + radius + 1)
                ]
            )
        )
        >= threshold_rad_s
    ]


def analyze_controller_run(
    run: ControllerRun,
    scenario: ClosedLoopScenario,
    *,
    controller_name: str,
    world_seed: int,
    training_seed: int,
    analysis: FailureAtlasConfig | None = None,
) -> dict[str, Any]:
    """Measure one run and attribute each loss-of-view onset."""

    analysis = analysis or FailureAtlasConfig()
    frames = run.episode.frames[1:]
    if not frames:
        raise ValueError("cannot analyze an empty controller run")
    config = run.episode.config
    half_fov = 0.5 * config.camera.selected_axis_fov_rad
    period_s = config.timing.control_period_s
    times = np.asarray([frame.diagnostics.time_s for frame in frames])
    errors = np.asarray(
        [frame.diagnostics.true_image_error_normalized for frame in frames]
    )
    lost = np.asarray(
        [not frame.diagnostics.target_in_view for frame in frames],
        dtype=bool,
    )
    target_body = np.asarray(
        [
            angle_delta_rad(
                frame.diagnostics.target_bearing_rad,
                frame.diagnostics.body_bearing_rad,
            )
            for frame in frames
        ]
    )
    relative_rates = np.asarray(
        [
            frame.diagnostics.target_rate_rad_s
            - frame.diagnostics.body_rate_rad_s
            for frame in frames
        ]
    )
    relative_accelerations = np.diff(
        relative_rates,
        prepend=relative_rates[0],
    ) / period_s
    reachable = np.asarray(
        [_mechanically_reachable(scenario, value) for value in target_body],
        dtype=bool,
    )
    detection_valid = np.asarray(
        [frame.observation.detection_valid for frame in frames],
        dtype=bool,
    )
    rate_capacity = (
        np.abs(relative_rates)
        >= analysis.rate_capacity_fraction * config.servo.max_rate_rad_s
    )
    acceleration_capacity = (
        np.abs(relative_accelerations)
        >= analysis.acceleration_capacity_fraction
        * config.servo.max_acceleration_rad_s2
    )
    gimbal_angles = np.asarray(
        [frame.diagnostics.gimbal_angle_rad for frame in frames]
    )
    near_mechanical_limit = np.minimum(
        gimbal_angles - config.servo.min_angle_rad,
        config.servo.max_angle_rad - gimbal_angles,
    ) <= analysis.mechanical_limit_margin_fov_fraction * half_fov
    saturated = np.asarray(
        [
            frame.diagnostics.rate_saturated
            or frame.diagnostics.angle_saturated
            for frame in frames
        ],
        dtype=bool,
    )

    forecast_errors = np.full(len(frames), np.nan, dtype=np.float64)
    for array_index in range(len(frames)):
        estimate_index = array_index + 1
        if estimate_index >= len(run.estimates):
            continue
        estimate = run.estimates[estimate_index]
        if not estimate.valid:
            continue
        target_angle, _ = scenario.target_motion.state_at(estimate.time_s)
        body_angle, _ = scenario.body_motion.state_at(estimate.time_s)
        truth = angle_delta_rad(target_angle, body_angle)
        forecast_errors[array_index] = abs(
            angle_delta_rad(
                estimate.body_relative_bearing_rad.value,
                truth,
            )
        ) / half_fov

    command_errors = np.empty(len(frames), dtype=np.float64)
    plant_tracking_errors = np.empty(len(frames), dtype=np.float64)
    oracle_preview_s = (
        config.servo.command_latency_s + config.servo.rate_time_constant_s
    )
    for index, frame in enumerate(frames):
        command = config.servo.position_from_normalized(
            frame.action.command_normalized
        )
        oracle_time_s = frame.diagnostics.time_s + oracle_preview_s
        target_angle, _ = scenario.target_motion.state_at(oracle_time_s)
        body_angle, _ = scenario.body_motion.state_at(oracle_time_s)
        oracle = angle_delta_rad(target_angle, body_angle)
        oracle = float(
            np.clip(
                oracle,
                config.servo.min_angle_rad,
                config.servo.max_angle_rad,
            )
        )
        command_errors[index] = abs(angle_delta_rad(command, oracle)) / half_fov
        plant_tracking_errors[index] = abs(
            angle_delta_rad(command, frame.diagnostics.gimbal_angle_rad)
        ) / half_fov

    reversal_indices = _reversal_indices(
        relative_rates,
        control_period_s=period_s,
        window_s=analysis.reversal_window_s,
        threshold_rad_s=(
            analysis.reversal_rate_threshold_fraction
            * config.servo.max_rate_rad_s
        ),
    )
    reversal_radius = max(1, int(round(analysis.reversal_window_s / period_s)))
    lookback = max(1, int(round(analysis.onset_lookback_s / period_s)))
    loss_events = []
    for start, end in _loss_segments(lost):
        context_start = max(0, start - lookback)
        # Attribute the onset only from causal evidence available through the
        # first lost frame. Future invalid detections are an effect of leaving
        # the FOV and must not be mislabeled as the cause of the loss.
        context_end = start + 1
        context = slice(context_start, context_end)
        onset_reachable = bool(reachable[start])
        recent_detection_fraction = float(np.mean(detection_valid[context]))
        forecast_error = _finite_median(forecast_errors[context])
        command_error = _finite_median(command_errors[context])
        plant_error = _finite_median(plant_tracking_errors[context])
        if not onset_reachable:
            cause = "physical_envelope"
        elif bool(np.any(near_mechanical_limit[context])):
            cause = "mechanical_limit"
        elif bool(np.any(rate_capacity[context])):
            cause = "servo_rate_capacity"
        elif bool(np.any(acceleration_capacity[context])):
            cause = "servo_acceleration_capacity"
        elif (
            recent_detection_fraction
            < analysis.minimum_recent_detection_valid_fraction
        ):
            cause = "detector_gap"
        elif forecast_error > analysis.forecast_error_threshold_fov_fraction:
            cause = "forecast_error"
        elif command_error > analysis.command_error_threshold_fov_fraction:
            cause = "command_timing_or_shaping"
        elif (
            plant_error > analysis.plant_tracking_error_threshold_fov_fraction
            or bool(np.any(saturated[context]))
        ):
            cause = "plant_tracking"
        else:
            cause = "unattributed_tracking"
        near_reversal = any(
            abs(index - start) <= reversal_radius for index in reversal_indices
        )
        loss_events.append(
            {
                "start_time_s": float(times[start]),
                "end_time_s": (
                    float(times[end]) if end < len(times) else None
                ),
                "duration_s": float((end - start) * period_s),
                "unrecovered_at_episode_end": end == len(lost),
                "terminally_censored": end == len(lost),
                "mechanically_reachable_at_onset": onset_reachable,
                "near_target_or_body_reversal": near_reversal,
                "recent_detection_valid_fraction": recent_detection_fraction,
                "median_forecast_error_fov_fraction": (
                    forecast_error if math.isfinite(forecast_error) else None
                ),
                "median_command_error_fov_fraction": command_error,
                "median_plant_tracking_error_fov_fraction": plant_error,
                "cause": cause,
            }
        )

    valid_forecast = forecast_errors[np.isfinite(forecast_errors)]
    return {
        "controller": controller_name,
        "scenario_name": scenario.name,
        "world_seed": world_seed,
        "training_seed": training_seed,
        "frame_count": len(frames),
        "mean_absolute_error_deg": run.metrics.mean_absolute_error_deg,
        "p95_absolute_error_deg": run.metrics.p95_absolute_error_deg,
        "mean_absolute_error_fov_fraction": float(np.mean(np.abs(errors))),
        "p95_absolute_error_fov_fraction": float(
            np.percentile(np.abs(errors), 95)
        ),
        "loss_of_view_fraction": float(np.mean(lost)),
        "avoidable_loss_fraction": float(np.mean(lost & reachable)),
        "physical_unreachable_fraction": float(np.mean(~reachable)),
        "detector_invalid_while_visible_fraction": float(
            np.mean((~detection_valid) & (~lost))
        ),
        "rate_capacity_exceeded_fraction": float(np.mean(rate_capacity)),
        "acceleration_capacity_exceeded_fraction": float(
            np.mean(acceleration_capacity)
        ),
        "forecast_error_fov_fraction": (
            float(np.mean(valid_forecast)) if len(valid_forecast) else None
        ),
        "command_oracle_error_fov_fraction": float(np.mean(command_errors)),
        "plant_tracking_error_fov_fraction": float(
            np.mean(plant_tracking_errors)
        ),
        "command_variation_per_s": run.metrics.command_variation_per_s,
        "actuator_acceleration_rms_normalized": (
            run.metrics.actuator_acceleration_rms_normalized
        ),
        "rate_saturation_fraction": run.metrics.rate_saturation_fraction,
        "angle_saturation_fraction": run.metrics.angle_saturation_fraction,
        "loss_of_view_events": run.metrics.loss_of_view_events,
        "unrecovered_loss_events": run.metrics.unrecovered_loss_events,
        "maximum_recovery_time_s": run.metrics.max_recovery_time_s,
        "reversal_count": len(reversal_indices),
        "loss_events_near_reversal": sum(
            bool(event["near_target_or_body_reversal"])
            for event in loss_events
        ),
        "loss_events": loss_events,
    }


def _weighted_mean(records: list[dict[str, Any]], key: str) -> float:
    pairs = [
        (float(record[key]), int(record["frame_count"]))
        for record in records
        if record.get(key) is not None
    ]
    if not pairs:
        return 0.0
    return float(
        sum(value * count for value, count in pairs)
        / sum(count for _, count in pairs)
    )


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty failure-atlas record set")
    events = [event for record in records for event in record["loss_events"]]
    causes = Counter(str(event["cause"]) for event in events)
    return {
        "episode_count": len(records),
        "frame_count": sum(int(record["frame_count"]) for record in records),
        "mean_absolute_error_deg": _weighted_mean(
            records, "mean_absolute_error_deg"
        ),
        "p95_absolute_error_deg": float(
            np.mean([record["p95_absolute_error_deg"] for record in records])
        ),
        "mean_absolute_error_fov_fraction": _weighted_mean(
            records, "mean_absolute_error_fov_fraction"
        ),
        "p95_absolute_error_fov_fraction": float(
            np.mean(
                [
                    record["p95_absolute_error_fov_fraction"]
                    for record in records
                ]
            )
        ),
        "loss_of_view_fraction": _weighted_mean(
            records, "loss_of_view_fraction"
        ),
        "avoidable_loss_fraction": _weighted_mean(
            records, "avoidable_loss_fraction"
        ),
        "physical_unreachable_fraction": _weighted_mean(
            records, "physical_unreachable_fraction"
        ),
        "detector_invalid_while_visible_fraction": _weighted_mean(
            records, "detector_invalid_while_visible_fraction"
        ),
        "rate_capacity_exceeded_fraction": _weighted_mean(
            records, "rate_capacity_exceeded_fraction"
        ),
        "acceleration_capacity_exceeded_fraction": _weighted_mean(
            records, "acceleration_capacity_exceeded_fraction"
        ),
        "forecast_error_fov_fraction": _weighted_mean(
            records, "forecast_error_fov_fraction"
        ),
        "command_oracle_error_fov_fraction": _weighted_mean(
            records, "command_oracle_error_fov_fraction"
        ),
        "plant_tracking_error_fov_fraction": _weighted_mean(
            records, "plant_tracking_error_fov_fraction"
        ),
        "command_variation_per_s": float(
            np.mean([record["command_variation_per_s"] for record in records])
        ),
        "actuator_acceleration_rms_normalized": float(
            np.mean(
                [
                    record["actuator_acceleration_rms_normalized"]
                    for record in records
                ]
            )
        ),
        "rate_saturation_fraction": _weighted_mean(
            records, "rate_saturation_fraction"
        ),
        "angle_saturation_fraction": _weighted_mean(
            records, "angle_saturation_fraction"
        ),
        "loss_of_view_events": sum(
            int(record["loss_of_view_events"]) for record in records
        ),
        "unrecovered_loss_events": sum(
            int(record["unrecovered_loss_events"]) for record in records
        ),
        "terminally_censored_loss_events": sum(
            bool(event["terminally_censored"]) for event in events
        ),
        "maximum_recovery_time_s": max(
            float(record["maximum_recovery_time_s"]) for record in records
        ),
        "reversal_count": sum(int(record["reversal_count"]) for record in records),
        "loss_events_near_reversal": sum(
            int(record["loss_events_near_reversal"]) for record in records
        ),
        "loss_event_causes": dict(sorted(causes.items())),
    }


def evaluate_contract(
    summary: dict[str, Any],
    contract: PerformanceContract,
) -> dict[str, Any]:
    """Evaluate an aggregate against every absolute threshold."""

    checks = {
        "mean_absolute_error": summary[
            "mean_absolute_error_fov_fraction"
        ]
        <= contract.maximum_mean_absolute_error_fov_fraction,
        "p95_absolute_error": summary[
            "p95_absolute_error_fov_fraction"
        ]
        <= contract.maximum_p95_absolute_error_fov_fraction,
        "loss_of_view": summary["loss_of_view_fraction"]
        <= contract.maximum_loss_of_view_fraction,
        "avoidable_loss": summary["avoidable_loss_fraction"]
        <= contract.maximum_avoidable_loss_fraction,
        "unrecovered_loss_events": summary["unrecovered_loss_events"]
        <= contract.maximum_unrecovered_loss_events,
        "recovery_time": summary["maximum_recovery_time_s"]
        <= contract.maximum_recovery_time_s,
        "command_variation": summary["command_variation_per_s"]
        <= contract.maximum_command_variation_per_s,
        "actuator_acceleration": summary[
            "actuator_acceleration_rms_normalized"
        ]
        <= contract.maximum_actuator_acceleration_rms_normalized,
        "rate_saturation": summary["rate_saturation_fraction"]
        <= contract.maximum_rate_saturation_fraction,
        "angle_saturation": summary["angle_saturation_fraction"]
        <= contract.maximum_angle_saturation_fraction,
    }
    return {
        "passed": all(checks.values()),
        "passed_check_count": sum(checks.values()),
        "check_count": len(checks),
        "checks": checks,
    }


def _load_models(
    result: dict[str, Any],
    training_seeds: tuple[int, ...],
    device: str,
) -> tuple[dict[int, CausalTargetStateGRU], dict[int, int]]:
    checkpoints = result.get("checkpoints")
    fixed_horizons = result.get("fixed_horizons")
    if not isinstance(checkpoints, dict) or not isinstance(fixed_horizons, dict):
        raise ValueError("visibility-risk result is missing model references")
    models: dict[int, CausalTargetStateGRU] = {}
    horizon_indices: dict[int, int] = {}
    for seed in training_seeds:
        checkpoint = checkpoints.get(str(seed))
        horizon = fixed_horizons.get(str(seed))
        if not isinstance(checkpoint, str) or not isinstance(horizon, dict):
            raise ValueError(f"training seed {seed} is unavailable")
        model, metadata = load_gru_checkpoint(checkpoint, device=device)
        recorded_seed = metadata.get("training_config", {}).get("seed")
        if recorded_seed is not None and int(recorded_seed) != seed:
            raise ValueError(f"training seed {seed} checkpoint mismatch")
        horizon_index = int(horizon["horizon_index"])
        if not 0 <= horizon_index < model.horizon_count:
            raise ValueError(f"training seed {seed} fixed horizon is invalid")
        models[seed] = model
        horizon_indices[seed] = horizon_index
    return models, horizon_indices


def _recommend_v3_priorities(
    controller: dict[str, Any],
) -> list[dict[str, Any]]:
    tracked = controller["tracked_summary"]
    by_scenario = controller["by_scenario"]
    failed_scenarios = [
        (name, values)
        for name, values in by_scenario.items()
        if values["contract_applicable"] and not values["contract"]["passed"]
    ]
    failed_scenarios.sort(
        key=lambda item: (
            item[1]["summary"]["avoidable_loss_fraction"],
            item[1]["summary"]["p95_absolute_error_fov_fraction"],
        ),
        reverse=True,
    )
    scenario_evidence = ", ".join(
        f"{name} ({100.0 * values['summary']['avoidable_loss_fraction']:.1f}% avoidable loss)"
        for name, values in failed_scenarios[:3]
    ) or "no tracked scenario failures"
    causes = Counter(tracked["loss_event_causes"])
    cause_actions = {
        "physical_envelope": (
            "Hardware and mission-envelope feasibility",
            "Keep travel, FOV, and motion bounds configurable; enlarge the selected hardware envelope or exclude impossible worlds from the controller acceptance claim.",
        ),
        "forecast_error": (
            "Control-aware multi-horizon training",
            "Penalize servo-arrival bearing error and visibility risk in the GRU objective.",
        ),
        "command_timing_or_shaping": (
            "Servo-aware constrained predictive control",
            "Optimize the position trajectory over the forecast instead of selecting one heuristic lead.",
        ),
        "servo_rate_capacity": (
            "Capacity-aware horizon and interception",
            "Use reachable-set constraints and start interception before rate demand reaches the configured limit.",
        ),
        "servo_acceleration_capacity": (
            "Acceleration-aware trajectory planning",
            "Include configured acceleration and jerk limits in the command optimizer.",
        ),
        "detector_gap": (
            "Dropout-robust state propagation",
            "Train longer causal gaps and make uncertainty explicitly gate recovery behavior.",
        ),
        "mechanical_limit": (
            "Mechanical-envelope planning",
            "Reserve travel margin and distinguish reachable re-entry from impossible pointing.",
        ),
        "plant_tracking": (
            "Identified servo dynamics in the controller",
            "Predict applied angle/rate from the configurable plant state before issuing the next setpoint.",
        ),
        "unattributed_tracking": (
            "Residual failure trace review",
            "Replay the unattributed onsets and add the missing state or constraint to the model.",
        ),
    }
    ranked = []
    for cause, count in causes.most_common():
        title, action = cause_actions[cause]
        ranked.append(
            {
                "failure_mode": cause,
                "event_count": int(count),
                "title": title,
                "evidence": scenario_evidence,
                "recommended_change": action,
            }
        )
    contract_checks = controller["contract"]["checks"]
    if (
        not contract_checks["mean_absolute_error"]
        or not contract_checks["p95_absolute_error"]
    ) and "forecast_error" not in causes:
        ranked.append(
            {
                "failure_mode": "forecast_and_command_error_budget",
                "event_count": 0,
                "title": "Control-aware forecasting and command optimization",
                "evidence": (
                    f"Mean forecast, oracle-command, and plant-tracking errors are "
                    f"{100.0 * tracked['forecast_error_fov_fraction']:.1f}%, "
                    f"{100.0 * tracked['command_oracle_error_fov_fraction']:.1f}%, "
                    f"and {100.0 * tracked['plant_tracking_error_fov_fraction']:.1f}% "
                    "of camera half-FOV."
                ),
                "recommended_change": (
                    "Train on servo-arrival pointing/visibility loss and consume the "
                    "multi-horizon forecast in a constrained position optimizer."
                ),
            }
        )
    if not ranked:
        ranked.append(
            {
                "failure_mode": "absolute_contract",
                "event_count": tracked["loss_of_view_events"],
                "title": "Reduce absolute tracking error",
                "evidence": scenario_evidence,
                "recommended_change": (
                    "Use constrained predictive position control and a control-aware training loss."
                ),
            }
        )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def build_position_failure_atlas(
    result: dict[str, Any],
    *,
    contract: PerformanceContract | None = None,
    analysis: FailureAtlasConfig | None = None,
    world_seeds: tuple[int, ...] | None = None,
    training_seeds: tuple[int, ...] | None = None,
    scenario_names: tuple[str, ...] | None = None,
    device: str = "cpu",
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replay frozen fixed/V2/V2.1 controllers and build the failure atlas."""

    if result.get("experiment") != ADAPTIVE_POSITION_V21_SCHEMA_VERSION:
        raise ValueError("unsupported visibility-risk result")
    confirmation = result.get("confirmation")
    if not isinstance(confirmation, dict) or not confirmation.get("opened"):
        raise ValueError("visibility-risk confirmation was not opened")
    contract = contract or PerformanceContract()
    analysis = analysis or FailureAtlasConfig()
    available_world_seeds = tuple(int(seed) for seed in confirmation["world_seeds"])
    available_training_seeds = tuple(int(seed) for seed in result["training_seeds"])
    selected_world_seeds = world_seeds or available_world_seeds
    selected_training_seeds = training_seeds or available_training_seeds
    if not selected_world_seeds or not set(selected_world_seeds) <= set(
        available_world_seeds
    ):
        raise ValueError("atlas world seeds must come from confirmation")
    if not selected_training_seeds or not set(selected_training_seeds) <= set(
        available_training_seeds
    ):
        raise ValueError("atlas training seeds are unavailable")

    scenario_catalog = {
        scenario.name: (index, scenario)
        for index, scenario in enumerate(closed_loop_scenarios())
    }
    selected_scenarios = scenario_names or tuple(scenario_catalog)
    if not selected_scenarios or not set(selected_scenarios) <= set(
        scenario_catalog
    ):
        raise ValueError("atlas scenario names are unavailable")
    tracked_selected = tuple(
        name for name in selected_scenarios if name in contract.tracked_scenarios
    )
    if not tracked_selected:
        raise ValueError("atlas selection contains no contract-tracked scenario")

    variants = tuple(
        (
            seed,
            scenario_catalog[name][0],
            randomize_closed_loop_scenario(
                scenario_catalog[name][1],
                seed=seed,
                config=GimbalDomainRandomizationConfig(),
            ),
        )
        for seed in selected_world_seeds
        for name in selected_scenarios
    )
    models, horizon_indices = _load_models(
        result,
        selected_training_seeds,
        device,
    )
    selected_candidate, v21_config = _selected_candidate_config(result)
    candidates = (
        AdaptivePositionCandidate("adaptive_v2", adaptive_position_v2_config()),
        AdaptivePositionCandidate("visibility_risk_v21", v21_config),
    )
    runtime = AdaptivePositionProtocolConfig(
        maximum_staleness_s=float(result["protocol"]["maximum_staleness_s"]),
        device=device,
        candidates=candidates,
    )
    fixed, adaptive = _run_grid(
        variants=variants,
        models=models,
        horizon_indices=horizon_indices,
        protocol=runtime,
        candidates=candidates,
    )
    run_sets = {
        "fixed_horizon": fixed,
        "adaptive_v2": adaptive["adaptive_v2"],
        "visibility_risk_v21": adaptive["visibility_risk_v21"],
    }

    all_records: list[dict[str, Any]] = []
    controller_results: dict[str, Any] = {}
    for controller_name, by_seed in run_sets.items():
        records = []
        for training_seed in selected_training_seeds:
            for variant_index, (world_seed, _scenario_index, scenario) in enumerate(
                variants
            ):
                record = analyze_controller_run(
                    by_seed[training_seed][variant_index],
                    scenario,
                    controller_name=controller_name,
                    world_seed=world_seed,
                    training_seed=training_seed,
                    analysis=analysis,
                )
                records.append(record)
                all_records.append(record)
        by_scenario = {}
        for scenario_name in selected_scenarios:
            scenario_records = [
                record
                for record in records
                if record["scenario_name"] == scenario_name
            ]
            summary = _aggregate_records(scenario_records)
            applicable = scenario_name in contract.tracked_scenarios
            by_scenario[scenario_name] = {
                "summary": summary,
                "contract_applicable": applicable,
                "contract": evaluate_contract(summary, contract) if applicable else None,
            }
        tracked_records = [
            record
            for record in records
            if record["scenario_name"] in contract.tracked_scenarios
        ]
        aggregate = _aggregate_records(records)
        tracked_summary = _aggregate_records(tracked_records)
        controller_results[controller_name] = {
            "aggregate_all_scenarios": aggregate,
            "tracked_summary": tracked_summary,
            "contract": evaluate_contract(tracked_summary, contract),
            "by_scenario": by_scenario,
        }

    metric_names = (
        "mean_absolute_error_fov_fraction",
        "p95_absolute_error_fov_fraction",
        "loss_of_view_fraction",
        "avoidable_loss_fraction",
        "command_variation_per_s",
        "actuator_acceleration_rms_normalized",
    )
    v21_summary = controller_results["visibility_risk_v21"]["tracked_summary"]
    comparisons = {}
    for reference_name in ("fixed_horizon", "adaptive_v2"):
        reference = controller_results[reference_name]["tracked_summary"]
        comparisons[f"v21_minus_{reference_name}"] = {
            key: float(v21_summary[key]) - float(reference[key])
            for key in metric_names
        } | {
            "unrecovered_loss_event_delta": (
                v21_summary["unrecovered_loss_events"]
                - reference["unrecovered_loss_events"]
            )
        }

    source_hash = None
    if source_path is not None:
        source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    atlas = {
        "experiment": FAILURE_ATLAS_SCHEMA_VERSION,
        "source_visibility_risk_result": (
            str(source_path) if source_path is not None else None
        ),
        "source_sha256": source_hash,
        "selected_v21_candidate": selected_candidate,
        "world_seeds": list(selected_world_seeds),
        "training_seeds": list(selected_training_seeds),
        "scenario_names": list(selected_scenarios),
        "contract": asdict(contract),
        "analysis_config": asdict(analysis),
        "controllers": controller_results,
        "comparisons": comparisons,
        "v3_priorities": _recommend_v3_priorities(
            controller_results["visibility_risk_v21"]
        ),
        "run_records": all_records,
    }
    return atlas


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an absolute performance contract and failure atlas."
    )
    parser.add_argument(
        "--visibility-risk-results",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_position_v21.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_position_failure_atlas.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        help="optional JSON file overriding provisional absolute thresholds",
    )
    parser.add_argument("--world-seed", type=int, action="append")
    parser.add_argument("--training-seed", type=int, action="append")
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = json.loads(
        args.visibility_risk_results.read_text(encoding="utf-8")
    )
    contract = (
        load_performance_contract(args.contract)
        if args.contract is not None
        else PerformanceContract()
    )
    atlas = build_position_failure_atlas(
        result,
        contract=contract,
        world_seeds=(tuple(args.world_seed) if args.world_seed else None),
        training_seeds=(
            tuple(args.training_seed) if args.training_seed else None
        ),
        scenario_names=(tuple(args.scenario) if args.scenario else None),
        device=args.device,
        source_path=args.visibility_risk_results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(atlas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict = atlas["controllers"]["visibility_risk_v21"]["contract"]
    print(f"wrote {args.output}")
    print(
        "V2.1 absolute contract: "
        f"{'PASS' if verdict['passed'] else 'FAIL'} "
        f"({verdict['passed_check_count']}/{verdict['check_count']} checks)"
    )


if __name__ == "__main__":
    main()
