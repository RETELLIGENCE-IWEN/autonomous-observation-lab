"""Versioned baseline registry and development-only C2 PID protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .adaptive_position import (
    AdaptivePositionProtocolConfig,
    _adaptive_run,
    _fresh_variants,
    _summary,
)
from .closed_loop import ClosedLoopScenario, ControllerRun, run_closed_loop_controller
from .config import GimbalCommandMode, ObservationProfile
from .controllers import RobustPIDControllerConfig, RobustPIDGimbalController
from .conventional_champion import (
    _selected_adapter_config,
    conventional_champion_run,
    practical_feedback_run,
)
from .estimators import (
    ConstantVelocityEstimatorConfig,
    MultiHorizonConstantVelocityTargetEstimator,
)
from .gru_control import _aggregate_runs
from .gru import load_gru_checkpoint
from .model_predictive_control import (
    IdentifiedMPCControllerConfig,
    IdentifiedMPCGimbalController,
)
from .randomization import GimbalDomainRandomizationConfig


BASELINE_BENCHMARK_SCHEMA_VERSION = "gimbal_baseline_benchmark_v1_development"
DEFAULT_DEVELOPMENT_SEEDS = tuple(range(120000, 120008))
DEFAULT_SEALED_TEST_SEEDS = tuple(range(121000, 121008))
PRIMARY_SCENARIOS = (
    "nominal_combined",
    "high_latency",
    "dropout_noise",
    "slow_servo",
    "aggressive_motion",
)


@dataclass(frozen=True)
class BaselineMethodSpec:
    identifier: str
    name: str
    family: str
    scientific_purpose: str
    implementation_status: str
    deployable: bool

    def __post_init__(self) -> None:
        if not self.identifier or not self.name:
            raise ValueError("baseline identifier and name must be non-empty")
        if self.family not in {
            "conventional",
            "learned",
            "proposed",
            "upper_bound",
        }:
            raise ValueError("unsupported baseline family")
        if self.implementation_status not in {
            "ready",
            "partial",
            "missing",
        }:
            raise ValueError("unsupported implementation status")


def baseline_method_registry() -> tuple[BaselineMethodSpec, ...]:
    """Return the concept-locked ten-method comparison registry."""

    return (
        BaselineMethodSpec(
            "C0",
            "Hold / zero action",
            "conventional",
            "Sanity lower bound",
            "ready",
            True,
        ),
        BaselineMethodSpec(
            "C1",
            "Latency-scheduled proportional visual servo",
            "conventional",
            "Minimum credible feedback controller",
            "ready",
            True,
        ),
        BaselineMethodSpec(
            "C2",
            "Filtered PID with anti-windup and IMU feed-forward",
            "conventional",
            "Strong conventional feedback baseline",
            "ready",
            True,
        ),
        BaselineMethodSpec(
            "C3",
            "Estimator plus constrained position adapter",
            "conventional",
            "Tests whether simple predictive state estimation is enough",
            "ready",
            True,
        ),
        BaselineMethodSpec(
            "C4",
            "Identified DMC / MPC",
            "conventional",
            "Strong explicit-model predictive baseline",
            "ready",
            True,
        ),
        BaselineMethodSpec(
            "L0",
            "Feed-forward SAC or TD3",
            "learned",
            "Tests whether an instantaneous nonlinear policy is enough",
            "missing",
            True,
        ),
        BaselineMethodSpec(
            "L1",
            "Recurrent SAC or TD3",
            "learned",
            "Tests model-free learned memory",
            "missing",
            True,
        ),
        BaselineMethodSpec(
            "L2",
            "Supervised neural inverse controller",
            "learned",
            "Tests whether reinforcement learning is necessary",
            "partial",
            True,
        ),
        BaselineMethodSpec(
            "P",
            "Recurrent predictive Dream-to-Center controller",
            "proposed",
            "Proposed deployable mechanism",
            "ready",
            True,
        ),
        BaselineMethodSpec(
            "UB",
            "Privileged constrained sequence oracle",
            "upper_bound",
            "Approximate non-deployable performance ceiling",
            "ready",
            False,
        ),
    )


@dataclass(frozen=True)
class RobustPIDCandidate:
    name: str
    controller: RobustPIDControllerConfig

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("PID candidate name must be identifier-like")


@dataclass(frozen=True)
class IdentifiedMPCCandidate:
    name: str
    prediction_horizon_s: float
    velocity_filter_coefficient: float
    controller: IdentifiedMPCControllerConfig

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("MPC candidate name must be identifier-like")
        if (
            not math.isfinite(self.prediction_horizon_s)
            or self.prediction_horizon_s <= 0.0
        ):
            raise ValueError("MPC prediction horizon must be positive")
        if (
            not math.isfinite(self.velocity_filter_coefficient)
            or not 0.0 < self.velocity_filter_coefficient <= 1.0
        ):
            raise ValueError("MPC velocity filter coefficient must be in (0, 1]")


def default_robust_pid_candidates() -> tuple[RobustPIDCandidate, ...]:
    """Predeclared C2 candidates; no grid may be added after test opening."""

    shared = {
        "body_rate_feedforward_gain": 1.0,
        "maximum_detection_gap_s": 0.25,
        "gain_schedule_reference_delay_s": 0.25,
        "minimum_gain_multiplier": 0.35,
        "maximum_gain_multiplier": 1.50,
        "command_rate_limit_scale": 1.0,
        "command_acceleration_limit_scale": 1.0,
        "command_jerk_rise_time_s": 0.04,
    }
    return (
        RobustPIDCandidate(
            "imu_p_low_025",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=0.25,
                integral_gain_s_inv2=0.0,
                derivative_gain=0.0,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_p_low_050",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=0.50,
                integral_gain_s_inv2=0.0,
                derivative_gain=0.0,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_p_low_075",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=0.75,
                integral_gain_s_inv2=0.0,
                derivative_gain=0.0,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_p_low_100",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=1.00,
                integral_gain_s_inv2=0.0,
                derivative_gain=0.0,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_p_conservative",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=1.8,
                integral_gain_s_inv2=0.0,
                derivative_gain=0.0,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_p_balanced",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=2.6,
                integral_gain_s_inv2=0.0,
                derivative_gain=0.0,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_pd_light",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=2.2,
                integral_gain_s_inv2=0.0,
                derivative_gain=0.12,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_pd_balanced",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=2.6,
                integral_gain_s_inv2=0.0,
                derivative_gain=0.24,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_pid_light",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=2.2,
                integral_gain_s_inv2=0.12,
                derivative_gain=0.12,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_pid_balanced",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=2.6,
                integral_gain_s_inv2=0.22,
                derivative_gain=0.18,
                **shared,
            ),
        ),
        RobustPIDCandidate(
            "imu_pid_fast",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=3.2,
                integral_gain_s_inv2=0.25,
                derivative_gain=0.12,
                gain_schedule_reference_delay_s=0.20,
                **{
                    key: value
                    for key, value in shared.items()
                    if key != "gain_schedule_reference_delay_s"
                },
            ),
        ),
        RobustPIDCandidate(
            "imu_pid_target_ff",
            RobustPIDControllerConfig(
                proportional_gain_s_inv=2.0,
                integral_gain_s_inv2=0.12,
                derivative_gain=0.08,
                target_rate_feedforward_gain=0.35,
                **shared,
            ),
        ),
    )


def default_identified_mpc_candidates() -> tuple[IdentifiedMPCCandidate, ...]:
    """Predeclared C4 development candidates for both command interfaces."""

    balanced = IdentifiedMPCControllerConfig()
    return (
        IdentifiedMPCCandidate(
            "mpc_h020_w5_s075",
            0.20,
            0.70,
            replace(
                balanced,
                command_change_weight=5.0,
                rate_command_slew_scale=0.75,
                position_command_slew_scale=0.75,
            ),
        ),
        IdentifiedMPCCandidate(
            "mpc_h030_w5_s075",
            0.30,
            0.70,
            replace(
                balanced,
                command_change_weight=5.0,
                rate_command_slew_scale=0.75,
                position_command_slew_scale=0.75,
            ),
        ),
        IdentifiedMPCCandidate(
            "mpc_h040_w5_s075",
            0.40,
            0.70,
            replace(
                balanced,
                command_change_weight=5.0,
                rate_command_slew_scale=0.75,
                position_command_slew_scale=0.75,
            ),
        ),
        IdentifiedMPCCandidate(
            "mpc_h030_w10_s075",
            0.30,
            0.70,
            replace(
                balanced,
                command_change_weight=10.0,
                rate_command_slew_scale=0.75,
                position_command_slew_scale=0.75,
            ),
        ),
        IdentifiedMPCCandidate(
            "mpc_h030_w5_s100",
            0.30,
            0.70,
            replace(
                balanced,
                command_change_weight=5.0,
                rate_command_slew_scale=1.0,
                position_command_slew_scale=1.0,
            ),
        ),
        IdentifiedMPCCandidate(
            "mpc_h030_w10_s100",
            0.30,
            0.70,
            replace(
                balanced,
                command_change_weight=10.0,
                rate_command_slew_scale=1.0,
                position_command_slew_scale=1.0,
            ),
        ),
    )


@dataclass(frozen=True)
class BaselineBenchmarkProtocolConfig:
    selection_scenarios: tuple[str, ...] = PRIMARY_SCENARIOS
    command_modes: tuple[GimbalCommandMode, ...] = (
        GimbalCommandMode.POSITION,
        GimbalCommandMode.RATE,
    )
    worst_scenario_cost_weight: float = 0.25
    command_variation_budget_per_s: float = 1.25
    variation_excess_weight: float = 2.0
    device: str = "cpu"
    candidates: tuple[RobustPIDCandidate, ...] = (
        default_robust_pid_candidates()
    )
    mpc_candidates: tuple[IdentifiedMPCCandidate, ...] = (
        default_identified_mpc_candidates()
    )

    def __post_init__(self) -> None:
        if not self.selection_scenarios:
            raise ValueError("selection scenarios must be non-empty")
        if len(set(self.selection_scenarios)) != len(self.selection_scenarios):
            raise ValueError("selection scenarios must be unique")
        if not self.command_modes or len(set(self.command_modes)) != len(
            self.command_modes
        ):
            raise ValueError("command modes must be non-empty and unique")
        for mode in self.command_modes:
            if not isinstance(mode, GimbalCommandMode):
                raise ValueError("command modes must be GimbalCommandMode values")
        for name in (
            "worst_scenario_cost_weight",
            "command_variation_budget_per_s",
            "variation_excess_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.candidates:
            raise ValueError("at least one PID candidate is required")
        names = [candidate.name for candidate in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError("PID candidate names must be unique")
        if not self.mpc_candidates:
            raise ValueError("at least one MPC candidate is required")
        mpc_names = [candidate.name for candidate in self.mpc_candidates]
        if len(set(mpc_names)) != len(mpc_names):
            raise ValueError("MPC candidate names must be unique")
        if not self.device:
            raise ValueError("benchmark device must be non-empty")


def locked_robust_pid_config(
    command_mode: GimbalCommandMode,
) -> RobustPIDControllerConfig:
    """Return the development-selected C2 configuration for one adapter."""

    candidates = {
        item.name: item.controller for item in default_robust_pid_candidates()
    }
    selected = {
        GimbalCommandMode.POSITION: "imu_p_low_050",
        GimbalCommandMode.RATE: "imu_p_low_050",
    }
    return candidates[selected[command_mode]]


def locked_identified_mpc_candidate(
    command_mode: GimbalCommandMode,
) -> IdentifiedMPCCandidate:
    """Return the development-selected C4 configuration for one adapter."""

    candidates = {
        item.name: item for item in default_identified_mpc_candidates()
    }
    selected = {
        GimbalCommandMode.POSITION: "mpc_h020_w5_s075",
        GimbalCommandMode.RATE: "mpc_h030_w10_s100",
    }
    return candidates[selected[command_mode]]


def robust_pid_run(
    *,
    scenario: ClosedLoopScenario,
    seed: int,
    command_mode: GimbalCommandMode,
    controller_config: RobustPIDControllerConfig | None = None,
    name: str | None = None,
) -> ControllerRun:
    config = replace(
        scenario.config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        command_mode=command_mode,
    )
    selected_config = controller_config or locked_robust_pid_config(command_mode)
    selected_name = name or f"robust_pid_{command_mode.name.lower()}"
    return run_closed_loop_controller(
        name=selected_name,
        description=(
            "Timestamped filtered PID with anti-windup, IMU body-rate "
            "feed-forward, delay scheduling, dropout hold, and hardware-bound "
            f"{command_mode.name.lower()} commands."
        ),
        scenario=scenario,
        config=config,
        controller=RobustPIDGimbalController(
            servo=config.servo,
            camera=config.camera,
            command_mode=command_mode,
            config=selected_config,
            name=selected_name,
        ),
        seed=seed,
    )


def _identified_mpc_estimator(
    *,
    scenario: ClosedLoopScenario,
    candidate: IdentifiedMPCCandidate,
    maximum_staleness_s: float,
) -> MultiHorizonConstantVelocityTargetEstimator:
    config = scenario.config
    period_s = config.timing.control_period_s
    horizon_steps = max(
        2,
        int(math.ceil(candidate.prediction_horizon_s / period_s - 1e-12)),
    )
    horizons_s = tuple(period_s * index for index in range(horizon_steps + 1))
    maximum_projection_s = maximum_staleness_s + horizons_s[-1]
    return MultiHorizonConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            center_noise_std_normalized=(
                config.camera.center_noise_std_normalized
            ),
            velocity_filter_coefficient=(
                candidate.velocity_filter_coefficient
            ),
            uncertainty_filter_coefficient=0.20,
            max_prediction_horizon_s=maximum_projection_s,
            history_horizon_s=max(1.0, maximum_projection_s + 0.50),
            process_acceleration_std_rad_s2=math.radians(80.0),
            body_rate_compensation=True,
        ),
        prediction_horizons_s=horizons_s,
    )


def identified_mpc_run(
    *,
    scenario: ClosedLoopScenario,
    seed: int,
    command_mode: GimbalCommandMode,
    candidate: IdentifiedMPCCandidate | None = None,
    maximum_staleness_s: float = 0.50,
    name: str | None = None,
) -> ControllerRun:
    config = replace(
        scenario.config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        command_mode=command_mode,
    )
    selected_candidate = candidate or locked_identified_mpc_candidate(
        command_mode
    )
    selected_name = name or f"identified_mpc_{command_mode.name.lower()}"
    estimator = _identified_mpc_estimator(
        scenario=replace(scenario, config=config),
        candidate=selected_candidate,
        maximum_staleness_s=maximum_staleness_s,
    )
    return run_closed_loop_controller(
        name=selected_name,
        description=(
            "Causal constant-velocity target estimator feeding an identified "
            "finite-horizon dynamic-matrix controller with explicit command "
            f"constraints and {command_mode.name.lower()} commands."
        ),
        scenario=scenario,
        config=config,
        controller=IdentifiedMPCGimbalController(
            estimator=estimator,
            servo=config.servo,
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            command_mode=command_mode,
            nominal_control_period_s=config.timing.control_period_s,
            config=selected_candidate.controller,
            name=selected_name,
        ),
        seed=seed,
    )


def _summarize(runs: list[ControllerRun]) -> dict[str, Any]:
    aggregate = _aggregate_runs(runs)
    summary = _summary(aggregate)
    episode_count = int(aggregate["episode_count"])
    return {
        **summary,
        "episode_count": episode_count,
        "mean_unrecovered_loss_events_per_episode": (
            summary["total_unrecovered_loss_events"] / episode_count
        ),
    }


def _by_scenario(
    variants: Sequence[tuple[int, int, ClosedLoopScenario]],
    runs: Sequence[ControllerRun],
) -> dict[str, Any]:
    buckets: dict[str, list[ControllerRun]] = {}
    for variant, run in zip(variants, runs, strict=True):
        buckets.setdefault(variant[2].name, []).append(run)
    return {name: _summarize(values) for name, values in buckets.items()}


def _selection_score(
    *,
    aggregate: dict[str, Any],
    by_scenario: dict[str, Any],
    protocol: BaselineBenchmarkProtocolConfig,
) -> dict[str, float]:
    worst_scenario_cost = max(
        float(value["mean_control_cost"]) for value in by_scenario.values()
    )
    variation_excess = max(
        0.0,
        float(aggregate["command_variation_per_s"])
        - protocol.command_variation_budget_per_s,
    )
    score = (
        float(aggregate["mean_control_cost"])
        + protocol.worst_scenario_cost_weight * worst_scenario_cost
        + protocol.variation_excess_weight * variation_excess**2
    )
    return {
        "value": score,
        "aggregate_control_cost": float(aggregate["mean_control_cost"]),
        "worst_scenario_control_cost": worst_scenario_cost,
        "command_variation_excess_per_s": variation_excess,
    }


def _dream_to_center_reference(
    *,
    visibility_risk_result: dict[str, Any],
    variants: Sequence[tuple[int, int, ClosedLoopScenario]],
    adapter: Any,
    maximum_staleness_s: float,
    device: str,
) -> dict[str, Any]:
    training_seeds = visibility_risk_result.get("training_seeds")
    checkpoints = visibility_risk_result.get("checkpoints")
    if not isinstance(training_seeds, list) or not training_seeds:
        raise ValueError("visibility-risk result has no GRU training seeds")
    if not isinstance(checkpoints, dict):
        raise ValueError("visibility-risk result has no GRU checkpoints")
    evaluation = AdaptivePositionProtocolConfig(
        maximum_staleness_s=maximum_staleness_s,
        device=device,
    )
    runs: list[ControllerRun] = []
    paired_variants: list[tuple[int, int, ClosedLoopScenario]] = []
    for training_seed_value in training_seeds:
        training_seed = int(training_seed_value)
        checkpoint = checkpoints.get(str(training_seed))
        if not isinstance(checkpoint, str):
            raise ValueError("GRU checkpoint is missing from visibility result")
        model, metadata = load_gru_checkpoint(Path(checkpoint), device=device)
        recorded_seed = metadata.get("training_config", {}).get("seed")
        if recorded_seed is not None and int(recorded_seed) != training_seed:
            raise ValueError("GRU checkpoint training seed mismatch")
        for seed, scenario_index, scenario in variants:
            runs.append(
                _adaptive_run(
                    scenario=scenario,
                    seed=seed,
                    model=model,
                    adapter=adapter,
                    evaluation=evaluation,
                    name=f"dream_to_center_seed_{training_seed}",
                )
            )
            paired_variants.append((seed, scenario_index, scenario))
    return {
        "training_seeds": [int(value) for value in training_seeds],
        "summary": _summarize(runs),
        "by_scenario": _by_scenario(paired_variants, runs),
    }


def _protocol_hash(
    protocol: BaselineBenchmarkProtocolConfig,
    development_seeds: tuple[int, ...],
    sealed_test_seeds: tuple[int, ...],
) -> str:
    value = {
        "schema": BASELINE_BENCHMARK_SCHEMA_VERSION,
        "registry": [asdict(item) for item in baseline_method_registry()],
        "protocol": asdict(protocol),
        "development_seeds": development_seeds,
        "sealed_test_seeds": sealed_test_seeds,
        "domain_randomization": asdict(GimbalDomainRandomizationConfig()),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_seed_blocks(
    development_seeds: tuple[int, ...],
    sealed_test_seeds: tuple[int, ...],
) -> None:
    if not development_seeds or not sealed_test_seeds:
        raise ValueError("development and sealed test seeds must be non-empty")
    if len(set(development_seeds)) != len(development_seeds) or len(
        set(sealed_test_seeds)
    ) != len(sealed_test_seeds):
        raise ValueError("seed blocks must be unique")
    if set(development_seeds) & set(sealed_test_seeds):
        raise ValueError("development and sealed test seeds overlap")


def develop_baseline_benchmark_v1(
    *,
    visibility_risk_result: dict[str, Any],
    protocol: BaselineBenchmarkProtocolConfig | None = None,
    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS,
    sealed_test_seeds: tuple[int, ...] = DEFAULT_SEALED_TEST_SEEDS,
) -> dict[str, Any]:
    """Tune C2/C4 on development worlds while leaving final test sealed."""

    protocol = protocol or BaselineBenchmarkProtocolConfig()
    _validate_seed_blocks(development_seeds, sealed_test_seeds)
    adapter = _selected_adapter_config(visibility_risk_result)
    variants = _fresh_variants(development_seeds)
    selection_variants = tuple(
        variant
        for variant in variants
        if variant[2].name in protocol.selection_scenarios
    )
    observed_names = {variant[2].name for variant in selection_variants}
    if observed_names != set(protocol.selection_scenarios):
        raise ValueError("selection scenarios are unavailable")

    modes: dict[str, Any] = {}
    for mode in protocol.command_modes:
        candidate_records = []
        for candidate in protocol.candidates:
            runs = [
                robust_pid_run(
                    scenario=scenario,
                    seed=seed,
                    command_mode=mode,
                    controller_config=candidate.controller,
                    name=f"c2_{candidate.name}_{mode.name.lower()}",
                )
                for seed, _scenario_index, scenario in selection_variants
            ]
            summary = _summarize(runs)
            scenario_summary = _by_scenario(selection_variants, runs)
            candidate_records.append(
                {
                    "name": candidate.name,
                    "controller_config": asdict(candidate.controller),
                    "summary": summary,
                    "by_scenario": scenario_summary,
                    "selection_score": _selection_score(
                        aggregate=summary,
                        by_scenario=scenario_summary,
                        protocol=protocol,
                    ),
                }
            )
        selected_record = min(
            candidate_records,
            key=lambda item: (
                item["selection_score"]["value"],
                item["summary"]["command_variation_per_s"],
            ),
        )
        selected_candidate = next(
            item
            for item in protocol.candidates
            if item.name == selected_record["name"]
        )
        all_runs = [
            robust_pid_run(
                scenario=scenario,
                seed=seed,
                command_mode=mode,
                controller_config=selected_candidate.controller,
                name=f"c2_robust_pid_{mode.name.lower()}",
            )
            for seed, _scenario_index, scenario in variants
        ]
        modes[mode.value] = {
            "candidates": candidate_records,
            "selected_candidate": selected_record["name"],
            "selected_config": asdict(selected_candidate.controller),
            "all_scenarios_summary": _summarize(all_runs),
            "all_scenarios": _by_scenario(variants, all_runs),
        }

    maximum_staleness_s = float(
        visibility_risk_result["protocol"]["maximum_staleness_s"]
    )
    mpc_modes: dict[str, Any] = {}
    for mode in protocol.command_modes:
        candidate_records = []
        for candidate in protocol.mpc_candidates:
            runs = [
                identified_mpc_run(
                    scenario=scenario,
                    seed=seed,
                    command_mode=mode,
                    candidate=candidate,
                    maximum_staleness_s=maximum_staleness_s,
                    name=f"c4_{candidate.name}_{mode.name.lower()}",
                )
                for seed, _scenario_index, scenario in selection_variants
            ]
            summary = _summarize(runs)
            scenario_summary = _by_scenario(selection_variants, runs)
            candidate_records.append(
                {
                    "name": candidate.name,
                    "prediction_horizon_s": candidate.prediction_horizon_s,
                    "velocity_filter_coefficient": (
                        candidate.velocity_filter_coefficient
                    ),
                    "controller_config": asdict(candidate.controller),
                    "summary": summary,
                    "by_scenario": scenario_summary,
                    "selection_score": _selection_score(
                        aggregate=summary,
                        by_scenario=scenario_summary,
                        protocol=protocol,
                    ),
                }
            )
        selected_record = min(
            candidate_records,
            key=lambda item: (
                item["selection_score"]["value"],
                item["summary"]["command_variation_per_s"],
            ),
        )
        selected_candidate = next(
            item
            for item in protocol.mpc_candidates
            if item.name == selected_record["name"]
        )
        all_runs = [
            identified_mpc_run(
                scenario=scenario,
                seed=seed,
                command_mode=mode,
                candidate=selected_candidate,
                maximum_staleness_s=maximum_staleness_s,
                name=f"c4_identified_mpc_{mode.name.lower()}",
            )
            for seed, _scenario_index, scenario in variants
        ]
        mpc_modes[mode.value] = {
            "candidates": candidate_records,
            "selected_candidate": selected_record["name"],
            "selected_config": {
                "prediction_horizon_s": (
                    selected_candidate.prediction_horizon_s
                ),
                "velocity_filter_coefficient": (
                    selected_candidate.velocity_filter_coefficient
                ),
                "controller": asdict(selected_candidate.controller),
            },
            "all_scenarios_summary": _summarize(all_runs),
            "all_scenarios": _by_scenario(variants, all_runs),
        }

    reference_runs = [
        practical_feedback_run(scenario=scenario, seed=seed)
        for seed, _scenario_index, scenario in variants
    ]
    champion_runs = [
        conventional_champion_run(
            scenario=scenario,
            seed=seed,
            adapter=adapter,
            maximum_staleness_s=float(
                visibility_risk_result["protocol"]["maximum_staleness_s"]
            ),
        )
        for seed, _scenario_index, scenario in variants
    ]
    learned_reference = _dream_to_center_reference(
        visibility_risk_result=visibility_risk_result,
        variants=variants,
        adapter=adapter,
        maximum_staleness_s=maximum_staleness_s,
        device=protocol.device,
    )
    registry = baseline_method_registry()
    return {
        "experiment": BASELINE_BENCHMARK_SCHEMA_VERSION,
        "protocol_hash": _protocol_hash(
            protocol,
            development_seeds,
            sealed_test_seeds,
        ),
        "registry": [asdict(item) for item in registry],
        "registry_status": {
            "method_count": len(registry),
            "ready_count": sum(
                item.implementation_status == "ready" for item in registry
            ),
            "partial_count": sum(
                item.implementation_status == "partial" for item in registry
            ),
            "missing_count": sum(
                item.implementation_status == "missing" for item in registry
            ),
        },
        "protocol": asdict(protocol),
        "development": {
            "opened": True,
            "world_seeds": list(development_seeds),
            "selection_scenarios": list(protocol.selection_scenarios),
            "all_scenario_count": len({item[2].name for item in variants}),
            "variant_count": len(variants),
            "robust_pid": modes,
            "identified_mpc": mpc_modes,
            "references": {
                "practical_feedback": {
                    "summary": _summarize(reference_runs),
                    "by_scenario": _by_scenario(variants, reference_runs),
                },
                "conventional_champion_v1": {
                    "summary": _summarize(champion_runs),
                    "by_scenario": _by_scenario(variants, champion_runs),
                },
                "dream_to_center": learned_reference,
            },
        },
        "test": {
            "opened": False,
            "sealed": True,
            "world_seeds": list(sealed_test_seeds),
            "reason": (
                "The final block remains unopened until all ten registry "
                "entries are implemented, tuned, frozen, and hash-locked."
            ),
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop C2/C4 within the sealed ten-method benchmark."
    )
    parser.add_argument(
        "--visibility-risk-results",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_position_v21.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_baseline_benchmark_v1_development.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    visibility_result = json.loads(
        args.visibility_risk_results.read_text(encoding="utf-8")
    )
    result = develop_baseline_benchmark_v1(
        visibility_risk_result=visibility_result
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selection = {
        family: {
            mode: record["selected_candidate"]
            for mode, record in modes.items()
        }
        for family, modes in (
            ("robust_pid", result["development"]["robust_pid"]),
            ("identified_mpc", result["development"]["identified_mpc"]),
        )
    }
    print(json.dumps(selection, indent=2))
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
