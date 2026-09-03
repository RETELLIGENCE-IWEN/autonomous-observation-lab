"""Closed-loop baselines for the predictive gimbal servoing research track."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from .config import (
    CameraConfig,
    GimbalCommandMode,
    GimbalServoingConfig,
    ObservationProfile,
    ScenarioConfig,
    ServoConfig,
    TimingConfig,
)
from .controllers import (
    ProportionalController,
    ProportionalPositionController,
    TargetStatePositionController,
    TargetStateRateController,
)
from .demos import DemoEpisode, DemoFrame
from .disturbances import AngularMotion, SinusoidalAngularMotion, SumAngularMotion
from .env import GimbalServoEnv
from .estimators import (
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
    TargetStateEstimate,
    angle_delta_rad,
)
from .types import GimbalAction, GimbalObservation


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: GimbalObservation) -> GimbalAction: ...


@dataclass(frozen=True)
class TrackingMetrics:
    rms_error_normalized: float
    mean_absolute_error_deg: float
    p95_absolute_error_deg: float
    tracking_lag_s: float
    loss_of_view_fraction: float
    loss_of_view_events: int
    mean_recovery_time_s: float
    max_recovery_time_s: float
    unrecovered_loss_events: int
    rate_saturation_fraction: float
    angle_saturation_fraction: float
    command_rms_normalized: float
    actuator_rate_rms_normalized: float
    actuator_acceleration_rms_normalized: float
    command_variation_per_s: float


@dataclass(frozen=True)
class EstimatorMetrics:
    valid_fraction: float
    bearing_rmse_deg: float
    rate_rmse_deg_s: float
    two_sigma_bearing_coverage: float


@dataclass(frozen=True)
class ControllerRun:
    episode: DemoEpisode
    metrics: TrackingMetrics
    estimates: tuple[TargetStateEstimate, ...]
    estimator_metrics: EstimatorMetrics | None
    adapter_diagnostics: tuple[dict[str, float | bool], ...] = ()
    forecasts: tuple[tuple[TargetStateEstimate, ...], ...] = ()


@dataclass(frozen=True)
class ClosedLoopScenario:
    name: str
    description: str
    config: GimbalServoingConfig
    target_motion: AngularMotion
    body_motion: AngularMotion


@dataclass(frozen=True)
class ClosedLoopComparison:
    scenario_name: str
    description: str
    runs: tuple[ControllerRun, ...]


@dataclass(frozen=True)
class ClosedLoopBenchmarkSuite:
    description: str
    comparisons: tuple[ClosedLoopComparison, ...]


def closed_loop_config(command_mode: GimbalCommandMode) -> GimbalServoingConfig:
    """A visible non-ideal plant; every hardware assumption remains explicit."""
    return GimbalServoingConfig(
        servo=ServoConfig(
            min_angle_rad=math.radians(-45.0),
            max_angle_rad=math.radians(45.0),
            max_rate_rad_s=math.radians(70.0),
            max_acceleration_rad_s2=math.radians(280.0),
            rate_time_constant_s=0.070,
            command_latency_s=0.040,
            position_gain_s_inv=5.0,
            position_tolerance_rad=math.radians(0.10),
        ),
        camera=CameraConfig(
            selected_axis_fov_rad=math.radians(60.0),
            orthogonal_fov_rad=math.radians(45.0),
            frame_rate_hz=30.0,
            detection_latency_s=0.120,
            center_noise_std_normalized=0.004,
            confidence_mean=0.95,
            confidence_noise_std=0.01,
            miss_probability=0.02,
        ),
        timing=TimingConfig(
            control_rate_hz=30.0,
            integration_rate_hz=1000.0,
            episode_duration_s=12.0,
        ),
        scenario=ScenarioConfig(
            target_angular_width_rad=math.radians(4.0),
            target_angular_height_rad=math.radians(4.0),
        ),
        observation_profile=ObservationProfile.SERVO_AWARE,
        command_mode=command_mode,
    )


def shared_target_motion() -> SumAngularMotion:
    return SumAngularMotion(
        (
            SinusoidalAngularMotion(
                amplitude_rad=math.radians(12.0),
                frequency_hz=0.24,
                phase_rad=0.20,
            ),
            SinusoidalAngularMotion(
                amplitude_rad=math.radians(4.0),
                frequency_hz=0.63,
                phase_rad=-0.70,
            ),
        )
    )


def shared_body_motion() -> SumAngularMotion:
    return SumAngularMotion(
        (
            SinusoidalAngularMotion(
                amplitude_rad=math.radians(8.0),
                frequency_hz=0.34,
                phase_rad=0.90,
            ),
            SinusoidalAngularMotion(
                amplitude_rad=math.radians(3.0),
                frequency_hz=0.81,
                phase_rad=0.10,
            ),
        )
    )


def nominal_scenario() -> ClosedLoopScenario:
    return ClosedLoopScenario(
        name="nominal_combined",
        description=(
            "Nominal target motion and quadcopter rotation with 120 ms "
            "detector latency."
        ),
        config=closed_loop_config(GimbalCommandMode.RATE),
        target_motion=shared_target_motion(),
        body_motion=shared_body_motion(),
    )


def closed_loop_scenarios() -> tuple[ClosedLoopScenario, ...]:
    """Named deterministic cases spanning sensing, plant, and motion stress."""
    nominal = nominal_scenario()
    high_latency_config = replace(
        nominal.config,
        camera=replace(
            nominal.config.camera,
            detection_latency_s=0.220,
            detection_latency_jitter_s=0.025,
        ),
        servo=replace(nominal.config.servo, command_latency_s=0.080),
    )
    dropout_config = replace(
        nominal.config,
        camera=replace(
            nominal.config.camera,
            detection_latency_jitter_s=0.030,
            center_noise_std_normalized=0.015,
            confidence_noise_std=0.04,
            miss_probability=0.18,
        ),
    )
    slow_servo_config = replace(
        nominal.config,
        servo=replace(
            nominal.config.servo,
            max_rate_rad_s=math.radians(42.0),
            max_acceleration_rad_s2=math.radians(125.0),
            rate_time_constant_s=0.130,
            command_latency_s=0.080,
            position_gain_s_inv=3.5,
        ),
    )
    aggressive_config = replace(
        nominal.config,
        timing=replace(nominal.config.timing, episode_duration_s=10.0),
    )
    recovery_config = replace(
        nominal.config,
        timing=replace(nominal.config.timing, episode_duration_s=16.0),
        camera=replace(
            nominal.config.camera,
            require_full_bbox_in_view=True,
        ),
    )
    return (
        nominal,
        ClosedLoopScenario(
            name="high_latency",
            description="Long and jittered vision/command latency.",
            config=high_latency_config,
            target_motion=shared_target_motion(),
            body_motion=shared_body_motion(),
        ),
        ClosedLoopScenario(
            name="dropout_noise",
            description="Noisy bbox centers with detector jitter and 18% misses.",
            config=dropout_config,
            target_motion=shared_target_motion(),
            body_motion=shared_body_motion(),
        ),
        ClosedLoopScenario(
            name="slow_servo",
            description="Rate-, acceleration-, bandwidth-, and latency-limited servo.",
            config=slow_servo_config,
            target_motion=shared_target_motion(),
            body_motion=shared_body_motion(),
        ),
        ClosedLoopScenario(
            name="aggressive_motion",
            description="Higher-frequency target and body motion.",
            config=aggressive_config,
            target_motion=SumAngularMotion(
                (
                    SinusoidalAngularMotion(
                        amplitude_rad=math.radians(15.0),
                        frequency_hz=0.40,
                        phase_rad=0.20,
                    ),
                    SinusoidalAngularMotion(
                        amplitude_rad=math.radians(6.0),
                        frequency_hz=0.92,
                        phase_rad=-0.40,
                    ),
                )
            ),
            body_motion=SumAngularMotion(
                (
                    SinusoidalAngularMotion(
                        amplitude_rad=math.radians(10.0),
                        frequency_hz=0.55,
                        phase_rad=0.70,
                    ),
                    SinusoidalAngularMotion(
                        amplitude_rad=math.radians(4.0),
                        frequency_hz=1.10,
                        phase_rad=0.10,
                    ),
                )
            ),
        ),
        ClosedLoopScenario(
            name="travel_limit_recovery",
            description="Target exits the reachable/FOV envelope and later returns.",
            config=recovery_config,
            target_motion=SinusoidalAngularMotion(
                amplitude_rad=math.radians(82.0),
                frequency_hz=0.06,
                phase_rad=0.0,
            ),
            body_motion=SinusoidalAngularMotion(
                amplitude_rad=math.radians(6.0),
                frequency_hz=0.31,
                phase_rad=0.50,
            ),
        ),
    )


def _rollout(
    *,
    name: str,
    description: str,
    config: GimbalServoingConfig,
    controller: Controller,
    target_motion: AngularMotion,
    body_motion: AngularMotion,
    seed: int,
) -> tuple[
    DemoEpisode,
    tuple[TargetStateEstimate, ...],
    tuple[dict[str, float | bool], ...],
    tuple[tuple[TargetStateEstimate, ...], ...],
]:
    env = GimbalServoEnv(
        config,
        target_motion=target_motion,
        body_motion=body_motion,
    )
    observation, diagnostics = env.reset(seed)
    controller.reset()
    action = controller.act(observation)
    frames = [DemoFrame(0, action, observation, diagnostics)]
    estimates = [_controller_estimate(controller, observation.time_s)]
    adapter_diagnostics = [_controller_adapter_diagnostics(controller)]
    forecasts = [_controller_forecasts(controller)]
    step = 0
    while True:
        result = env.step(action)
        step += 1
        frames.append(
            DemoFrame(
                step=step,
                action=action,
                observation=result.observation,
                diagnostics=result.diagnostics,
            )
        )
        if result.truncated:
            break
        observation = result.observation
        action = controller.act(observation)
        estimates.append(_controller_estimate(controller, observation.time_s))
        adapter_diagnostics.append(_controller_adapter_diagnostics(controller))
        forecasts.append(_controller_forecasts(controller))
    episode = DemoEpisode(name, description, config, tuple(frames))
    return (
        episode,
        tuple(estimates),
        tuple(adapter_diagnostics),
        tuple(forecasts),
    )


def _controller_estimate(
    controller: Controller, time_s: float
) -> TargetStateEstimate:
    estimate = getattr(controller, "last_estimate", None)
    if isinstance(estimate, TargetStateEstimate):
        return estimate
    return TargetStateEstimate.missing(time_s)


def _controller_adapter_diagnostics(
    controller: Controller,
) -> dict[str, float | bool]:
    diagnostics = getattr(controller, "last_diagnostics", None)
    serializer = getattr(diagnostics, "to_dict", None)
    if callable(serializer):
        value = serializer()
        if isinstance(value, dict):
            return value
    return {}


def _controller_forecasts(
    controller: Controller,
) -> tuple[TargetStateEstimate, ...]:
    estimator = getattr(controller, "estimator", None)
    values = getattr(estimator, "last_estimates", None)
    if isinstance(values, tuple) and all(
        isinstance(value, TargetStateEstimate) for value in values
    ):
        return values
    estimate = getattr(controller, "last_estimate", None)
    if isinstance(estimate, TargetStateEstimate):
        return (estimate,)
    return ()


def _tracking_lag_s(episode: DemoEpisode, max_lag_s: float = 0.50) -> float:
    frames = episode.frames
    times = np.array([frame.diagnostics.time_s for frame in frames])
    start = int(np.searchsorted(times, 1.0))
    if len(frames) - start < 3:
        start = 0
    desired = np.unwrap(
        np.array(
            [
                frame.diagnostics.target_bearing_rad
                - frame.diagnostics.body_bearing_rad
                for frame in frames
            ]
        )
    )[start:]
    actual = np.unwrap(
        np.array([frame.diagnostics.gimbal_angle_rad for frame in frames])
    )[start:]
    if len(desired) < 2:
        return 0.0
    period_s = episode.config.timing.control_period_s
    max_lag_steps = min(int(round(max_lag_s / period_s)), len(desired) // 4)
    best_lag = 0
    best_correlation = -math.inf
    for lag in range(max_lag_steps + 1):
        if lag:
            reference = desired[:-lag]
            response = actual[lag:]
        else:
            reference = desired
            response = actual
        reference = reference - np.mean(reference)
        response = response - np.mean(response)
        denominator = float(np.linalg.norm(reference) * np.linalg.norm(response))
        correlation = (
            float(np.dot(reference, response)) / denominator
            if denominator > 0.0
            else -math.inf
        )
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    return best_lag * period_s


def tracking_metrics(episode: DemoEpisode) -> TrackingMetrics:
    frames = episode.frames[1:]
    errors = np.array(
        [frame.diagnostics.true_image_error_normalized for frame in frames]
    )
    absolute_error_deg = (
        np.abs(errors)
        * math.degrees(0.5 * episode.config.camera.selected_axis_fov_rad)
    )
    visible = np.array([frame.diagnostics.target_in_view for frame in frames])
    lost = ~visible
    loss_events = int(lost[0]) + int(np.sum(lost[1:] & visible[:-1]))
    period_s = episode.config.timing.control_period_s
    recovery_times_s: list[float] = []
    unrecovered_loss_events = 0
    index = 0
    while index < len(lost):
        if not lost[index]:
            index += 1
            continue
        start = index
        while index < len(lost) and lost[index]:
            index += 1
        if index < len(lost):
            recovery_times_s.append((index - start) * period_s)
        else:
            unrecovered_loss_events += 1
    actions = np.array([frame.action.command_normalized for frame in frames])
    rates = np.array([frame.diagnostics.gimbal_rate_rad_s for frame in frames])
    accelerations = np.diff(rates, prepend=rates[0]) / period_s
    duration_s = frames[-1].diagnostics.time_s - frames[0].diagnostics.time_s
    return TrackingMetrics(
        rms_error_normalized=float(np.sqrt(np.mean(errors**2))),
        mean_absolute_error_deg=float(np.mean(absolute_error_deg)),
        p95_absolute_error_deg=float(np.percentile(absolute_error_deg, 95)),
        tracking_lag_s=_tracking_lag_s(episode),
        loss_of_view_fraction=float(np.mean(lost)),
        loss_of_view_events=loss_events,
        mean_recovery_time_s=(
            float(np.mean(recovery_times_s)) if recovery_times_s else 0.0
        ),
        max_recovery_time_s=(
            float(np.max(recovery_times_s)) if recovery_times_s else 0.0
        ),
        unrecovered_loss_events=unrecovered_loss_events,
        rate_saturation_fraction=float(
            np.mean([frame.diagnostics.rate_saturated for frame in frames])
        ),
        angle_saturation_fraction=float(
            np.mean([frame.diagnostics.angle_saturated for frame in frames])
        ),
        command_rms_normalized=float(np.sqrt(np.mean(actions**2))),
        actuator_rate_rms_normalized=float(
            np.sqrt(np.mean((rates / episode.config.servo.max_rate_rad_s) ** 2))
        ),
        actuator_acceleration_rms_normalized=float(
            np.sqrt(
                np.mean(
                    (
                        accelerations
                        / episode.config.servo.max_acceleration_rad_s2
                    )
                    ** 2
                )
            )
        ),
        command_variation_per_s=float(
            np.sum(np.abs(np.diff(actions))) / duration_s
        ),
    )


def run_closed_loop_controller(
    *,
    name: str,
    description: str,
    scenario: ClosedLoopScenario,
    config: GimbalServoingConfig,
    controller: Controller,
    seed: int,
) -> ControllerRun:
    """Roll out one controller and compute tracking and estimator metrics."""
    episode, estimates, adapter_diagnostics, forecasts = _rollout(
        name=name,
        description=description,
        config=config,
        controller=controller,
        target_motion=scenario.target_motion,
        body_motion=scenario.body_motion,
        seed=seed,
    )
    return ControllerRun(
        episode=episode,
        metrics=tracking_metrics(episode),
        estimates=estimates,
        estimator_metrics=estimator_metrics(episode, estimates),
        adapter_diagnostics=adapter_diagnostics,
        forecasts=forecasts,
    )


def estimator_metrics(
    episode: DemoEpisode,
    estimates: tuple[TargetStateEstimate, ...],
) -> EstimatorMetrics | None:
    if not estimates:
        return None
    paired = tuple(zip(episode.frames, estimates, strict=False))
    valid = [(frame, estimate) for frame, estimate in paired if estimate.valid]
    if not valid:
        return None

    bearing_errors = np.array(
        [
            angle_delta_rad(
                estimate.body_relative_bearing_rad.value,
                frame.diagnostics.target_bearing_rad
                - frame.diagnostics.body_bearing_rad,
            )
            for frame, estimate in valid
        ]
    )
    rate_errors = np.array(
        [
            estimate.body_relative_rate_rad_s.value
            - (
                frame.diagnostics.target_rate_rad_s
                - frame.diagnostics.body_rate_rad_s
            )
            for frame, estimate in valid
        ]
    )
    bearing_std = np.array(
        [estimate.bearing_std_rad.value for _, estimate in valid]
    )
    return EstimatorMetrics(
        valid_fraction=len(valid) / len(estimates),
        bearing_rmse_deg=math.degrees(
            float(np.sqrt(np.mean(bearing_errors**2)))
        ),
        rate_rmse_deg_s=math.degrees(
            float(np.sqrt(np.mean(rate_errors**2)))
        ),
        two_sigma_bearing_coverage=float(
            np.mean(np.abs(bearing_errors) <= 2.0 * bearing_std)
        ),
    )


def _target_estimator(
    config: GimbalServoingConfig,
) -> ConstantVelocityTargetEstimator:
    minimum_horizon_s = (
        config.camera.detection_latency_s
        + config.camera.detection_latency_jitter_s
        + 2.0 * config.camera.frame_period_s
    )
    prediction_horizon_s = max(0.30, minimum_horizon_s)
    return ConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            center_noise_std_normalized=(
                config.camera.center_noise_std_normalized
            ),
            velocity_filter_coefficient=0.40,
            uncertainty_filter_coefficient=0.20,
            max_prediction_horizon_s=prediction_horizon_s,
            history_horizon_s=max(1.0, prediction_horizon_s + 0.50),
        )
    )


def _controller_specifications(
    rate_config: GimbalServoingConfig,
) -> tuple[tuple[str, str, GimbalServoingConfig, Controller], ...]:
    position_config = replace(rate_config, command_mode=GimbalCommandMode.POSITION)
    return (
        (
            "proportional_rate",
            "Delayed visual feedback driving desired gimbal rate.",
            rate_config,
            ProportionalController(gain=1.35),
        ),
        (
            "proportional_position",
            "Delayed visual feedback updating an absolute position setpoint.",
            position_config,
            ProportionalPositionController(
                servo=position_config.servo,
                selected_axis_fov_rad=position_config.camera.selected_axis_fov_rad,
                gain=0.85,
            ),
        ),
        (
            "predictive_rate",
            "Causal target-bearing prediction driving desired gimbal rate.",
            rate_config,
            TargetStateRateController(
                estimator=_target_estimator(rate_config),
                max_rate_rad_s=rate_config.servo.max_rate_rad_s,
                proportional_gain_s_inv=2.5,
                name="predictive_rate",
            ),
        ),
        (
            "predictive_position",
            "The same target-state estimate driving an absolute setpoint.",
            position_config,
            TargetStatePositionController(
                estimator=_target_estimator(position_config),
                servo=position_config.servo,
                command_preview_s=(
                    position_config.servo.command_latency_s
                    + position_config.servo.rate_time_constant_s
                ),
                name="predictive_position",
            ),
        ),
    )


def closed_loop_comparison(
    seed: int = 31,
    scenario: ClosedLoopScenario | None = None,
) -> ClosedLoopComparison:
    scenario = scenario or nominal_scenario()
    specifications = _controller_specifications(scenario.config)
    runs = []
    for name, description, config, controller in specifications:
        episode, estimates, adapter_diagnostics, forecasts = _rollout(
            name=name,
            description=description,
            config=config,
            controller=controller,
            target_motion=scenario.target_motion,
            body_motion=scenario.body_motion,
            seed=seed,
        )
        runs.append(
            ControllerRun(
                episode=episode,
                metrics=tracking_metrics(episode),
                estimates=estimates,
                estimator_metrics=estimator_metrics(episode, estimates),
                adapter_diagnostics=adapter_diagnostics,
                forecasts=forecasts,
            )
        )
    return ClosedLoopComparison(
        scenario_name=scenario.name,
        description=(
            f"{scenario.description} Four causal controllers share identical "
            "target motion, body motion, and detector randomness."
        ),
        runs=tuple(runs),
    )


def closed_loop_benchmark_suite(seed: int = 41) -> ClosedLoopBenchmarkSuite:
    comparisons = tuple(
        closed_loop_comparison(seed=seed, scenario=scenario)
        for scenario in closed_loop_scenarios()
    )
    return ClosedLoopBenchmarkSuite(
        description=(
            "Controller matrix across nominal, sensing, actuator, motion, and "
            "loss-of-view recovery cases."
        ),
        comparisons=comparisons,
    )
