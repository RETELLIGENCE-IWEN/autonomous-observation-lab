"""Deterministic domain randomization for predictive gimbal learning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np

from .closed_loop import ClosedLoopScenario
from .disturbances import (
    AngularMotion,
    ConstantRateAngularMotion,
    RatePulseAngularMotion,
    SampledAngularMotion,
    SinusoidalAngularMotion,
    StaticAngularMotion,
    SumAngularMotion,
)


@dataclass(frozen=True)
class UniformRange:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("uniform range bounds must be finite")
        if self.low > self.high:
            raise ValueError("uniform range low bound must not exceed high bound")

    def sample(self, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return self.low
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class MotionRandomizationConfig:
    amplitude_scale: UniformRange = UniformRange(0.65, 1.35)
    frequency_scale: UniformRange = UniformRange(0.70, 1.40)
    phase_offset_rad: UniformRange = UniformRange(-math.pi, math.pi)
    bias_offset_rad: UniformRange = UniformRange(
        math.radians(-4.0), math.radians(4.0)
    )
    drift_rate_rad_s: UniformRange = UniformRange(
        math.radians(-4.0), math.radians(4.0)
    )
    pulse_probability: float = 0.35
    pulse_onset_fraction: UniformRange = UniformRange(0.15, 0.75)
    pulse_duration_s: UniformRange = UniformRange(0.08, 0.35)
    pulse_rate_magnitude_rad_s: UniformRange = UniformRange(
        math.radians(20.0), math.radians(100.0)
    )

    def __post_init__(self) -> None:
        if self.amplitude_scale.low <= 0.0:
            raise ValueError("motion amplitude scale must be positive")
        if self.frequency_scale.low <= 0.0:
            raise ValueError("motion frequency scale must be positive")
        if not 0.0 <= self.pulse_probability <= 1.0:
            raise ValueError("pulse probability must be in [0, 1]")
        if not 0.0 <= self.pulse_onset_fraction.low:
            raise ValueError("pulse onset fraction must be non-negative")
        if self.pulse_onset_fraction.high > 1.0:
            raise ValueError("pulse onset fraction must not exceed one")
        if self.pulse_duration_s.low <= 0.0:
            raise ValueError("pulse duration must be positive")
        if self.pulse_rate_magnitude_rad_s.low < 0.0:
            raise ValueError("pulse rate magnitude must be non-negative")


@dataclass(frozen=True)
class HardwareRandomizationConfig:
    travel_scale: UniformRange = UniformRange(0.85, 1.15)
    maximum_rate_scale: UniformRange = UniformRange(0.70, 1.30)
    maximum_acceleration_scale: UniformRange = UniformRange(0.65, 1.35)
    rate_time_constant_scale: UniformRange = UniformRange(0.65, 1.50)
    command_latency_scale: UniformRange = UniformRange(0.60, 1.50)
    command_latency_add_s: UniformRange = UniformRange(0.0, 0.020)
    rate_deadband_rad_s: UniformRange = UniformRange(
        0.0, math.radians(0.40)
    )
    rate_quantization_rad_s: UniformRange = UniformRange(
        0.0, math.radians(0.20)
    )
    position_gain_scale: UniformRange = UniformRange(0.75, 1.25)
    position_quantization_rad: UniformRange = UniformRange(
        0.0, math.radians(0.10)
    )
    selected_axis_fov_scale: UniformRange = UniformRange(0.85, 1.15)
    orthogonal_fov_scale: UniformRange = UniformRange(0.90, 1.10)
    frame_rate_scale: UniformRange = UniformRange(0.75, 1.25)
    detection_latency_scale: UniformRange = UniformRange(0.60, 1.50)
    detection_latency_add_s: UniformRange = UniformRange(0.0, 0.020)
    detection_jitter_add_s: UniformRange = UniformRange(0.0, 0.015)
    center_noise_add_normalized: UniformRange = UniformRange(0.0, 0.006)
    size_noise_add_fraction: UniformRange = UniformRange(0.0, 0.010)
    confidence_mean_offset: UniformRange = UniformRange(-0.05, 0.02)
    confidence_noise_add: UniformRange = UniformRange(0.0, 0.025)
    miss_probability_add: UniformRange = UniformRange(0.0, 0.050)
    control_rate_scale: UniformRange = UniformRange(0.80, 1.20)
    target_extent_scale: UniformRange = UniformRange(0.70, 1.35)
    initial_angle_fraction: UniformRange = UniformRange(-0.12, 0.12)
    initial_rate_fraction: UniformRange = UniformRange(-0.08, 0.08)
    episode_duration_s: float | None = 8.0

    def __post_init__(self) -> None:
        for name in (
            "travel_scale",
            "maximum_rate_scale",
            "maximum_acceleration_scale",
            "rate_time_constant_scale",
            "command_latency_scale",
            "position_gain_scale",
            "selected_axis_fov_scale",
            "orthogonal_fov_scale",
            "frame_rate_scale",
            "detection_latency_scale",
            "control_rate_scale",
            "target_extent_scale",
        ):
            if getattr(self, name).low <= 0.0:
                raise ValueError(f"{name} must be strictly positive")
        if self.episode_duration_s is not None and self.episode_duration_s <= 0.0:
            raise ValueError("episode duration must be positive when provided")


@dataclass(frozen=True)
class GimbalDomainRandomizationConfig:
    target_motion: MotionRandomizationConfig = MotionRandomizationConfig()
    body_motion: MotionRandomizationConfig = MotionRandomizationConfig(
        drift_rate_rad_s=UniformRange(
            math.radians(-7.0), math.radians(7.0)
        ),
        pulse_rate_magnitude_rad_s=UniformRange(
            math.radians(30.0), math.radians(130.0)
        ),
    )
    hardware: HardwareRandomizationConfig = HardwareRandomizationConfig()
    seed_salt: int = 0x47494D42

    def __post_init__(self) -> None:
        if self.seed_salt < 0:
            raise ValueError("seed salt must be non-negative")


def _randomize_motion_component(
    motion: AngularMotion,
    rng: np.random.Generator,
    config: MotionRandomizationConfig,
) -> AngularMotion:
    if isinstance(motion, SinusoidalAngularMotion):
        return replace(
            motion,
            bias_rad=(
                motion.bias_rad + config.bias_offset_rad.sample(rng)
            ),
            amplitude_rad=(
                motion.amplitude_rad * config.amplitude_scale.sample(rng)
            ),
            frequency_hz=(
                motion.frequency_hz * config.frequency_scale.sample(rng)
            ),
            phase_rad=(
                motion.phase_rad + config.phase_offset_rad.sample(rng)
            ),
        )
    if isinstance(motion, ConstantRateAngularMotion):
        return replace(
            motion,
            initial_angle_rad=(
                motion.initial_angle_rad + config.bias_offset_rad.sample(rng)
            ),
            rate_rad_s=(
                motion.rate_rad_s * config.frequency_scale.sample(rng)
            ),
        )
    if isinstance(motion, RatePulseAngularMotion):
        duration_scale = config.frequency_scale.sample(rng)
        return replace(
            motion,
            initial_angle_rad=(
                motion.initial_angle_rad + config.bias_offset_rad.sample(rng)
            ),
            duration_s=motion.duration_s / duration_scale,
            rate_rad_s=motion.rate_rad_s * config.amplitude_scale.sample(rng),
        )
    if isinstance(motion, StaticAngularMotion):
        return replace(
            motion,
            angle_rad=motion.angle_rad + config.bias_offset_rad.sample(rng),
        )
    if isinstance(motion, SumAngularMotion):
        return SumAngularMotion(
            tuple(
                _randomize_motion_component(component, rng, config)
                for component in motion.components
            )
        )
    if isinstance(motion, SampledAngularMotion):
        offset = config.bias_offset_rad.sample(rng)
        scale = config.amplitude_scale.sample(rng)
        return SampledAngularMotion(
            times_s=motion.times_s,
            angles_rad=tuple(
                offset + scale * angle for angle in motion.angles_rad
            ),
        )
    raise TypeError(f"unsupported angular motion type: {type(motion).__name__}")


def randomize_angular_motion(
    motion: AngularMotion,
    *,
    episode_duration_s: float,
    rng: np.random.Generator,
    config: MotionRandomizationConfig,
) -> AngularMotion:
    components: list[AngularMotion] = [
        _randomize_motion_component(motion, rng, config)
    ]
    drift_rate = config.drift_rate_rad_s.sample(rng)
    if drift_rate != 0.0:
        components.append(ConstantRateAngularMotion(rate_rad_s=drift_rate))
    if rng.random() < config.pulse_probability:
        onset = (
            episode_duration_s
            * config.pulse_onset_fraction.sample(rng)
        )
        magnitude = config.pulse_rate_magnitude_rad_s.sample(rng)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        components.append(
            RatePulseAngularMotion(
                onset_s=onset,
                duration_s=config.pulse_duration_s.sample(rng),
                rate_rad_s=sign * magnitude,
            )
        )
    if len(components) == 1:
        return components[0]
    return SumAngularMotion(tuple(components))


def _randomize_hardware(
    scenario: ClosedLoopScenario,
    rng: np.random.Generator,
    config: HardwareRandomizationConfig,
):
    base = scenario.config
    servo = base.servo
    min_angle = servo.min_angle_rad * config.travel_scale.sample(rng)
    max_angle = servo.max_angle_rad * config.travel_scale.sample(rng)
    max_rate = servo.max_rate_rad_s * config.maximum_rate_scale.sample(rng)
    randomized_servo = replace(
        servo,
        min_angle_rad=min_angle,
        max_angle_rad=max_angle,
        max_rate_rad_s=max_rate,
        max_acceleration_rad_s2=(
            servo.max_acceleration_rad_s2
            * config.maximum_acceleration_scale.sample(rng)
        ),
        rate_time_constant_s=(
            servo.rate_time_constant_s
            * config.rate_time_constant_scale.sample(rng)
        ),
        command_latency_s=(
            servo.command_latency_s
            * config.command_latency_scale.sample(rng)
            + config.command_latency_add_s.sample(rng)
        ),
        rate_deadband_rad_s=config.rate_deadband_rad_s.sample(rng),
        rate_quantization_rad_s=config.rate_quantization_rad_s.sample(rng),
        position_gain_s_inv=(
            servo.position_gain_s_inv
            * config.position_gain_scale.sample(rng)
        ),
        position_quantization_rad=(
            config.position_quantization_rad.sample(rng)
        ),
    )
    camera = base.camera
    randomized_camera = replace(
        camera,
        selected_axis_fov_rad=(
            camera.selected_axis_fov_rad
            * config.selected_axis_fov_scale.sample(rng)
        ),
        orthogonal_fov_rad=(
            camera.orthogonal_fov_rad
            * config.orthogonal_fov_scale.sample(rng)
        ),
        frame_rate_hz=(
            camera.frame_rate_hz * config.frame_rate_scale.sample(rng)
        ),
        detection_latency_s=(
            camera.detection_latency_s
            * config.detection_latency_scale.sample(rng)
            + config.detection_latency_add_s.sample(rng)
        ),
        detection_latency_jitter_s=(
            camera.detection_latency_jitter_s
            + config.detection_jitter_add_s.sample(rng)
        ),
        center_noise_std_normalized=(
            camera.center_noise_std_normalized
            + config.center_noise_add_normalized.sample(rng)
        ),
        size_noise_std_fraction=(
            camera.size_noise_std_fraction
            + config.size_noise_add_fraction.sample(rng)
        ),
        confidence_mean=float(
            np.clip(
                camera.confidence_mean
                + config.confidence_mean_offset.sample(rng),
                0.0,
                1.0,
            )
        ),
        confidence_noise_std=(
            camera.confidence_noise_std
            + config.confidence_noise_add.sample(rng)
        ),
        miss_probability=float(
            np.clip(
                camera.miss_probability
                + config.miss_probability_add.sample(rng),
                0.0,
                1.0,
            )
        ),
    )
    timing = replace(
        base.timing,
        control_rate_hz=(
            base.timing.control_rate_hz
            * config.control_rate_scale.sample(rng)
        ),
        episode_duration_s=(
            config.episode_duration_s
            if config.episode_duration_s is not None
            else base.timing.episode_duration_s
        ),
    )
    initial_fraction = config.initial_angle_fraction.sample(rng)
    initial_angle = (
        initial_fraction
        * (max_angle if initial_fraction >= 0.0 else -min_angle)
    )
    initial_rate = (
        config.initial_rate_fraction.sample(rng) * max_rate
    )
    scenario_config = replace(
        base.scenario,
        initial_gimbal_angle_rad=initial_angle,
        initial_gimbal_rate_rad_s=initial_rate,
        target_angular_width_rad=(
            base.scenario.target_angular_width_rad
            * config.target_extent_scale.sample(rng)
        ),
        target_angular_height_rad=(
            base.scenario.target_angular_height_rad
            * config.target_extent_scale.sample(rng)
        ),
    )
    return replace(
        base,
        servo=randomized_servo,
        camera=randomized_camera,
        timing=timing,
        scenario=scenario_config,
    )


def randomize_closed_loop_scenario(
    scenario: ClosedLoopScenario,
    *,
    seed: int,
    config: GimbalDomainRandomizationConfig | None = None,
) -> ClosedLoopScenario:
    """Produce one replayable target/body/plant variant from a split seed."""
    config = config or GimbalDomainRandomizationConfig()
    scenario_key = int.from_bytes(
        hashlib.sha256(scenario.name.encode("utf-8")).digest()[:4],
        byteorder="little",
    )
    streams = np.random.SeedSequence(
        [seed, config.seed_salt, scenario_key]
    ).spawn(3)
    hardware_rng = np.random.default_rng(streams[0])
    target_rng = np.random.default_rng(streams[1])
    body_rng = np.random.default_rng(streams[2])
    randomized_config = _randomize_hardware(
        scenario, hardware_rng, config.hardware
    )
    duration_s = randomized_config.timing.episode_duration_s
    return ClosedLoopScenario(
        name=scenario.name,
        description=(
            f"{scenario.description} Deterministic domain-randomized variant "
            f"for seed {seed}."
        ),
        config=randomized_config,
        target_motion=randomize_angular_motion(
            scenario.target_motion,
            episode_duration_s=duration_s,
            rng=target_rng,
            config=config.target_motion,
        ),
        body_motion=randomize_angular_motion(
            scenario.body_motion,
            episode_duration_s=duration_s,
            rng=body_rng,
            config=config.body_motion,
        ),
    )
