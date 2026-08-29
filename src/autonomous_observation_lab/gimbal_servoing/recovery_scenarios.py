"""Constructed scenarios for loss-of-view recovery evaluation."""

from __future__ import annotations

import math
from dataclasses import replace

from .closed_loop import (
    ClosedLoopScenario,
    nominal_scenario,
    shared_body_motion,
    shared_target_motion,
)
from .config import GimbalServoingConfig
from .disturbances import (
    SampledAngularMotion,
    SinusoidalAngularMotion,
    StaticAngularMotion,
)
from .randomization import (
    GimbalDomainRandomizationConfig,
    HardwareRandomizationConfig,
    MotionRandomizationConfig,
    UniformRange,
)


EXPANDED_RECOVERY_SUITE_VERSION = "gimbal_expanded_recovery_suite_v1"


def _recovery_base_config() -> GimbalServoingConfig:
    nominal = nominal_scenario()
    return replace(
        nominal.config,
        timing=replace(nominal.config.timing, episode_duration_s=8.0),
        camera=replace(
            nominal.config.camera,
            miss_probability=0.0,
            require_full_bbox_in_view=True,
        ),
    )


def recovery_scenarios() -> tuple[ClosedLoopScenario, ...]:
    """Return sensor-loss, returning-target, and unreachable-target cases."""
    base_config = _recovery_base_config()
    detector_outage = replace(
        base_config,
        camera=replace(
            base_config.camera,
            forced_dropout_intervals_s=((2.0, 3.4), (5.2, 6.0)),
        ),
    )
    returning_motion = SampledAngularMotion(
        times_s=(0.0, 1.5, 2.3, 4.4, 5.2, 8.0),
        angles_rad=tuple(
            map(math.radians, (0.0, 0.0, 86.0, 86.0, 0.0, 0.0))
        ),
    )
    unreachable_motion = SampledAngularMotion(
        times_s=(0.0, 1.5, 2.3, 8.0),
        angles_rad=tuple(map(math.radians, (0.0, 0.0, 86.0, 86.0))),
    )
    mild_body_motion = SinusoidalAngularMotion(
        amplitude_rad=math.radians(3.0),
        frequency_hz=0.27,
        phase_rad=0.4,
    )
    return (
        ClosedLoopScenario(
            name="detector_burst_recovery",
            description=(
                "The target remains reachable while the detector has two "
                "scheduled output outages."
            ),
            config=detector_outage,
            target_motion=shared_target_motion(),
            body_motion=shared_body_motion(),
        ),
        ClosedLoopScenario(
            name="travel_limit_reentry",
            description=(
                "The target moves beyond the physical travel/FOV envelope "
                "and later re-enters it."
            ),
            config=base_config,
            target_motion=returning_motion,
            body_motion=mild_body_motion,
        ),
        ClosedLoopScenario(
            name="physically_unreachable",
            description=(
                "The target moves beyond the physical travel/FOV envelope "
                "and remains unreachable through episode end."
            ),
            config=base_config,
            target_motion=unreachable_motion,
            body_motion=StaticAngularMotion(),
        ),
    )


def expanded_recovery_scenarios() -> tuple[ClosedLoopScenario, ...]:
    """Return the recovery suite with threshold- and direction-sensitive cases."""
    base_config = _recovery_base_config()
    micro_burst_config = replace(
        base_config,
        camera=replace(
            base_config.camera,
            forced_dropout_intervals_s=(
                (1.60, 1.90),
                (2.60, 3.05),
                (4.20, 4.85),
                (6.30, 6.55),
            ),
        ),
    )
    reversal_config = replace(
        base_config,
        camera=replace(
            base_config.camera,
            forced_dropout_intervals_s=((2.0, 4.1),),
        ),
    )
    body_maneuver_config = replace(
        base_config,
        camera=replace(
            base_config.camera,
            forced_dropout_intervals_s=((1.9, 3.6),),
        ),
    )
    negative_reentry = SampledAngularMotion(
        times_s=(0.0, 1.5, 2.3, 4.4, 5.2, 8.0),
        angles_rad=tuple(
            map(math.radians, (0.0, 0.0, -86.0, -86.0, 0.0, 0.0))
        ),
    )
    reversal_motion = SampledAngularMotion(
        times_s=(0.0, 1.8, 2.5, 3.3, 4.1, 5.0, 8.0),
        angles_rad=tuple(
            map(math.radians, (0.0, 5.0, 24.0, -20.0, 8.0, 0.0, 0.0))
        ),
    )
    body_maneuver = SampledAngularMotion(
        times_s=(0.0, 1.7, 2.35, 3.05, 3.8, 5.0, 8.0),
        angles_rad=tuple(
            map(math.radians, (0.0, 0.0, 28.0, -18.0, 4.0, 0.0, 0.0))
        ),
    )
    mild_body_motion = SinusoidalAngularMotion(
        amplitude_rad=math.radians(3.0),
        frequency_hz=0.27,
        phase_rad=0.4,
    )
    return (
        *recovery_scenarios(),
        ClosedLoopScenario(
            name="detector_micro_bursts",
            description=(
                "Four detector interruptions span the coast-duration "
                "candidate boundaries while the target remains reachable."
            ),
            config=micro_burst_config,
            target_motion=shared_target_motion(),
            body_motion=shared_body_motion(),
        ),
        ClosedLoopScenario(
            name="target_reversal_outage",
            description=(
                "A reachable target reverses direction during one long "
                "detector outage, challenging constant-rate projection."
            ),
            config=reversal_config,
            target_motion=reversal_motion,
            body_motion=mild_body_motion,
        ),
        ClosedLoopScenario(
            name="negative_travel_limit_reentry",
            description=(
                "The target exits the negative travel/FOV boundary and later "
                "re-enters, checking directional symmetry."
            ),
            config=base_config,
            target_motion=negative_reentry,
            body_motion=mild_body_motion,
        ),
        ClosedLoopScenario(
            name="body_maneuver_outage",
            description=(
                "The target is stationary while the vehicle body reverses "
                "rotation during a detector outage."
            ),
            config=body_maneuver_config,
            target_motion=StaticAngularMotion(),
            body_motion=body_maneuver,
        ),
    )


def recovery_domain_randomization(
    *, episode_duration_s: float = 8.0
) -> GimbalDomainRandomizationConfig:
    """Randomize hardware while preserving each constructed motion event."""
    fixed_motion = MotionRandomizationConfig(
        amplitude_scale=UniformRange(1.0, 1.0),
        frequency_scale=UniformRange(1.0, 1.0),
        phase_offset_rad=UniformRange(0.0, 0.0),
        bias_offset_rad=UniformRange(0.0, 0.0),
        drift_rate_rad_s=UniformRange(0.0, 0.0),
        pulse_probability=0.0,
    )
    return GimbalDomainRandomizationConfig(
        target_motion=fixed_motion,
        body_motion=fixed_motion,
        hardware=HardwareRandomizationConfig(
            episode_duration_s=episode_duration_s
        ),
    )
