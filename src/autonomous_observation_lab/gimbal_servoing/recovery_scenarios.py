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


def recovery_scenarios() -> tuple[ClosedLoopScenario, ...]:
    """Return sensor-loss, returning-target, and unreachable-target cases."""
    nominal = nominal_scenario()
    base_config = replace(
        nominal.config,
        timing=replace(nominal.config.timing, episode_duration_s=8.0),
        camera=replace(
            nominal.config.camera,
            miss_probability=0.0,
            require_full_bbox_in_view=True,
        ),
    )
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
