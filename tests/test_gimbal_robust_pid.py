import math
from dataclasses import replace

import pytest

from autonomous_observation_lab.gimbal_servoing import (
    GimbalAction,
    GimbalCommandMode,
    GimbalObservation,
    MaskedScalar,
    ObservationProfile,
    RobustPIDControllerConfig,
    RobustPIDGimbalController,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    ClosedLoopScenario,
    closed_loop_config,
    run_closed_loop_controller,
)
from autonomous_observation_lab.gimbal_servoing.disturbances import (
    StaticAngularMotion,
)


def _observation(
    *,
    time_s: float,
    mode: GimbalCommandMode,
    error: float | None,
    frame_updated: bool = True,
    body_rate_rad_s: float = 0.0,
) -> GimbalObservation:
    measured = error is not None
    return GimbalObservation(
        time_s=time_s,
        control_dt_s=0.05,
        frame_updated=frame_updated,
        measurement_age_s=(
            MaskedScalar(0.10, True) if measured else MaskedScalar.missing()
        ),
        image_error_normalized=(
            MaskedScalar(error, True) if measured else MaskedScalar.missing()
        ),
        bbox_width_fraction=MaskedScalar.missing(),
        bbox_height_fraction=MaskedScalar.missing(),
        confidence=MaskedScalar.missing(),
        gimbal_angle_rad=MaskedScalar(0.0, True),
        gimbal_rate_rad_s=MaskedScalar(0.0, True),
        body_rate_rad_s=MaskedScalar(body_rate_rad_s, True),
        command_mode=mode,
        previous_action_normalized=0.0,
    )


def test_robust_pid_emits_both_command_modes() -> None:
    for mode in (GimbalCommandMode.RATE, GimbalCommandMode.POSITION):
        config = closed_loop_config(mode)
        controller = RobustPIDGimbalController(
            config.servo,
            config.camera,
            mode,
        )
        action = controller.act(
            _observation(time_s=0.1, mode=mode, error=0.4)
        )

        assert action.mode is mode
        assert -1.0 <= action.command_normalized <= 1.0
        assert controller.last_diagnostics.valid_tracking_evidence


def test_robust_pid_filters_derivative_and_uses_body_feedforward() -> None:
    mode = GimbalCommandMode.RATE
    config = closed_loop_config(mode)
    controller = RobustPIDGimbalController(
        config.servo,
        config.camera,
        mode,
    )
    controller.act(_observation(time_s=0.2, mode=mode, error=0.1))
    controller.act(
        _observation(
            time_s=0.3,
            mode=mode,
            error=0.2,
            body_rate_rad_s=math.radians(12.0),
        )
    )

    diagnostics = controller.last_diagnostics
    assert diagnostics.filtered_error_rate_rad_s > 0.0
    assert diagnostics.derivative_rate_rad_s > 0.0
    assert diagnostics.body_feedforward_rate_rad_s == pytest.approx(
        math.radians(-12.0)
    )


def test_robust_pid_anti_windup_blocks_integrator_during_saturation() -> None:
    mode = GimbalCommandMode.RATE
    config = closed_loop_config(mode)
    controller = RobustPIDGimbalController(
        config.servo,
        config.camera,
        mode,
        RobustPIDControllerConfig(
            proportional_gain_s_inv=100.0,
            integral_gain_s_inv2=10.0,
            derivative_gain=0.0,
        ),
    )
    controller.act(_observation(time_s=0.1, mode=mode, error=1.0))

    diagnostics = controller.last_diagnostics
    assert diagnostics.anti_windup_active
    assert diagnostics.rate_limited
    assert diagnostics.integral_error_rad_s == 0.0


def test_robust_pid_watchdog_expires_held_detection() -> None:
    mode = GimbalCommandMode.POSITION
    config = closed_loop_config(mode)
    controller = RobustPIDGimbalController(
        config.servo,
        config.camera,
        mode,
        RobustPIDControllerConfig(maximum_detection_gap_s=0.15),
    )
    controller.act(_observation(time_s=0.1, mode=mode, error=0.2))
    controller.act(
        _observation(
            time_s=0.2,
            mode=mode,
            error=None,
            frame_updated=True,
        )
    )
    assert controller.last_diagnostics.valid_tracking_evidence
    controller.act(
        _observation(
            time_s=0.3,
            mode=mode,
            error=None,
            frame_updated=True,
        )
    )
    assert not controller.last_diagnostics.valid_tracking_evidence
    assert controller.last_diagnostics.error_rad == 0.0


@pytest.mark.parametrize(
    "mode",
    (GimbalCommandMode.RATE, GimbalCommandMode.POSITION),
)
def test_robust_pid_centers_a_delayed_static_target(
    mode: GimbalCommandMode,
) -> None:
    base = closed_loop_config(mode)
    config = replace(
        base,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        camera=replace(
            base.camera,
            detection_latency_s=0.08,
            miss_probability=0.0,
        ),
        timing=replace(base.timing, episode_duration_s=3.0),
    )
    scenario = ClosedLoopScenario(
        "pid_static_target",
        "Delayed static target PID test.",
        config,
        StaticAngularMotion(math.radians(10.0)),
        StaticAngularMotion(),
    )
    run = run_closed_loop_controller(
        name="pid",
        description="pid",
        scenario=scenario,
        config=config,
        controller=RobustPIDGimbalController(
            config.servo,
            config.camera,
            mode,
        ),
        seed=5,
    )

    final_error = run.episode.frames[-1].diagnostics.true_image_error_normalized
    assert abs(final_error) < 0.02


def test_robust_pid_configuration_and_mode_mismatch_are_rejected() -> None:
    with pytest.raises(ValueError, match="minimum gain multiplier"):
        RobustPIDControllerConfig(
            minimum_gain_multiplier=2.0,
            maximum_gain_multiplier=1.0,
        )
    config = closed_loop_config(GimbalCommandMode.RATE)
    controller = RobustPIDGimbalController(
        config.servo,
        config.camera,
        GimbalCommandMode.RATE,
    )
    with pytest.raises(ValueError, match="command mode"):
        controller.act(
            _observation(
                time_s=0.1,
                mode=GimbalCommandMode.POSITION,
                error=0.1,
            )
        )
