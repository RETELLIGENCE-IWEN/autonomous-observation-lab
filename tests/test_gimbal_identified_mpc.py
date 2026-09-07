import math
from dataclasses import replace

import pytest

from autonomous_observation_lab.gimbal_servoing import (
    GimbalCommandMode,
    IdentifiedMPCControllerConfig,
    IdentifiedMPCGimbalController,
    ObservationProfile,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    ClosedLoopScenario,
    closed_loop_config,
    run_closed_loop_controller,
)
from autonomous_observation_lab.gimbal_servoing.disturbances import (
    StaticAngularMotion,
)
from autonomous_observation_lab.gimbal_servoing.estimators import (
    ConstantVelocityEstimatorConfig,
    MultiHorizonConstantVelocityTargetEstimator,
)


def _controller(
    scenario: ClosedLoopScenario,
    mode: GimbalCommandMode,
    *,
    controller_config: IdentifiedMPCControllerConfig | None = None,
) -> IdentifiedMPCGimbalController:
    runtime = scenario.config
    period_s = runtime.timing.control_period_s
    horizons = tuple(index * period_s for index in range(10))
    estimator = MultiHorizonConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=runtime.camera.selected_axis_fov_rad,
            center_noise_std_normalized=(
                runtime.camera.center_noise_std_normalized
            ),
            max_prediction_horizon_s=(
                runtime.camera.detection_latency_s
                + runtime.camera.detection_latency_jitter_s
                + horizons[-1]
                + 2.0 * runtime.camera.frame_period_s
            ),
            history_horizon_s=2.0,
            body_rate_compensation=True,
        ),
        prediction_horizons_s=horizons,
    )
    return IdentifiedMPCGimbalController(
        estimator=estimator,
        servo=runtime.servo,
        selected_axis_fov_rad=runtime.camera.selected_axis_fov_rad,
        command_mode=mode,
        nominal_control_period_s=period_s,
        config=controller_config or IdentifiedMPCControllerConfig(),
    )


@pytest.mark.parametrize(
    "mode",
    (GimbalCommandMode.RATE, GimbalCommandMode.POSITION),
)
def test_identified_mpc_centers_delayed_static_target(
    mode: GimbalCommandMode,
) -> None:
    base = closed_loop_config(mode)
    runtime = replace(
        base,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        camera=replace(
            base.camera,
            detection_latency_s=0.08,
            miss_probability=0.0,
        ),
        timing=replace(base.timing, episode_duration_s=4.0),
    )
    scenario = ClosedLoopScenario(
        "mpc_static_target",
        "Delayed static target MPC test.",
        runtime,
        StaticAngularMotion(math.radians(10.0)),
        StaticAngularMotion(),
    )
    controller = _controller(scenario, mode)
    run = run_closed_loop_controller(
        name="mpc",
        description="mpc",
        scenario=scenario,
        config=runtime,
        controller=controller,
        seed=7,
    )

    final_error = run.episode.frames[-1].diagnostics.true_image_error_normalized
    assert abs(final_error) < 0.05
    assert run.metrics.loss_of_view_fraction == 0.0
    assert controller.last_diagnostics.valid
    assert controller.last_diagnostics.warm_started
    assert all(
        -1.0 <= frame.action.command_normalized <= 1.0
        for frame in run.episode.frames
    )
    if mode is GimbalCommandMode.RATE:
        physical_commands = [
            frame.action.command_normalized * runtime.servo.max_rate_rad_s
            for frame in run.episode.frames
        ]
        maximum_step = (
            controller.config.rate_command_slew_scale
            * runtime.servo.max_acceleration_rad_s2
            * runtime.timing.control_period_s
        )
    else:
        physical_commands = [
            runtime.servo.position_from_normalized(
                frame.action.command_normalized
            )
            for frame in run.episode.frames
        ]
        maximum_step = (
            controller.config.position_command_slew_scale
            * runtime.servo.max_rate_rad_s
            * runtime.timing.control_period_s
        )
    assert max(
        abs(right - left)
        for left, right in zip(physical_commands, physical_commands[1:])
    ) <= maximum_step + 1e-9


def test_identified_mpc_reports_scaled_identified_plant() -> None:
    mode = GimbalCommandMode.RATE
    scenario = ClosedLoopScenario(
        "mpc_config",
        "MPC configuration test.",
        closed_loop_config(mode),
        StaticAngularMotion(math.radians(5.0)),
        StaticAngularMotion(),
    )
    selected = IdentifiedMPCControllerConfig(
        model_rate_time_constant_scale=1.2,
        model_command_latency_scale=0.8,
        model_position_gain_scale=0.9,
    )
    controller = _controller(
        scenario,
        mode,
        controller_config=selected,
    )
    runtime = replace(
        scenario.config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
    )
    run_closed_loop_controller(
        name="mpc",
        description="mpc",
        scenario=scenario,
        config=runtime,
        controller=controller,
        seed=3,
    )

    diagnostics = controller.last_diagnostics
    assert diagnostics.modeled_rate_time_constant_s == pytest.approx(
        1.2 * runtime.servo.rate_time_constant_s
    )
    assert diagnostics.modeled_command_latency_s == pytest.approx(
        0.8 * runtime.servo.command_latency_s
    )
    assert diagnostics.modeled_position_gain_s_inv == pytest.approx(
        0.9 * runtime.servo.position_gain_s_inv
    )


def test_identified_mpc_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="visibility margin"):
        IdentifiedMPCControllerConfig(visibility_margin_fraction=0.0)
    with pytest.raises(ValueError, match="iterations"):
        IdentifiedMPCControllerConfig(optimization_iterations=0)
