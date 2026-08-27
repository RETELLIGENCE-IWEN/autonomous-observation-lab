from dataclasses import replace

import pytest

from autonomous_observation_lab.gimbal_servoing import (
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
    GimbalAction,
    GimbalCommandMode,
    GimbalServoEnv,
    TargetStateEstimate,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    closed_loop_config,
)


def test_missing_target_state_uses_explicit_validity_masks() -> None:
    estimate = TargetStateEstimate.missing(time_s=1.25)

    assert not estimate.valid
    assert not estimate.measurement_time_s.valid
    assert not estimate.body_relative_bearing_rad.valid
    assert not estimate.body_relative_rate_rad_s.valid
    assert not estimate.bearing_std_rad.valid
    assert not estimate.rate_std_rad_s.valid
    assert not estimate.prediction_horizon_s.valid


def test_estimator_rejects_measurements_older_than_prediction_horizon() -> None:
    config = closed_loop_config(GimbalCommandMode.RATE)
    estimator = ConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            max_prediction_horizon_s=0.05,
            history_horizon_s=1.0,
        )
    )
    env = GimbalServoEnv(config)
    observation, _ = env.reset(seed=12)
    estimate = estimator.update(observation)
    for _ in range(10):
        observation = env.step(GimbalAction.rate(0.0)).observation
        estimate = estimator.update(observation)

    assert not estimate.valid


def test_estimator_configuration_validates_uncertainty_and_timing() -> None:
    base = ConstantVelocityEstimatorConfig(selected_axis_fov_rad=1.0)
    with pytest.raises(ValueError, match="history horizon"):
        replace(base, history_horizon_s=base.max_prediction_horizon_s)
    with pytest.raises(ValueError, match="center noise"):
        replace(base, center_noise_std_normalized=-0.1)
    with pytest.raises(ValueError, match="uncertainty_filter_coefficient"):
        replace(base, uncertainty_filter_coefficient=0.0)
