from dataclasses import replace
import math

import numpy as np

import pytest

from autonomous_observation_lab.gimbal_servoing import (
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
    GimbalAction,
    GimbalCommandMode,
    GimbalServoEnv,
    MultiHorizonConstantVelocityTargetEstimator,
    TargetStateEstimate,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    closed_loop_config,
)
from autonomous_observation_lab.gimbal_servoing.config import ObservationProfile
from autonomous_observation_lab.gimbal_servoing.disturbances import (
    SinusoidalAngularMotion,
    StaticAngularMotion,
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


def test_multi_horizon_estimator_projects_one_causal_state() -> None:
    config = closed_loop_config(GimbalCommandMode.POSITION)
    horizons = (0.0, 0.1, 0.2, 0.3)
    estimator = MultiHorizonConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            max_prediction_horizon_s=0.75,
            history_horizon_s=1.5,
        ),
        prediction_horizons_s=horizons,
    )
    env = GimbalServoEnv(config)
    observation, _ = env.reset(seed=21)
    estimates = estimator.update_all(observation)
    for _ in range(20):
        observation = env.step(GimbalAction.position(0.0)).observation
        estimates = estimator.update_all(observation)

    assert estimator.last_estimate is estimates[0]
    assert estimator.last_estimates is estimates
    assert all(estimate.valid for estimate in estimates)
    assert tuple(
        estimate.time_s - observation.time_s for estimate in estimates
    ) == pytest.approx(horizons)
    assert len(
        {
            estimate.measurement_time_s.value
            for estimate in estimates
        }
    ) == 1


def test_multi_horizon_estimator_rejects_invalid_horizon_grid() -> None:
    config = ConstantVelocityEstimatorConfig(selected_axis_fov_rad=1.0)
    with pytest.raises(ValueError, match="non-empty"):
        MultiHorizonConstantVelocityTargetEstimator(
            config,
            prediction_horizons_s=(),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        MultiHorizonConstantVelocityTargetEstimator(
            config,
            prediction_horizons_s=(0.0, 0.2, 0.1),
        )


def test_body_rate_compensation_removes_delayed_body_motion_from_rate() -> None:
    base = closed_loop_config(GimbalCommandMode.POSITION)
    config = replace(
        base,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        camera=replace(
            base.camera,
            detection_latency_s=0.12,
            center_noise_std_normalized=0.0,
            miss_probability=0.0,
        ),
    )
    rms_errors_deg_s = []
    for compensated in (False, True):
        env = GimbalServoEnv(
            config,
            target_motion=StaticAngularMotion(0.0),
            body_motion=SinusoidalAngularMotion(
                amplitude_rad=math.radians(15.0),
                frequency_hz=0.5,
                phase_rad=0.1,
            ),
        )
        observation, diagnostics = env.reset(seed=5)
        estimator = ConstantVelocityTargetEstimator(
            ConstantVelocityEstimatorConfig(
                selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
                velocity_filter_coefficient=0.70,
                max_prediction_horizon_s=0.50,
                history_horizon_s=1.0,
                body_rate_compensation=compensated,
            )
        )
        errors = []
        for _ in range(150):
            estimate = estimator.update(observation)
            if estimate.valid and observation.time_s > 1.0:
                true_relative_rate = -diagnostics.body_rate_rad_s
                errors.append(
                    estimate.body_relative_rate_rad_s.value
                    - true_relative_rate
                )
            result = env.step(GimbalAction.position(0.0))
            observation = result.observation
            diagnostics = result.diagnostics
        rms_errors_deg_s.append(
            math.degrees(float(np.sqrt(np.mean(np.square(errors)))))
        )

    assert rms_errors_deg_s[1] < 0.05
    assert rms_errors_deg_s[1] < 0.02 * rms_errors_deg_s[0]
