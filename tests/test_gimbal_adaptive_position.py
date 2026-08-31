import math
from dataclasses import dataclass, field

import pytest

from autonomous_observation_lab.gimbal_servoing.config import (
    GimbalCommandMode,
    ServoConfig,
)
from autonomous_observation_lab.gimbal_servoing.controllers import (
    AdaptivePositionControllerConfig,
    AdaptiveTargetStatePositionController,
    _interpolate_target_estimate,
)
from autonomous_observation_lab.gimbal_servoing.estimators import (
    TargetStateEstimate,
)
from autonomous_observation_lab.gimbal_servoing.types import (
    GimbalObservation,
    MaskedScalar,
)


def _scalar(value: float = 0.0, valid: bool = True) -> MaskedScalar:
    return MaskedScalar(value, valid)


def _observation(
    *,
    time_s: float = 1.0,
    dt_s: float = 0.1,
    gimbal_angle_rad: float = 0.0,
) -> GimbalObservation:
    return GimbalObservation(
        time_s=time_s,
        control_dt_s=dt_s,
        frame_updated=True,
        measurement_age_s=_scalar(0.05),
        image_error_normalized=_scalar(0.0),
        bbox_width_fraction=_scalar(0.1),
        bbox_height_fraction=_scalar(0.1),
        confidence=_scalar(0.95),
        gimbal_angle_rad=_scalar(gimbal_angle_rad),
        gimbal_rate_rad_s=_scalar(0.0),
        body_rate_rad_s=_scalar(0.0),
        command_mode=GimbalCommandMode.POSITION,
        previous_action_normalized=0.0,
    )


@dataclass
class FakeMultiHorizonEstimator:
    prediction_horizons_s: tuple[float, ...]
    bearings_rad: tuple[float, ...]
    bearing_std_rad: tuple[float, ...]
    valid: bool = True
    name: str = "fake_multi_horizon"
    last_estimate: TargetStateEstimate = field(init=False)
    last_estimates: tuple[TargetStateEstimate, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self.last_estimates = tuple(
            TargetStateEstimate.missing(0.0)
            for _ in self.prediction_horizons_s
        )

    def update_all(
        self, observation: GimbalObservation
    ) -> tuple[TargetStateEstimate, ...]:
        if not self.valid:
            self.last_estimates = tuple(
                TargetStateEstimate.missing(observation.time_s)
                for _ in self.prediction_horizons_s
            )
        else:
            measurement_time_s = observation.time_s - 0.05
            self.last_estimates = tuple(
                TargetStateEstimate(
                    time_s=observation.time_s + horizon_s,
                    measurement_time_s=_scalar(measurement_time_s),
                    body_relative_bearing_rad=_scalar(bearing_rad),
                    body_relative_rate_rad_s=_scalar(math.radians(20.0)),
                    bearing_std_rad=_scalar(bearing_std_rad),
                    rate_std_rad_s=_scalar(math.radians(5.0)),
                    prediction_horizon_s=_scalar(horizon_s + 0.05),
                )
                for horizon_s, bearing_rad, bearing_std_rad in zip(
                    self.prediction_horizons_s,
                    self.bearings_rad,
                    self.bearing_std_rad,
                    strict=True,
                )
            )
        self.last_estimate = self.last_estimates[0]
        return self.last_estimates


def _servo() -> ServoConfig:
    return ServoConfig(
        min_angle_rad=math.radians(-60.0),
        max_angle_rad=math.radians(60.0),
        max_rate_rad_s=math.radians(100.0),
        max_acceleration_rad_s2=math.radians(1000.0),
        command_latency_s=0.04,
        rate_time_constant_s=0.06,
        position_gain_s_inv=5.0,
    )


def test_adaptive_position_configuration_rejects_invalid_trust_and_limits():
    with pytest.raises(ValueError, match="zero_trust"):
        AdaptivePositionControllerConfig(
            full_trust_std_ratio=2.0,
            zero_trust_std_ratio=2.0,
        )
    with pytest.raises(ValueError, match="rate_limit_scale"):
        AdaptivePositionControllerConfig(setpoint_rate_limit_scale=0.0)
    with pytest.raises(ValueError, match="jerk_rise_time"):
        AdaptivePositionControllerConfig(setpoint_jerk_rise_time_s=0.0)
    with pytest.raises(ValueError, match="full_fraction"):
        AdaptivePositionControllerConfig(
            visibility_risk_onset_fraction=0.8,
            visibility_risk_full_fraction=0.8,
        )
    with pytest.raises(ValueError, match="at least one"):
        AdaptivePositionControllerConfig(risk_jerk_limit_multiplier=0.5)


def test_adaptive_position_rejects_duplicate_prediction_horizons():
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1, 0.1),
        bearings_rad=(0.0, 0.1, 0.2),
        bearing_std_rad=(0.1, 0.1, 0.1),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        AdaptiveTargetStatePositionController(estimator, _servo())


def test_arrival_horizon_is_interpolated_without_extra_inference():
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1, 0.2, 0.3),
        bearings_rad=tuple(math.radians(value) for value in (0.0, 10.0, 20.0, 30.0)),
        bearing_std_rad=(0.1, 0.1, 0.1, 0.1),
    )
    controller = AdaptiveTargetStatePositionController(
        estimator=estimator,
        servo=_servo(),
        config=AdaptivePositionControllerConfig(position_response_fraction=0.25),
    )

    controller.act(_observation())

    assert controller.last_diagnostics.requested_horizon_s == pytest.approx(0.15)
    assert controller.last_diagnostics.effective_horizon_s == pytest.approx(0.15)
    assert controller.last_diagnostics.prediction_weight == pytest.approx(1.0)
    assert math.degrees(controller.last_diagnostics.raw_target_angle_rad) == pytest.approx(
        15.0
    )


def test_visibility_risk_adds_configured_horizon_boost():
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1, 0.2, 0.3),
        bearings_rad=tuple(math.radians(20.0) for _ in range(4)),
        bearing_std_rad=(0.01, 0.01, 0.01, 0.01),
    )
    controller = AdaptiveTargetStatePositionController(
        estimator=estimator,
        servo=_servo(),
        selected_axis_fov_rad=math.radians(60.0),
        config=AdaptivePositionControllerConfig(
            position_response_fraction=0.25,
            visibility_risk_onset_fraction=0.50,
            visibility_risk_full_fraction=0.75,
            risk_horizon_boost_s=0.10,
        ),
    )

    controller.act(_observation())

    assert controller.last_diagnostics.predicted_fov_fraction == pytest.approx(
        2.0 / 3.0
    )
    assert controller.last_diagnostics.visibility_risk == pytest.approx(2.0 / 3.0)
    assert controller.last_diagnostics.horizon_boost_s == pytest.approx(1.0 / 15.0)
    assert controller.last_diagnostics.requested_horizon_s == pytest.approx(
        0.15 + 1.0 / 15.0
    )


def test_visibility_guard_requires_configured_camera_fov():
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1),
        bearings_rad=(0.0, 0.0),
        bearing_std_rad=(0.1, 0.1),
    )

    with pytest.raises(ValueError, match="selected_axis_fov_rad"):
        AdaptiveTargetStatePositionController(
            estimator=estimator,
            servo=_servo(),
            config=AdaptivePositionControllerConfig(risk_horizon_boost_s=0.05),
        )


def test_visibility_guard_can_ignore_motion_returning_toward_center():
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1),
        bearings_rad=(math.radians(-20.0), math.radians(-20.0)),
        bearing_std_rad=(0.01, 0.01),
    )
    controller = AdaptiveTargetStatePositionController(
        estimator=estimator,
        servo=_servo(),
        selected_axis_fov_rad=math.radians(60.0),
        config=AdaptivePositionControllerConfig(
            position_response_fraction=0.0,
            visibility_risk_onset_fraction=0.50,
            visibility_risk_full_fraction=0.75,
            risk_requires_outward_motion=True,
            risk_horizon_boost_s=0.10,
        ),
    )

    controller.act(_observation())

    assert controller.last_diagnostics.predicted_fov_fraction == pytest.approx(
        2.0 / 3.0
    )
    assert controller.last_diagnostics.visibility_risk == 0.0
    assert controller.last_diagnostics.horizon_boost_s == 0.0


def test_uncertainty_ratio_blends_forecast_back_to_current_state():
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1, 0.2, 0.3),
        bearings_rad=tuple(math.radians(value) for value in (5.0, 15.0, 25.0, 35.0)),
        bearing_std_rad=(0.1, 0.2, 0.3, 0.4),
    )
    controller = AdaptiveTargetStatePositionController(
        estimator=estimator,
        servo=_servo(),
        config=AdaptivePositionControllerConfig(
            position_response_fraction=0.25,
            full_trust_std_ratio=1.0,
            zero_trust_std_ratio=2.0,
        ),
    )

    controller.act(_observation())

    assert controller.last_diagnostics.uncertainty_ratio == pytest.approx(
        math.sqrt(6.5),
    )
    assert controller.last_diagnostics.prediction_weight == pytest.approx(0.0)
    assert controller.last_diagnostics.effective_horizon_s == pytest.approx(0.0)
    assert math.degrees(controller.last_diagnostics.raw_target_angle_rad) == pytest.approx(
        5.0
    )


def test_circular_horizon_interpolation_uses_shortest_angle():
    observation = _observation()
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1),
        bearings_rad=(math.radians(179.0), math.radians(-179.0)),
        bearing_std_rad=(0.1, 0.1),
    )
    estimates = estimator.update_all(observation)

    interpolated = _interpolate_target_estimate(
        estimates,
        estimator.prediction_horizons_s,
        0.05,
        observation.time_s,
    )

    assert abs(math.degrees(interpolated.body_relative_bearing_rad.value)) == pytest.approx(
        180.0
    )


def test_trajectory_shaper_obeys_configured_rate_acceleration_and_jerk():
    servo = ServoConfig(
        min_angle_rad=math.radians(-60.0),
        max_angle_rad=math.radians(60.0),
        max_rate_rad_s=math.radians(10.0),
        max_acceleration_rad_s2=math.radians(20.0),
        command_latency_s=0.0,
        rate_time_constant_s=0.0,
        position_gain_s_inv=5.0,
    )
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1),
        bearings_rad=(math.radians(30.0), math.radians(30.0)),
        bearing_std_rad=(0.1, 0.1),
    )
    config = AdaptivePositionControllerConfig(
        position_response_fraction=0.0,
        setpoint_rate_limit_scale=0.5,
        setpoint_acceleration_limit_scale=0.5,
        setpoint_jerk_rise_time_s=0.5,
    )
    controller = AdaptiveTargetStatePositionController(
        estimator=estimator,
        servo=servo,
        config=config,
    )
    rates = []
    accelerations = []
    for step in range(10):
        controller.act(_observation(time_s=step * 0.1, dt_s=0.1))
        rates.append(controller.last_diagnostics.setpoint_rate_rad_s)
        accelerations.append(
            controller.last_diagnostics.setpoint_acceleration_rad_s2
        )

    max_rate = 0.5 * servo.max_rate_rad_s
    max_acceleration = 0.5 * servo.max_acceleration_rad_s2
    max_jerk = max_acceleration / config.setpoint_jerk_rise_time_s
    assert max(abs(value) for value in rates) <= max_rate + 1e-12
    assert max(abs(value) for value in accelerations) <= max_acceleration + 1e-12
    for left, right in zip(accelerations, accelerations[1:]):
        assert abs(right - left) / 0.1 <= max_jerk + 1e-12
    assert controller.last_diagnostics.shaped_target_angle_rad < math.radians(30.0)


def test_invalid_estimate_holds_last_shaped_position():
    estimator = FakeMultiHorizonEstimator(
        prediction_horizons_s=(0.0, 0.1),
        bearings_rad=(math.radians(30.0), math.radians(30.0)),
        bearing_std_rad=(0.1, 0.1),
    )
    controller = AdaptiveTargetStatePositionController(estimator, _servo())
    first = controller.act(_observation())
    estimator.valid = False

    held = controller.act(_observation(time_s=1.1))

    assert held.command_normalized == pytest.approx(first.command_normalized)
    assert not controller.last_diagnostics.valid
    assert controller.last_diagnostics.setpoint_rate_rad_s == 0.0
