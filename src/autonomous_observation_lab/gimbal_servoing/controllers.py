from dataclasses import dataclass, field

import numpy as np

from .config import ServoConfig
from .estimators import (
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
    TargetStateEstimate,
    TargetStateEstimator,
    angle_delta_rad,
    wrap_angle_rad,
)
from .types import GimbalAction, GimbalObservation


class ZeroController:
    name = "zero"

    def reset(self) -> None:
        return None

    def act(self, observation: GimbalObservation) -> GimbalAction:
        del observation
        return GimbalAction.rate(0.0)


@dataclass
class ProportionalController:
    gain: float = 1.0
    name: str = "proportional"

    def reset(self) -> None:
        return None

    def act(self, observation: GimbalObservation) -> GimbalAction:
        error = observation.image_error_normalized
        if not error.valid:
            return GimbalAction.rate(0.0)
        action = float(np.clip(self.gain * error.value, -1.0, 1.0))
        return GimbalAction.rate(action)


@dataclass
class WrongSignController(ProportionalController):
    name: str = "wrong_sign"

    def act(self, observation: GimbalObservation) -> GimbalAction:
        action = super().act(observation)
        return GimbalAction.rate(-action.command_normalized)


@dataclass
class BodyRateFeedforwardController:
    """Proportional visual feedback with deployable body-rate feedforward."""

    max_rate_rad_s: float
    proportional_gain: float = 1.0
    name: str = "body_rate_feedforward"

    def __post_init__(self) -> None:
        if self.max_rate_rad_s <= 0.0:
            raise ValueError("max_rate_rad_s must be positive")

    def reset(self) -> None:
        return None

    def act(self, observation: GimbalObservation) -> GimbalAction:
        feedback = 0.0
        if observation.image_error_normalized.valid:
            feedback = self.proportional_gain * observation.image_error_normalized.value
        feedforward = 0.0
        if observation.body_rate_rad_s.valid:
            feedforward = -observation.body_rate_rad_s.value / self.max_rate_rad_s
        return GimbalAction.rate(
            float(np.clip(feedback + feedforward, -1.0, 1.0))
        )


@dataclass
class ProportionalPositionController:
    """Convert visual error into an absolute body-relative position setpoint."""

    servo: ServoConfig
    selected_axis_fov_rad: float
    gain: float = 1.0
    name: str = "proportional_position"

    def __post_init__(self) -> None:
        if self.selected_axis_fov_rad <= 0.0:
            raise ValueError("selected_axis_fov_rad must be positive")
        self._last_command_normalized = 0.0

    def reset(self) -> None:
        self._last_command_normalized = 0.0

    def act(self, observation: GimbalObservation) -> GimbalAction:
        angle = observation.gimbal_angle_rad
        if not angle.valid:
            return GimbalAction.position(self._last_command_normalized)

        desired_angle = angle.value
        error = observation.image_error_normalized
        if error.valid:
            error_rad = error.value * 0.5 * self.selected_axis_fov_rad
            desired_angle += self.gain * error_rad
        desired_angle = float(
            np.clip(
                desired_angle,
                self.servo.min_angle_rad,
                self.servo.max_angle_rad,
            )
        )
        self._last_command_normalized = self.servo.normalized_from_position(
            desired_angle
        )
        return GimbalAction.position(self._last_command_normalized)


@dataclass
class TargetStateRateController:
    """Adapt a body-relative target-state estimate to desired gimbal rate."""

    estimator: TargetStateEstimator
    max_rate_rad_s: float
    proportional_gain_s_inv: float = 4.0
    name: str = "target_state_rate"
    last_estimate: TargetStateEstimate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_rate_rad_s <= 0.0:
            raise ValueError("max_rate_rad_s must be positive")
        if self.proportional_gain_s_inv < 0.0:
            raise ValueError("proportional gain must be non-negative")
        self.last_estimate = TargetStateEstimate.missing(0.0)

    def reset(self) -> None:
        self.estimator.reset()
        self.last_estimate = TargetStateEstimate.missing(0.0)

    def act(self, observation: GimbalObservation) -> GimbalAction:
        self.last_estimate = self.estimator.update(observation)
        gimbal_angle = observation.gimbal_angle_rad
        if not gimbal_angle.valid or not self.last_estimate.valid:
            return GimbalAction.rate(0.0)
        predicted_error = angle_delta_rad(
            self.last_estimate.body_relative_bearing_rad.value,
            gimbal_angle.value,
        )
        desired_rate = (
            self.last_estimate.body_relative_rate_rad_s.value
            + self.proportional_gain_s_inv * predicted_error
        )
        return GimbalAction.rate(
            float(np.clip(desired_rate / self.max_rate_rad_s, -1.0, 1.0))
        )


@dataclass
class TargetStatePositionController:
    """Adapt a target-state estimate to an absolute body-relative setpoint."""

    estimator: TargetStateEstimator
    servo: ServoConfig
    command_preview_s: float = 0.0
    name: str = "target_state_position"
    last_estimate: TargetStateEstimate = field(init=False, repr=False)
    _last_command_normalized: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.command_preview_s < 0.0:
            raise ValueError("command preview must be non-negative")
        self.last_estimate = TargetStateEstimate.missing(0.0)

    def reset(self) -> None:
        self.estimator.reset()
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self._last_command_normalized = 0.0

    def act(self, observation: GimbalObservation) -> GimbalAction:
        self.last_estimate = self.estimator.update(observation)
        if not self.last_estimate.valid:
            return GimbalAction.position(self._last_command_normalized)
        desired_angle = wrap_angle_rad(
            self.last_estimate.body_relative_bearing_rad.value
            + self.command_preview_s
            * self.last_estimate.body_relative_rate_rad_s.value
        )
        desired_angle = float(
            np.clip(
                desired_angle,
                self.servo.min_angle_rad,
                self.servo.max_angle_rad,
            )
        )
        self._last_command_normalized = self.servo.normalized_from_position(
            desired_angle
        )
        return GimbalAction.position(self._last_command_normalized)


@dataclass
class PredictiveRateController:
    """Compatibility facade for the analytical estimator plus rate adapter."""

    max_rate_rad_s: float
    selected_axis_fov_rad: float
    proportional_gain_s_inv: float = 4.0
    velocity_filter_coefficient: float = 0.45
    max_prediction_horizon_s: float = 0.30
    history_horizon_s: float = 1.0
    center_noise_std_normalized: float = 0.0
    name: str = "predictive_rate"
    _delegate: TargetStateRateController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        estimator = ConstantVelocityTargetEstimator(
            ConstantVelocityEstimatorConfig(
                selected_axis_fov_rad=self.selected_axis_fov_rad,
                center_noise_std_normalized=self.center_noise_std_normalized,
                velocity_filter_coefficient=self.velocity_filter_coefficient,
                max_prediction_horizon_s=self.max_prediction_horizon_s,
                history_horizon_s=self.history_horizon_s,
            )
        )
        self._delegate = TargetStateRateController(
            estimator=estimator,
            max_rate_rad_s=self.max_rate_rad_s,
            proportional_gain_s_inv=self.proportional_gain_s_inv,
            name=self.name,
        )

    @property
    def last_estimate(self) -> TargetStateEstimate:
        return self._delegate.last_estimate

    def reset(self) -> None:
        self._delegate.reset()

    def act(self, observation: GimbalObservation) -> GimbalAction:
        return self._delegate.act(observation)
