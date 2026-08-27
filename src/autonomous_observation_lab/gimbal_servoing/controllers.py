from dataclasses import dataclass

import numpy as np

from .config import ServoConfig
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
