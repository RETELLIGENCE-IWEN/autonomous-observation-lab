import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import GimbalCommandMode, ServoConfig
from .estimators import (
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
    MultiHorizonTargetStateEstimator,
    TargetStateEstimate,
    TargetStateEstimator,
    angle_delta_rad,
    wrap_angle_rad,
)
from .types import GimbalAction, GimbalObservation, MaskedScalar


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


@dataclass(frozen=True)
class AdaptivePositionControllerConfig:
    """Hardware-relative timing, trust, and trajectory settings for V2.

    The forecast lead is derived from the configured servo command latency,
    rate time constant, and a configurable fraction of the position-loop time
    constant. Detection age is already a GRU input and is intentionally not
    added again. Trajectory limits are fractions of the configured plant
    limits. Values above one are valid because this shapes a requested
    setpoint, not physical motion; the independent inner servo still enforces
    the plant limits. Jerk is set by the time allowed to reach the setpoint
    acceleration limit.
    """

    actuator_arrival_time_scale: float = 1.0
    position_response_fraction: float = 0.25
    additional_preview_s: float = 0.0
    full_trust_std_ratio: float = 1.0
    zero_trust_std_ratio: float = 2.0
    minimum_prediction_weight: float = 0.0
    setpoint_rate_limit_scale: float = 1.0
    setpoint_acceleration_limit_scale: float = 1.0
    setpoint_jerk_rise_time_s: float = 0.075

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.actuator_arrival_time_scale)
            or self.actuator_arrival_time_scale <= 0.0
        ):
            raise ValueError(
                "actuator_arrival_time_scale must be finite and positive"
            )
        for name in ("position_response_fraction", "additional_preview_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.full_trust_std_ratio)
            or self.full_trust_std_ratio <= 0.0
        ):
            raise ValueError("full_trust_std_ratio must be finite and positive")
        if (
            not math.isfinite(self.zero_trust_std_ratio)
            or self.zero_trust_std_ratio <= self.full_trust_std_ratio
        ):
            raise ValueError(
                "zero_trust_std_ratio must exceed full_trust_std_ratio"
            )
        if not 0.0 <= self.minimum_prediction_weight <= 1.0:
            raise ValueError("minimum_prediction_weight must be in [0, 1]")
        for name in (
            "setpoint_rate_limit_scale",
            "setpoint_acceleration_limit_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.setpoint_jerk_rise_time_s)
            or self.setpoint_jerk_rise_time_s <= 0.0
        ):
            raise ValueError(
                "setpoint_jerk_rise_time_s must be finite and positive"
            )


@dataclass(frozen=True)
class AdaptivePositionDiagnostics:
    valid: bool
    requested_horizon_s: float
    effective_horizon_s: float
    prediction_weight: float
    uncertainty_ratio: float
    raw_target_angle_rad: float
    shaped_target_angle_rad: float
    setpoint_rate_rad_s: float
    setpoint_acceleration_rad_s2: float
    rate_limited: bool
    acceleration_limited: bool
    jerk_limited: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _interpolate_target_estimate(
    estimates: tuple[TargetStateEstimate, ...],
    horizons_s: tuple[float, ...],
    requested_horizon_s: float,
    observation_time_s: float,
) -> TargetStateEstimate:
    if len(estimates) != len(horizons_s) or not estimates:
        raise ValueError("estimates must match non-empty prediction horizons")
    if any(not estimate.valid for estimate in estimates):
        return TargetStateEstimate.missing(observation_time_s)
    requested = float(
        np.clip(requested_horizon_s, horizons_s[0], horizons_s[-1])
    )
    right = int(np.searchsorted(horizons_s, requested, side="left"))
    if right == 0:
        left = right = 0
        fraction = 0.0
    elif right >= len(horizons_s):
        left = right = len(horizons_s) - 1
        fraction = 0.0
    else:
        left = right - 1
        span = horizons_s[right] - horizons_s[left]
        fraction = (requested - horizons_s[left]) / span
    left_estimate = estimates[left]
    right_estimate = estimates[right]

    def linear(left_value: float, right_value: float) -> float:
        return left_value + fraction * (right_value - left_value)

    left_bearing = left_estimate.body_relative_bearing_rad.value
    bearing = wrap_angle_rad(
        left_bearing
        + fraction
        * angle_delta_rad(
            right_estimate.body_relative_bearing_rad.value,
            left_bearing,
        )
    )
    bearing_variance = linear(
        left_estimate.bearing_std_rad.value**2,
        right_estimate.bearing_std_rad.value**2,
    )
    rate_variance = linear(
        left_estimate.rate_std_rad_s.value**2,
        right_estimate.rate_std_rad_s.value**2,
    )
    measurement_time_s = left_estimate.measurement_time_s.value
    estimate_time_s = observation_time_s + requested
    return TargetStateEstimate(
        time_s=estimate_time_s,
        measurement_time_s=MaskedScalar(measurement_time_s, True),
        body_relative_bearing_rad=MaskedScalar(bearing, True),
        body_relative_rate_rad_s=MaskedScalar(
            linear(
                left_estimate.body_relative_rate_rad_s.value,
                right_estimate.body_relative_rate_rad_s.value,
            ),
            True,
        ),
        bearing_std_rad=MaskedScalar(math.sqrt(max(0.0, bearing_variance)), True),
        rate_std_rad_s=MaskedScalar(math.sqrt(max(0.0, rate_variance)), True),
        prediction_horizon_s=MaskedScalar(
            estimate_time_s - measurement_time_s,
            True,
        ),
    )


@dataclass
class AdaptiveTargetStatePositionController:
    """V2 absolute-position adapter with timing, trust, and shaping."""

    estimator: MultiHorizonTargetStateEstimator
    servo: ServoConfig
    config: AdaptivePositionControllerConfig = AdaptivePositionControllerConfig()
    name: str = "adaptive_target_state_position"
    last_estimate: TargetStateEstimate = field(init=False, repr=False)
    last_diagnostics: AdaptivePositionDiagnostics = field(init=False, repr=False)
    _setpoint_angle_rad: float | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _setpoint_rate_rad_s: float = field(init=False, default=0.0, repr=False)
    _setpoint_acceleration_rad_s2: float = field(
        init=False,
        default=0.0,
        repr=False,
    )

    def __post_init__(self) -> None:
        horizons = self.estimator.prediction_horizons_s
        if not horizons or any(
            right <= left for left, right in zip(horizons, horizons[1:])
        ):
            raise ValueError(
                "prediction horizons must be non-empty and strictly increasing"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in horizons):
            raise ValueError("prediction horizons must be finite and non-negative")
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self.last_diagnostics = self._missing_diagnostics()

    def _missing_diagnostics(self) -> AdaptivePositionDiagnostics:
        angle = self._setpoint_angle_rad or 0.0
        return AdaptivePositionDiagnostics(
            valid=False,
            requested_horizon_s=0.0,
            effective_horizon_s=0.0,
            prediction_weight=0.0,
            uncertainty_ratio=0.0,
            raw_target_angle_rad=angle,
            shaped_target_angle_rad=angle,
            setpoint_rate_rad_s=0.0,
            setpoint_acceleration_rad_s2=0.0,
            rate_limited=False,
            acceleration_limited=False,
            jerk_limited=False,
        )

    def reset(self) -> None:
        self.estimator.reset()
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self._setpoint_angle_rad = None
        self._setpoint_rate_rad_s = 0.0
        self._setpoint_acceleration_rad_s2 = 0.0
        self.last_diagnostics = self._missing_diagnostics()

    def _arrival_horizon_s(self) -> float:
        position_response_s = (
            self.config.position_response_fraction / self.servo.position_gain_s_inv
        )
        requested = (
            self.config.actuator_arrival_time_scale
            * (
                self.servo.command_latency_s
                + self.servo.rate_time_constant_s
                + position_response_s
            )
            + self.config.additional_preview_s
        )
        horizons = self.estimator.prediction_horizons_s
        return float(np.clip(requested, horizons[0], horizons[-1]))

    def _prediction_weight(
        self,
        current: TargetStateEstimate,
        forecast: TargetStateEstimate,
    ) -> tuple[float, float]:
        current_std = max(current.bearing_std_rad.value, 1e-9)
        ratio = forecast.bearing_std_rad.value / current_std
        if ratio <= self.config.full_trust_std_ratio:
            weight = 1.0
        elif ratio >= self.config.zero_trust_std_ratio:
            weight = self.config.minimum_prediction_weight
        else:
            fraction = (
                ratio - self.config.full_trust_std_ratio
            ) / (
                self.config.zero_trust_std_ratio
                - self.config.full_trust_std_ratio
            )
            weight = 1.0 - fraction * (
                1.0 - self.config.minimum_prediction_weight
            )
        return float(np.clip(weight, 0.0, 1.0)), ratio

    def _blend_forecast(
        self,
        current: TargetStateEstimate,
        forecast: TargetStateEstimate,
        requested_horizon_s: float,
        prediction_weight: float,
        observation_time_s: float,
    ) -> TargetStateEstimate:
        current_horizon_s = self.estimator.prediction_horizons_s[0]
        effective_horizon_s = current_horizon_s + prediction_weight * (
            requested_horizon_s - current_horizon_s
        )
        bearing = wrap_angle_rad(
            current.body_relative_bearing_rad.value
            + prediction_weight
            * angle_delta_rad(
                forecast.body_relative_bearing_rad.value,
                current.body_relative_bearing_rad.value,
            )
        )
        measurement_time_s = current.measurement_time_s.value
        return TargetStateEstimate(
            time_s=observation_time_s + effective_horizon_s,
            measurement_time_s=MaskedScalar(measurement_time_s, True),
            body_relative_bearing_rad=MaskedScalar(bearing, True),
            body_relative_rate_rad_s=MaskedScalar(
                current.body_relative_rate_rad_s.value
                + prediction_weight
                * (
                    forecast.body_relative_rate_rad_s.value
                    - current.body_relative_rate_rad_s.value
                ),
                True,
            ),
            bearing_std_rad=MaskedScalar(
                current.bearing_std_rad.value
                + prediction_weight
                * (
                    forecast.bearing_std_rad.value
                    - current.bearing_std_rad.value
                ),
                True,
            ),
            rate_std_rad_s=MaskedScalar(
                current.rate_std_rad_s.value
                + prediction_weight
                * (
                    forecast.rate_std_rad_s.value
                    - current.rate_std_rad_s.value
                ),
                True,
            ),
            prediction_horizon_s=MaskedScalar(
                observation_time_s + effective_horizon_s - measurement_time_s,
                True,
            ),
        )

    @staticmethod
    def _move_toward(value: float, target: float, maximum_delta: float) -> float:
        return value + float(
            np.clip(target - value, -maximum_delta, maximum_delta)
        )

    def _shape_setpoint(
        self,
        target_angle_rad: float,
        observation: GimbalObservation,
    ) -> tuple[float, bool, bool, bool]:
        target = float(
            np.clip(
                target_angle_rad,
                self.servo.min_angle_rad,
                self.servo.max_angle_rad,
            )
        )
        if self._setpoint_angle_rad is None:
            self._setpoint_angle_rad = (
                observation.gimbal_angle_rad.value
                if observation.gimbal_angle_rad.valid
                else 0.0
            )
        dt_s = observation.control_dt_s
        if dt_s <= 0.0:
            self._setpoint_rate_rad_s = 0.0
            self._setpoint_acceleration_rad_s2 = 0.0
            return self._setpoint_angle_rad, False, False, False
        max_rate = (
            self.config.setpoint_rate_limit_scale * self.servo.max_rate_rad_s
        )
        max_acceleration = (
            self.config.setpoint_acceleration_limit_scale
            * self.servo.max_acceleration_rad_s2
        )
        max_jerk = max_acceleration / self.config.setpoint_jerk_rise_time_s
        error = target - self._setpoint_angle_rad
        stopping_speed = math.sqrt(2.0 * max_acceleration * abs(error))
        desired_rate = math.copysign(
            min(max_rate, stopping_speed),
            error,
        ) if abs(error) > 1e-12 else 0.0
        raw_acceleration = (desired_rate - self._setpoint_rate_rad_s) / dt_s
        desired_acceleration = float(
            np.clip(raw_acceleration, -max_acceleration, max_acceleration)
        )
        acceleration = self._move_toward(
            self._setpoint_acceleration_rad_s2,
            desired_acceleration,
            max_jerk * dt_s,
        )
        rate = float(
            np.clip(
                self._setpoint_rate_rad_s + acceleration * dt_s,
                -max_rate,
                max_rate,
            )
        )
        step = rate * dt_s
        if step * error > 0.0 and abs(step) >= abs(error):
            self._setpoint_angle_rad = target
            rate = 0.0
            acceleration = 0.0
        else:
            self._setpoint_angle_rad = float(
                np.clip(
                    self._setpoint_angle_rad + step,
                    self.servo.min_angle_rad,
                    self.servo.max_angle_rad,
                )
            )
        self._setpoint_rate_rad_s = rate
        self._setpoint_acceleration_rad_s2 = acceleration
        return (
            self._setpoint_angle_rad,
            abs(desired_rate) >= max_rate - 1e-12,
            abs(raw_acceleration) > max_acceleration + 1e-12,
            abs(acceleration - desired_acceleration) > 1e-12,
        )

    def act(self, observation: GimbalObservation) -> GimbalAction:
        estimates = self.estimator.update_all(observation)
        if not estimates or not all(estimate.valid for estimate in estimates):
            self.last_estimate = TargetStateEstimate.missing(observation.time_s)
            if self._setpoint_angle_rad is None:
                self._setpoint_angle_rad = (
                    observation.gimbal_angle_rad.value
                    if observation.gimbal_angle_rad.valid
                    else 0.0
                )
            self._setpoint_rate_rad_s = 0.0
            self._setpoint_acceleration_rad_s2 = 0.0
            self.last_diagnostics = self._missing_diagnostics()
            command = self.servo.normalized_from_position(
                float(
                    np.clip(
                        self._setpoint_angle_rad,
                        self.servo.min_angle_rad,
                        self.servo.max_angle_rad,
                    )
                )
            )
            return GimbalAction.position(command)

        requested_horizon_s = self._arrival_horizon_s()
        forecast = _interpolate_target_estimate(
            estimates,
            self.estimator.prediction_horizons_s,
            requested_horizon_s,
            observation.time_s,
        )
        current = estimates[0]
        prediction_weight, uncertainty_ratio = self._prediction_weight(
            current,
            forecast,
        )
        self.last_estimate = self._blend_forecast(
            current,
            forecast,
            requested_horizon_s,
            prediction_weight,
            observation.time_s,
        )
        raw_target = float(
            np.clip(
                self.last_estimate.body_relative_bearing_rad.value,
                self.servo.min_angle_rad,
                self.servo.max_angle_rad,
            )
        )
        shaped_target, rate_limited, acceleration_limited, jerk_limited = (
            self._shape_setpoint(raw_target, observation)
        )
        self.last_diagnostics = AdaptivePositionDiagnostics(
            valid=True,
            requested_horizon_s=requested_horizon_s,
            effective_horizon_s=(
                self.last_estimate.time_s - observation.time_s
            ),
            prediction_weight=prediction_weight,
            uncertainty_ratio=uncertainty_ratio,
            raw_target_angle_rad=raw_target,
            shaped_target_angle_rad=shaped_target,
            setpoint_rate_rad_s=self._setpoint_rate_rad_s,
            setpoint_acceleration_rad_s2=(
                self._setpoint_acceleration_rad_s2
            ),
            rate_limited=rate_limited,
            acceleration_limited=acceleration_limited,
            jerk_limited=jerk_limited,
        )
        return GimbalAction.position(
            self.servo.normalized_from_position(shaped_target)
        )


@dataclass
class SearchFallbackController:
    """Sweep the configured travel envelope while an estimate is unavailable."""

    delegate: TargetStateRateController | TargetStatePositionController
    servo: ServoConfig
    command_mode: GimbalCommandMode
    search_rate_normalized: float = 0.25
    search_position_fraction: float = 0.90
    reversal_margin_rad: float = 0.0
    name: str = "target_state_search_fallback"
    _direction: float = field(init=False, default=1.0, repr=False)

    def __post_init__(self) -> None:
        for name in ("search_rate_normalized", "search_position_fraction"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.reversal_margin_rad < 0.0:
            raise ValueError("reversal margin must be non-negative")
        available_margin = min(
            -self.servo.min_angle_rad,
            self.servo.max_angle_rad,
        )
        if self.reversal_margin_rad >= available_margin:
            raise ValueError("reversal margin must fit inside servo travel")

    @property
    def last_estimate(self) -> TargetStateEstimate:
        return self.delegate.last_estimate

    def reset(self) -> None:
        self.delegate.reset()
        self._direction = 1.0

    def _remember_target_direction(
        self, observation: GimbalObservation
    ) -> None:
        if not observation.gimbal_angle_rad.valid:
            return
        error = angle_delta_rad(
            self.last_estimate.body_relative_bearing_rad.value,
            observation.gimbal_angle_rad.value,
        )
        if abs(error) > 1e-6:
            self._direction = 1.0 if error > 0.0 else -1.0
            return
        rate = self.last_estimate.body_relative_rate_rad_s.value
        if abs(rate) > 1e-6:
            self._direction = 1.0 if rate > 0.0 else -1.0

    def _reverse_at_travel_limit(
        self, observation: GimbalObservation
    ) -> None:
        angle = observation.gimbal_angle_rad
        if not angle.valid:
            return
        if angle.value >= self.servo.max_angle_rad - self.reversal_margin_rad:
            self._direction = -1.0
        elif angle.value <= self.servo.min_angle_rad + self.reversal_margin_rad:
            self._direction = 1.0

    def act(self, observation: GimbalObservation) -> GimbalAction:
        nominal_action = self.delegate.act(observation)
        if self.last_estimate.valid:
            self._remember_target_direction(observation)
            return nominal_action
        self._reverse_at_travel_limit(observation)
        if self.command_mode is GimbalCommandMode.RATE:
            return GimbalAction.rate(
                self._direction * self.search_rate_normalized
            )
        return GimbalAction.position(
            self._direction * self.search_position_fraction
        )


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
