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
    acceleration limit. Visibility risk is computed from the configured camera
    FOV, predicted bearing error, and predicted uncertainty. Its horizon and
    shaping multipliers are neutral by default.
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
    visibility_risk_onset_fraction: float = 0.65
    visibility_risk_full_fraction: float = 0.90
    visibility_uncertainty_sigma: float = 0.0
    risk_requires_outward_motion: bool = False
    risk_horizon_boost_s: float = 0.0
    risk_rate_limit_multiplier: float = 1.0
    risk_acceleration_limit_multiplier: float = 1.0
    risk_jerk_limit_multiplier: float = 1.0

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
        if (
            not math.isfinite(self.visibility_risk_onset_fraction)
            or self.visibility_risk_onset_fraction < 0.0
        ):
            raise ValueError(
                "visibility_risk_onset_fraction must be finite and non-negative"
            )
        if (
            not math.isfinite(self.visibility_risk_full_fraction)
            or self.visibility_risk_full_fraction
            <= self.visibility_risk_onset_fraction
        ):
            raise ValueError(
                "visibility_risk_full_fraction must exceed the onset fraction"
            )
        for name in ("visibility_uncertainty_sigma", "risk_horizon_boost_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.risk_requires_outward_motion, bool):
            raise ValueError("risk_requires_outward_motion must be boolean")
        for name in (
            "risk_rate_limit_multiplier",
            "risk_acceleration_limit_multiplier",
            "risk_jerk_limit_multiplier",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be finite and at least one")


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
    visibility_risk: float
    predicted_fov_fraction: float
    horizon_boost_s: float

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
    selected_axis_fov_rad: float | None = None
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
        if self.selected_axis_fov_rad is not None and (
            not math.isfinite(self.selected_axis_fov_rad)
            or self.selected_axis_fov_rad <= 0.0
        ):
            raise ValueError("selected_axis_fov_rad must be finite and positive")
        risk_changes_control = (
            self.config.risk_horizon_boost_s > 0.0
            or self.config.risk_rate_limit_multiplier > 1.0
            or self.config.risk_acceleration_limit_multiplier > 1.0
            or self.config.risk_jerk_limit_multiplier > 1.0
        )
        if risk_changes_control and self.selected_axis_fov_rad is None:
            raise ValueError(
                "selected_axis_fov_rad is required when visibility risk changes control"
            )
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
            visibility_risk=0.0,
            predicted_fov_fraction=0.0,
            horizon_boost_s=0.0,
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

    def _visibility_risk(
        self,
        estimate: TargetStateEstimate,
        observation: GimbalObservation,
    ) -> tuple[float, float]:
        if (
            self.selected_axis_fov_rad is None
            or not estimate.valid
            or not observation.gimbal_angle_rad.valid
        ):
            return 0.0, 0.0
        image_error_rad = angle_delta_rad(
            estimate.body_relative_bearing_rad.value,
            observation.gimbal_angle_rad.value,
        )
        angular_margin = abs(image_error_rad) + (
            self.config.visibility_uncertainty_sigma
            * estimate.bearing_std_rad.value
        )
        fov_fraction = angular_margin / (0.5 * self.selected_axis_fov_rad)
        if self.config.risk_requires_outward_motion:
            if not observation.gimbal_rate_rad_s.valid:
                return 0.0, fov_fraction
            image_error_rate_rad_s = (
                estimate.body_relative_rate_rad_s.value
                - observation.gimbal_rate_rad_s.value
            )
            if image_error_rad * image_error_rate_rad_s <= 0.0:
                return 0.0, fov_fraction
        risk = (
            fov_fraction - self.config.visibility_risk_onset_fraction
        ) / (
            self.config.visibility_risk_full_fraction
            - self.config.visibility_risk_onset_fraction
        )
        return float(np.clip(risk, 0.0, 1.0)), fov_fraction

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
        visibility_risk: float = 0.0,
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
        risk = float(np.clip(visibility_risk, 0.0, 1.0))
        rate_multiplier = 1.0 + risk * (
            self.config.risk_rate_limit_multiplier - 1.0
        )
        acceleration_multiplier = 1.0 + risk * (
            self.config.risk_acceleration_limit_multiplier - 1.0
        )
        jerk_multiplier = 1.0 + risk * (
            self.config.risk_jerk_limit_multiplier - 1.0
        )
        max_rate = (
            self.config.setpoint_rate_limit_scale * self.servo.max_rate_rad_s
            * rate_multiplier
        )
        base_max_acceleration = (
            self.config.setpoint_acceleration_limit_scale
            * self.servo.max_acceleration_rad_s2
        )
        max_acceleration = base_max_acceleration * acceleration_multiplier
        max_jerk = (
            base_max_acceleration
            / self.config.setpoint_jerk_rise_time_s
            * jerk_multiplier
        )
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

        base_requested_horizon_s = self._arrival_horizon_s()
        base_forecast = _interpolate_target_estimate(
            estimates,
            self.estimator.prediction_horizons_s,
            base_requested_horizon_s,
            observation.time_s,
        )
        current = estimates[0]
        base_prediction_weight, _base_uncertainty_ratio = self._prediction_weight(
            current,
            base_forecast,
        )
        base_estimate = self._blend_forecast(
            current,
            base_forecast,
            base_requested_horizon_s,
            base_prediction_weight,
            observation.time_s,
        )
        visibility_risk, predicted_fov_fraction = self._visibility_risk(
            base_estimate,
            observation,
        )
        horizons = self.estimator.prediction_horizons_s
        requested_horizon_s = float(
            np.clip(
                base_requested_horizon_s
                + visibility_risk * self.config.risk_horizon_boost_s,
                horizons[0],
                horizons[-1],
            )
        )
        horizon_boost_s = requested_horizon_s - base_requested_horizon_s
        forecast = _interpolate_target_estimate(
            estimates,
            horizons,
            requested_horizon_s,
            observation.time_s,
        )
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
            self._shape_setpoint(raw_target, observation, visibility_risk)
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
            visibility_risk=visibility_risk,
            predicted_fov_fraction=predicted_fov_fraction,
            horizon_boost_s=horizon_boost_s,
        )
        return GimbalAction.position(
            self.servo.normalized_from_position(shaped_target)
        )


@dataclass(frozen=True)
class PredictivePositionOptimizerConfig:
    """Hardware-relative objective and search settings for position V3."""

    candidate_grid_size: int = 31
    servo_simulation_step_s: float = 0.005
    maximum_optimization_horizon_s: float = 0.10
    forecast_full_trust_std_ratio: float = 1.0
    forecast_zero_trust_std_ratio: float = 4.0
    minimum_forecast_weight: float = 0.50
    tracking_weight: float = 1.0
    terminal_tracking_weight: float = 3.0
    rate_matching_weight: float = 0.35
    visibility_weight: float = 8.0
    visibility_onset_fov_fraction: float = 0.70
    uncertainty_sigma: float = 0.5
    command_change_weight: float = 0.03
    command_rate_change_weight: float = 0.005
    travel_margin_weight: float = 0.01
    travel_margin_fraction: float = 0.92
    activation_rate_onset_fraction: float = 0.65
    activation_rate_full_fraction: float = 1.00
    activation_visibility_onset_fraction: float = 0.70
    activation_visibility_full_fraction: float = 0.95
    activation_gate_mode: str = "maximum"
    require_command_effect_within_horizon: bool = True
    command_effect_response_fraction: float = 0.25
    minimum_optimizer_position_gain_s_inv: float = 4.0
    fallback_arrival_time_scale: float = 0.85
    fallback_risk_horizon_boost_s: float = 0.125
    fallback_visibility_onset_fraction: float = 0.55
    fallback_visibility_full_fraction: float = 0.85
    fallback_uncertainty_sigma: float = 1.0
    setpoint_rate_limit_scale: float = 6.0
    setpoint_acceleration_limit_scale: float = 12.0
    setpoint_jerk_rise_time_s: float = 0.015

    def __post_init__(self) -> None:
        if self.candidate_grid_size < 3 or self.candidate_grid_size % 2 == 0:
            raise ValueError("candidate_grid_size must be odd and at least three")
        for name in (
            "servo_simulation_step_s",
            "maximum_optimization_horizon_s",
            "forecast_full_trust_std_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.forecast_zero_trust_std_ratio)
            or self.forecast_zero_trust_std_ratio
            <= self.forecast_full_trust_std_ratio
        ):
            raise ValueError(
                "forecast_zero_trust_std_ratio must exceed full trust"
            )
        if not 0.0 <= self.minimum_forecast_weight <= 1.0:
            raise ValueError("minimum_forecast_weight must be in [0, 1]")
        for name in (
            "tracking_weight",
            "terminal_tracking_weight",
            "rate_matching_weight",
            "visibility_weight",
            "uncertainty_sigma",
            "command_change_weight",
            "command_rate_change_weight",
            "travel_margin_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.visibility_onset_fov_fraction < 1.0:
            raise ValueError(
                "visibility_onset_fov_fraction must be in [0, 1)"
            )
        if not 0.0 < self.travel_margin_fraction <= 1.0:
            raise ValueError("travel_margin_fraction must be in (0, 1]")
        for onset_name, full_name in (
            (
                "activation_rate_onset_fraction",
                "activation_rate_full_fraction",
            ),
            (
                "activation_visibility_onset_fraction",
                "activation_visibility_full_fraction",
            ),
            (
                "fallback_visibility_onset_fraction",
                "fallback_visibility_full_fraction",
            ),
        ):
            onset = getattr(self, onset_name)
            full = getattr(self, full_name)
            if not math.isfinite(onset) or onset < 0.0:
                raise ValueError(f"{onset_name} must be finite and non-negative")
            if not math.isfinite(full) or full <= onset:
                raise ValueError(f"{full_name} must exceed its onset")
        if not isinstance(self.require_command_effect_within_horizon, bool):
            raise ValueError(
                "require_command_effect_within_horizon must be boolean"
            )
        if self.activation_gate_mode not in {"maximum", "minimum", "product"}:
            raise ValueError(
                "activation_gate_mode must be maximum, minimum, or product"
            )
        for name in (
            "fallback_arrival_time_scale",
            "fallback_risk_horizon_boost_s",
            "fallback_uncertainty_sigma",
            "command_effect_response_fraction",
            "minimum_optimizer_position_gain_s_inv",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
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
class PredictivePositionOptimizerDiagnostics:
    valid: bool
    selected_command_angle_rad: float
    raw_candidate_angle_rad: float
    selected_objective: float
    predicted_terminal_error_fov_fraction: float
    predicted_peak_error_fov_fraction: float
    predicted_terminal_rate_error_normalized: float
    predicted_rate_utilization: float
    predicted_acceleration_utilization: float
    setpoint_rate_rad_s: float
    setpoint_acceleration_rad_s2: float
    rate_limited: bool
    acceleration_limited: bool
    jerk_limited: bool
    evaluated_candidate_count: int
    optimizer_active: bool
    activation_score: float
    activation_rate_score: float
    activation_visibility_score: float
    fallback_target_angle_rad: float
    optimization_horizon_s: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass
class ConstrainedPredictivePositionController:
    """Short-horizon position optimizer over configurable servo dynamics.

    Each candidate is first passed through a configurable setpoint trajectory
    constraint, then evaluated by simulating the actual position-loop plant.
    Issued command history is retained so configured command latency is
    represented without exposing simulator-only state to the controller.
    """

    estimator: MultiHorizonTargetStateEstimator
    servo: ServoConfig
    selected_axis_fov_rad: float
    config: PredictivePositionOptimizerConfig = (
        PredictivePositionOptimizerConfig()
    )
    name: str = "constrained_predictive_position_v3"
    last_estimate: TargetStateEstimate = field(init=False, repr=False)
    last_diagnostics: PredictivePositionOptimizerDiagnostics = field(
        init=False,
        repr=False,
    )
    _initial_command_angle_rad: float | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_command_angle_rad: float | None = field(
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
    _command_history: list[tuple[float, float]] = field(
        init=False,
        default_factory=list,
        repr=False,
    )

    def __post_init__(self) -> None:
        horizons = self.estimator.prediction_horizons_s
        if not horizons or horizons[0] != 0.0:
            raise ValueError("V3 prediction horizons must start at zero")
        if any(
            right <= left for left, right in zip(horizons, horizons[1:])
        ):
            raise ValueError("V3 prediction horizons must be strictly increasing")
        if not math.isfinite(self.selected_axis_fov_rad) or (
            self.selected_axis_fov_rad <= 0.0
        ):
            raise ValueError("selected_axis_fov_rad must be finite and positive")
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self.last_diagnostics = self._missing_diagnostics()

    def _missing_diagnostics(self) -> PredictivePositionOptimizerDiagnostics:
        angle = self._last_command_angle_rad or 0.0
        return PredictivePositionOptimizerDiagnostics(
            valid=False,
            selected_command_angle_rad=angle,
            raw_candidate_angle_rad=angle,
            selected_objective=0.0,
            predicted_terminal_error_fov_fraction=0.0,
            predicted_peak_error_fov_fraction=0.0,
            predicted_terminal_rate_error_normalized=0.0,
            predicted_rate_utilization=0.0,
            predicted_acceleration_utilization=0.0,
            setpoint_rate_rad_s=0.0,
            setpoint_acceleration_rad_s2=0.0,
            rate_limited=False,
            acceleration_limited=False,
            jerk_limited=False,
            evaluated_candidate_count=0,
            optimizer_active=False,
            activation_score=0.0,
            activation_rate_score=0.0,
            activation_visibility_score=0.0,
            fallback_target_angle_rad=angle,
            optimization_horizon_s=0.0,
        )

    def reset(self) -> None:
        self.estimator.reset()
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self._initial_command_angle_rad = None
        self._last_command_angle_rad = None
        self._setpoint_rate_rad_s = 0.0
        self._setpoint_acceleration_rad_s2 = 0.0
        self._command_history.clear()
        self.last_diagnostics = self._missing_diagnostics()

    @staticmethod
    def _move_toward(value: float, target: float, maximum_delta: float) -> float:
        return value + float(
            np.clip(target - value, -maximum_delta, maximum_delta)
        )

    def _shape_candidate(
        self,
        raw_target_angle_rad: float,
        dt_s: float,
    ) -> tuple[float, float, float, bool, bool, bool]:
        assert self._last_command_angle_rad is not None
        target = float(
            np.clip(
                raw_target_angle_rad,
                self.servo.min_angle_rad,
                self.servo.max_angle_rad,
            )
        )
        if dt_s <= 0.0:
            return (
                self._last_command_angle_rad,
                0.0,
                0.0,
                False,
                False,
                False,
            )
        max_rate = (
            self.config.setpoint_rate_limit_scale * self.servo.max_rate_rad_s
        )
        max_acceleration = (
            self.config.setpoint_acceleration_limit_scale
            * self.servo.max_acceleration_rad_s2
        )
        max_jerk = max_acceleration / self.config.setpoint_jerk_rise_time_s
        error = target - self._last_command_angle_rad
        stopping_speed = math.sqrt(2.0 * max_acceleration * abs(error))
        desired_rate = (
            math.copysign(min(max_rate, stopping_speed), error)
            if abs(error) > 1e-12
            else 0.0
        )
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
            command = target
            rate = 0.0
            acceleration = 0.0
        else:
            command = float(
                np.clip(
                    self._last_command_angle_rad + step,
                    self.servo.min_angle_rad,
                    self.servo.max_angle_rad,
                )
            )
        return (
            command,
            rate,
            acceleration,
            abs(desired_rate) >= max_rate - 1e-12,
            abs(raw_acceleration) > max_acceleration + 1e-12,
            abs(acceleration - desired_acceleration) > 1e-12,
        )

    def _applied_position_at(
        self,
        time_s: float,
        candidate_issue: tuple[float, float],
    ) -> float:
        assert self._initial_command_angle_rad is not None
        cutoff = time_s - self.servo.command_latency_s
        selected = self._initial_command_angle_rad
        for issue_time_s, command_angle_rad in (
            *self._command_history,
            candidate_issue,
        ):
            if issue_time_s <= cutoff + 1e-12:
                selected = command_angle_rad
            else:
                break
        physical = selected * self.servo.command_polarity
        return float(
            np.clip(
                physical,
                self.servo.min_angle_rad,
                self.servo.max_angle_rad,
            )
        )

    def _simulate_candidate(
        self,
        command_angle_rad: float,
        observation: GimbalObservation,
        estimates: tuple[TargetStateEstimate, ...],
    ) -> tuple[float, dict[str, float]]:
        assert observation.gimbal_angle_rad.valid
        assert observation.gimbal_rate_rad_s.valid
        half_fov = 0.5 * self.selected_axis_fov_rad
        angle = observation.gimbal_angle_rad.value
        rate = observation.gimbal_rate_rad_s.value
        simulation_time_s = observation.time_s
        issue = (observation.time_s, command_angle_rad)
        predicted_angles = [angle]
        predicted_rates = [rate]
        peak_rate_utilization = abs(rate) / self.servo.max_rate_rad_s
        peak_acceleration_utilization = 0.0
        for estimate in estimates[1:]:
            end_time_s = estimate.time_s
            while simulation_time_s < end_time_s - 1e-12:
                dt_s = min(
                    self.config.servo_simulation_step_s,
                    end_time_s - simulation_time_s,
                )
                applied = self._applied_position_at(simulation_time_s, issue)
                position_error = applied - angle
                if abs(position_error) <= self.servo.position_tolerance_rad:
                    desired_rate = 0.0
                else:
                    desired_rate = self.servo.position_gain_s_inv * position_error
                desired_rate = float(
                    np.clip(
                        desired_rate,
                        -self.servo.max_rate_rad_s,
                        self.servo.max_rate_rad_s,
                    )
                )
                if self.servo.rate_time_constant_s > 0.0:
                    raw_acceleration = (
                        desired_rate - rate
                    ) / self.servo.rate_time_constant_s
                else:
                    raw_acceleration = (desired_rate - rate) / dt_s
                acceleration = float(
                    np.clip(
                        raw_acceleration,
                        -self.servo.max_acceleration_rad_s2,
                        self.servo.max_acceleration_rad_s2,
                    )
                )
                rate = float(
                    np.clip(
                        rate + acceleration * dt_s,
                        -self.servo.max_rate_rad_s,
                        self.servo.max_rate_rad_s,
                    )
                )
                angle = float(
                    np.clip(
                        angle + rate * dt_s,
                        self.servo.min_angle_rad,
                        self.servo.max_angle_rad,
                    )
                )
                simulation_time_s += dt_s
                peak_rate_utilization = max(
                    peak_rate_utilization,
                    abs(rate) / self.servo.max_rate_rad_s,
                )
                peak_acceleration_utilization = max(
                    peak_acceleration_utilization,
                    abs(acceleration) / self.servo.max_acceleration_rad_s2,
                )
            predicted_angles.append(angle)
            predicted_rates.append(rate)

        objective = 0.0
        error_fractions = []
        rate_error_fractions = []
        last_index = len(estimates) - 1
        for index, (estimate, predicted_angle, predicted_rate) in enumerate(
            zip(estimates, predicted_angles, predicted_rates, strict=True)
        ):
            error = angle_delta_rad(
                estimate.body_relative_bearing_rad.value,
                predicted_angle,
            )
            error_fraction = abs(error) / half_fov
            rate_error_fraction = abs(
                estimate.body_relative_rate_rad_s.value - predicted_rate
            ) / self.servo.max_rate_rad_s
            error_fractions.append(error_fraction)
            rate_error_fractions.append(rate_error_fraction)
            if index == 0:
                continue
            terminal_multiplier = (
                self.config.terminal_tracking_weight
                if index == last_index
                else 1.0
            )
            objective += (
                self.config.tracking_weight
                * terminal_multiplier
                * error_fraction**2
            )
            objective += (
                self.config.rate_matching_weight
                * terminal_multiplier
                * rate_error_fraction**2
            )
            robust_error_fraction = (
                abs(error)
                + self.config.uncertainty_sigma
                * estimate.bearing_std_rad.value
            ) / half_fov
            visibility_excess = max(
                0.0,
                (
                    robust_error_fraction
                    - self.config.visibility_onset_fov_fraction
                )
                / (1.0 - self.config.visibility_onset_fov_fraction),
            )
            objective += self.config.visibility_weight * visibility_excess**2

        previous_normalized = self.servo.normalized_from_position(
            self._last_command_angle_rad or 0.0
        )
        candidate_normalized = self.servo.normalized_from_position(
            command_angle_rad
        )
        command_delta = candidate_normalized - previous_normalized
        objective += self.config.command_change_weight * command_delta**2
        objective += (
            self.config.command_rate_change_weight
            * (
                (
                    command_angle_rad - (self._last_command_angle_rad or 0.0)
                )
                / max(
                    self.servo.max_rate_rad_s
                    * max(observation.control_dt_s, 1e-9),
                    1e-9,
                )
            )
            ** 2
        )
        travel_excess = max(
            0.0,
            abs(candidate_normalized) - self.config.travel_margin_fraction,
        ) / max(1.0 - self.config.travel_margin_fraction, 1e-9)
        objective += self.config.travel_margin_weight * travel_excess**2
        return objective, {
            "terminal_error_fov_fraction": error_fractions[-1],
            "peak_error_fov_fraction": max(error_fractions),
            "terminal_rate_error_normalized": rate_error_fractions[-1],
            "rate_utilization": peak_rate_utilization,
            "acceleration_utilization": peak_acceleration_utilization,
        }

    def _trusted_estimates(
        self,
        estimates: tuple[TargetStateEstimate, ...],
        observation_time_s: float,
    ) -> tuple[TargetStateEstimate, ...]:
        current = estimates[0]
        current_std = max(current.bearing_std_rad.value, 1e-9)
        trusted = [current]
        for estimate in estimates[1:]:
            ratio = estimate.bearing_std_rad.value / current_std
            if ratio <= self.config.forecast_full_trust_std_ratio:
                weight = 1.0
            elif ratio >= self.config.forecast_zero_trust_std_ratio:
                weight = self.config.minimum_forecast_weight
            else:
                fraction = (
                    ratio - self.config.forecast_full_trust_std_ratio
                ) / (
                    self.config.forecast_zero_trust_std_ratio
                    - self.config.forecast_full_trust_std_ratio
                )
                weight = 1.0 - fraction * (
                    1.0 - self.config.minimum_forecast_weight
                )
            bearing = wrap_angle_rad(
                current.body_relative_bearing_rad.value
                + weight
                * angle_delta_rad(
                    estimate.body_relative_bearing_rad.value,
                    current.body_relative_bearing_rad.value,
                )
            )
            trusted.append(
                TargetStateEstimate(
                    time_s=estimate.time_s,
                    measurement_time_s=estimate.measurement_time_s,
                    body_relative_bearing_rad=MaskedScalar(bearing, True),
                    body_relative_rate_rad_s=MaskedScalar(
                        current.body_relative_rate_rad_s.value
                        + weight
                        * (
                            estimate.body_relative_rate_rad_s.value
                            - current.body_relative_rate_rad_s.value
                        ),
                        True,
                    ),
                    bearing_std_rad=estimate.bearing_std_rad,
                    rate_std_rad_s=estimate.rate_std_rad_s,
                    prediction_horizon_s=MaskedScalar(
                        estimate.time_s - observation_time_s,
                        True,
                    ),
                )
            )
        return tuple(trusted)

    @staticmethod
    def _ramp(value: float, onset: float, full: float) -> float:
        return float(np.clip((value - onset) / (full - onset), 0.0, 1.0))

    def _fallback_target(
        self,
        estimates: tuple[TargetStateEstimate, ...],
        observation: GimbalObservation,
    ) -> tuple[float, float]:
        horizons = self.estimator.prediction_horizons_s
        base_horizon_s = float(
            np.clip(
                self.config.fallback_arrival_time_scale
                * (
                    self.servo.command_latency_s
                    + self.servo.rate_time_constant_s
                ),
                horizons[0],
                horizons[-1],
            )
        )
        current = estimates[0]
        base_forecast = _interpolate_target_estimate(
            estimates,
            horizons,
            base_horizon_s,
            observation.time_s,
        )
        base_ratio = (
            base_forecast.bearing_std_rad.value
            / max(current.bearing_std_rad.value, 1e-9)
        )
        if base_ratio <= self.config.forecast_full_trust_std_ratio:
            base_weight = 1.0
        elif base_ratio >= self.config.forecast_zero_trust_std_ratio:
            base_weight = self.config.minimum_forecast_weight
        else:
            fraction = (
                base_ratio - self.config.forecast_full_trust_std_ratio
            ) / (
                self.config.forecast_zero_trust_std_ratio
                - self.config.forecast_full_trust_std_ratio
            )
            base_weight = 1.0 - fraction * (
                1.0 - self.config.minimum_forecast_weight
            )
        base_bearing = wrap_angle_rad(
            current.body_relative_bearing_rad.value
            + base_weight
            * angle_delta_rad(
                base_forecast.body_relative_bearing_rad.value,
                current.body_relative_bearing_rad.value,
            )
        )
        base_std = current.bearing_std_rad.value + base_weight * (
            base_forecast.bearing_std_rad.value
            - current.bearing_std_rad.value
        )
        assert observation.gimbal_angle_rad.valid
        half_fov = 0.5 * self.selected_axis_fov_rad
        robust_error_fraction = (
            abs(
                angle_delta_rad(
                    base_bearing,
                    observation.gimbal_angle_rad.value,
                )
            )
            + self.config.fallback_uncertainty_sigma
            * base_std
        ) / half_fov
        risk = self._ramp(
            robust_error_fraction,
            self.config.fallback_visibility_onset_fraction,
            self.config.fallback_visibility_full_fraction,
        )
        requested_horizon_s = float(
            np.clip(
                base_horizon_s
                + risk * self.config.fallback_risk_horizon_boost_s,
                horizons[0],
                horizons[-1],
            )
        )
        forecast = _interpolate_target_estimate(
            estimates,
            horizons,
            requested_horizon_s,
            observation.time_s,
        )
        ratio = (
            forecast.bearing_std_rad.value
            / max(current.bearing_std_rad.value, 1e-9)
        )
        if ratio <= self.config.forecast_full_trust_std_ratio:
            weight = 1.0
        elif ratio >= self.config.forecast_zero_trust_std_ratio:
            weight = self.config.minimum_forecast_weight
        else:
            fraction = (
                ratio - self.config.forecast_full_trust_std_ratio
            ) / (
                self.config.forecast_zero_trust_std_ratio
                - self.config.forecast_full_trust_std_ratio
            )
            weight = 1.0 - fraction * (
                1.0 - self.config.minimum_forecast_weight
            )
        bearing = wrap_angle_rad(
            current.body_relative_bearing_rad.value
            + weight
            * angle_delta_rad(
                forecast.body_relative_bearing_rad.value,
                current.body_relative_bearing_rad.value,
            )
        )
        return (
            float(
                np.clip(
                    bearing,
                    self.servo.min_angle_rad,
                    self.servo.max_angle_rad,
                )
            ),
            requested_horizon_s,
        )

    def _activation_scores(
        self,
        estimates: tuple[TargetStateEstimate, ...],
        observation: GimbalObservation,
    ) -> tuple[float, float, float]:
        assert observation.gimbal_angle_rad.valid
        half_fov = 0.5 * self.selected_axis_fov_rad
        rate_fraction = max(
            abs(estimate.body_relative_rate_rad_s.value)
            / self.servo.max_rate_rad_s
            for estimate in estimates
        )
        visibility_fraction = max(
            (
                abs(
                    angle_delta_rad(
                        estimate.body_relative_bearing_rad.value,
                        observation.gimbal_angle_rad.value,
                    )
                )
                + self.config.uncertainty_sigma
                * estimate.bearing_std_rad.value
            )
            / half_fov
            for estimate in estimates
        )
        rate_score = self._ramp(
            rate_fraction,
            self.config.activation_rate_onset_fraction,
            self.config.activation_rate_full_fraction,
        )
        visibility_score = self._ramp(
            visibility_fraction,
            self.config.activation_visibility_onset_fraction,
            self.config.activation_visibility_full_fraction,
        )
        if self.config.activation_gate_mode == "maximum":
            score = max(rate_score, visibility_score)
        elif self.config.activation_gate_mode == "minimum":
            score = min(rate_score, visibility_score)
        else:
            score = rate_score * visibility_score
        horizon_s = estimates[-1].time_s - observation.time_s
        effect_delay_s = self.servo.command_latency_s + (
            self.config.command_effect_response_fraction
            * self.servo.rate_time_constant_s
        )
        if (
            self.config.require_command_effect_within_horizon
            and (
                effect_delay_s >= horizon_s
                or self.servo.position_gain_s_inv
                < self.config.minimum_optimizer_position_gain_s_inv
            )
        ):
            score = 0.0
        return score, rate_score, visibility_score

    def _raw_candidates(
        self,
        estimates: tuple[TargetStateEstimate, ...],
        extra_anchors: tuple[float, ...] = (),
    ) -> tuple[float, ...]:
        grid = np.linspace(
            self.servo.min_angle_rad,
            self.servo.max_angle_rad,
            self.config.candidate_grid_size,
        )
        anchors = [
            self._last_command_angle_rad or 0.0,
            *(estimate.body_relative_bearing_rad.value for estimate in estimates),
            *extra_anchors,
        ]
        candidates = np.concatenate((grid, np.asarray(anchors)))
        clipped = np.clip(
            candidates,
            self.servo.min_angle_rad,
            self.servo.max_angle_rad,
        )
        return tuple(float(value) for value in np.unique(np.round(clipped, 12)))

    def act(self, observation: GimbalObservation) -> GimbalAction:
        if not observation.gimbal_angle_rad.valid:
            raise ValueError("V3 requires servo angle in its observation profile")
        if not observation.gimbal_rate_rad_s.valid:
            raise ValueError("V3 requires servo rate in its observation profile")
        if self._initial_command_angle_rad is None:
            self._initial_command_angle_rad = observation.gimbal_angle_rad.value
            self._last_command_angle_rad = observation.gimbal_angle_rad.value
        all_estimates = self.estimator.update_all(observation)
        if not all_estimates or not all(
            estimate.valid for estimate in all_estimates
        ):
            self.last_estimate = TargetStateEstimate.missing(observation.time_s)
            self._setpoint_rate_rad_s = 0.0
            self._setpoint_acceleration_rad_s2 = 0.0
            self.last_diagnostics = self._missing_diagnostics()
            assert self._last_command_angle_rad is not None
            command = self.servo.normalized_from_position(
                self._last_command_angle_rad
            )
            self._command_history.append(
                (observation.time_s, self._last_command_angle_rad)
            )
            return GimbalAction.position(command)

        trusted_all_estimates = self._trusted_estimates(
            all_estimates,
            observation.time_s,
        )
        estimates = tuple(
            estimate
            for estimate in trusted_all_estimates
            if estimate.time_s - observation.time_s
            <= self.config.maximum_optimization_horizon_s + 1e-12
        )
        if len(estimates) < 2:
            raise ValueError(
                "maximum_optimization_horizon_s excludes every future forecast"
            )
        self.last_estimate = estimates[0]
        fallback_target, _fallback_horizon_s = self._fallback_target(
            all_estimates,
            observation,
        )
        (
            activation_score,
            activation_rate_score,
            activation_visibility_score,
        ) = self._activation_scores(estimates, observation)
        evaluated = []
        if activation_score > 0.0:
            candidates = self._raw_candidates(
                estimates,
                extra_anchors=(fallback_target,),
            )
            for candidate in candidates:
                shaped_candidate = self._shape_candidate(
                    candidate,
                    observation.control_dt_s,
                )
                candidate_objective, candidate_prediction = (
                    self._simulate_candidate(
                        shaped_candidate[0],
                        observation,
                        estimates,
                    )
                )
                evaluated.append(
                    (
                        candidate_objective,
                        candidate,
                        shaped_candidate,
                        candidate_prediction,
                    )
                )
            _, optimized_raw, _optimized_shape, _optimized_prediction = min(
                evaluated,
                key=lambda item: item[0],
            )
            raw_candidate = fallback_target + activation_score * (
                optimized_raw - fallback_target
            )
        else:
            raw_candidate = fallback_target
        shaped = self._shape_candidate(
            raw_candidate,
            observation.control_dt_s,
        )
        objective, prediction = self._simulate_candidate(
            shaped[0],
            observation,
            estimates,
        )
        (
            command,
            rate,
            acceleration,
            rate_limited,
            acceleration_limited,
            jerk_limited,
        ) = shaped
        self._last_command_angle_rad = command
        self._setpoint_rate_rad_s = rate
        self._setpoint_acceleration_rad_s2 = acceleration
        self._command_history.append((observation.time_s, command))
        self.last_diagnostics = PredictivePositionOptimizerDiagnostics(
            valid=True,
            selected_command_angle_rad=command,
            raw_candidate_angle_rad=raw_candidate,
            selected_objective=objective,
            predicted_terminal_error_fov_fraction=prediction[
                "terminal_error_fov_fraction"
            ],
            predicted_peak_error_fov_fraction=prediction[
                "peak_error_fov_fraction"
            ],
            predicted_terminal_rate_error_normalized=prediction[
                "terminal_rate_error_normalized"
            ],
            predicted_rate_utilization=prediction["rate_utilization"],
            predicted_acceleration_utilization=prediction[
                "acceleration_utilization"
            ],
            setpoint_rate_rad_s=rate,
            setpoint_acceleration_rad_s2=acceleration,
            rate_limited=rate_limited,
            acceleration_limited=acceleration_limited,
            jerk_limited=jerk_limited,
            evaluated_candidate_count=max(1, len(evaluated)),
            optimizer_active=activation_score > 0.0,
            activation_score=activation_score,
            activation_rate_score=activation_rate_score,
            activation_visibility_score=activation_visibility_score,
            fallback_target_angle_rad=fallback_target,
            optimization_horizon_s=(
                estimates[-1].time_s - observation.time_s
            ),
        )
        return GimbalAction.position(
            self.servo.normalized_from_position(command)
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
