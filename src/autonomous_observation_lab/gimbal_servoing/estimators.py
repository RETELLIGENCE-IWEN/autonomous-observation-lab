"""Deployable target-state estimators for predictive visual servoing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .types import GimbalObservation, MaskedScalar


def angle_delta_rad(left_rad: float, right_rad: float) -> float:
    return math.atan2(
        math.sin(left_rad - right_rad),
        math.cos(left_rad - right_rad),
    )


def wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class TargetStateEstimate:
    """Timestamped body-relative target state with explicit uncertainty masks."""

    time_s: float
    measurement_time_s: MaskedScalar
    body_relative_bearing_rad: MaskedScalar
    body_relative_rate_rad_s: MaskedScalar
    bearing_std_rad: MaskedScalar
    rate_std_rad_s: MaskedScalar
    prediction_horizon_s: MaskedScalar

    @property
    def valid(self) -> bool:
        return (
            self.body_relative_bearing_rad.valid
            and self.body_relative_rate_rad_s.valid
        )

    @classmethod
    def missing(cls, time_s: float) -> "TargetStateEstimate":
        missing = MaskedScalar.missing()
        return cls(
            time_s=time_s,
            measurement_time_s=missing,
            body_relative_bearing_rad=missing,
            body_relative_rate_rad_s=missing,
            bearing_std_rad=missing,
            rate_std_rad_s=missing,
            prediction_horizon_s=missing,
        )


class TargetStateEstimator(Protocol):
    name: str
    last_estimate: TargetStateEstimate

    def reset(self) -> None: ...

    def update(self, observation: GimbalObservation) -> TargetStateEstimate: ...


class MultiHorizonTargetStateEstimator(Protocol):
    """Estimator exposing simultaneous causal forecasts from one update."""

    name: str
    prediction_horizons_s: tuple[float, ...]
    last_estimate: TargetStateEstimate
    last_estimates: tuple[TargetStateEstimate, ...]

    def reset(self) -> None: ...

    def update_all(
        self, observation: GimbalObservation
    ) -> tuple[TargetStateEstimate, ...]: ...


@dataclass(frozen=True)
class ConstantVelocityEstimatorConfig:
    selected_axis_fov_rad: float
    center_noise_std_normalized: float = 0.0
    velocity_filter_coefficient: float = 0.40
    uncertainty_filter_coefficient: float = 0.20
    max_prediction_horizon_s: float = 0.30
    history_horizon_s: float = 1.0
    minimum_bearing_std_rad: float = math.radians(0.05)
    initial_rate_std_rad_s: float = math.radians(30.0)
    process_acceleration_std_rad_s2: float = math.radians(80.0)
    body_rate_compensation: bool = False

    def __post_init__(self) -> None:
        if self.selected_axis_fov_rad <= 0.0:
            raise ValueError("selected_axis_fov_rad must be positive")
        if self.center_noise_std_normalized < 0.0:
            raise ValueError("center noise must be non-negative")
        for name in (
            "velocity_filter_coefficient",
            "uncertainty_filter_coefficient",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.max_prediction_horizon_s < 0.0:
            raise ValueError("maximum prediction horizon must be non-negative")
        if self.history_horizon_s <= self.max_prediction_horizon_s:
            raise ValueError("history horizon must exceed prediction horizon")
        if self.minimum_bearing_std_rad < 0.0:
            raise ValueError("minimum bearing uncertainty must be non-negative")
        if self.initial_rate_std_rad_s < 0.0:
            raise ValueError("initial rate uncertainty must be non-negative")
        if self.process_acceleration_std_rad_s2 < 0.0:
            raise ValueError("process acceleration uncertainty must be non-negative")
        if not isinstance(self.body_rate_compensation, bool):
            raise ValueError("body_rate_compensation must be boolean")


@dataclass
class ConstantVelocityTargetEstimator:
    """Estimate body-relative target bearing from delayed bbox observations.

    The estimator reconstructs target bearing at detector capture time by
    adding the bbox angular error to an interpolated historical gimbal angle.
    It estimates velocity from successive reconstructed samples, then projects
    the state causally to the current control time.
    """

    config: ConstantVelocityEstimatorConfig
    name: str = "constant_velocity"
    _gimbal_history: list[tuple[float, float]] = field(
        init=False, default_factory=list, repr=False
    )
    _body_rate_history: list[tuple[float, float]] = field(
        init=False, default_factory=list, repr=False
    )
    _last_target_sample: tuple[float, float] | None = field(
        init=False, default=None, repr=False
    )
    _target_rate_rad_s: float = field(init=False, default=0.0, repr=False)
    _rate_variance_rad_s2: float = field(init=False, default=0.0, repr=False)
    last_estimate: TargetStateEstimate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._gimbal_history.clear()
        self._body_rate_history.clear()
        self._last_target_sample = None
        self._target_rate_rad_s = 0.0
        self._rate_variance_rad_s2 = self.config.initial_rate_std_rad_s**2
        self.last_estimate = TargetStateEstimate.missing(0.0)

    def _remember_gimbal(self, observation: GimbalObservation) -> None:
        angle = observation.gimbal_angle_rad
        if not angle.valid:
            return
        time_s = observation.time_s
        if not self._gimbal_history or time_s > self._gimbal_history[-1][0]:
            self._gimbal_history.append((time_s, angle.value))
        cutoff = time_s - self.config.history_horizon_s
        while (
            len(self._gimbal_history) > 2
            and self._gimbal_history[1][0] < cutoff
        ):
            self._gimbal_history.pop(0)
        body_rate = observation.body_rate_rad_s
        if body_rate.valid and (
            not self._body_rate_history
            or time_s > self._body_rate_history[-1][0]
        ):
            self._body_rate_history.append((time_s, body_rate.value))
        while (
            len(self._body_rate_history) > 2
            and self._body_rate_history[1][0] < cutoff
        ):
            self._body_rate_history.pop(0)

    def _gimbal_angle_at(self, time_s: float) -> float | None:
        history = self._gimbal_history
        if not history or time_s < history[0][0] - 1e-9:
            return None
        if time_s >= history[-1][0]:
            return history[-1][1]
        for (left_time, left_angle), (right_time, right_angle) in zip(
            history, history[1:]
        ):
            if left_time <= time_s <= right_time:
                fraction = (time_s - left_time) / (right_time - left_time)
                return left_angle + fraction * angle_delta_rad(
                    right_angle, left_angle
                )
        return None

    def _body_rate_at(self, time_s: float) -> float | None:
        history = self._body_rate_history
        if not history or time_s < history[0][0] - 1e-9:
            return None
        if time_s >= history[-1][0]:
            return history[-1][1]
        for (left_time, left_rate), (right_time, right_rate) in zip(
            history, history[1:]
        ):
            if left_time <= time_s <= right_time:
                fraction = (time_s - left_time) / (right_time - left_time)
                return left_rate + fraction * (right_rate - left_rate)
        return None

    def _integrated_body_rotation(
        self,
        start_time_s: float,
        end_time_s: float,
    ) -> float | None:
        """Integrate causal IMU yaw-rate history with a constant future hold."""

        if end_time_s < start_time_s:
            raise ValueError("body-rate integration interval must be ordered")
        start_rate = self._body_rate_at(start_time_s)
        end_rate = self._body_rate_at(end_time_s)
        if start_rate is None or end_rate is None:
            return None
        knots = [(start_time_s, start_rate)]
        knots.extend(
            (time_s, rate_rad_s)
            for time_s, rate_rad_s in self._body_rate_history
            if start_time_s < time_s < end_time_s
        )
        knots.append((end_time_s, end_rate))
        return sum(
            0.5 * (left_rate + right_rate) * (right_time - left_time)
            for (left_time, left_rate), (right_time, right_rate) in zip(
                knots,
                knots[1:],
            )
        )

    def _update_measurement(self, observation: GimbalObservation) -> None:
        if not observation.frame_updated or not observation.detection_valid:
            return
        if not observation.measurement_age_s.valid:
            return
        capture_time_s = observation.time_s - observation.measurement_age_s.value
        capture_gimbal_angle = self._gimbal_angle_at(capture_time_s)
        if capture_gimbal_angle is None:
            return
        image_error_rad = (
            observation.image_error_normalized.value
            * 0.5
            * self.config.selected_axis_fov_rad
        )
        target_bearing_rad = wrap_angle_rad(
            capture_gimbal_angle + image_error_rad
        )
        previous = self._last_target_sample
        if previous is not None and capture_time_s > previous[0] + 1e-9:
            sample_period_s = capture_time_s - previous[0]
            sample_rate = angle_delta_rad(
                target_bearing_rad, previous[1]
            ) / sample_period_s
            if self.config.body_rate_compensation:
                body_rotation = self._integrated_body_rotation(
                    previous[0],
                    capture_time_s,
                )
                if body_rotation is None:
                    self._last_target_sample = (
                        capture_time_s,
                        target_bearing_rad,
                    )
                    return
                sample_rate += body_rotation / sample_period_s
            innovation = sample_rate - self._target_rate_rad_s
            rate_alpha = self.config.velocity_filter_coefficient
            self._target_rate_rad_s += rate_alpha * innovation
            variance_alpha = self.config.uncertainty_filter_coefficient
            self._rate_variance_rad_s2 = (
                (1.0 - variance_alpha) * self._rate_variance_rad_s2
                + variance_alpha * innovation**2
            )
        self._last_target_sample = (capture_time_s, target_bearing_rad)

    def _project(self, time_s: float) -> TargetStateEstimate:
        sample = self._last_target_sample
        if sample is None:
            return TargetStateEstimate.missing(time_s)
        sample_time_s, sampled_bearing_rad = sample
        horizon_s = max(0.0, time_s - sample_time_s)
        if horizon_s > self.config.max_prediction_horizon_s + 1e-9:
            return TargetStateEstimate.missing(time_s)

        rate_std_rad_s = math.sqrt(max(0.0, self._rate_variance_rad_s2))
        rate_std_rad_s = math.hypot(
            rate_std_rad_s,
            self.config.process_acceleration_std_rad_s2 * horizon_s,
        )
        measurement_std_rad = max(
            self.config.minimum_bearing_std_rad,
            self.config.center_noise_std_normalized
            * 0.5
            * self.config.selected_axis_fov_rad,
        )
        bearing_std_rad = math.sqrt(
            measurement_std_rad**2
            + (horizon_s * rate_std_rad_s) ** 2
            + (
                0.5
                * self.config.process_acceleration_std_rad_s2
                * horizon_s**2
            )
            ** 2
        )
        projected_bearing_rad = wrap_angle_rad(
            sampled_bearing_rad + self._target_rate_rad_s * horizon_s
        )
        projected_rate_rad_s = self._target_rate_rad_s
        if self.config.body_rate_compensation:
            body_rotation = self._integrated_body_rotation(
                sample_time_s,
                time_s,
            )
            body_rate = self._body_rate_at(time_s)
            if body_rotation is None or body_rate is None:
                return TargetStateEstimate.missing(time_s)
            projected_bearing_rad = wrap_angle_rad(
                sampled_bearing_rad
                + self._target_rate_rad_s * horizon_s
                - body_rotation
            )
            projected_rate_rad_s = self._target_rate_rad_s - body_rate
        return TargetStateEstimate(
            time_s=time_s,
            measurement_time_s=MaskedScalar(sample_time_s, True),
            body_relative_bearing_rad=MaskedScalar(
                projected_bearing_rad,
                True,
            ),
            body_relative_rate_rad_s=MaskedScalar(
                projected_rate_rad_s, True
            ),
            bearing_std_rad=MaskedScalar(bearing_std_rad, True),
            rate_std_rad_s=MaskedScalar(rate_std_rad_s, True),
            prediction_horizon_s=MaskedScalar(horizon_s, True),
        )

    def update(self, observation: GimbalObservation) -> TargetStateEstimate:
        self._remember_gimbal(observation)
        self._update_measurement(observation)
        self.last_estimate = self._project(observation.time_s)
        return self.last_estimate


@dataclass
class MultiHorizonConstantVelocityTargetEstimator(
    ConstantVelocityTargetEstimator
):
    """Expose one causal constant-velocity state at several future horizons.

    This is the classical counterpart to a learned multi-horizon estimator.
    Measurement reconstruction and state filtering happen exactly once per
    control update; every forecast is then projected from that same filtered
    state. This lets classical and learned estimators use the identical
    downstream command adapter without duplicating observations or state.
    """

    prediction_horizons_s: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    name: str = "multi_horizon_constant_velocity"
    last_estimates: tuple[TargetStateEstimate, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.prediction_horizons_s:
            raise ValueError("prediction horizons must be non-empty")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in self.prediction_horizons_s
        ):
            raise ValueError(
                "prediction horizons must be finite and non-negative"
            )
        if any(
            right <= left
            for left, right in zip(
                self.prediction_horizons_s,
                self.prediction_horizons_s[1:],
            )
        ):
            raise ValueError("prediction horizons must be strictly increasing")
        super().__post_init__()

    def reset(self) -> None:
        super().reset()
        self.last_estimates = tuple(
            TargetStateEstimate.missing(0.0)
            for _ in self.prediction_horizons_s
        )

    def update_all(
        self,
        observation: GimbalObservation,
    ) -> tuple[TargetStateEstimate, ...]:
        self._remember_gimbal(observation)
        self._update_measurement(observation)
        self.last_estimates = tuple(
            self._project(observation.time_s + horizon_s)
            for horizon_s in self.prediction_horizons_s
        )
        self.last_estimate = self.last_estimates[0]
        return self.last_estimates

    def update(self, observation: GimbalObservation) -> TargetStateEstimate:
        self.update_all(observation)
        return self.last_estimate
