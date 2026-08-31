"""Belief-guided loss-of-view recovery for target-state controllers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import GimbalCommandMode, ServoConfig
from .controllers import TargetStatePositionController, TargetStateRateController
from .estimators import TargetStateEstimate, angle_delta_rad, wrap_angle_rad
from .types import GimbalAction, GimbalObservation, MaskedScalar


class RecoveryState(str, Enum):
    TRACK = "track"
    COAST = "coast"
    SEARCH = "search"
    REACQUIRE = "reacquire"


@dataclass(frozen=True)
class BeliefRecoveryConfig:
    dropout_grace_s: float = 0.15
    maximum_coast_s: float = 0.65
    maximum_coast_bearing_std_rad: float = math.radians(18.0)
    process_acceleration_std_rad_s2: float = math.radians(90.0)
    maximum_search_projection_s: float = 0.75
    search_feedback_gain_s_inv: float = 2.0
    search_maximum_rate_normalized: float = 0.25
    search_boundary_margin_rad: float = math.radians(2.0)
    reacquire_confirmation_frames: int = 3
    reacquire_blend_s: float = 0.25
    edge_conditioned_search: bool = False
    search_activation_edge_fraction: float = 0.65
    search_activation_minimum_outward_rate_normalized_s: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "dropout_grace_s",
            "maximum_coast_s",
            "maximum_coast_bearing_std_rad",
            "process_acceleration_std_rad_s2",
            "maximum_search_projection_s",
            "search_feedback_gain_s_inv",
            "search_boundary_margin_rad",
            "reacquire_blend_s",
            "search_activation_minimum_outward_rate_normalized_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.maximum_coast_s < self.dropout_grace_s:
            raise ValueError("maximum coast must not be shorter than dropout grace")
        if not (
            math.isfinite(self.search_maximum_rate_normalized)
            and 0.0 < self.search_maximum_rate_normalized <= 1.0
        ):
            raise ValueError("search maximum rate must be in (0, 1]")
        if self.reacquire_confirmation_frames <= 0:
            raise ValueError("reacquire confirmation frames must be positive")
        if not isinstance(self.edge_conditioned_search, bool):
            raise ValueError("edge-conditioned search flag must be boolean")
        if not (
            math.isfinite(self.search_activation_edge_fraction)
            and 0.0 <= self.search_activation_edge_fraction <= 1.0
        ):
            raise ValueError("search activation edge fraction must be in [0, 1]")


@dataclass(frozen=True)
class RecoveryTransition:
    time_s: float
    previous: RecoveryState
    current: RecoveryState
    reason: str


@dataclass
class BeliefRecoveryController:
    """Wrap a target-state adapter with explicit recovery-state behavior."""

    delegate: TargetStateRateController | TargetStatePositionController
    servo: ServoConfig
    command_mode: GimbalCommandMode
    recovery: BeliefRecoveryConfig = BeliefRecoveryConfig()
    name: str = "belief_recovery"
    state: RecoveryState = field(
        init=False, default=RecoveryState.TRACK
    )
    transitions: list[RecoveryTransition] = field(
        init=False, default_factory=list
    )
    state_trace: list[tuple[float, RecoveryState]] = field(
        init=False, default_factory=list
    )
    action_trace: list[tuple[float, GimbalAction]] = field(
        init=False, default_factory=list
    )
    edge_evidence_trace: list[tuple[float, bool]] = field(
        init=False, default_factory=list
    )
    _belief: TargetStateEstimate = field(init=False, repr=False)
    _current_estimate: TargetStateEstimate = field(init=False, repr=False)
    _last_measurement_time_s: float | None = field(
        init=False, default=None, repr=False
    )
    _last_fresh_arrival_time_s: float | None = field(
        init=False, default=None, repr=False
    )
    _last_action: GimbalAction = field(init=False, repr=False)
    _state_enter_time_s: float = field(init=False, default=0.0, repr=False)
    _reacquire_confirmation_count: int = field(
        init=False, default=0, repr=False
    )
    _reacquire_anchor_action: GimbalAction = field(init=False, repr=False)
    _reacquire_blend_start_s: float | None = field(
        init=False, default=None, repr=False
    )
    _last_valid_image_error_normalized: float | None = field(
        init=False, default=None, repr=False
    )
    _last_valid_image_error_time_s: float | None = field(
        init=False, default=None, repr=False
    )
    _last_valid_image_error_rate_normalized_s: float = field(
        init=False, default=0.0, repr=False
    )

    def __post_init__(self) -> None:
        available_margin = min(
            -self.servo.min_angle_rad,
            self.servo.max_angle_rad,
        )
        if self.recovery.search_boundary_margin_rad >= available_margin:
            raise ValueError("search boundary margin must fit inside travel")
        self.reset()

    @property
    def last_estimate(self) -> TargetStateEstimate:
        return self._current_estimate

    @property
    def edge_search_evidence_supported(self) -> bool:
        if not self.recovery.edge_conditioned_search:
            return True
        error = self._last_valid_image_error_normalized
        if error is None:
            return False
        outward_rate = (
            math.copysign(1.0, error)
            * self._last_valid_image_error_rate_normalized_s
            if abs(error) > 1e-9
            else 0.0
        )
        return (
            abs(error) >= self.recovery.search_activation_edge_fraction
            and outward_rate
            > self.recovery.search_activation_minimum_outward_rate_normalized_s
        )

    def _zero_action(self) -> GimbalAction:
        if self.command_mode is GimbalCommandMode.RATE:
            return GimbalAction.rate(0.0)
        return GimbalAction.position(0.0)

    def reset(self) -> None:
        self.delegate.reset()
        self.state = RecoveryState.TRACK
        self.transitions.clear()
        self.state_trace.clear()
        self.action_trace.clear()
        self.edge_evidence_trace.clear()
        self._belief = TargetStateEstimate.missing(0.0)
        self._current_estimate = TargetStateEstimate.missing(0.0)
        self._last_measurement_time_s = None
        self._last_fresh_arrival_time_s = None
        self._last_action = self._zero_action()
        self._state_enter_time_s = 0.0
        self._reacquire_confirmation_count = 0
        self._reacquire_anchor_action = self._zero_action()
        self._reacquire_blend_start_s = None
        self._last_valid_image_error_normalized = None
        self._last_valid_image_error_time_s = None
        self._last_valid_image_error_rate_normalized_s = 0.0

    def _update_edge_evidence(
        self,
        observation: GimbalObservation,
        measurement_time_s: float,
    ) -> None:
        error = observation.image_error_normalized.value
        previous_error = self._last_valid_image_error_normalized
        previous_time = self._last_valid_image_error_time_s
        if previous_error is not None and previous_time is not None:
            elapsed_s = measurement_time_s - previous_time
            if elapsed_s > 1e-9:
                self._last_valid_image_error_rate_normalized_s = (
                    error - previous_error
                ) / elapsed_s
        self._last_valid_image_error_normalized = error
        self._last_valid_image_error_time_s = measurement_time_s

    def _transition(
        self,
        state: RecoveryState,
        time_s: float,
        reason: str,
    ) -> None:
        if state is self.state:
            return
        self.transitions.append(
            RecoveryTransition(time_s, self.state, state, reason)
        )
        self.state = state
        self._state_enter_time_s = time_s
        if state is RecoveryState.REACQUIRE:
            self._reacquire_confirmation_count = 1
            self._reacquire_anchor_action = self._last_action
            self._reacquire_blend_start_s = None

    def _project_belief(
        self,
        time_s: float,
        *,
        propagation_limit_s: float | None = None,
    ) -> TargetStateEstimate:
        belief = self._belief
        if not belief.valid:
            return TargetStateEstimate.missing(time_s)
        elapsed_s = max(0.0, time_s - belief.time_s)
        propagation_s = (
            min(elapsed_s, propagation_limit_s)
            if propagation_limit_s is not None
            else elapsed_s
        )
        bearing = wrap_angle_rad(
            belief.body_relative_bearing_rad.value
            + propagation_s * belief.body_relative_rate_rad_s.value
        )
        acceleration_std = self.recovery.process_acceleration_std_rad_s2
        bearing_std = math.hypot(
            belief.bearing_std_rad.value,
            0.5 * acceleration_std * elapsed_s**2,
        )
        rate_std = math.hypot(
            belief.rate_std_rad_s.value,
            acceleration_std * elapsed_s,
        )
        measurement_time = belief.measurement_time_s
        horizon_s = (
            max(0.0, time_s - measurement_time.value)
            if measurement_time.valid
            else 0.0
        )
        return TargetStateEstimate(
            time_s=time_s,
            measurement_time_s=measurement_time,
            body_relative_bearing_rad=MaskedScalar(bearing, True),
            body_relative_rate_rad_s=belief.body_relative_rate_rad_s,
            bearing_std_rad=MaskedScalar(bearing_std, True),
            rate_std_rad_s=MaskedScalar(rate_std, True),
            prediction_horizon_s=MaskedScalar(horizon_s, True),
        )

    def _belief_action(
        self,
        observation: GimbalObservation,
        estimate: TargetStateEstimate,
        *,
        searching: bool,
    ) -> GimbalAction:
        angle = observation.gimbal_angle_rad
        if not estimate.valid or not angle.valid:
            return self._last_action
        margin = self.recovery.search_boundary_margin_rad if searching else 0.0
        lower = self.servo.min_angle_rad + margin
        upper = self.servo.max_angle_rad - margin
        raw_target = estimate.body_relative_bearing_rad.value
        desired_angle = float(np.clip(raw_target, lower, upper))
        if self.command_mode is GimbalCommandMode.POSITION:
            return GimbalAction.position(
                self.servo.normalized_from_position(desired_angle)
            )
        target_rate = estimate.body_relative_rate_rad_s.value
        if searching and not lower < raw_target < upper:
            target_rate = 0.0
        desired_rate = target_rate + (
            self.recovery.search_feedback_gain_s_inv
            * angle_delta_rad(desired_angle, angle.value)
        )
        limit = (
            self.recovery.search_maximum_rate_normalized
            if searching
            else 1.0
        )
        normalized = float(
            np.clip(desired_rate / self.servo.max_rate_rad_s, -limit, limit)
        )
        return GimbalAction.rate(normalized)

    def _blend_action(
        self,
        nominal_action: GimbalAction,
        time_s: float,
    ) -> GimbalAction:
        if self._reacquire_blend_start_s is None:
            return self._reacquire_anchor_action
        duration = self.recovery.reacquire_blend_s
        alpha = (
            1.0
            if duration == 0.0
            else float(
                np.clip(
                    (time_s - self._reacquire_blend_start_s) / duration,
                    0.0,
                    1.0,
                )
            )
        )
        value = (
            (1.0 - alpha) * self._reacquire_anchor_action.command_normalized
            + alpha * nominal_action.command_normalized
        )
        if alpha >= 1.0:
            self._transition(RecoveryState.TRACK, time_s, "blend complete")
        if self.command_mode is GimbalCommandMode.RATE:
            return GimbalAction.rate(float(value))
        return GimbalAction.position(float(value))

    def act(self, observation: GimbalObservation) -> GimbalAction:
        previous_measurement_time = self._last_measurement_time_s
        nominal_action = self.delegate.act(observation)
        raw_estimate = self.delegate.last_estimate
        measurement_time = raw_estimate.measurement_time_s
        if measurement_time.valid and (
            self._last_measurement_time_s is None
            or measurement_time.value > self._last_measurement_time_s
        ):
            self._last_measurement_time_s = measurement_time.value
        fresh_detection = (
            observation.frame_updated
            and observation.detection_valid
            and raw_estimate.valid
            and measurement_time.valid
            and (
                previous_measurement_time is None
                or measurement_time.value > previous_measurement_time + 1e-9
            )
        )
        if fresh_detection:
            self._belief = raw_estimate
            self._last_fresh_arrival_time_s = observation.time_s
            self._update_edge_evidence(
                observation,
                measurement_time.value,
            )
        projected = self._project_belief(observation.time_s)
        search_projection = self._project_belief(
            observation.time_s,
            propagation_limit_s=self.recovery.maximum_search_projection_s,
        )
        self._current_estimate = raw_estimate if raw_estimate.valid else projected
        gap_s = (
            observation.time_s - self._last_fresh_arrival_time_s
            if self._last_fresh_arrival_time_s is not None
            else 0.0
        )

        if fresh_detection:
            if self.state in {RecoveryState.COAST, RecoveryState.SEARCH}:
                self._transition(
                    RecoveryState.REACQUIRE,
                    observation.time_s,
                    "fresh detection",
                )
            elif self.state is RecoveryState.REACQUIRE:
                self._reacquire_confirmation_count += 1
        elif observation.frame_updated and not observation.detection_valid:
            if self.state is RecoveryState.REACQUIRE:
                self._reacquire_confirmation_count = 0
                self._reacquire_blend_start_s = None

        if (
            self.state in {RecoveryState.TRACK, RecoveryState.REACQUIRE}
            and self._last_fresh_arrival_time_s is not None
            and gap_s > self.recovery.dropout_grace_s
        ):
            self._transition(
                RecoveryState.COAST,
                observation.time_s,
                "measurement gap",
            )
        coast_limit_reached = self.state is RecoveryState.COAST and (
            gap_s > self.recovery.maximum_coast_s
            or (
                projected.valid
                and projected.bearing_std_rad.value
                > self.recovery.maximum_coast_bearing_std_rad
            )
        )
        if coast_limit_reached and self.edge_search_evidence_supported:
            self._transition(
                RecoveryState.SEARCH,
                observation.time_s,
                (
                    "coast limit with edge evidence"
                    if self.recovery.edge_conditioned_search
                    else "coast limit"
                ),
            )

        if self.state is RecoveryState.TRACK:
            action = nominal_action
        elif self.state is RecoveryState.COAST:
            action = (
                nominal_action
                if (
                    raw_estimate.valid
                    or (
                        self.recovery.edge_conditioned_search
                        and not self.edge_search_evidence_supported
                    )
                )
                else self._belief_action(
                    observation, projected, searching=False
                )
            )
        elif self.state is RecoveryState.SEARCH:
            action = self._belief_action(
                observation, search_projection, searching=True
            )
        else:
            if (
                self._reacquire_confirmation_count
                >= self.recovery.reacquire_confirmation_frames
                and self._reacquire_blend_start_s is None
            ):
                self._reacquire_blend_start_s = observation.time_s
            action = self._blend_action(nominal_action, observation.time_s)

        self._last_action = action
        self.state_trace.append((observation.time_s, self.state))
        self.action_trace.append((observation.time_s, action))
        self.edge_evidence_trace.append(
            (
                observation.time_s,
                self.recovery.edge_conditioned_search
                and self.edge_search_evidence_supported,
            )
        )
        return action
