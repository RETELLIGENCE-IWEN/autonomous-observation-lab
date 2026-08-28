import heapq
import math
from dataclasses import dataclass

import numpy as np

from .config import GimbalCommandMode, GimbalServoingConfig, ObservationProfile
from .disturbances import AngularMotion, StaticAngularMotion
from .types import (
    GimbalAction,
    GimbalDiagnostics,
    GimbalObservation,
    GimbalStepResult,
    MaskedScalar,
)


_EPSILON = 1e-12


def wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class _Measurement:
    frame_index: int
    capture_time_s: float
    valid: bool
    image_error_normalized: float
    bbox_width_fraction: float
    bbox_height_fraction: float
    confidence: float


class GimbalServoEnv:
    """Deterministic-by-seed feature-level one-axis gimbal environment.

    Coordinate convention:

    - body bearing is expressed in a world-fixed one-axis angular coordinate;
    - gimbal angle is relative to the body and zero points body-forward;
    - camera optical-axis bearing is ``body bearing + gimbal angle``;
    - positive target error is reduced by a positive logical gimbal rate.

    The controller receives only :class:`GimbalObservation`. Simulator truth is
    returned separately as :class:`GimbalDiagnostics` for tests and evaluation.
    """

    def __init__(
        self,
        config: GimbalServoingConfig | None = None,
        *,
        body_motion: AngularMotion | None = None,
        target_motion: AngularMotion | None = None,
    ):
        self.config = config or GimbalServoingConfig()
        self.body_motion = body_motion or StaticAngularMotion()
        self.target_motion = target_motion or StaticAngularMotion()

        self._time_s = 0.0
        self._gimbal_angle_rad = 0.0
        self._gimbal_rate_rad_s = 0.0
        self._requested_command_normalized = 0.0
        self._requested_rate_rad_s: float | None = 0.0
        self._requested_position_rad: float | None = None
        self._applied_rate_command_rad_s: float | None = 0.0
        self._applied_position_command_rad: float | None = None
        self._inner_rate_target_rad_s = 0.0
        self._previous_action_normalized = 0.0
        self._previous_cost_action_normalized = 0.0
        self._last_control_dt_s = 0.0
        self._next_capture_time_s = 0.0
        self._frame_index = 0
        self._latest_measurement: _Measurement | None = None
        self._latest_released_frame_index = -1
        self._command_sequence = 0
        self._measurement_sequence = 0
        self._command_queue: list[
            tuple[float, int, GimbalCommandMode, float]
        ] = []
        self._measurement_queue: list[tuple[float, int, _Measurement]] = []
        self._done = False
        self._rate_saturated = False
        self._angle_saturated = False
        self._rng_jitter = np.random.default_rng()
        self._rng_dropout = np.random.default_rng()
        self._rng_center = np.random.default_rng()
        self._rng_size = np.random.default_rng()
        self._rng_confidence = np.random.default_rng()

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def diagnostics(self) -> GimbalDiagnostics:
        return self._diagnostics()

    def reset(self, seed: int) -> tuple[GimbalObservation, GimbalDiagnostics]:
        streams = np.random.SeedSequence(seed).spawn(5)
        self._rng_jitter = np.random.default_rng(streams[0])
        self._rng_dropout = np.random.default_rng(streams[1])
        self._rng_center = np.random.default_rng(streams[2])
        self._rng_size = np.random.default_rng(streams[3])
        self._rng_confidence = np.random.default_rng(streams[4])

        scenario = self.config.scenario
        self._time_s = 0.0
        self._gimbal_angle_rad = scenario.initial_gimbal_angle_rad
        self._gimbal_rate_rad_s = scenario.initial_gimbal_rate_rad_s
        self._requested_command_normalized = 0.0
        if self.config.command_mode is GimbalCommandMode.RATE:
            self._requested_rate_rad_s = 0.0
            self._requested_position_rad = None
            self._applied_rate_command_rad_s = 0.0
            self._applied_position_command_rad = None
        else:
            self._requested_rate_rad_s = None
            self._requested_position_rad = self._gimbal_angle_rad
            self._applied_rate_command_rad_s = None
            self._applied_position_command_rad = self._gimbal_angle_rad
        self._inner_rate_target_rad_s = 0.0
        self._previous_action_normalized = 0.0
        self._previous_cost_action_normalized = 0.0
        self._last_control_dt_s = 0.0
        self._next_capture_time_s = 0.0
        self._frame_index = 0
        self._latest_measurement = None
        self._latest_released_frame_index = -1
        self._command_sequence = 0
        self._measurement_sequence = 0
        self._command_queue = []
        self._measurement_queue = []
        self._done = False
        self._rate_saturated = False
        self._angle_saturated = False

        self._capture_due_measurements()
        self._release_due_measurements()
        observation = self._make_observation(frame_updated=self._latest_measurement is not None)
        return observation, self._diagnostics()

    def step(self, action: GimbalAction) -> GimbalStepResult:
        if self._done:
            raise RuntimeError("step called after episode completion")
        if action.mode is not self.config.command_mode:
            raise ValueError(
                f"environment expects {self.config.command_mode.value}, "
                f"received {action.mode.value}"
            )

        servo = self.config.servo
        command_normalized = action.command_normalized
        self._requested_command_normalized = command_normalized
        if action.mode is GimbalCommandMode.RATE:
            command_physical = command_normalized * servo.max_rate_rad_s
            self._requested_rate_rad_s = command_physical
            self._requested_position_rad = None
        else:
            command_physical = servo.position_from_normalized(command_normalized)
            self._requested_rate_rad_s = None
            self._requested_position_rad = command_physical
        self._command_sequence += 1
        heapq.heappush(
            self._command_queue,
            (
                self._time_s + servo.command_latency_s,
                self._command_sequence,
                action.mode,
                command_physical,
            ),
        )
        self._rate_saturated = False
        self._angle_saturated = False
        previous_frame = self._latest_released_frame_index

        step_start_time_s = self._time_s
        end_time_s = min(
            self._time_s + self.config.timing.control_period_s,
            self.config.timing.episode_duration_s,
        )
        self._advance_to(end_time_s)
        self._last_control_dt_s = self._time_s - step_start_time_s
        self._previous_action_normalized = command_normalized

        frame_updated = self._latest_released_frame_index > previous_frame
        observation = self._make_observation(frame_updated=frame_updated)
        reward = -self._step_cost(
            command_normalized, self._last_control_dt_s
        )
        self._previous_cost_action_normalized = command_normalized
        truncated = self._time_s >= self.config.timing.episode_duration_s - _EPSILON
        self._done = truncated
        return GimbalStepResult(
            observation=observation,
            reward=reward,
            terminated=False,
            truncated=truncated,
            diagnostics=self._diagnostics(),
        )

    def _advance_to(self, end_time_s: float) -> None:
        integration_period = self.config.timing.integration_period_s
        while self._time_s < end_time_s - _EPSILON:
            self._apply_due_commands()
            self._capture_due_measurements()
            self._release_due_measurements()

            candidates = [end_time_s, self._time_s + integration_period]
            if self._command_queue:
                candidates.append(self._command_queue[0][0])
            candidates.append(self._next_capture_time_s)
            future = [candidate for candidate in candidates if candidate > self._time_s + _EPSILON]
            next_time_s = min(future)
            self._integrate_servo(next_time_s - self._time_s)
            self._time_s = next_time_s

        self._time_s = end_time_s
        self._apply_due_commands()
        self._capture_due_measurements()
        self._release_due_measurements()

    def _apply_due_commands(self) -> None:
        while self._command_queue and self._command_queue[0][0] <= self._time_s + _EPSILON:
            _, _, mode, requested_command = heapq.heappop(self._command_queue)
            if mode is GimbalCommandMode.RATE:
                self._applied_rate_command_rad_s = self._rate_command_adapter(
                    requested_command
                )
                self._applied_position_command_rad = None
            else:
                self._applied_position_command_rad = self._position_command_adapter(
                    requested_command
                )
                self._applied_rate_command_rad_s = None

    def _rate_command_adapter(self, requested_rate_rad_s: float) -> float:
        servo = self.config.servo
        command = requested_rate_rad_s * servo.command_polarity
        if abs(command) < servo.rate_deadband_rad_s:
            command = 0.0
        if servo.rate_quantization_rad_s > 0.0:
            quantum = servo.rate_quantization_rad_s
            command = round(command / quantum) * quantum
        clipped = float(np.clip(command, -servo.max_rate_rad_s, servo.max_rate_rad_s))
        at_rate_limit = abs(clipped) >= servo.max_rate_rad_s - _EPSILON
        self._rate_saturated = (
            self._rate_saturated
            or at_rate_limit
            or not math.isclose(command, clipped)
        )
        return clipped

    def _position_command_adapter(self, requested_position_rad: float) -> float:
        servo = self.config.servo
        command = requested_position_rad * servo.command_polarity
        if servo.position_quantization_rad > 0.0:
            quantum = servo.position_quantization_rad
            command = round(command / quantum) * quantum
        return float(np.clip(command, servo.min_angle_rad, servo.max_angle_rad))

    def _integrate_servo(self, dt_s: float) -> None:
        servo = self.config.servo
        if self.config.command_mode is GimbalCommandMode.RATE:
            assert self._applied_rate_command_rad_s is not None
            desired_rate = self._applied_rate_command_rad_s
        else:
            assert self._applied_position_command_rad is not None
            position_error = (
                self._applied_position_command_rad - self._gimbal_angle_rad
            )
            if abs(position_error) <= servo.position_tolerance_rad:
                desired_rate = 0.0
            else:
                desired_rate = servo.position_gain_s_inv * position_error
            desired_rate = float(
                np.clip(desired_rate, -servo.max_rate_rad_s, servo.max_rate_rad_s)
            )
        self._inner_rate_target_rad_s = desired_rate
        if servo.rate_time_constant_s > 0.0:
            acceleration = (
                desired_rate - self._gimbal_rate_rad_s
            ) / servo.rate_time_constant_s
        else:
            acceleration = (desired_rate - self._gimbal_rate_rad_s) / dt_s
        limited_acceleration = float(
            np.clip(
                acceleration,
                -servo.max_acceleration_rad_s2,
                servo.max_acceleration_rad_s2,
            )
        )
        rate = self._gimbal_rate_rad_s + limited_acceleration * dt_s
        clipped_rate = float(np.clip(rate, -servo.max_rate_rad_s, servo.max_rate_rad_s))
        self._rate_saturated = self._rate_saturated or not math.isclose(rate, clipped_rate)

        angle = self._gimbal_angle_rad + clipped_rate * dt_s
        clipped_angle = float(np.clip(angle, servo.min_angle_rad, servo.max_angle_rad))
        if not math.isclose(angle, clipped_angle):
            self._angle_saturated = True
            pushing_lower = clipped_angle == servo.min_angle_rad and clipped_rate < 0.0
            pushing_upper = clipped_angle == servo.max_angle_rad and clipped_rate > 0.0
            if pushing_lower or pushing_upper:
                clipped_rate = 0.0
        self._gimbal_angle_rad = clipped_angle
        self._gimbal_rate_rad_s = clipped_rate

    def _capture_due_measurements(self) -> None:
        camera = self.config.camera
        while self._next_capture_time_s <= self._time_s + _EPSILON:
            capture_time_s = self._next_capture_time_s
            measurement = self._capture(capture_time_s)
            jitter = 0.0
            if camera.detection_latency_jitter_s > 0.0:
                jitter = float(
                    self._rng_jitter.uniform(
                        -camera.detection_latency_jitter_s,
                        camera.detection_latency_jitter_s,
                    )
                )
            available_time_s = max(
                capture_time_s,
                capture_time_s + camera.detection_latency_s + jitter,
            )
            self._measurement_sequence += 1
            heapq.heappush(
                self._measurement_queue,
                (available_time_s, self._measurement_sequence, measurement),
            )
            self._frame_index += 1
            self._next_capture_time_s = self._frame_index * camera.frame_period_s

    def _capture(self, capture_time_s: float) -> _Measurement:
        camera = self.config.camera
        scenario = self.config.scenario
        target_bearing, _ = self.target_motion.state_at(capture_time_s)
        body_bearing, _ = self.body_motion.state_at(capture_time_s)
        optical_axis = body_bearing + self._gimbal_angle_rad
        error_rad = wrap_angle_rad(target_bearing - optical_axis)
        half_fov = 0.5 * camera.selected_axis_fov_rad
        half_target_width = 0.5 * scenario.target_angular_width_rad
        visible_limit = half_fov - half_target_width if camera.require_full_bbox_in_view else half_fov
        geometrically_visible = abs(error_rad) <= max(0.0, visible_limit)

        missed = any(
            start_s <= capture_time_s < end_s
            for start_s, end_s in camera.forced_dropout_intervals_s
        )
        if not missed and camera.miss_probability > 0.0:
            missed = bool(self._rng_dropout.random() < camera.miss_probability)
        center_noise = 0.0
        if camera.center_noise_std_normalized > 0.0:
            center_noise = float(
                self._rng_center.normal(0.0, camera.center_noise_std_normalized)
            )
        width_noise = height_noise = 0.0
        if camera.size_noise_std_fraction > 0.0:
            width_noise, height_noise = self._rng_size.normal(
                0.0, camera.size_noise_std_fraction, size=2
            )
        confidence_noise = 0.0
        if camera.confidence_noise_std > 0.0:
            confidence_noise = float(
                self._rng_confidence.normal(0.0, camera.confidence_noise_std)
            )

        valid = geometrically_visible and not missed
        width = scenario.target_angular_width_rad / camera.selected_axis_fov_rad
        height = scenario.target_angular_height_rad / camera.orthogonal_fov_rad
        return _Measurement(
            frame_index=self._frame_index,
            capture_time_s=capture_time_s,
            valid=valid,
            image_error_normalized=error_rad / half_fov + center_noise,
            bbox_width_fraction=float(np.clip(width + width_noise, 0.0, 1.0)),
            bbox_height_fraction=float(np.clip(height + height_noise, 0.0, 1.0)),
            confidence=float(
                np.clip(camera.confidence_mean + confidence_noise, 0.0, 1.0)
            ),
        )

    def _release_due_measurements(self) -> None:
        while self._measurement_queue and self._measurement_queue[0][0] <= self._time_s + _EPSILON:
            _, _, measurement = heapq.heappop(self._measurement_queue)
            if measurement.frame_index >= self._latest_released_frame_index:
                self._latest_measurement = measurement
                self._latest_released_frame_index = measurement.frame_index

    def _make_observation(self, *, frame_updated: bool) -> GimbalObservation:
        measurement = self._latest_measurement
        if measurement is None:
            age = MaskedScalar.missing()
            image_error = MaskedScalar.missing()
            bbox_width = MaskedScalar.missing()
            bbox_height = MaskedScalar.missing()
            confidence = MaskedScalar.missing()
        else:
            age = MaskedScalar(self._time_s - measurement.capture_time_s, True)
            image_error = MaskedScalar(
                measurement.image_error_normalized if measurement.valid else 0.0,
                measurement.valid,
            )
            bbox_width = MaskedScalar(
                measurement.bbox_width_fraction if measurement.valid else 0.0,
                measurement.valid,
            )
            bbox_height = MaskedScalar(
                measurement.bbox_height_fraction if measurement.valid else 0.0,
                measurement.valid,
            )
            confidence = MaskedScalar(
                measurement.confidence if measurement.valid else 0.0,
                measurement.valid,
            )

        profile = self.config.observation_profile
        servo_available = profile in {
            ObservationProfile.SERVO_AWARE,
            ObservationProfile.DISTURBANCE_AWARE,
        }
        body_available = profile is ObservationProfile.DISTURBANCE_AWARE
        _, body_rate = self.body_motion.state_at(self._time_s)
        return GimbalObservation(
            time_s=self._time_s,
            control_dt_s=self._last_control_dt_s,
            frame_updated=frame_updated,
            measurement_age_s=age,
            image_error_normalized=image_error,
            bbox_width_fraction=bbox_width,
            bbox_height_fraction=bbox_height,
            confidence=confidence,
            gimbal_angle_rad=MaskedScalar(self._gimbal_angle_rad, servo_available),
            gimbal_rate_rad_s=MaskedScalar(self._gimbal_rate_rad_s, servo_available),
            body_rate_rad_s=MaskedScalar(body_rate, body_available),
            command_mode=self.config.command_mode,
            previous_action_normalized=self._previous_action_normalized,
        )

    def _true_geometry(self) -> tuple[float, float, float, float, float, bool]:
        target_bearing, target_rate = self.target_motion.state_at(self._time_s)
        body_bearing, body_rate = self.body_motion.state_at(self._time_s)
        optical_axis = body_bearing + self._gimbal_angle_rad
        error_rad = wrap_angle_rad(target_bearing - optical_axis)
        half_fov = 0.5 * self.config.camera.selected_axis_fov_rad
        if self.config.camera.require_full_bbox_in_view:
            limit = half_fov - 0.5 * self.config.scenario.target_angular_width_rad
        else:
            limit = half_fov
        in_view = abs(error_rad) <= max(0.0, limit)
        return target_bearing, target_rate, body_bearing, body_rate, error_rad, in_view

    def _step_cost(self, action_normalized: float, dt_s: float) -> float:
        _, _, _, _, error_rad, in_view = self._true_geometry()
        objective = self.config.objective
        half_fov = 0.5 * self.config.camera.selected_axis_fov_rad
        normalized_error = error_rad / half_fov
        action_change = action_normalized - self._previous_cost_action_normalized
        instantaneous = (
            objective.error_weight * normalized_error**2
            + objective.loss_of_view_penalty * float(not in_view)
            + objective.action_effort_weight * action_normalized**2
            + objective.action_change_weight * action_change**2
        )
        return instantaneous * dt_s

    def _diagnostics(self) -> GimbalDiagnostics:
        (
            target_bearing,
            target_rate,
            body_bearing,
            body_rate,
            error_rad,
            in_view,
        ) = self._true_geometry()
        half_fov = 0.5 * self.config.camera.selected_axis_fov_rad
        return GimbalDiagnostics(
            time_s=self._time_s,
            target_bearing_rad=target_bearing,
            target_rate_rad_s=target_rate,
            body_bearing_rad=body_bearing,
            body_rate_rad_s=body_rate,
            gimbal_angle_rad=self._gimbal_angle_rad,
            gimbal_rate_rad_s=self._gimbal_rate_rad_s,
            optical_axis_bearing_rad=body_bearing + self._gimbal_angle_rad,
            true_image_error_normalized=error_rad / half_fov,
            command_mode=self.config.command_mode,
            requested_command_normalized=self._requested_command_normalized,
            requested_rate_rad_s=self._requested_rate_rad_s,
            requested_position_rad=self._requested_position_rad,
            applied_rate_command_rad_s=self._applied_rate_command_rad_s,
            applied_position_command_rad=self._applied_position_command_rad,
            inner_rate_target_rad_s=self._inner_rate_target_rad_s,
            target_in_view=in_view,
            rate_saturated=self._rate_saturated,
            angle_saturated=self._angle_saturated,
        )
