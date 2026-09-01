"""Privileged V2.1 position-adapter teacher actions for GRU training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import ObservationProfile
from .controllers import AdaptivePositionControllerConfig
from .dataset import FEATURE_NAMES, GimbalTargetStateDataset
from .control_criticality import _scenario_config


@dataclass(frozen=True)
class AdaptivePositionSupervision:
    teacher_action_normalized: np.ndarray
    mask: np.ndarray
    gimbal_angle_rad: np.ndarray
    gimbal_rate_rad_s: np.ndarray
    control_dt_s: np.ndarray
    selected_axis_fov_rad: np.ndarray
    servo_min_angle_rad: np.ndarray
    servo_max_angle_rad: np.ndarray
    servo_max_rate_rad_s: np.ndarray
    servo_max_acceleration_rad_s2: np.ndarray
    servo_position_gain_s_inv: np.ndarray
    servo_command_latency_s: np.ndarray
    servo_rate_time_constant_s: np.ndarray


def _wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _interpolate_bearing(
    bearings: np.ndarray,
    horizons_s: np.ndarray,
    requested_horizon_s: float,
) -> tuple[float, int, int]:
    requested = float(
        np.clip(requested_horizon_s, horizons_s[0], horizons_s[-1])
    )
    if requested <= horizons_s[0]:
        return float(bearings[0]), 0, 0
    right = int(np.searchsorted(horizons_s, requested, side="left"))
    right = min(right, len(horizons_s) - 1)
    if horizons_s[right] == requested:
        return float(bearings[right]), right, right
    left = right - 1
    fraction = (requested - horizons_s[left]) / (
        horizons_s[right] - horizons_s[left]
    )
    bearing = _wrap(
        float(bearings[left])
        + fraction * _wrap(float(bearings[right] - bearings[left]))
    )
    return bearing, left, right


def _interpolate_scalar(
    values: np.ndarray,
    horizons_s: np.ndarray,
    requested_horizon_s: float,
) -> float:
    requested = float(
        np.clip(requested_horizon_s, horizons_s[0], horizons_s[-1])
    )
    if requested <= horizons_s[0]:
        return float(values[0])
    right = min(
        int(np.searchsorted(horizons_s, requested, side="left")),
        len(horizons_s) - 1,
    )
    if horizons_s[right] == requested:
        return float(values[right])
    left = right - 1
    fraction = (requested - horizons_s[left]) / (
        horizons_s[right] - horizons_s[left]
    )
    return float(values[left] + fraction * (values[right] - values[left]))


def _position_normalized(
    position_rad: float,
    minimum_angle_rad: float,
    maximum_angle_rad: float,
) -> float:
    limit = maximum_angle_rad if position_rad >= 0.0 else -minimum_angle_rad
    return float(np.clip(position_rad / limit, -1.0, 1.0))


def compute_adaptive_position_supervision(
    dataset: GimbalTargetStateDataset,
    *,
    adapter: AdaptivePositionControllerConfig,
    profile: ObservationProfile = ObservationProfile.DISTURBANCE_AWARE,
) -> AdaptivePositionSupervision:
    """Replay the selected V2.1 adapter against privileged target trajectories."""

    try:
        profile_index = dataset.manifest.observation_profiles.index(profile.value)
    except ValueError as error:
        raise ValueError("adaptive-position supervision profile is absent") from error
    if profile is ObservationProfile.VISION_ONLY:
        raise ValueError("adaptive-position supervision requires gimbal telemetry")
    shape = dataset.sequence_mask.shape
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name in (
            "selected_axis_fov_rad",
            "servo_min_angle_rad",
            "servo_max_angle_rad",
            "servo_max_rate_rad_s",
            "servo_max_acceleration_rad_s2",
            "servo_position_gain_s_inv",
            "servo_command_latency_s",
            "servo_rate_time_constant_s",
        )
    }
    feature_indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    gimbal_angle = dataset.features[
        :, profile_index, :, feature_indices["gimbal_angle_rad"]
    ].astype(np.float32, copy=False)
    gimbal_rate = dataset.features[
        :, profile_index, :, feature_indices["gimbal_rate_rad_s"]
    ].astype(np.float32, copy=False)
    control_dt = dataset.features[
        :, profile_index, :, feature_indices["control_dt_s"]
    ].astype(np.float32, copy=False)
    teacher = np.zeros(shape, dtype=np.float32)
    teacher_mask = np.zeros(shape, dtype=np.bool_)
    horizons = np.asarray(
        dataset.manifest.prediction_horizons_s,
        dtype=np.float64,
    )

    for episode_index in range(dataset.episode_count):
        hardware = _scenario_config(dataset, episode_index)
        servo = hardware["servo"]
        camera = hardware["camera"]
        values = {
            "selected_axis_fov_rad": float(camera["selected_axis_fov_rad"]),
            "servo_min_angle_rad": float(servo["min_angle_rad"]),
            "servo_max_angle_rad": float(servo["max_angle_rad"]),
            "servo_max_rate_rad_s": float(servo["max_rate_rad_s"]),
            "servo_max_acceleration_rad_s2": float(
                servo["max_acceleration_rad_s2"]
            ),
            "servo_position_gain_s_inv": float(servo["position_gain_s_inv"]),
            "servo_command_latency_s": float(servo["command_latency_s"]),
            "servo_rate_time_constant_s": float(
                servo["rate_time_constant_s"]
            ),
        }
        for name, value in values.items():
            arrays[name][episode_index] = value
        minimum_angle = values["servo_min_angle_rad"]
        maximum_angle = values["servo_max_angle_rad"]
        half_fov = 0.5 * values["selected_axis_fov_rad"]
        arrival_horizon = adapter.actuator_arrival_time_scale * (
            values["servo_command_latency_s"]
            + values["servo_rate_time_constant_s"]
            + adapter.position_response_fraction
            / values["servo_position_gain_s_inv"]
        ) + adapter.additional_preview_s
        arrival_horizon = float(np.clip(arrival_horizon, horizons[0], horizons[-1]))
        setpoint_angle = float(gimbal_angle[episode_index, 0])
        setpoint_rate = 0.0
        setpoint_acceleration = 0.0
        length = int(np.sum(dataset.sequence_mask[episode_index]))
        for time_index in range(length):
            bearings = dataset.targets[episode_index, time_index, :, 0]
            base_bearing, base_left, base_right = _interpolate_bearing(
                bearings,
                horizons,
                arrival_horizon,
            )
            image_error = _wrap(
                base_bearing - float(gimbal_angle[episode_index, time_index])
            )
            fov_fraction = abs(image_error) / half_fov
            risk = float(
                np.clip(
                    (
                        fov_fraction
                        - adapter.visibility_risk_onset_fraction
                    )
                    / (
                        adapter.visibility_risk_full_fraction
                        - adapter.visibility_risk_onset_fraction
                    ),
                    0.0,
                    1.0,
                )
            )
            if adapter.risk_requires_outward_motion:
                base_rate = _interpolate_scalar(
                    dataset.targets[episode_index, time_index, :, 1],
                    horizons,
                    arrival_horizon,
                )
                image_error_rate = base_rate - float(
                    gimbal_rate[episode_index, time_index]
                )
                if image_error * image_error_rate <= 0.0:
                    risk = 0.0
            requested_horizon = float(
                np.clip(
                    arrival_horizon + risk * adapter.risk_horizon_boost_s,
                    horizons[0],
                    horizons[-1],
                )
            )
            target, left, right = _interpolate_bearing(
                bearings,
                horizons,
                requested_horizon,
            )
            valid = bool(
                dataset.target_mask[episode_index, time_index, base_left]
                and dataset.target_mask[episode_index, time_index, base_right]
                and dataset.target_mask[episode_index, time_index, left]
                and dataset.target_mask[episode_index, time_index, right]
            )
            target = float(np.clip(target, minimum_angle, maximum_angle))
            dt_s = float(control_dt[episode_index, time_index])
            if dt_s > 0.0:
                rate_multiplier = 1.0 + risk * (
                    adapter.risk_rate_limit_multiplier - 1.0
                )
                acceleration_multiplier = 1.0 + risk * (
                    adapter.risk_acceleration_limit_multiplier - 1.0
                )
                jerk_multiplier = 1.0 + risk * (
                    adapter.risk_jerk_limit_multiplier - 1.0
                )
                maximum_rate = (
                    adapter.setpoint_rate_limit_scale
                    * values["servo_max_rate_rad_s"]
                    * rate_multiplier
                )
                base_acceleration = (
                    adapter.setpoint_acceleration_limit_scale
                    * values["servo_max_acceleration_rad_s2"]
                )
                maximum_acceleration = (
                    base_acceleration * acceleration_multiplier
                )
                maximum_jerk = (
                    base_acceleration
                    / adapter.setpoint_jerk_rise_time_s
                    * jerk_multiplier
                )
                error = target - setpoint_angle
                stopping_speed = math.sqrt(
                    2.0 * maximum_acceleration * abs(error)
                )
                desired_rate = (
                    math.copysign(min(maximum_rate, stopping_speed), error)
                    if abs(error) > 1e-12
                    else 0.0
                )
                desired_acceleration = float(
                    np.clip(
                        (desired_rate - setpoint_rate) / dt_s,
                        -maximum_acceleration,
                        maximum_acceleration,
                    )
                )
                setpoint_acceleration += float(
                    np.clip(
                        desired_acceleration - setpoint_acceleration,
                        -maximum_jerk * dt_s,
                        maximum_jerk * dt_s,
                    )
                )
                setpoint_rate = float(
                    np.clip(
                        setpoint_rate + setpoint_acceleration * dt_s,
                        -maximum_rate,
                        maximum_rate,
                    )
                )
                step = setpoint_rate * dt_s
                if step * error > 0.0 and abs(step) >= abs(error):
                    setpoint_angle = target
                    setpoint_rate = 0.0
                    setpoint_acceleration = 0.0
                else:
                    setpoint_angle = float(
                        np.clip(
                            setpoint_angle + step,
                            minimum_angle,
                            maximum_angle,
                        )
                    )
            teacher[episode_index, time_index] = _position_normalized(
                setpoint_angle,
                minimum_angle,
                maximum_angle,
            )
            teacher_mask[episode_index, time_index] = valid

    return AdaptivePositionSupervision(
        teacher_action_normalized=teacher,
        mask=teacher_mask & dataset.sequence_mask,
        gimbal_angle_rad=gimbal_angle,
        gimbal_rate_rad_s=gimbal_rate,
        control_dt_s=control_dt,
        **arrays,
    )
