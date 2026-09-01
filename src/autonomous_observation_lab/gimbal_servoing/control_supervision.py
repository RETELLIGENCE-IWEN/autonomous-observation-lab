"""Privileged, hardware-normalized action supervision for GRU training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ObservationProfile
from .dataset import (
    FEATURE_NAMES,
    ORACLE_ACTION_NAMES,
    GimbalTargetStateDataset,
)
from .control_criticality import _scenario_config


@dataclass(frozen=True)
class ControlActionSupervision:
    """Training-only context for a differentiable oracle-action surrogate."""

    oracle_actions: np.ndarray
    gimbal_angle_rad: np.ndarray
    servo_max_rate_rad_s: np.ndarray
    servo_min_angle_rad: np.ndarray
    servo_max_angle_rad: np.ndarray
    rate_feedback_gain_s_inv: np.ndarray
    position_preview_s: np.ndarray
    mask: np.ndarray


def compute_control_action_supervision(
    dataset: GimbalTargetStateDataset,
    *,
    profile: ObservationProfile = ObservationProfile.DISTURBANCE_AWARE,
) -> ControlActionSupervision:
    """Materialize oracle actions and per-episode actuator scales.

    This data is consumed by the loss only. It is never appended to the
    deployable observation vector.
    """

    if dataset.manifest.feature_names != FEATURE_NAMES:
        raise ValueError("control supervision requires the current feature schema")
    if dataset.manifest.oracle_action_names != ORACLE_ACTION_NAMES:
        raise ValueError("control supervision requires both oracle actions")
    try:
        profile_index = dataset.manifest.observation_profiles.index(profile.value)
    except ValueError as error:
        raise ValueError("control supervision profile is absent") from error
    if profile is ObservationProfile.VISION_ONLY:
        raise ValueError("control supervision requires deployable gimbal angle")

    shape = dataset.sequence_mask.shape
    maximum_rate = np.zeros(shape, dtype=np.float32)
    minimum_angle = np.zeros(shape, dtype=np.float32)
    maximum_angle = np.zeros(shape, dtype=np.float32)
    rate_gain = np.zeros(shape, dtype=np.float32)
    position_preview = np.zeros(shape, dtype=np.float32)
    request = dataset.manifest.generation.get("request", {})
    oracle_control = request.get("oracle_control", {})
    if not isinstance(oracle_control, dict):
        raise ValueError("dataset manifest lacks oracle control configuration")
    configured_rate_gain = float(oracle_control["rate_feedback_gain_s_inv"])
    configured_position_preview = float(oracle_control["position_preview_s"])

    for episode_index in range(dataset.episode_count):
        hardware = _scenario_config(dataset, episode_index)
        if not isinstance(hardware.get("servo"), dict):
            raise ValueError("dataset scenario lacks servo configuration")
        servo = hardware["servo"]
        maximum_rate[episode_index] = float(servo["max_rate_rad_s"])
        minimum_angle[episode_index] = float(servo["min_angle_rad"])
        maximum_angle[episode_index] = float(servo["max_angle_rad"])
        rate_gain[episode_index] = configured_rate_gain
        position_preview[episode_index] = configured_position_preview

    feature_index = FEATURE_NAMES.index("gimbal_angle_rad")
    gimbal_angle = dataset.features[:, profile_index, :, feature_index]
    return ControlActionSupervision(
        oracle_actions=dataset.oracle_actions.astype(np.float32, copy=False),
        gimbal_angle_rad=gimbal_angle.astype(np.float32, copy=False),
        servo_max_rate_rad_s=maximum_rate,
        servo_min_angle_rad=minimum_angle,
        servo_max_angle_rad=maximum_angle,
        rate_feedback_gain_s_inv=rate_gain,
        position_preview_s=position_preview,
        mask=dataset.sequence_mask.copy(),
    )
