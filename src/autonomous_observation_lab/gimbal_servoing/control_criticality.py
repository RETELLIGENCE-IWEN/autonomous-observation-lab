"""Training-only control criticality for predictive gimbal supervision."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import ObservationProfile
from .dataset import FEATURE_NAMES, GimbalTargetStateDataset


@dataclass(frozen=True)
class ControlCriticalityConfig:
    """Hardware-relative weights for rare, control-relevant target states.

    Criticality is used only to weight privileged training labels. It is not a
    deployable model input. Camera and actuator scales come from the serialized
    configuration for each randomized episode.
    """

    visibility_onset_fov_fraction: float = 0.45
    visibility_full_fov_fraction: float = 0.90
    rate_onset_capacity_fraction: float = 0.35
    rate_full_capacity_fraction: float = 0.90
    acceleration_onset_capacity_fraction: float = 0.20
    acceleration_full_capacity_fraction: float = 0.80
    visibility_weight: float = 2.0
    rate_weight: float = 1.0
    acceleration_weight: float = 2.0
    reversal_weight: float = 3.0
    detector_gap_weight: float = 1.0
    joint_capacity_visibility_weight: float = 2.0
    maximum_label_weight: float = 10.0
    weighting_strength: float = 1.0
    critical_weight_threshold: float = 5.0
    normalize_mean_weight: bool = True
    mechanically_reachable_only: bool = True

    def __post_init__(self) -> None:
        for onset_name, full_name in (
            (
                "visibility_onset_fov_fraction",
                "visibility_full_fov_fraction",
            ),
            (
                "rate_onset_capacity_fraction",
                "rate_full_capacity_fraction",
            ),
            (
                "acceleration_onset_capacity_fraction",
                "acceleration_full_capacity_fraction",
            ),
        ):
            onset = getattr(self, onset_name)
            full = getattr(self, full_name)
            if not math.isfinite(onset) or onset < 0.0:
                raise ValueError(f"{onset_name} must be finite and non-negative")
            if not math.isfinite(full) or full <= onset:
                raise ValueError(f"{full_name} must exceed its onset")
        for name in (
            "visibility_weight",
            "rate_weight",
            "acceleration_weight",
            "reversal_weight",
            "detector_gap_weight",
            "joint_capacity_visibility_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.maximum_label_weight) or (
            self.maximum_label_weight < 1.0
        ):
            raise ValueError("maximum_label_weight must be finite and at least one")
        if not math.isfinite(self.critical_weight_threshold) or (
            self.critical_weight_threshold < 1.0
        ):
            raise ValueError(
                "critical_weight_threshold must be finite and at least one"
            )
        if not math.isfinite(self.weighting_strength) or not (
            0.0 <= self.weighting_strength <= 1.0
        ):
            raise ValueError("weighting_strength must be in [0, 1]")


@dataclass(frozen=True)
class ControlCriticality:
    weights: np.ndarray
    raw_weights: np.ndarray
    visibility_risk: np.ndarray
    rate_risk: np.ndarray
    acceleration_risk: np.ndarray
    reversal: np.ndarray
    detector_gap: np.ndarray
    mechanically_reachable: np.ndarray
    critical_mask: np.ndarray


def _ramp(values: np.ndarray, onset: float, full: float) -> np.ndarray:
    return np.clip((values - onset) / (full - onset), 0.0, 1.0)


def _angle_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = left - right
    return np.arctan2(np.sin(difference), np.cos(difference))


def _scenario_config(
    dataset: GimbalTargetStateDataset,
    episode_index: int,
) -> dict[str, Any]:
    generation = dataset.manifest.generation
    seed = int(dataset.episode_seed[episode_index])
    scenario_index = int(dataset.scenario_index[episode_index])
    for item in generation.get("scenario_variants", []):
        if int(item["seed"]) == seed and int(item["scenario_index"]) == (
            scenario_index
        ):
            return item["scenario"]["config"]
    scenarios = generation.get("scenarios", [])
    if scenario_index >= len(scenarios):
        raise ValueError("dataset manifest has no configuration for an episode")
    return scenarios[scenario_index]["config"]


def compute_control_criticality(
    dataset: GimbalTargetStateDataset,
    *,
    profile: ObservationProfile = ObservationProfile.DISTURBANCE_AWARE,
    config: ControlCriticalityConfig | None = None,
) -> ControlCriticality:
    """Compute hardware-normalized privileged label weights.

    The returned tensors have shape ``[episode, time, horizon]``. Only oracle
    labels and serialized hardware configuration affect the weighting; the
    deployable feature schema remains unchanged.
    """

    config = config or ControlCriticalityConfig()
    if dataset.manifest.feature_names != FEATURE_NAMES:
        raise ValueError("control criticality requires the current feature schema")
    try:
        profile_index = dataset.manifest.observation_profiles.index(profile.value)
    except ValueError as error:
        raise ValueError("criticality profile is absent from the dataset") from error
    if profile is ObservationProfile.VISION_ONLY:
        raise ValueError("control criticality requires deployable servo angle")

    shape = dataset.target_mask.shape
    visibility = np.zeros(shape, dtype=np.float32)
    rate_risk = np.zeros(shape, dtype=np.float32)
    acceleration = np.zeros(shape, dtype=np.float32)
    reversal = np.zeros(shape, dtype=np.float32)
    detector_gap = np.zeros(shape, dtype=np.float32)
    reachable = np.zeros(shape, dtype=np.bool_)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    horizons = np.asarray(
        dataset.manifest.prediction_horizons_s,
        dtype=np.float64,
    )

    for episode_index in range(dataset.episode_count):
        hardware = _scenario_config(dataset, episode_index)
        camera = hardware["camera"]
        servo = hardware["servo"]
        half_fov = 0.5 * float(camera["selected_axis_fov_rad"])
        maximum_rate = float(servo["max_rate_rad_s"])
        maximum_acceleration = float(servo["max_acceleration_rad_s2"])
        minimum_angle = float(servo["min_angle_rad"])
        maximum_angle = float(servo["max_angle_rad"])
        features = dataset.features[episode_index, profile_index]
        gimbal_angle = features[:, feature_index["gimbal_angle_rad"]]
        image_valid = features[:, feature_index["image_error_valid"]] > 0.5
        control_dt = np.maximum(
            features[:, feature_index["control_dt_s"]],
            1e-6,
        )
        bearing = dataset.targets[episode_index, ..., 0]
        rate = dataset.targets[episode_index, ..., 1]
        current_rate = rate[:, :1]

        error_fraction = np.abs(
            _angle_delta(bearing, gimbal_angle[:, None])
        ) / half_fov
        visibility[episode_index] = _ramp(
            error_fraction,
            config.visibility_onset_fov_fraction,
            config.visibility_full_fov_fraction,
        )
        rate_risk[episode_index] = _ramp(
            np.abs(rate) / maximum_rate,
            config.rate_onset_capacity_fraction,
            config.rate_full_capacity_fraction,
        )

        interval = np.maximum(horizons[None, :], control_dt[:, None])
        acceleration_fraction = np.abs(rate - current_rate) / (
            maximum_acceleration * interval
        )
        if horizons[0] == 0.0:
            current_rate_series = rate[:, 0]
            temporal_change = np.zeros_like(current_rate_series)
            temporal_change[1:] = np.abs(
                np.diff(current_rate_series)
            ) / (maximum_acceleration * control_dt[1:])
            acceleration_fraction[:, 0] = temporal_change
        acceleration[episode_index] = _ramp(
            acceleration_fraction,
            config.acceleration_onset_capacity_fraction,
            config.acceleration_full_capacity_fraction,
        )
        reversal[episode_index] = (
            (current_rate * rate < 0.0)
            & (np.abs(current_rate - rate) > 0.05 * maximum_rate)
        ).astype(np.float32)
        detector_gap[episode_index] = (~image_valid[:, None]).astype(np.float32)
        reachable[episode_index] = (
            (bearing >= minimum_angle) & (bearing <= maximum_angle)
        )

    valid = dataset.target_mask & dataset.sequence_mask[:, :, None]
    extra = (
        config.visibility_weight * visibility
        + config.rate_weight * rate_risk
        + config.acceleration_weight * acceleration
        + config.reversal_weight * reversal
        + config.detector_gap_weight * detector_gap
        + config.joint_capacity_visibility_weight
        * np.minimum(visibility, np.maximum(rate_risk, acceleration))
    )
    if config.mechanically_reachable_only:
        extra = np.where(reachable, extra, 0.0)
    raw_weights = np.clip(
        1.0 + extra,
        1.0,
        config.maximum_label_weight,
    ).astype(np.float32)
    weights = raw_weights.copy()
    if config.normalize_mean_weight and np.any(valid):
        weights /= float(np.mean(weights[valid]))
    weights = 1.0 + config.weighting_strength * (weights - 1.0)
    weights = np.where(valid, weights, 0.0).astype(np.float32)
    critical = valid & (raw_weights >= config.critical_weight_threshold)
    return ControlCriticality(
        weights=weights,
        raw_weights=raw_weights,
        visibility_risk=visibility,
        rate_risk=rate_risk,
        acceleration_risk=acceleration,
        reversal=reversal,
        detector_gap=detector_gap,
        mechanically_reachable=reachable,
        critical_mask=critical,
    )


def control_criticality_report(
    dataset: GimbalTargetStateDataset,
    criticality: ControlCriticality,
    *,
    config: ControlCriticalityConfig | None = None,
) -> dict[str, Any]:
    """Summarize raw coverage and effective supervised weight by scenario."""

    config = config or ControlCriticalityConfig()
    valid = dataset.target_mask & dataset.sequence_mask[:, :, None]

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        selected = valid & mask
        valid_count = int(np.sum(valid & mask))
        critical_count = int(np.sum(criticality.critical_mask & mask))
        total_weight = float(np.sum(criticality.weights[valid & mask]))
        critical_weight = float(
            np.sum(criticality.weights[criticality.critical_mask & mask])
        )
        return {
            "valid_label_count": valid_count,
            "critical_label_count": critical_count,
            "critical_label_fraction": (
                critical_count / valid_count if valid_count else 0.0
            ),
            "effective_weight_on_critical_fraction": (
                critical_weight / total_weight if total_weight else 0.0
            ),
        }

    all_mask = np.ones_like(valid, dtype=np.bool_)
    by_scenario = {}
    for scenario_index, scenario_name in enumerate(
        dataset.manifest.scenario_names
    ):
        episode_mask = dataset.scenario_index == scenario_index
        mask = np.broadcast_to(episode_mask[:, None, None], valid.shape)
        by_scenario[scenario_name] = summarize(mask)

    component_fractions = {}
    for name in (
        "visibility_risk",
        "rate_risk",
        "acceleration_risk",
        "reversal",
        "detector_gap",
    ):
        values = getattr(criticality, name)
        component_fractions[name] = float(np.mean(values[valid] > 0.0))
    return {
        "configuration": asdict(config),
        "episode_count": dataset.episode_count,
        "overall": summarize(all_mask),
        "component_active_fractions": component_fractions,
        "by_scenario": by_scenario,
    }
