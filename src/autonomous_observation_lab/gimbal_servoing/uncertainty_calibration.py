"""Validation-fit uncertainty calibration for causal gimbal GRUs.

This module requires the optional ``learning`` dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ObservationProfile
from .dataset import (
    FEATURE_NAMES,
    GimbalTargetStateDataset,
    load_gimbal_dataset,
    validate_disjoint_seed_blocks,
)
from .gru import CausalTargetStateGRU, load_gru_checkpoint
from .gru_training import (
    GimbalTorchSequenceDataset,
    _angle_residual_numpy,
    _metrics_from_predictions,
)
from .types import GimbalObservation


CALIBRATION_SCHEMA_VERSION = "gimbal_uncertainty_calibration_v1"
CONTEXTUAL_CALIBRATION_SCHEMA_VERSION = (
    "gimbal_contextual_uncertainty_calibration_v1"
)
DEFAULT_NOMINAL_COVERAGES = (0.50, 0.6827, 0.80, 0.90, 0.9545, 0.99)
CONTEXT_NAMES = (
    "fresh_young",
    "fresh_old",
    "between_young",
    "between_old",
    "invalid_short_gap",
    "invalid_medium_gap",
    "invalid_long_gap",
    "no_valid_detection_history",
)


@dataclass(frozen=True)
class GaussianUncertaintyCalibration:
    schema_version: str
    profile: ObservationProfile
    prediction_horizons_s: tuple[float, ...]
    bearing_std_scale: tuple[float, ...]
    rate_std_scale: tuple[float, ...]
    validation_dataset_hash: str
    test_dataset_hash: str
    checkpoint_sha256: str
    minimum_scale: float
    maximum_scale: float

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported uncertainty calibration schema")
        horizon_count = len(self.prediction_horizons_s)
        if horizon_count == 0:
            raise ValueError("calibration requires prediction horizons")
        if not (
            len(self.bearing_std_scale)
            == len(self.rate_std_scale)
            == horizon_count
        ):
            raise ValueError("calibration scale/horizon lengths differ")
        if not 0.0 < self.minimum_scale <= self.maximum_scale:
            raise ValueError("calibration scale bounds are invalid")
        for scale in self.bearing_std_scale + self.rate_std_scale:
            if (
                not math.isfinite(scale)
                or not self.minimum_scale <= scale <= self.maximum_scale
            ):
                raise ValueError("calibration scale is outside declared bounds")

    def scales_for_horizon(self, horizon_index: int) -> tuple[float, float]:
        if not 0 <= horizon_index < len(self.prediction_horizons_s):
            raise ValueError("calibration horizon index is out of range")
        return (
            self.bearing_std_scale[horizon_index],
            self.rate_std_scale[horizon_index],
        )

    def scales_for_observation(
        self,
        horizon_index: int,
        observation: GimbalObservation,
        detection_gap_s: float | None,
    ) -> tuple[float, float]:
        del observation, detection_gap_s
        return self.scales_for_horizon(horizon_index)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GaussianUncertaintyCalibration":
        payload = dict(value)
        payload["profile"] = ObservationProfile(payload["profile"])
        for name in (
            "prediction_horizons_s",
            "bearing_std_scale",
            "rate_std_scale",
        ):
            payload[name] = tuple(payload[name])
        return cls(**payload)


@dataclass(frozen=True)
class ContextualGaussianUncertaintyCalibration:
    """Piecewise variance scaling from deployable detector context."""

    schema_version: str
    profile: ObservationProfile
    prediction_horizons_s: tuple[float, ...]
    context_names: tuple[str, ...]
    bearing_std_scale_by_context: tuple[tuple[float, ...], ...]
    rate_std_scale_by_context: tuple[tuple[float, ...], ...]
    measurement_age_threshold_s: float
    detection_gap_thresholds_s: tuple[float, float]
    prior_strength: float
    validation_dataset_hash: str
    test_dataset_hash: str
    checkpoint_sha256: str
    minimum_scale: float
    maximum_scale: float

    def __post_init__(self) -> None:
        if self.schema_version != CONTEXTUAL_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported contextual calibration schema")
        if self.context_names != CONTEXT_NAMES:
            raise ValueError("unsupported contextual calibration contexts")
        horizon_count = len(self.prediction_horizons_s)
        if horizon_count == 0:
            raise ValueError("calibration requires prediction horizons")
        if not 0.0 < self.minimum_scale <= self.maximum_scale:
            raise ValueError("calibration scale bounds are invalid")
        if not math.isfinite(self.prior_strength) or self.prior_strength < 0.0:
            raise ValueError("prior strength must be finite and non-negative")
        if (
            not math.isfinite(self.measurement_age_threshold_s)
            or self.measurement_age_threshold_s <= 0.0
        ):
            raise ValueError("measurement-age threshold must be positive")
        short_gap, medium_gap = self.detection_gap_thresholds_s
        if not 0.0 < short_gap < medium_gap:
            raise ValueError("detection-gap thresholds must be increasing")
        for table in (
            self.bearing_std_scale_by_context,
            self.rate_std_scale_by_context,
        ):
            if len(table) != len(self.context_names):
                raise ValueError("calibration context table length differs")
            if any(len(row) != horizon_count for row in table):
                raise ValueError("calibration horizon table length differs")
            for scale in (value for row in table for value in row):
                if (
                    not math.isfinite(scale)
                    or not self.minimum_scale <= scale <= self.maximum_scale
                ):
                    raise ValueError("calibration scale is outside bounds")

    def context_index(
        self,
        observation: GimbalObservation,
        detection_gap_s: float | None,
    ) -> int:
        age_is_young = (
            observation.measurement_age_s.valid
            and observation.measurement_age_s.value
            < self.measurement_age_threshold_s
        )
        if observation.frame_updated and observation.detection_valid:
            return 0 if age_is_young else 1
        if observation.detection_valid:
            return 2 if age_is_young else 3
        if detection_gap_s is None:
            return 7
        short_gap, medium_gap = self.detection_gap_thresholds_s
        if detection_gap_s < short_gap:
            return 4
        if detection_gap_s < medium_gap:
            return 5
        return 6

    def scales_for_observation(
        self,
        horizon_index: int,
        observation: GimbalObservation,
        detection_gap_s: float | None,
    ) -> tuple[float, float]:
        if not 0 <= horizon_index < len(self.prediction_horizons_s):
            raise ValueError("calibration horizon index is out of range")
        context_index = self.context_index(observation, detection_gap_s)
        return (
            self.bearing_std_scale_by_context[context_index][horizon_index],
            self.rate_std_scale_by_context[context_index][horizon_index],
        )

    @classmethod
    def from_dict(
        cls, value: dict[str, Any]
    ) -> "ContextualGaussianUncertaintyCalibration":
        payload = dict(value)
        payload["profile"] = ObservationProfile(payload["profile"])
        payload["prediction_horizons_s"] = tuple(
            payload["prediction_horizons_s"]
        )
        payload["context_names"] = tuple(payload["context_names"])
        payload["detection_gap_thresholds_s"] = tuple(
            payload["detection_gap_thresholds_s"]
        )
        for name in (
            "bearing_std_scale_by_context",
            "rate_std_scale_by_context",
        ):
            payload[name] = tuple(tuple(row) for row in payload[name])
        return cls(**payload)


UncertaintyCalibration = (
    GaussianUncertaintyCalibration | ContextualGaussianUncertaintyCalibration
)


def uncertainty_calibration_from_dict(
    value: dict[str, Any],
) -> UncertaintyCalibration:
    schema = value.get("schema_version")
    if schema == CALIBRATION_SCHEMA_VERSION:
        return GaussianUncertaintyCalibration.from_dict(value)
    if schema == CONTEXTUAL_CALIBRATION_SCHEMA_VERSION:
        return ContextualGaussianUncertaintyCalibration.from_dict(value)
    raise ValueError("unsupported uncertainty calibration schema")


@dataclass(frozen=True)
class CalibrationExperimentConfig:
    profile: ObservationProfile = ObservationProfile.DISTURBANCE_AWARE
    batch_size: int = 32
    minimum_scale: float = 0.25
    maximum_scale: float = 4.0
    nominal_coverages: tuple[float, ...] = DEFAULT_NOMINAL_COVERAGES
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not 0.0 < self.minimum_scale <= self.maximum_scale:
            raise ValueError("calibration scale bounds are invalid")
        if not self.nominal_coverages:
            raise ValueError("nominal coverage grid must be non-empty")
        if any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.nominal_coverages
        ):
            raise ValueError("nominal coverages must be finite and in (0, 1)")
        if tuple(sorted(set(self.nominal_coverages))) != self.nominal_coverages:
            raise ValueError("nominal coverages must be unique and increasing")


@dataclass(frozen=True)
class ContextualCalibrationExperimentConfig(CalibrationExperimentConfig):
    measurement_age_threshold_s: float = 0.16
    detection_gap_thresholds_s: tuple[float, float] = (0.15, 0.65)
    prior_strength: float = 512.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not math.isfinite(self.measurement_age_threshold_s)
            or self.measurement_age_threshold_s <= 0.0
        ):
            raise ValueError("measurement-age threshold must be positive")
        short_gap, medium_gap = self.detection_gap_thresholds_s
        if not 0.0 < short_gap < medium_gap:
            raise ValueError("detection-gap thresholds must be increasing")
        if not math.isfinite(self.prior_strength) or self.prior_strength < 0.0:
            raise ValueError("prior strength must be finite and non-negative")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _gru_predictions(
    model: CausalTargetStateGRU,
    dataset: GimbalTargetStateDataset,
    profile: ObservationProfile,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    loader = DataLoader(
        GimbalTorchSequenceDataset(dataset, profile),
        batch_size=batch_size,
        shuffle=False,
    )
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for batch in loader:
        output = model(batch["features"].to(device))
        means.append(output.mean.cpu().numpy())
        stds.append(output.std.cpu().numpy())
    possible_mask = dataset.target_mask & dataset.sequence_mask[..., None]
    return np.concatenate(means), np.concatenate(stds), possible_mask


def _fit_scales(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_scale: float,
    maximum_scale: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    bearing_error = _angle_residual_numpy(mean[..., 0], targets[..., 0])
    rate_error = mean[..., 1] - targets[..., 1]
    dimension_errors = (bearing_error, rate_error)
    dimension_scales: list[tuple[float, ...]] = []
    for dimension, error in enumerate(dimension_errors):
        scales = []
        for horizon_index in range(mean.shape[-2]):
            horizon_mask = mask[..., horizon_index]
            standardized_squared = (
                error[..., horizon_index][horizon_mask]
                / std[..., horizon_index, dimension][horizon_mask]
            ) ** 2
            if standardized_squared.size == 0:
                raise ValueError("calibration split has an empty horizon")
            optimum = math.sqrt(float(np.mean(standardized_squared)))
            scales.append(
                float(np.clip(optimum, minimum_scale, maximum_scale))
            )
        dimension_scales.append(tuple(scales))
    sample_counts = tuple(int(mask[..., index].sum()) for index in range(mask.shape[-1]))
    return dimension_scales[0], dimension_scales[1], sample_counts


def _detection_gap_array(
    dataset: GimbalTargetStateDataset,
    features: np.ndarray,
) -> np.ndarray:
    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    frame_updated = features[..., indices["frame_updated"]] > 0.5
    detection_valid = features[..., indices["image_error_valid"]] > 0.5
    detection_gap = np.full(
        dataset.sequence_mask.shape, np.inf, dtype=np.float64
    )
    for episode_index in range(dataset.episode_count):
        length = int(dataset.sequence_mask[episode_index].sum())
        last_valid_arrival: float | None = None
        for time_index in range(length):
            if (
                frame_updated[episode_index, time_index]
                and detection_valid[episode_index, time_index]
            ):
                last_valid_arrival = float(
                    dataset.time_s[episode_index, time_index]
                )
            if last_valid_arrival is not None:
                detection_gap[episode_index, time_index] = max(
                    0.0,
                    float(dataset.time_s[episode_index, time_index])
                    - last_valid_arrival,
                )
    return detection_gap


def _context_index_array(
    dataset: GimbalTargetStateDataset,
    profile: ObservationProfile,
    *,
    measurement_age_threshold_s: float,
    detection_gap_thresholds_s: tuple[float, float],
) -> np.ndarray:
    profile_index = dataset.manifest.observation_profiles.index(profile.value)
    features = dataset.features[:, profile_index]
    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    sequence = dataset.sequence_mask
    frame_updated = features[..., indices["frame_updated"]] > 0.5
    detection_valid = features[..., indices["image_error_valid"]] > 0.5
    age_valid = features[..., indices["measurement_age_valid"]] > 0.5
    measurement_age = features[..., indices["measurement_age_s"]]
    young = age_valid & (measurement_age < measurement_age_threshold_s)
    detection_gap = _detection_gap_array(dataset, features)
    short_gap, medium_gap = detection_gap_thresholds_s
    context = np.full(sequence.shape, 7, dtype=np.int64)
    context[sequence & frame_updated & detection_valid & young] = 0
    context[sequence & frame_updated & detection_valid & ~young] = 1
    context[sequence & ~frame_updated & detection_valid & young] = 2
    context[sequence & ~frame_updated & detection_valid & ~young] = 3
    invalid = sequence & ~detection_valid & np.isfinite(detection_gap)
    context[invalid & (detection_gap < short_gap)] = 4
    context[
        invalid
        & (detection_gap >= short_gap)
        & (detection_gap < medium_gap)
    ] = 5
    context[invalid & (detection_gap >= medium_gap)] = 6
    return context


def _fit_contextual_scales(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    context_index: np.ndarray,
    *,
    global_bearing_scale: tuple[float, ...],
    global_rate_scale: tuple[float, ...],
    prior_strength: float,
    minimum_scale: float,
    maximum_scale: float,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    errors = (
        _angle_residual_numpy(mean[..., 0], targets[..., 0]),
        mean[..., 1] - targets[..., 1],
    )
    global_scales = (global_bearing_scale, global_rate_scale)
    tables: list[tuple[tuple[float, ...], ...]] = []
    sample_counts: list[tuple[int, ...]] = []
    for context in range(len(CONTEXT_NAMES)):
        context_counts = []
        for horizon_index in range(mean.shape[-2]):
            context_counts.append(
                int(
                    (
                        mask[..., horizon_index]
                        & (context_index == context)
                    ).sum()
                )
            )
        sample_counts.append(tuple(context_counts))
    for dimension, error in enumerate(errors):
        rows = []
        for context in range(len(CONTEXT_NAMES)):
            scales = []
            for horizon_index in range(mean.shape[-2]):
                selected = mask[..., horizon_index] & (
                    context_index == context
                )
                standardized_squared = (
                    error[..., horizon_index][selected]
                    / std[..., horizon_index, dimension][selected]
                ) ** 2
                count = standardized_squared.size
                prior_variance = global_scales[dimension][horizon_index] ** 2
                denominator = count + prior_strength
                if denominator == 0.0:
                    raise ValueError("empty context requires positive shrinkage")
                variance = (
                    float(np.sum(standardized_squared))
                    + prior_strength * prior_variance
                ) / denominator
                scales.append(
                    float(
                        np.clip(
                            math.sqrt(variance),
                            minimum_scale,
                            maximum_scale,
                        )
                    )
                )
            rows.append(tuple(scales))
        tables.append(tuple(rows))
    return tables[0], tables[1], tuple(sample_counts)


def apply_uncertainty_calibration(
    std: np.ndarray,
    calibration: GaussianUncertaintyCalibration,
) -> np.ndarray:
    if std.shape[-2] != len(calibration.prediction_horizons_s) or std.shape[-1] != 2:
        raise ValueError("prediction standard-deviation shape is incompatible")
    calibrated = std.copy()
    calibrated[..., 0] *= np.asarray(calibration.bearing_std_scale)
    calibrated[..., 1] *= np.asarray(calibration.rate_std_scale)
    return calibrated


def apply_contextual_uncertainty_calibration(
    std: np.ndarray,
    dataset: GimbalTargetStateDataset,
    calibration: ContextualGaussianUncertaintyCalibration,
) -> np.ndarray:
    if (
        std.shape[-2] != len(calibration.prediction_horizons_s)
        or std.shape[-1] != 2
    ):
        raise ValueError("prediction standard-deviation shape is incompatible")
    context_index = _context_index_array(
        dataset,
        calibration.profile,
        measurement_age_threshold_s=(
            calibration.measurement_age_threshold_s
        ),
        detection_gap_thresholds_s=calibration.detection_gap_thresholds_s,
    )
    calibrated = std.copy()
    bearing_table = np.asarray(calibration.bearing_std_scale_by_context)
    rate_table = np.asarray(calibration.rate_std_scale_by_context)
    calibrated[..., 0] *= bearing_table[context_index]
    calibrated[..., 1] *= rate_table[context_index]
    return calibrated


def _metrics(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    dataset: GimbalTargetStateDataset,
    mask: np.ndarray,
) -> dict[str, Any]:
    return asdict(
        _metrics_from_predictions(
            mean=mean,
            std=std,
            targets=dataset.targets,
            prediction_mask=mask,
            possible_mask=mask,
            horizons_s=dataset.manifest.prediction_horizons_s,
        )
    )


def _reliability(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    nominal_coverages: tuple[float, ...],
) -> dict[str, Any]:
    errors = (
        _angle_residual_numpy(mean[..., 0], targets[..., 0]),
        mean[..., 1] - targets[..., 1],
    )
    records: dict[str, Any] = {}
    for dimension, name in enumerate(("bearing", "rate")):
        empirical = []
        per_horizon = [[] for _ in range(mask.shape[-1])]
        for nominal in nominal_coverages:
            quantile = NormalDist().inv_cdf(0.5 * (1.0 + nominal))
            covered = np.abs(errors[dimension]) <= quantile * std[..., dimension]
            empirical.append(float(np.mean(covered[mask])))
            for horizon_index in range(mask.shape[-1]):
                horizon_mask = mask[..., horizon_index]
                per_horizon[horizon_index].append(
                    float(
                        np.mean(covered[..., horizon_index][horizon_mask])
                    )
                )
        records[name] = {
            "nominal_coverage": list(nominal_coverages),
            "empirical_coverage": empirical,
            "mean_absolute_calibration_error": float(
                np.mean(
                    np.abs(
                        np.asarray(empirical)
                        - np.asarray(nominal_coverages)
                    )
                )
            ),
            "per_horizon_empirical_coverage": per_horizon,
        }
    return records


def _episode_scenario_payload(
    dataset: GimbalTargetStateDataset,
    episode_index: int,
) -> dict[str, Any]:
    variants = dataset.manifest.generation.get("scenario_variants")
    scenario_index = int(dataset.scenario_index[episode_index])
    if isinstance(variants, list):
        seed = int(dataset.episode_seed[episode_index])
        for variant in variants:
            if (
                int(variant["seed"]) == seed
                and int(variant["scenario_index"]) == scenario_index
            ):
                return variant["scenario"]
        raise ValueError("dataset manifest is missing an episode scenario")
    scenarios = dataset.manifest.generation.get("scenarios")
    if not isinstance(scenarios, list) or not 0 <= scenario_index < len(scenarios):
        raise ValueError("dataset manifest has no canonical scenario payload")
    return scenarios[scenario_index]


def _context_masks(
    dataset: GimbalTargetStateDataset,
    profile: ObservationProfile,
) -> dict[str, np.ndarray]:
    profile_index = dataset.manifest.observation_profiles.index(profile.value)
    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    features = dataset.features[:, profile_index]
    sequence = dataset.sequence_mask
    frame_updated = features[..., indices["frame_updated"]] > 0.5
    detection_valid = features[..., indices["image_error_valid"]] > 0.5
    age_valid = features[..., indices["measurement_age_valid"]] > 0.5
    measurement_age = features[..., indices["measurement_age_s"]]
    detection_gap = np.full(sequence.shape, np.inf, dtype=np.float64)
    target_visible = np.zeros(sequence.shape, dtype=np.bool_)
    zero_horizon = dataset.manifest.prediction_horizons_s.index(0.0)
    gimbal_angle = features[..., indices["gimbal_angle_rad"]]

    for episode_index in range(dataset.episode_count):
        length = int(sequence[episode_index].sum())
        last_valid_arrival: float | None = None
        for time_index in range(length):
            if (
                frame_updated[episode_index, time_index]
                and detection_valid[episode_index, time_index]
            ):
                last_valid_arrival = float(
                    dataset.time_s[episode_index, time_index]
                )
            if last_valid_arrival is not None:
                detection_gap[episode_index, time_index] = max(
                    0.0,
                    float(dataset.time_s[episode_index, time_index])
                    - last_valid_arrival,
                )
        scenario = _episode_scenario_payload(dataset, episode_index)
        camera = scenario["config"]["camera"]
        scenario_config = scenario["config"]["scenario"]
        limit = 0.5 * float(camera["selected_axis_fov_rad"])
        if bool(camera["require_full_bbox_in_view"]):
            limit -= 0.5 * float(scenario_config["target_angular_width_rad"])
        bearing = dataset.targets[
            episode_index, :length, zero_horizon, 0
        ]
        error = np.arctan2(
            np.sin(bearing - gimbal_angle[episode_index, :length]),
            np.cos(bearing - gimbal_angle[episode_index, :length]),
        )
        target_visible[episode_index, :length] = np.abs(error) <= max(0.0, limit)

    masks = {
        "target_visible": sequence & target_visible,
        "target_out_of_view": sequence & ~target_visible,
        "fresh_valid_detection": sequence & frame_updated & detection_valid,
        "between_valid_frames": sequence & ~frame_updated & detection_valid,
        "detector_invalid": sequence & ~detection_valid,
        "measurement_age_lt_80ms": sequence & age_valid & (measurement_age < 0.08),
        "measurement_age_80_to_160ms": (
            sequence
            & age_valid
            & (measurement_age >= 0.08)
            & (measurement_age < 0.16)
        ),
        "measurement_age_160_to_300ms": (
            sequence
            & age_valid
            & (measurement_age >= 0.16)
            & (measurement_age < 0.30)
        ),
        "measurement_age_ge_300ms": sequence & age_valid & (measurement_age >= 0.30),
        "measurement_age_invalid": sequence & ~age_valid,
        "detection_gap_lt_150ms": sequence & (detection_gap < 0.15),
        "detection_gap_150_to_650ms": (
            sequence & (detection_gap >= 0.15) & (detection_gap < 0.65)
        ),
        "detection_gap_ge_650ms": sequence & np.isfinite(detection_gap) & (detection_gap >= 0.65),
        "no_valid_detection_history": sequence & ~np.isfinite(detection_gap),
    }
    return masks


def _stratified_metrics(
    *,
    mean: np.ndarray,
    uncalibrated_std: np.ndarray,
    calibrated_std: np.ndarray,
    dataset: GimbalTargetStateDataset,
    possible_mask: np.ndarray,
    profile: ObservationProfile,
) -> dict[str, Any]:
    result = {}
    for name, context_mask in _context_masks(dataset, profile).items():
        mask = possible_mask & context_mask[..., None]
        if not mask.any():
            continue
        result[name] = {
            "uncalibrated": _metrics(
                mean=mean,
                std=uncalibrated_std,
                dataset=dataset,
                mask=mask,
            ),
            "calibrated": _metrics(
                mean=mean,
                std=calibrated_std,
                dataset=dataset,
                mask=mask,
            ),
        }
    return result


def load_uncertainty_calibration(
    path: str | Path,
) -> UncertaintyCalibration:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("experiment") not in {
        CALIBRATION_SCHEMA_VERSION,
        CONTEXTUAL_CALIBRATION_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported uncertainty calibration result")
    calibration = result.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("calibration result has no calibration payload")
    return uncertainty_calibration_from_dict(calibration)


def calibrate_gru_uncertainty(
    *,
    validation_data: str | Path,
    test_data: str | Path,
    checkpoint: str | Path,
    config: CalibrationExperimentConfig | None = None,
) -> dict[str, Any]:
    """Fit on validation predictions and evaluate once on the test split."""
    config = config or CalibrationExperimentConfig()
    validation = load_gimbal_dataset(validation_data)
    test = load_gimbal_dataset(test_data)
    validate_disjoint_seed_blocks((validation.manifest, test.manifest))
    if validation.manifest.feature_names != test.manifest.feature_names:
        raise ValueError("validation and test feature schemas differ")
    if (
        validation.manifest.prediction_horizons_s
        != test.manifest.prediction_horizons_s
    ):
        raise ValueError("validation and test horizons differ")
    model, metadata = load_gru_checkpoint(checkpoint, device=config.device)
    if metadata.get("profile") != config.profile.value:
        raise ValueError("checkpoint profile does not match calibration profile")
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema does not match calibration")
    dataset_hashes = metadata.get("dataset_hashes")
    if not isinstance(dataset_hashes, dict):
        raise ValueError("checkpoint is missing dataset hashes")
    if dataset_hashes.get("validation") != validation.manifest.configuration_hash:
        raise ValueError("checkpoint validation dataset hash mismatch")
    if dataset_hashes.get("test") != test.manifest.configuration_hash:
        raise ValueError("checkpoint test dataset hash mismatch")
    horizons = validation.manifest.prediction_horizons_s
    if model.config.prediction_horizons_s != horizons:
        raise ValueError("checkpoint and dataset horizons differ")

    validation_mean, validation_std, validation_mask = _gru_predictions(
        model,
        validation,
        config.profile,
        batch_size=config.batch_size,
        device=config.device,
    )
    bearing_scale, rate_scale, sample_counts = _fit_scales(
        validation_mean,
        validation_std,
        validation.targets,
        validation_mask,
        minimum_scale=config.minimum_scale,
        maximum_scale=config.maximum_scale,
    )
    calibration = GaussianUncertaintyCalibration(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        profile=config.profile,
        prediction_horizons_s=horizons,
        bearing_std_scale=bearing_scale,
        rate_std_scale=rate_scale,
        validation_dataset_hash=validation.manifest.configuration_hash,
        test_dataset_hash=test.manifest.configuration_hash,
        checkpoint_sha256=_sha256(checkpoint),
        minimum_scale=config.minimum_scale,
        maximum_scale=config.maximum_scale,
    )
    test_mean, test_std, test_mask = _gru_predictions(
        model,
        test,
        config.profile,
        batch_size=config.batch_size,
        device=config.device,
    )

    def split_result(
        dataset: GimbalTargetStateDataset,
        mean: np.ndarray,
        std: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, Any]:
        calibrated_std = apply_uncertainty_calibration(std, calibration)
        return {
            "uncalibrated": _metrics(
                mean=mean, std=std, dataset=dataset, mask=mask
            ),
            "calibrated": _metrics(
                mean=mean,
                std=calibrated_std,
                dataset=dataset,
                mask=mask,
            ),
            "reliability": {
                "uncalibrated": _reliability(
                    mean=mean,
                    std=std,
                    targets=dataset.targets,
                    mask=mask,
                    nominal_coverages=config.nominal_coverages,
                ),
                "calibrated": _reliability(
                    mean=mean,
                    std=calibrated_std,
                    targets=dataset.targets,
                    mask=mask,
                    nominal_coverages=config.nominal_coverages,
                ),
            },
            "strata": _stratified_metrics(
                mean=mean,
                uncalibrated_std=std,
                calibrated_std=calibrated_std,
                dataset=dataset,
                possible_mask=mask,
                profile=config.profile,
            ),
        }

    return {
        "experiment": CALIBRATION_SCHEMA_VERSION,
        "method": "per_horizon_closed_form_gaussian_std_scaling",
        "fit_split": "validation",
        "evaluation_split": "test",
        "config": asdict(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": calibration.checkpoint_sha256,
        "dataset_hashes": dataset_hashes,
        "fit_sample_count_per_horizon": list(sample_counts),
        "calibration": asdict(calibration),
        "validation": split_result(
            validation, validation_mean, validation_std, validation_mask
        ),
        "test": split_result(test, test_mean, test_std, test_mask),
    }


def _stratified_variant_metrics(
    *,
    mean: np.ndarray,
    std_by_variant: dict[str, np.ndarray],
    dataset: GimbalTargetStateDataset,
    possible_mask: np.ndarray,
    profile: ObservationProfile,
) -> dict[str, Any]:
    result = {}
    for name, context_mask in _context_masks(dataset, profile).items():
        mask = possible_mask & context_mask[..., None]
        if not mask.any():
            continue
        result[name] = {
            variant: _metrics(
                mean=mean,
                std=std,
                dataset=dataset,
                mask=mask,
            )
            for variant, std in std_by_variant.items()
        }
    return result


def calibrate_contextual_gru_uncertainty(
    *,
    validation_data: str | Path,
    test_data: str | Path,
    checkpoint: str | Path,
    config: ContextualCalibrationExperimentConfig | None = None,
) -> dict[str, Any]:
    """Fit context-conditioned scales on validation and evaluate test once."""
    config = config or ContextualCalibrationExperimentConfig()
    validation = load_gimbal_dataset(validation_data)
    test = load_gimbal_dataset(test_data)
    validate_disjoint_seed_blocks((validation.manifest, test.manifest))
    if validation.manifest.feature_names != test.manifest.feature_names:
        raise ValueError("validation and test feature schemas differ")
    if (
        validation.manifest.prediction_horizons_s
        != test.manifest.prediction_horizons_s
    ):
        raise ValueError("validation and test horizons differ")
    model, metadata = load_gru_checkpoint(checkpoint, device=config.device)
    if metadata.get("profile") != config.profile.value:
        raise ValueError("checkpoint profile does not match calibration profile")
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema does not match calibration")
    dataset_hashes = metadata.get("dataset_hashes")
    if not isinstance(dataset_hashes, dict):
        raise ValueError("checkpoint is missing dataset hashes")
    if dataset_hashes.get("validation") != validation.manifest.configuration_hash:
        raise ValueError("checkpoint validation dataset hash mismatch")
    if dataset_hashes.get("test") != test.manifest.configuration_hash:
        raise ValueError("checkpoint test dataset hash mismatch")
    horizons = validation.manifest.prediction_horizons_s
    if model.config.prediction_horizons_s != horizons:
        raise ValueError("checkpoint and dataset horizons differ")

    validation_mean, validation_std, validation_mask = _gru_predictions(
        model,
        validation,
        config.profile,
        batch_size=config.batch_size,
        device=config.device,
    )
    global_bearing, global_rate, global_counts = _fit_scales(
        validation_mean,
        validation_std,
        validation.targets,
        validation_mask,
        minimum_scale=config.minimum_scale,
        maximum_scale=config.maximum_scale,
    )
    validation_context = _context_index_array(
        validation,
        config.profile,
        measurement_age_threshold_s=config.measurement_age_threshold_s,
        detection_gap_thresholds_s=config.detection_gap_thresholds_s,
    )
    bearing_table, rate_table, context_counts = _fit_contextual_scales(
        validation_mean,
        validation_std,
        validation.targets,
        validation_mask,
        validation_context,
        global_bearing_scale=global_bearing,
        global_rate_scale=global_rate,
        prior_strength=config.prior_strength,
        minimum_scale=config.minimum_scale,
        maximum_scale=config.maximum_scale,
    )
    checkpoint_sha256 = _sha256(checkpoint)
    calibration = ContextualGaussianUncertaintyCalibration(
        schema_version=CONTEXTUAL_CALIBRATION_SCHEMA_VERSION,
        profile=config.profile,
        prediction_horizons_s=horizons,
        context_names=CONTEXT_NAMES,
        bearing_std_scale_by_context=bearing_table,
        rate_std_scale_by_context=rate_table,
        measurement_age_threshold_s=config.measurement_age_threshold_s,
        detection_gap_thresholds_s=config.detection_gap_thresholds_s,
        prior_strength=config.prior_strength,
        validation_dataset_hash=validation.manifest.configuration_hash,
        test_dataset_hash=test.manifest.configuration_hash,
        checkpoint_sha256=checkpoint_sha256,
        minimum_scale=config.minimum_scale,
        maximum_scale=config.maximum_scale,
    )
    global_calibration = GaussianUncertaintyCalibration(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        profile=config.profile,
        prediction_horizons_s=horizons,
        bearing_std_scale=global_bearing,
        rate_std_scale=global_rate,
        validation_dataset_hash=validation.manifest.configuration_hash,
        test_dataset_hash=test.manifest.configuration_hash,
        checkpoint_sha256=checkpoint_sha256,
        minimum_scale=config.minimum_scale,
        maximum_scale=config.maximum_scale,
    )
    test_mean, test_std, test_mask = _gru_predictions(
        model,
        test,
        config.profile,
        batch_size=config.batch_size,
        device=config.device,
    )

    def split_result(
        dataset: GimbalTargetStateDataset,
        mean: np.ndarray,
        std: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, Any]:
        global_std = apply_uncertainty_calibration(std, global_calibration)
        contextual_std = apply_contextual_uncertainty_calibration(
            std, dataset, calibration
        )
        variants = {
            "uncalibrated": std,
            "global": global_std,
            "contextual": contextual_std,
        }
        return {
            **{
                name: _metrics(
                    mean=mean,
                    std=variant_std,
                    dataset=dataset,
                    mask=mask,
                )
                for name, variant_std in variants.items()
            },
            "reliability": {
                name: _reliability(
                    mean=mean,
                    std=variant_std,
                    targets=dataset.targets,
                    mask=mask,
                    nominal_coverages=config.nominal_coverages,
                )
                for name, variant_std in variants.items()
            },
            "strata": _stratified_variant_metrics(
                mean=mean,
                std_by_variant=variants,
                dataset=dataset,
                possible_mask=mask,
                profile=config.profile,
            ),
        }

    return {
        "experiment": CONTEXTUAL_CALIBRATION_SCHEMA_VERSION,
        "method": "shrinkage_contextual_gaussian_std_scaling",
        "fit_split": "validation",
        "evaluation_split": "test",
        "config": asdict(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_hashes": dataset_hashes,
        "global_fit_sample_count_per_horizon": list(global_counts),
        "context_fit_sample_count_per_horizon": {
            name: list(counts)
            for name, counts in zip(
                CONTEXT_NAMES, context_counts, strict=True
            )
        },
        "global_calibration": asdict(global_calibration),
        "calibration": asdict(calibration),
        "validation": split_result(
            validation, validation_mean, validation_std, validation_mask
        ),
        "test": split_result(test, test_mean, test_std, test_mask),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate causal gimbal GRU uncertainty on validation data."
    )
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in ObservationProfile],
        default=ObservationProfile.DISTURBANCE_AWARE.value,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-scale", type=float, default=0.25)
    parser.add_argument("--maximum-scale", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = calibrate_gru_uncertainty(
        validation_data=args.validation_data,
        test_data=args.test_data,
        checkpoint=args.checkpoint,
        config=CalibrationExperimentConfig(
            profile=ObservationProfile(args.profile),
            batch_size=args.batch_size,
            minimum_scale=args.minimum_scale,
            maximum_scale=args.maximum_scale,
            device=args.device,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "bearing_std_scale": result["calibration"]["bearing_std_scale"],
        "rate_std_scale": result["calibration"]["rate_std_scale"],
        "test_uncalibrated": result["test"]["uncalibrated"],
        "test_calibrated": result["test"]["calibrated"],
    }
    print(json.dumps(summary, indent=2))


def _parse_contextual_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit deployable context-aware gimbal GRU uncertainty scaling."
        )
    )
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--measurement-age-threshold-s", type=float, default=0.16)
    parser.add_argument("--short-gap-threshold-s", type=float, default=0.15)
    parser.add_argument("--medium-gap-threshold-s", type=float, default=0.65)
    parser.add_argument("--prior-strength", type=float, default=512.0)
    parser.add_argument("--minimum-scale", type=float, default=0.25)
    parser.add_argument("--maximum-scale", type=float, default=4.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def contextual_main(argv: Sequence[str] | None = None) -> None:
    args = _parse_contextual_args(argv)
    result = calibrate_contextual_gru_uncertainty(
        validation_data=args.validation_data,
        test_data=args.test_data,
        checkpoint=args.checkpoint,
        config=ContextualCalibrationExperimentConfig(
            batch_size=args.batch_size,
            measurement_age_threshold_s=args.measurement_age_threshold_s,
            detection_gap_thresholds_s=(
                args.short_gap_threshold_s,
                args.medium_gap_threshold_s,
            ),
            prior_strength=args.prior_strength,
            minimum_scale=args.minimum_scale,
            maximum_scale=args.maximum_scale,
            device=args.device,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "contexts": result["calibration"]["context_names"],
                "test_uncalibrated": result["test"]["uncalibrated"],
                "test_global": result["test"]["global"],
                "test_contextual": result["test"]["contextual"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
