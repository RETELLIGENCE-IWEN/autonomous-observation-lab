"""Training, evaluation, and analytical comparison for the gimbal GRU."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .adaptive_position_supervision import (
    AdaptivePositionSupervision,
    compute_adaptive_position_supervision,
)
from .config import GimbalCommandMode, ObservationProfile
from .control_supervision import (
    ControlActionSupervision,
    compute_control_action_supervision,
)
from .dataset import (
    FEATURE_NAMES,
    TARGET_NAMES,
    GimbalTargetStateDataset,
    load_gimbal_dataset,
    validate_disjoint_seed_blocks,
)
from .estimators import (
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
)
from .gru import (
    CausalTargetStateGRU,
    GRUAdaptivePositionLossContext,
    GRUControlLossContext,
    GRULossConfig,
    GRUTargetStateModelConfig,
    adaptive_position_surrogate_actions,
    angular_residual_rad,
    differentiable_position_servo_rollout,
    gru_parameter_count,
    save_gru_checkpoint,
    target_state_nll,
)
from .types import GimbalObservation, MaskedScalar


@dataclass(frozen=True)
class GRUTrainingConfig:
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning rate must be positive and decay non-negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")
        if self.seed < 0:
            raise ValueError("training seed must be non-negative")


@dataclass(frozen=True)
class GRUReferenceAnchorConfig:
    """Function-space trust region around the starting predictor.

    The anchor is a training-only regularizer. Its weights use the same
    physical units as the bearing/rate mean-squared-error terms, and the
    caller selects which labels are anchored independently of supervision
    weights.
    """

    bearing_weight: float = 0.0
    rate_weight: float = 0.0
    project_conflicting_gradients: bool = False
    projection_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("bearing_weight", "rate_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.projection_epsilon) or (
            self.projection_epsilon <= 0.0
        ):
            raise ValueError("projection epsilon must be finite and positive")

    @property
    def active(self) -> bool:
        return self.bearing_weight > 0.0 or self.rate_weight > 0.0


@dataclass(frozen=True)
class HorizonMetrics:
    horizon_s: float
    valid_samples: int
    availability_fraction: float
    bearing_rmse_deg: float | None
    rate_rmse_deg_s: float | None
    bearing_nll: float | None
    rate_nll: float | None
    bearing_one_sigma_coverage: float | None
    bearing_two_sigma_coverage: float | None
    rate_one_sigma_coverage: float | None
    rate_two_sigma_coverage: float | None


@dataclass(frozen=True)
class GRUEvaluationMetrics:
    loss: float | None
    valid_samples: int
    availability_fraction: float
    bearing_rmse_deg: float | None
    rate_rmse_deg_s: float | None
    bearing_nll: float | None
    rate_nll: float | None
    bearing_one_sigma_coverage: float | None
    bearing_two_sigma_coverage: float | None
    rate_one_sigma_coverage: float | None
    rate_two_sigma_coverage: float | None
    dynamic_consistency_rmse_deg: float | None
    rate_action_rmse_normalized: float | None
    position_action_rmse_normalized: float | None
    adaptive_position_action_rmse_normalized: float | None
    position_plant_tracking_rmse_normalized: float | None
    position_plant_response_rmse_normalized: float | None
    position_plant_regret_rmse_normalized: float | None
    position_plant_visibility_rmse_normalized: float | None
    position_plant_smoothness_rmse_normalized: float | None
    position_plant_saturation_rmse_normalized: float | None
    per_horizon: tuple[HorizonMetrics, ...]


@dataclass(frozen=True)
class GRUEpochRecord:
    epoch: int
    training_loss: float
    validation_loss: float
    validation_bearing_rmse_deg: float | None
    validation_rate_rmse_deg_s: float | None
    training_reference_anchor_loss: float = 0.0
    reference_anchor_conflict_fraction: float = 0.0


@dataclass
class GRUTrainingResult:
    model: CausalTargetStateGRU
    history: tuple[GRUEpochRecord, ...]
    best_epoch: int
    initial_validation: GRUEvaluationMetrics
    best_validation: GRUEvaluationMetrics
    epoch_state_dicts: tuple[dict[str, torch.Tensor], ...] = ()


class GimbalTorchSequenceDataset(Dataset):
    """One selected deployment profile from a multi-profile dataset."""

    def __init__(
        self,
        dataset: GimbalTargetStateDataset,
        profile: ObservationProfile,
        label_weights: np.ndarray | None = None,
        control_supervision: ControlActionSupervision | None = None,
        adaptive_position_supervision: AdaptivePositionSupervision | None = None,
        reference_anchor_weights: np.ndarray | None = None,
    ):
        try:
            self.profile_index = dataset.manifest.observation_profiles.index(
                profile.value
            )
        except ValueError as error:
            raise ValueError(
                f"profile {profile.value!r} is absent from the dataset"
            ) from error
        self.dataset = dataset
        if label_weights is not None:
            if label_weights.shape != dataset.target_mask.shape:
                raise ValueError("label weights do not match target masks")
            if np.any(~np.isfinite(label_weights)) or np.any(
                label_weights < 0.0
            ):
                raise ValueError(
                    "label weights must be finite and non-negative"
                )
        self.label_weights = label_weights
        if reference_anchor_weights is not None:
            if reference_anchor_weights.shape != dataset.target_mask.shape:
                raise ValueError(
                    "reference anchor weights do not match target masks"
                )
            if np.any(~np.isfinite(reference_anchor_weights)) or np.any(
                reference_anchor_weights < 0.0
            ):
                raise ValueError(
                    "reference anchor weights must be finite and non-negative"
                )
        self.reference_anchor_weights = reference_anchor_weights
        if control_supervision is not None:
            expected_shape = dataset.sequence_mask.shape
            for value in (
                control_supervision.gimbal_angle_rad,
                control_supervision.servo_max_rate_rad_s,
                control_supervision.servo_min_angle_rad,
                control_supervision.servo_max_angle_rad,
                control_supervision.rate_feedback_gain_s_inv,
                control_supervision.position_preview_s,
                control_supervision.mask,
            ):
                if value.shape != expected_shape:
                    raise ValueError("control supervision shape is invalid")
            if control_supervision.oracle_actions.shape != (*expected_shape, 2):
                raise ValueError("oracle action supervision shape is invalid")
        self.control_supervision = control_supervision
        if adaptive_position_supervision is not None:
            expected_shape = dataset.sequence_mask.shape
            for value in (
                adaptive_position_supervision.teacher_action_normalized,
                adaptive_position_supervision.mask,
                adaptive_position_supervision.gimbal_angle_rad,
                adaptive_position_supervision.gimbal_rate_rad_s,
                adaptive_position_supervision.control_dt_s,
                adaptive_position_supervision.selected_axis_fov_rad,
                adaptive_position_supervision.servo_min_angle_rad,
                adaptive_position_supervision.servo_max_angle_rad,
                adaptive_position_supervision.servo_max_rate_rad_s,
                adaptive_position_supervision.servo_max_acceleration_rad_s2,
                adaptive_position_supervision.servo_position_gain_s_inv,
                adaptive_position_supervision.servo_position_tolerance_rad,
                adaptive_position_supervision.servo_position_quantization_rad,
                adaptive_position_supervision.servo_command_polarity,
                adaptive_position_supervision.servo_command_latency_s,
                adaptive_position_supervision.servo_rate_time_constant_s,
                adaptive_position_supervision.control_period_s,
                adaptive_position_supervision.integration_period_s,
                adaptive_position_supervision.camera_frame_period_s,
            ):
                if value.shape != expected_shape:
                    raise ValueError(
                        "adaptive position supervision shape is invalid"
                    )
        self.adaptive_position_supervision = adaptive_position_supervision

    def __len__(self) -> int:
        return self.dataset.episode_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "features": torch.from_numpy(
                self.dataset.features[index, self.profile_index]
            ).float(),
            "sequence_mask": torch.from_numpy(
                self.dataset.sequence_mask[index]
            ).bool(),
            "targets": torch.from_numpy(self.dataset.targets[index]).float(),
            "target_mask": torch.from_numpy(
                self.dataset.target_mask[index]
            ).bool(),
        }
        if self.label_weights is not None:
            item["label_weights"] = torch.from_numpy(
                self.label_weights[index]
            ).float()
        if self.reference_anchor_weights is not None:
            item["reference_anchor_weights"] = torch.from_numpy(
                self.reference_anchor_weights[index]
            ).float()
        if self.control_supervision is not None:
            supervision = self.control_supervision
            item.update(
                {
                    "oracle_actions": torch.from_numpy(
                        supervision.oracle_actions[index]
                    ).float(),
                    "control_gimbal_angle_rad": torch.from_numpy(
                        supervision.gimbal_angle_rad[index]
                    ).float(),
                    "control_servo_max_rate_rad_s": torch.from_numpy(
                        supervision.servo_max_rate_rad_s[index]
                    ).float(),
                    "control_servo_min_angle_rad": torch.from_numpy(
                        supervision.servo_min_angle_rad[index]
                    ).float(),
                    "control_servo_max_angle_rad": torch.from_numpy(
                        supervision.servo_max_angle_rad[index]
                    ).float(),
                    "control_rate_feedback_gain_s_inv": torch.from_numpy(
                        supervision.rate_feedback_gain_s_inv[index]
                    ).float(),
                    "control_position_preview_s": torch.from_numpy(
                        supervision.position_preview_s[index]
                    ).float(),
                    "control_supervision_mask": torch.from_numpy(
                        supervision.mask[index]
                    ).bool(),
                }
            )
        if self.adaptive_position_supervision is not None:
            supervision = self.adaptive_position_supervision
            names = (
                "teacher_action_normalized",
                "mask",
                "gimbal_angle_rad",
                "gimbal_rate_rad_s",
                "control_dt_s",
                "selected_axis_fov_rad",
                "servo_min_angle_rad",
                "servo_max_angle_rad",
                "servo_max_rate_rad_s",
                "servo_max_acceleration_rad_s2",
                "servo_position_gain_s_inv",
                "servo_position_tolerance_rad",
                "servo_position_quantization_rad",
                "servo_command_polarity",
                "servo_command_latency_s",
                "servo_rate_time_constant_s",
                "control_period_s",
                "integration_period_s",
                "camera_frame_period_s",
            )
            for name in names:
                value = torch.from_numpy(getattr(supervision, name)[index])
                item[f"adaptive_{name}"] = (
                    value.bool() if name == "mask" else value.float()
                )
        return item


def set_gru_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _angle_residual_numpy(
    prediction_rad: np.ndarray, target_rad: np.ndarray
) -> np.ndarray:
    difference = prediction_rad - target_rad
    return np.arctan2(np.sin(difference), np.cos(difference))


def _masked_metric(
    values: np.ndarray, mask: np.ndarray, function
) -> float | None:
    selected = values[mask]
    return float(function(selected)) if selected.size else None


def _metrics_from_predictions(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    prediction_mask: np.ndarray,
    possible_mask: np.ndarray,
    horizons_s: tuple[float, ...],
    loss_config: GRULossConfig | None = None,
    label_weights: np.ndarray | None = None,
    control_supervision: ControlActionSupervision | None = None,
    interval_rate_rad_s: np.ndarray | None = None,
    adaptive_position_predictions: np.ndarray | None = None,
    adaptive_position_teacher: np.ndarray | None = None,
    adaptive_position_mask: np.ndarray | None = None,
    position_plant_tracking_error_normalized: np.ndarray | None = None,
    position_plant_response_error_normalized: np.ndarray | None = None,
    position_plant_regret_fraction: np.ndarray | None = None,
    position_plant_visibility_violation_normalized: np.ndarray | None = None,
    position_plant_saturation_fraction: np.ndarray | None = None,
    position_plant_mask: np.ndarray | None = None,
) -> GRUEvaluationMetrics:
    loss_config = loss_config or GRULossConfig()
    if mean.shape != targets.shape or std.shape != targets.shape:
        raise ValueError("prediction arrays must match target shape")
    if prediction_mask.shape != targets.shape[:-1]:
        raise ValueError("prediction mask shape is invalid")
    if possible_mask.shape != prediction_mask.shape:
        raise ValueError("possible mask shape is invalid")
    mask = prediction_mask & possible_mask
    possible_count = int(possible_mask.sum())
    valid_count = int(mask.sum())
    availability = valid_count / possible_count if possible_count else 0.0
    if label_weights is None:
        weights = np.ones_like(prediction_mask, dtype=np.float64)
    else:
        if label_weights.shape != prediction_mask.shape:
            raise ValueError("label weights shape is invalid")
        if np.any(~np.isfinite(label_weights)) or np.any(label_weights < 0.0):
            raise ValueError("label weights must be finite and non-negative")
        weights = label_weights.astype(np.float64, copy=False)
    if loss_config.horizon_weights:
        if len(loss_config.horizon_weights) != targets.shape[-2]:
            raise ValueError("horizon weights do not match predictions")
        weights = weights * np.asarray(
            loss_config.horizon_weights,
            dtype=np.float64,
        )[None, None, :]

    bearing_error = _angle_residual_numpy(mean[..., 0], targets[..., 0])
    rate_error = mean[..., 1] - targets[..., 1]
    bearing_std = std[..., 0]
    rate_std = std[..., 1]
    bearing_nll_values = np.log(bearing_std) + 0.5 * (
        bearing_error / bearing_std
    ) ** 2
    rate_nll_values = np.log(rate_std) + 0.5 * (
        rate_error / rate_std
    ) ** 2

    bearing_mse = _masked_metric(bearing_error**2, mask, np.mean)
    rate_mse = _masked_metric(rate_error**2, mask, np.mean)
    bearing_nll = _masked_metric(bearing_nll_values, mask, np.mean)
    rate_nll = _masked_metric(rate_nll_values, mask, np.mean)
    weighted_mask = weights * mask
    weight_sum = float(np.sum(weighted_mask))

    def weighted_mean(values: np.ndarray) -> float:
        if weight_sum <= 0.0:
            return 0.0
        return float(np.sum(values * weighted_mask) / weight_sum)

    weighted_bearing_nll = weighted_mean(bearing_nll_values)
    weighted_rate_nll = weighted_mean(rate_nll_values)
    weighted_bearing_mse = weighted_mean(bearing_error**2)
    weighted_rate_mse = weighted_mean(rate_error**2)
    dynamic_consistency_mse = 0.0
    if len(horizons_s) > 1:
        intervals = np.diff(np.asarray(horizons_s, dtype=np.float64))
        bearing_step = _angle_residual_numpy(
            mean[..., 1:, 0],
            mean[..., :-1, 0],
        )
        if interval_rate_rad_s is None:
            integrated_rate = 0.5 * (
                mean[..., 1:, 1] + mean[..., :-1, 1]
            ) * intervals
        else:
            if interval_rate_rad_s.shape != mean.shape[:-2] + (
                mean.shape[-2] - 1,
            ):
                raise ValueError("interval rate prediction shape is invalid")
            integrated_rate = (
                mean[..., :-1, 1]
                + 4.0 * interval_rate_rad_s
                + mean[..., 1:, 1]
            ) * intervals / 6.0
        consistency_error = _angle_residual_numpy(
            bearing_step,
            integrated_rate,
        )
        consistency_mask = mask[..., 1:] & mask[..., :-1]
        pair_weights = 0.5 * (weights[..., 1:] + weights[..., :-1])
        selected_pair_weights = pair_weights * consistency_mask
        pair_weight_sum = float(np.sum(selected_pair_weights))
        if pair_weight_sum > 0.0:
            dynamic_consistency_mse = float(
                np.sum(consistency_error**2 * selected_pair_weights)
                / pair_weight_sum
            )
    rate_action_mse = None
    position_action_mse = None
    adaptive_position_action_mse = None
    position_plant_tracking_mse = None
    position_plant_response_mse = None
    position_plant_regret_mse = None
    position_plant_visibility_mse = None
    position_plant_smoothness_mse = None
    position_plant_saturation_mse = None
    if (
        loss_config.rate_action_weight > 0.0
        or loss_config.position_action_weight > 0.0
    ):
        if control_supervision is None:
            raise ValueError(
                "control supervision is required for action-aware evaluation"
            )
        expected_time_shape = targets.shape[:2]
        if control_supervision.oracle_actions.shape != (*expected_time_shape, 2):
            raise ValueError("oracle action supervision shape is invalid")
        predicted_bearing = mean[..., 0, 0]
        predicted_rate = mean[..., 0, 1]
        bearing_to_gimbal = _angle_residual_numpy(
            predicted_bearing,
            control_supervision.gimbal_angle_rad,
        )
        predicted_rate_action = np.clip(
            (
                predicted_rate
                + control_supervision.rate_feedback_gain_s_inv
                * bearing_to_gimbal
            )
            / control_supervision.servo_max_rate_rad_s,
            -1.0,
            1.0,
        )
        predicted_position = _angle_residual_numpy(
            predicted_bearing
            + control_supervision.position_preview_s * predicted_rate,
            np.zeros_like(predicted_bearing),
        )
        predicted_position = np.clip(
            predicted_position,
            control_supervision.servo_min_angle_rad,
            control_supervision.servo_max_angle_rad,
        )
        predicted_position_action = np.where(
            predicted_position >= 0.0,
            predicted_position / control_supervision.servo_max_angle_rad,
            predicted_position / (-control_supervision.servo_min_angle_rad),
        )
        action_mask = mask[..., 0] & control_supervision.mask
        action_weights = weights[..., 0] * action_mask
        action_weight_sum = float(np.sum(action_weights))
        if action_weight_sum > 0.0:
            rate_action_mse = float(
                np.sum(
                    (
                        predicted_rate_action
                        - control_supervision.oracle_actions[..., 0]
                    )
                    ** 2
                    * action_weights
                )
                / action_weight_sum
            )
            position_action_mse = float(
                np.sum(
                    (
                        predicted_position_action
                        - control_supervision.oracle_actions[..., 1]
                    )
                    ** 2
                    * action_weights
                )
                / action_weight_sum
            )
    if loss_config.adaptive_position_action_weight > 0.0:
        expected_time_shape = targets.shape[:2]
        if (
            adaptive_position_predictions is None
            or adaptive_position_teacher is None
            or adaptive_position_mask is None
        ):
            raise ValueError(
                "adaptive position predictions and teacher are required"
            )
        if (
            adaptive_position_predictions.shape != expected_time_shape
            or adaptive_position_teacher.shape != expected_time_shape
            or adaptive_position_mask.shape != expected_time_shape
        ):
            raise ValueError("adaptive position evaluation shape is invalid")
        action_mask = mask[..., 0] & adaptive_position_mask
        action_weights = weights[..., 0] * action_mask
        action_weight_sum = float(np.sum(action_weights))
        if action_weight_sum > 0.0:
            adaptive_position_action_mse = float(
                np.sum(
                    (
                        adaptive_position_predictions
                        - adaptive_position_teacher
                    )
                    ** 2
                    * action_weights
                )
                / action_weight_sum
            )
    plant_aware = any(
        value > 0.0
        for value in (
            loss_config.position_plant_tracking_weight,
            loss_config.position_plant_response_weight,
            loss_config.position_plant_regret_weight,
            loss_config.position_plant_visibility_weight,
            loss_config.position_plant_smoothness_weight,
            loss_config.position_plant_saturation_weight,
        )
    )
    if plant_aware:
        if loss_config.position_plant_config is None:
            raise ValueError(
                "position plant config is required for plant-aware evaluation"
            )
        expected_time_shape = targets.shape[:2]
        values = (
            position_plant_tracking_error_normalized,
            position_plant_response_error_normalized,
            position_plant_regret_fraction,
            position_plant_visibility_violation_normalized,
            position_plant_saturation_fraction,
            position_plant_mask,
            adaptive_position_predictions,
        )
        if any(value is None for value in values):
            raise ValueError("position plant evaluation arrays are required")
        assert position_plant_tracking_error_normalized is not None
        assert position_plant_response_error_normalized is not None
        assert position_plant_regret_fraction is not None
        assert position_plant_visibility_violation_normalized is not None
        assert position_plant_saturation_fraction is not None
        assert position_plant_mask is not None
        assert adaptive_position_predictions is not None
        if any(value.shape != expected_time_shape for value in values):
            raise ValueError("position plant evaluation shape is invalid")
        horizon_index = loss_config.position_plant_config.horizon_index
        if horizon_index >= targets.shape[-2]:
            raise ValueError("position plant horizon index is out of range")
        rollout_mask = mask[..., horizon_index] & position_plant_mask
        rollout_weights = weights[..., horizon_index] * rollout_mask
        rollout_weight_sum = float(np.sum(rollout_weights))
        if rollout_weight_sum > 0.0:
            position_plant_tracking_mse = float(
                np.sum(
                    position_plant_tracking_error_normalized**2
                    * rollout_weights
                )
                / rollout_weight_sum
            )
            position_plant_response_mse = float(
                np.sum(
                    position_plant_response_error_normalized**2
                    * rollout_weights
                )
                / rollout_weight_sum
            )
            position_plant_regret_mse = float(
                np.sum(position_plant_regret_fraction * rollout_weights)
                / rollout_weight_sum
            )
            position_plant_visibility_mse = float(
                np.sum(
                    position_plant_visibility_violation_normalized**2
                    * rollout_weights
                )
                / rollout_weight_sum
            )
            position_plant_saturation_mse = float(
                np.sum(position_plant_saturation_fraction * rollout_weights)
                / rollout_weight_sum
            )
        smooth_mask = rollout_mask[:, 1:] & rollout_mask[:, :-1]
        smooth_weights = 0.5 * (
            weights[:, 1:, horizon_index]
            + weights[:, :-1, horizon_index]
        ) * smooth_mask
        smooth_weight_sum = float(np.sum(smooth_weights))
        if smooth_weight_sum > 0.0:
            position_plant_smoothness_mse = float(
                np.sum(
                    np.diff(adaptive_position_predictions, axis=1) ** 2
                    * smooth_weights
                )
                / smooth_weight_sum
            )
    if bearing_nll is None or rate_nll is None:
        loss = None
    else:
        assert bearing_mse is not None and rate_mse is not None
        loss = (
            loss_config.bearing_weight * weighted_bearing_nll
            + loss_config.rate_weight * weighted_rate_nll
            + loss_config.mean_error_weight
            * (weighted_bearing_mse + weighted_rate_mse)
            + loss_config.bearing_mean_error_weight
            * weighted_bearing_mse
            + loss_config.rate_mean_error_weight * weighted_rate_mse
            + loss_config.dynamic_consistency_weight
            * dynamic_consistency_mse
            + loss_config.rate_action_weight * (rate_action_mse or 0.0)
            + loss_config.position_action_weight
            * (position_action_mse or 0.0)
            + loss_config.adaptive_position_action_weight
            * (adaptive_position_action_mse or 0.0)
            + loss_config.position_plant_tracking_weight
            * (position_plant_tracking_mse or 0.0)
            + loss_config.position_plant_response_weight
            * (position_plant_response_mse or 0.0)
            + loss_config.position_plant_regret_weight
            * (position_plant_regret_mse or 0.0)
            + loss_config.position_plant_visibility_weight
            * (position_plant_visibility_mse or 0.0)
            + loss_config.position_plant_smoothness_weight
            * (position_plant_smoothness_mse or 0.0)
            + loss_config.position_plant_saturation_weight
            * (position_plant_saturation_mse or 0.0)
        )

    def coverage(error: np.ndarray, sigma: np.ndarray, multiple: float):
        return _masked_metric(
            np.abs(error) <= multiple * sigma,
            mask,
            np.mean,
        )

    per_horizon = []
    for horizon_index, horizon_s in enumerate(horizons_s):
        horizon_mask = mask[..., horizon_index]
        horizon_possible = possible_mask[..., horizon_index]
        horizon_valid_count = int(horizon_mask.sum())
        horizon_possible_count = int(horizon_possible.sum())

        def horizon_metric(values: np.ndarray, function):
            return _masked_metric(
                values[..., horizon_index], horizon_mask, function
            )

        bearing_horizon_mse = horizon_metric(bearing_error**2, np.mean)
        rate_horizon_mse = horizon_metric(rate_error**2, np.mean)
        per_horizon.append(
            HorizonMetrics(
                horizon_s=horizon_s,
                valid_samples=horizon_valid_count,
                availability_fraction=(
                    horizon_valid_count / horizon_possible_count
                    if horizon_possible_count
                    else 0.0
                ),
                bearing_rmse_deg=(
                    math.degrees(math.sqrt(bearing_horizon_mse))
                    if bearing_horizon_mse is not None
                    else None
                ),
                rate_rmse_deg_s=(
                    math.degrees(math.sqrt(rate_horizon_mse))
                    if rate_horizon_mse is not None
                    else None
                ),
                bearing_nll=horizon_metric(bearing_nll_values, np.mean),
                rate_nll=horizon_metric(rate_nll_values, np.mean),
                bearing_one_sigma_coverage=horizon_metric(
                    np.abs(bearing_error) <= bearing_std, np.mean
                ),
                bearing_two_sigma_coverage=horizon_metric(
                    np.abs(bearing_error) <= 2.0 * bearing_std, np.mean
                ),
                rate_one_sigma_coverage=horizon_metric(
                    np.abs(rate_error) <= rate_std, np.mean
                ),
                rate_two_sigma_coverage=horizon_metric(
                    np.abs(rate_error) <= 2.0 * rate_std, np.mean
                ),
            )
        )

    return GRUEvaluationMetrics(
        loss=loss,
        valid_samples=valid_count,
        availability_fraction=availability,
        bearing_rmse_deg=(
            math.degrees(math.sqrt(bearing_mse))
            if bearing_mse is not None
            else None
        ),
        rate_rmse_deg_s=(
            math.degrees(math.sqrt(rate_mse))
            if rate_mse is not None
            else None
        ),
        bearing_nll=bearing_nll,
        rate_nll=rate_nll,
        bearing_one_sigma_coverage=coverage(
            bearing_error, bearing_std, 1.0
        ),
        bearing_two_sigma_coverage=coverage(
            bearing_error, bearing_std, 2.0
        ),
        rate_one_sigma_coverage=coverage(rate_error, rate_std, 1.0),
        rate_two_sigma_coverage=coverage(rate_error, rate_std, 2.0),
        dynamic_consistency_rmse_deg=(
            math.degrees(math.sqrt(dynamic_consistency_mse))
            if valid_count
            else None
        ),
        rate_action_rmse_normalized=(
            math.sqrt(rate_action_mse)
            if rate_action_mse is not None
            else None
        ),
        position_action_rmse_normalized=(
            math.sqrt(position_action_mse)
            if position_action_mse is not None
            else None
        ),
        adaptive_position_action_rmse_normalized=(
            math.sqrt(adaptive_position_action_mse)
            if adaptive_position_action_mse is not None
            else None
        ),
        position_plant_tracking_rmse_normalized=(
            math.sqrt(position_plant_tracking_mse)
            if position_plant_tracking_mse is not None
            else None
        ),
        position_plant_response_rmse_normalized=(
            math.sqrt(position_plant_response_mse)
            if position_plant_response_mse is not None
            else None
        ),
        position_plant_regret_rmse_normalized=(
            math.sqrt(position_plant_regret_mse)
            if position_plant_regret_mse is not None
            else None
        ),
        position_plant_visibility_rmse_normalized=(
            math.sqrt(position_plant_visibility_mse)
            if position_plant_visibility_mse is not None
            else None
        ),
        position_plant_smoothness_rmse_normalized=(
            math.sqrt(position_plant_smoothness_mse)
            if position_plant_smoothness_mse is not None
            else None
        ),
        position_plant_saturation_rmse_normalized=(
            math.sqrt(position_plant_saturation_mse)
            if position_plant_saturation_mse is not None
            else None
        ),
        per_horizon=tuple(per_horizon),
    )


@torch.no_grad()
def evaluate_gru(
    model: CausalTargetStateGRU,
    dataset: GimbalTargetStateDataset,
    profile: ObservationProfile,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    loss_config: GRULossConfig | None = None,
    evaluation_mask: np.ndarray | None = None,
    label_weights: np.ndarray | None = None,
) -> GRUEvaluationMetrics:
    loss_config = loss_config or GRULossConfig()
    action_aware = (
        loss_config.rate_action_weight > 0.0
        or loss_config.position_action_weight > 0.0
    )
    adaptive_aware = (
        loss_config.adaptive_position_action_weight > 0.0
        or any(
            weight > 0.0
            for weight in (
                loss_config.position_plant_tracking_weight,
                loss_config.position_plant_response_weight,
                loss_config.position_plant_regret_weight,
                loss_config.position_plant_visibility_weight,
                loss_config.position_plant_smoothness_weight,
                loss_config.position_plant_saturation_weight,
            )
        )
    )
    plant_aware = any(
        weight > 0.0
        for weight in (
            loss_config.position_plant_tracking_weight,
            loss_config.position_plant_response_weight,
            loss_config.position_plant_regret_weight,
            loss_config.position_plant_visibility_weight,
            loss_config.position_plant_smoothness_weight,
            loss_config.position_plant_saturation_weight,
        )
    )
    if plant_aware and loss_config.position_plant_config is None:
        raise ValueError(
            "position plant config is required for plant-aware evaluation"
        )
    if adaptive_aware and loss_config.adaptive_position_config is None:
        raise ValueError(
            "adaptive position config is required for adaptive evaluation"
        )
    control_supervision = (
        compute_control_action_supervision(dataset, profile=profile)
        if action_aware
        else None
    )
    adaptive_position_supervision = (
        compute_adaptive_position_supervision(
            dataset,
            adapter=loss_config.adaptive_position_config,
            profile=profile,
        )
        if adaptive_aware and loss_config.adaptive_position_config is not None
        else None
    )
    model.to(device)
    model.eval()
    loader = DataLoader(
        GimbalTorchSequenceDataset(
            dataset,
            profile,
            label_weights=label_weights,
            adaptive_position_supervision=adaptive_position_supervision,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    means, stds, targets, prediction_masks, possible_masks = [], [], [], [], []
    interval_rates = []
    collected_weights = []
    adaptive_predictions = []
    adaptive_teachers = []
    adaptive_masks = []
    plant_tracking_errors = []
    plant_response_errors = []
    plant_regret_fractions = []
    plant_visibility_violations = []
    plant_saturation_fractions = []
    plant_masks = []
    for batch in loader:
        features = batch["features"].to(device)
        output = model(features)
        if adaptive_aware:
            assert loss_config.adaptive_position_config is not None
            adaptive_context = _adaptive_position_context_from_batch(
                batch,
                device,
            )
            predicted_actions = adaptive_position_surrogate_actions(
                output,
                adaptive_context,
                dataset.manifest.prediction_horizons_s,
                loss_config.adaptive_position_config,
                batch["sequence_mask"].to(device),
            )
            adaptive_predictions.append(predicted_actions.cpu().numpy())
            adaptive_teachers.append(
                batch["adaptive_teacher_action_normalized"].numpy()
            )
            adaptive_masks.append(batch["adaptive_mask"].numpy())
            if plant_aware:
                assert loss_config.position_plant_config is not None
                horizon_index = (
                    loss_config.position_plant_config.horizon_index
                )
                if horizon_index >= len(
                    dataset.manifest.prediction_horizons_s
                ):
                    raise ValueError(
                        "position plant horizon index is out of range"
                    )
                rollout = differentiable_position_servo_rollout(
                    predicted_actions,
                    adaptive_context,
                    duration_s=dataset.manifest.prediction_horizons_s[
                        horizon_index
                    ],
                    integration_period_override_s=(
                        loss_config.position_plant_config.
                        integration_period_override_s
                    ),
                )
                future_bearing = batch["targets"][
                    ..., horizon_index, 0
                ].to(device)
                tracking_error = angular_residual_rad(
                    future_bearing,
                    rollout.angle_rad,
                ) / (0.5 * adaptive_context.selected_axis_fov_rad)
                plant_tracking_errors.append(tracking_error.cpu().numpy())
                teacher_rollout = differentiable_position_servo_rollout(
                    batch["adaptive_teacher_action_normalized"].to(device),
                    adaptive_context,
                    duration_s=dataset.manifest.prediction_horizons_s[
                        horizon_index
                    ],
                    integration_period_override_s=(
                        loss_config.position_plant_config.
                        integration_period_override_s
                    ),
                )
                plant_response_errors.append(
                    (
                        angular_residual_rad(
                            rollout.angle_rad,
                            teacher_rollout.angle_rad,
                        )
                        / (0.5 * adaptive_context.selected_axis_fov_rad)
                    ).cpu().numpy()
                )
                teacher_tracking_error = angular_residual_rad(
                    future_bearing,
                    teacher_rollout.angle_rad,
                ) / (0.5 * adaptive_context.selected_axis_fov_rad)
                plant_regret_fractions.append(
                    torch.relu(
                        tracking_error.square()
                        - teacher_tracking_error.square()
                    ).cpu().numpy()
                )
                plant_visibility_violations.append(
                    torch.relu(
                        torch.abs(tracking_error)
                        - loss_config.position_plant_config.
                        visibility_margin_fraction
                    ).cpu().numpy()
                )
                plant_saturation_fractions.append(
                    rollout.saturation_fraction.cpu().numpy()
                )
                plant_masks.append(batch["adaptive_mask"].numpy())
        means.append(output.mean.cpu().numpy())
        stds.append(output.std.cpu().numpy())
        if output.interval_rate_rad_s is not None:
            interval_rates.append(output.interval_rate_rad_s.cpu().numpy())
        targets.append(batch["targets"].numpy())
        sequence_mask = batch["sequence_mask"].numpy()
        target_mask = batch["target_mask"].numpy()
        possible = target_mask & sequence_mask[..., None]
        prediction_masks.append(possible.copy())
        possible_masks.append(possible)
        if "label_weights" in batch:
            collected_weights.append(batch["label_weights"].numpy())
    prediction_mask = np.concatenate(prediction_masks)
    possible_mask = np.concatenate(possible_masks)
    if evaluation_mask is not None:
        if evaluation_mask.shape != prediction_mask.shape:
            raise ValueError("evaluation mask shape is invalid")
        prediction_mask &= evaluation_mask
    return _metrics_from_predictions(
        mean=np.concatenate(means),
        std=np.concatenate(stds),
        targets=np.concatenate(targets),
        prediction_mask=prediction_mask,
        possible_mask=possible_mask,
        horizons_s=dataset.manifest.prediction_horizons_s,
        loss_config=loss_config,
        label_weights=(
            np.concatenate(collected_weights) if collected_weights else None
        ),
        control_supervision=control_supervision,
        interval_rate_rad_s=(
            np.concatenate(interval_rates) if interval_rates else None
        ),
        adaptive_position_predictions=(
            np.concatenate(adaptive_predictions)
            if adaptive_predictions
            else None
        ),
        adaptive_position_teacher=(
            np.concatenate(adaptive_teachers) if adaptive_teachers else None
        ),
        adaptive_position_mask=(
            np.concatenate(adaptive_masks) if adaptive_masks else None
        ),
        position_plant_tracking_error_normalized=(
            np.concatenate(plant_tracking_errors)
            if plant_tracking_errors
            else None
        ),
        position_plant_response_error_normalized=(
            np.concatenate(plant_response_errors)
            if plant_response_errors
            else None
        ),
        position_plant_regret_fraction=(
            np.concatenate(plant_regret_fractions)
            if plant_regret_fractions
            else None
        ),
        position_plant_visibility_violation_normalized=(
            np.concatenate(plant_visibility_violations)
            if plant_visibility_violations
            else None
        ),
        position_plant_saturation_fraction=(
            np.concatenate(plant_saturation_fractions)
            if plant_saturation_fractions
            else None
        ),
        position_plant_mask=(
            np.concatenate(plant_masks) if plant_masks else None
        ),
    )


def _control_context_from_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> GRUControlLossContext:
    return GRUControlLossContext(
        oracle_actions=batch["oracle_actions"].to(device),
        gimbal_angle_rad=batch["control_gimbal_angle_rad"].to(device),
        servo_max_rate_rad_s=batch[
            "control_servo_max_rate_rad_s"
        ].to(device),
        servo_min_angle_rad=batch[
            "control_servo_min_angle_rad"
        ].to(device),
        servo_max_angle_rad=batch[
            "control_servo_max_angle_rad"
        ].to(device),
        rate_feedback_gain_s_inv=batch[
            "control_rate_feedback_gain_s_inv"
        ].to(device),
        position_preview_s=batch["control_position_preview_s"].to(device),
        mask=batch["control_supervision_mask"].to(device),
    )


def _adaptive_position_context_from_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> GRUAdaptivePositionLossContext:
    return GRUAdaptivePositionLossContext(
        teacher_action_normalized=batch[
            "adaptive_teacher_action_normalized"
        ].to(device),
        mask=batch["adaptive_mask"].to(device),
        gimbal_angle_rad=batch["adaptive_gimbal_angle_rad"].to(device),
        gimbal_rate_rad_s=batch["adaptive_gimbal_rate_rad_s"].to(device),
        control_dt_s=batch["adaptive_control_dt_s"].to(device),
        selected_axis_fov_rad=batch[
            "adaptive_selected_axis_fov_rad"
        ].to(device),
        servo_min_angle_rad=batch["adaptive_servo_min_angle_rad"].to(device),
        servo_max_angle_rad=batch["adaptive_servo_max_angle_rad"].to(device),
        servo_max_rate_rad_s=batch[
            "adaptive_servo_max_rate_rad_s"
        ].to(device),
        servo_max_acceleration_rad_s2=batch[
            "adaptive_servo_max_acceleration_rad_s2"
        ].to(device),
        servo_position_gain_s_inv=batch[
            "adaptive_servo_position_gain_s_inv"
        ].to(device),
        servo_position_tolerance_rad=batch[
            "adaptive_servo_position_tolerance_rad"
        ].to(device),
        servo_position_quantization_rad=batch[
            "adaptive_servo_position_quantization_rad"
        ].to(device),
        servo_command_polarity=batch[
            "adaptive_servo_command_polarity"
        ].to(device),
        servo_command_latency_s=batch[
            "adaptive_servo_command_latency_s"
        ].to(device),
        servo_rate_time_constant_s=batch[
            "adaptive_servo_rate_time_constant_s"
        ].to(device),
        control_period_s=batch["adaptive_control_period_s"].to(device),
        integration_period_s=batch["adaptive_integration_period_s"].to(device),
        camera_frame_period_s=batch[
            "adaptive_camera_frame_period_s"
        ].to(device),
    )


def _compatible_datasets(
    train: GimbalTargetStateDataset,
    validation: GimbalTargetStateDataset,
    profile: ObservationProfile,
) -> None:
    validate_disjoint_seed_blocks((train.manifest, validation.manifest))
    for dataset in (train, validation):
        if profile.value not in dataset.manifest.observation_profiles:
            raise ValueError(f"profile {profile.value!r} is absent from a dataset")
        if dataset.manifest.feature_names != FEATURE_NAMES:
            raise ValueError("feature schema mismatch")
        if dataset.manifest.target_names != TARGET_NAMES:
            raise ValueError("target schema mismatch")
    if (
        train.manifest.prediction_horizons_s
        != validation.manifest.prediction_horizons_s
    ):
        raise ValueError("training and validation horizons differ")


def train_gru(
    train: GimbalTargetStateDataset,
    validation: GimbalTargetStateDataset,
    profile: ObservationProfile,
    *,
    model_config: GRUTargetStateModelConfig | None = None,
    training_config: GRUTrainingConfig | None = None,
    loss_config: GRULossConfig | None = None,
    training_label_weights: np.ndarray | None = None,
    validation_label_weights: np.ndarray | None = None,
    training_episode_weights: np.ndarray | None = None,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    reference_anchor_config: GRUReferenceAnchorConfig | None = None,
    training_reference_anchor_weights: np.ndarray | None = None,
    retain_epoch_states: bool = False,
) -> GRUTrainingResult:
    training_config = training_config or GRUTrainingConfig()
    loss_config = loss_config or GRULossConfig()
    reference_anchor_config = (
        reference_anchor_config or GRUReferenceAnchorConfig()
    )
    _compatible_datasets(train, validation, profile)
    model_config = model_config or GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=train.manifest.prediction_horizons_s,
    )
    if model_config.input_dim != len(FEATURE_NAMES):
        raise ValueError("model input dimension does not match feature schema")
    if (
        model_config.prediction_horizons_s
        != train.manifest.prediction_horizons_s
    ):
        raise ValueError("model and dataset prediction horizons differ")

    set_gru_seed(training_config.seed)
    device = torch.device(training_config.device)
    action_aware = (
        loss_config.rate_action_weight > 0.0
        or loss_config.position_action_weight > 0.0
    )
    adaptive_aware = (
        loss_config.adaptive_position_action_weight > 0.0
        or any(
            weight > 0.0
            for weight in (
                loss_config.position_plant_tracking_weight,
                loss_config.position_plant_response_weight,
                loss_config.position_plant_regret_weight,
                loss_config.position_plant_visibility_weight,
                loss_config.position_plant_smoothness_weight,
                loss_config.position_plant_saturation_weight,
            )
        )
    )
    if adaptive_aware and loss_config.adaptive_position_config is None:
        raise ValueError(
            "adaptive position config is required for adaptive training"
        )
    training_control_supervision = (
        compute_control_action_supervision(train, profile=profile)
        if action_aware
        else None
    )
    training_adaptive_position_supervision = (
        compute_adaptive_position_supervision(
            train,
            adapter=loss_config.adaptive_position_config,
            profile=profile,
        )
        if adaptive_aware and loss_config.adaptive_position_config is not None
        else None
    )
    model = CausalTargetStateGRU(model_config).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    reference_model = None
    if reference_anchor_config.active:
        if initial_state_dict is None:
            raise ValueError(
                "reference anchoring requires an initial state dictionary"
            )
        reference_model = copy.deepcopy(model).eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    generator = torch.Generator().manual_seed(training_config.seed)
    sampler = None
    if training_episode_weights is not None:
        if training_episode_weights.shape != (train.episode_count,):
            raise ValueError("training episode weights shape is invalid")
        if np.any(~np.isfinite(training_episode_weights)) or np.any(
            training_episode_weights <= 0.0
        ):
            raise ValueError(
                "training episode weights must be finite and positive"
            )
        sampler = WeightedRandomSampler(
            torch.as_tensor(training_episode_weights, dtype=torch.double),
            num_samples=train.episode_count,
            replacement=True,
            generator=generator,
        )
    loader = DataLoader(
        GimbalTorchSequenceDataset(
            train,
            profile,
            training_label_weights,
            training_control_supervision,
            training_adaptive_position_supervision,
            training_reference_anchor_weights,
        ),
        batch_size=training_config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=(generator if sampler is None else None),
    )
    initial_validation = evaluate_gru(
        model,
        validation,
        profile,
        batch_size=training_config.batch_size,
        device=device,
        loss_config=loss_config,
        label_weights=validation_label_weights,
    )
    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    epoch_state_dicts = []

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        total_weighted_loss = 0.0
        total_weighted_anchor_loss = 0.0
        total_labels = 0
        anchor_conflict_batches = 0
        anchor_evaluated_batches = 0
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            target_mask = batch["target_mask"].to(device)
            sequence_mask = batch["sequence_mask"].to(device)
            output = model(features)
            loss = target_state_nll(
                output,
                targets,
                target_mask,
                sequence_mask,
                loss_config,
                label_weights=(
                    batch["label_weights"].to(device)
                    if "label_weights" in batch
                    else None
                ),
                prediction_horizons_s=(
                    train.manifest.prediction_horizons_s
                ),
                control_context=(
                    _control_context_from_batch(batch, device)
                    if action_aware
                    else None
                ),
                adaptive_position_context=(
                    _adaptive_position_context_from_batch(batch, device)
                    if adaptive_aware
                    else None
                ),
            )
            anchor_loss = targets.new_zeros(())
            if reference_model is not None:
                with torch.no_grad():
                    reference_output = reference_model(features)
                anchor_weights = (
                    batch["reference_anchor_weights"].to(device)
                    if "reference_anchor_weights" in batch
                    else torch.ones_like(target_mask, dtype=targets.dtype)
                )
                if loss_config.horizon_weights:
                    anchor_weights = anchor_weights * targets.new_tensor(
                        loss_config.horizon_weights
                    ).view(1, 1, -1)
                anchor_mask = (
                    target_mask.bool() & sequence_mask.bool().unsqueeze(-1)
                )
                selected_anchor_weights = anchor_weights * anchor_mask.to(
                    dtype=anchor_weights.dtype
                )
                anchor_weight_sum = selected_anchor_weights.sum().clamp_min(
                    1.0
                )
                bearing_anchor_mse = (
                    angular_residual_rad(
                        output.mean[..., 0],
                        reference_output.mean[..., 0],
                    ).square()
                    * selected_anchor_weights
                ).sum() / anchor_weight_sum
                rate_anchor_mse = (
                    (output.mean[..., 1] - reference_output.mean[..., 1])
                    .square()
                    * selected_anchor_weights
                ).sum() / anchor_weight_sum
                anchor_loss = (
                    reference_anchor_config.bearing_weight
                    * bearing_anchor_mse
                    + reference_anchor_config.rate_weight * rate_anchor_mse
                )
            total_loss = loss.total + anchor_loss
            optimizer.zero_grad(set_to_none=True)
            if (
                reference_model is not None
                and reference_anchor_config.project_conflicting_gradients
            ):
                parameters = tuple(
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                control_gradients = torch.autograd.grad(
                    loss.total,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                anchor_gradients = torch.autograd.grad(
                    anchor_loss,
                    parameters,
                    allow_unused=True,
                )
                gradient_dot = targets.new_zeros(())
                anchor_norm_squared = targets.new_zeros(())
                for control_gradient, anchor_gradient in zip(
                    control_gradients,
                    anchor_gradients,
                ):
                    if anchor_gradient is not None:
                        anchor_norm_squared = (
                            anchor_norm_squared
                            + anchor_gradient.square().sum()
                        )
                    if (
                        control_gradient is not None
                        and anchor_gradient is not None
                    ):
                        gradient_dot = gradient_dot + (
                            control_gradient * anchor_gradient
                        ).sum()
                conflict = float(gradient_dot.detach()) < 0.0
                anchor_evaluated_batches += 1
                if conflict:
                    anchor_conflict_batches += 1
                projection_scale = (
                    gradient_dot
                    / anchor_norm_squared.clamp_min(
                        reference_anchor_config.projection_epsilon
                    )
                    if conflict
                    else targets.new_zeros(())
                )
                for parameter, control_gradient, anchor_gradient in zip(
                    parameters,
                    control_gradients,
                    anchor_gradients,
                ):
                    combined_gradient = None
                    if control_gradient is not None:
                        combined_gradient = control_gradient
                        if conflict and anchor_gradient is not None:
                            combined_gradient = (
                                combined_gradient
                                - projection_scale * anchor_gradient
                            )
                    if anchor_gradient is not None:
                        combined_gradient = (
                            anchor_gradient
                            if combined_gradient is None
                            else combined_gradient + anchor_gradient
                        )
                    parameter.grad = combined_gradient
            else:
                total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm
            )
            optimizer.step()
            label_count = int(
                (target_mask & sequence_mask.unsqueeze(-1)).sum().item()
            )
            total_weighted_loss += float(total_loss.detach()) * label_count
            total_weighted_anchor_loss += (
                float(anchor_loss.detach()) * label_count
            )
            total_labels += label_count

        validation_metrics = evaluate_gru(
            model,
            validation,
            profile,
            batch_size=training_config.batch_size,
            device=device,
            loss_config=loss_config,
            label_weights=validation_label_weights,
        )
        if validation_metrics.loss is None:
            raise RuntimeError("validation set contains no valid targets")
        history.append(
            GRUEpochRecord(
                epoch=epoch,
                training_loss=total_weighted_loss / max(1, total_labels),
                validation_loss=validation_metrics.loss,
                validation_bearing_rmse_deg=(
                    validation_metrics.bearing_rmse_deg
                ),
                validation_rate_rmse_deg_s=(
                    validation_metrics.rate_rmse_deg_s
                ),
                training_reference_anchor_loss=(
                    total_weighted_anchor_loss / max(1, total_labels)
                ),
                reference_anchor_conflict_fraction=(
                    anchor_conflict_batches / anchor_evaluated_batches
                    if anchor_evaluated_batches
                    else 0.0
                ),
            )
        )
        if validation_metrics.loss < best_loss:
            best_loss = validation_metrics.loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if retain_epoch_states:
            epoch_state_dicts.append(
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            )

    model.load_state_dict(best_state)
    best_validation = evaluate_gru(
        model,
        validation,
        profile,
        batch_size=training_config.batch_size,
        device=device,
        loss_config=loss_config,
        label_weights=validation_label_weights,
    )
    return GRUTrainingResult(
        model=model,
        history=tuple(history),
        best_epoch=best_epoch,
        initial_validation=initial_validation,
        best_validation=best_validation,
        epoch_state_dicts=tuple(epoch_state_dicts),
    )


def _feature_value(
    row: np.ndarray,
    indices: dict[str, int],
    value_name: str,
    validity_name: str,
    transform=lambda value: value,
) -> MaskedScalar:
    valid = row[indices[validity_name]] > 0.5
    value = transform(float(row[indices[value_name]])) if valid else 0.0
    return MaskedScalar(value, bool(valid))


def _scenario_payload(
    dataset: GimbalTargetStateDataset,
    scenario_index: int,
    episode_seed: int,
) -> dict[str, object]:
    variants = dataset.manifest.generation.get("scenario_variants")
    if isinstance(variants, list):
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            if (
                int(variant.get("seed", -1)) == episode_seed
                and int(variant.get("scenario_index", -1)) == scenario_index
            ):
                payload = variant.get("scenario")
                if isinstance(payload, dict):
                    return payload
        raise ValueError("manifest is missing an episode scenario variant")
    scenarios = dataset.manifest.generation.get("scenarios")
    if not isinstance(scenarios, list) or not 0 <= scenario_index < len(scenarios):
        raise ValueError("manifest does not contain canonical scenario configs")
    payload = scenarios[scenario_index]
    if not isinstance(payload, dict):
        raise ValueError("scenario payload is invalid")
    return payload


def constant_velocity_predictions(
    dataset: GimbalTargetStateDataset,
    profile: ObservationProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay the causal analytical estimator and return distribution arrays."""
    try:
        profile_index = dataset.manifest.observation_profiles.index(profile.value)
    except ValueError as error:
        raise ValueError(f"profile {profile.value!r} is absent") from error
    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    mean = np.zeros_like(dataset.targets, dtype=np.float32)
    std = np.ones_like(dataset.targets, dtype=np.float32)
    prediction_mask = np.zeros_like(dataset.target_mask, dtype=np.bool_)
    possible_mask = dataset.target_mask & dataset.sequence_mask[..., None]

    for episode_index in range(dataset.episode_count):
        scenario_index = int(dataset.scenario_index[episode_index])
        episode_seed = int(dataset.episode_seed[episode_index])
        payload = _scenario_payload(dataset, scenario_index, episode_seed)
        config = payload["config"]
        assert isinstance(config, dict)
        servo = config["servo"]
        camera = config["camera"]
        assert isinstance(servo, dict) and isinstance(camera, dict)
        min_angle = float(servo["min_angle_rad"])
        max_angle = float(servo["max_angle_rad"])
        max_rate = float(servo["max_rate_rad_s"])
        fov = float(camera["selected_axis_fov_rad"])
        frame_rate = float(camera["frame_rate_hz"])
        maximum_projection_s = max(
            0.30,
            float(camera["detection_latency_s"])
            + float(camera["detection_latency_jitter_s"])
            + 2.0 / frame_rate,
        )
        estimator_config = ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=fov,
            center_noise_std_normalized=float(
                camera["center_noise_std_normalized"]
            ),
            velocity_filter_coefficient=0.40,
            uncertainty_filter_coefficient=0.20,
            max_prediction_horizon_s=maximum_projection_s,
            history_horizon_s=max(1.0, maximum_projection_s + 0.50),
        )
        estimator = ConstantVelocityTargetEstimator(estimator_config)
        length = int(dataset.sequence_mask[episode_index].sum())
        rows = dataset.features[episode_index, profile_index]

        def position_from_normalized(value: float) -> float:
            limit = max_angle if value >= 0.0 else -min_angle
            return value * limit

        for time_index in range(length):
            row = rows[time_index]
            rate_mode = row[indices["command_mode_rate"]] > 0.5
            observation = GimbalObservation(
                time_s=float(dataset.time_s[episode_index, time_index]),
                control_dt_s=float(row[indices["control_dt_s"]]),
                frame_updated=bool(row[indices["frame_updated"]] > 0.5),
                measurement_age_s=_feature_value(
                    row,
                    indices,
                    "measurement_age_s",
                    "measurement_age_valid",
                ),
                image_error_normalized=_feature_value(
                    row,
                    indices,
                    "image_error_normalized",
                    "image_error_valid",
                ),
                bbox_width_fraction=_feature_value(
                    row,
                    indices,
                    "bbox_width_fraction",
                    "bbox_width_valid",
                ),
                bbox_height_fraction=_feature_value(
                    row,
                    indices,
                    "bbox_height_fraction",
                    "bbox_height_valid",
                ),
                confidence=_feature_value(
                    row,
                    indices,
                    "confidence",
                    "confidence_valid",
                ),
                gimbal_angle_rad=_feature_value(
                    row,
                    indices,
                    "gimbal_position_normalized",
                    "gimbal_position_valid",
                    position_from_normalized,
                ),
                gimbal_rate_rad_s=_feature_value(
                    row,
                    indices,
                    "gimbal_rate_normalized",
                    "gimbal_rate_valid",
                    lambda value: value * max_rate,
                ),
                body_rate_rad_s=_feature_value(
                    row,
                    indices,
                    "body_rate_normalized",
                    "body_rate_valid",
                    lambda value: value * max_rate,
                ),
                command_mode=(
                    GimbalCommandMode.RATE
                    if rate_mode
                    else GimbalCommandMode.POSITION
                ),
                previous_action_normalized=float(
                    row[indices["previous_action_normalized"]]
                ),
            )
            estimate = estimator.update(observation)
            if not estimate.valid:
                continue
            for horizon_index, horizon_s in enumerate(
                dataset.manifest.prediction_horizons_s
            ):
                if not possible_mask[
                    episode_index, time_index, horizon_index
                ]:
                    continue
                bearing = (
                    estimate.body_relative_bearing_rad.value
                    + horizon_s * estimate.body_relative_rate_rad_s.value
                )
                mean[episode_index, time_index, horizon_index, 0] = math.atan2(
                    math.sin(bearing), math.cos(bearing)
                )
                mean[episode_index, time_index, horizon_index, 1] = (
                    estimate.body_relative_rate_rad_s.value
                )
                acceleration_std = (
                    estimator_config.process_acceleration_std_rad_s2
                )
                std[episode_index, time_index, horizon_index, 0] = math.sqrt(
                    estimate.bearing_std_rad.value**2
                    + (horizon_s * estimate.rate_std_rad_s.value) ** 2
                    + (0.5 * acceleration_std * horizon_s**2) ** 2
                )
                std[episode_index, time_index, horizon_index, 1] = math.hypot(
                    estimate.rate_std_rad_s.value,
                    acceleration_std * horizon_s,
                )
                prediction_mask[
                    episode_index, time_index, horizon_index
                ] = True

    return mean, std, prediction_mask, possible_mask


def evaluate_constant_velocity_baseline(
    dataset: GimbalTargetStateDataset,
    profile: ObservationProfile,
    *,
    loss_config: GRULossConfig | None = None,
) -> GRUEvaluationMetrics:
    """Score the existing causal analytical estimator on encoded features."""
    mean, std, prediction_mask, possible_mask = constant_velocity_predictions(
        dataset, profile
    )
    return _metrics_from_predictions(
        mean=mean,
        std=std,
        targets=dataset.targets,
        prediction_mask=prediction_mask,
        possible_mask=possible_mask,
        horizons_s=dataset.manifest.prediction_horizons_s,
        loss_config=loss_config,
    )


def run_gru_experiment(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    checkpoint_path: str | Path,
    profile: ObservationProfile = ObservationProfile.SERVO_AWARE,
    hidden_dim: int = 64,
    embedding_dim: int = 64,
    training_config: GRUTrainingConfig | None = None,
) -> dict[str, object]:
    training_config = training_config or GRUTrainingConfig()
    train = load_gimbal_dataset(train_path)
    validation = load_gimbal_dataset(validation_path)
    test = load_gimbal_dataset(test_path)
    validate_disjoint_seed_blocks(
        (train.manifest, validation.manifest, test.manifest)
    )
    _compatible_datasets(train, test, profile)
    model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=train.manifest.prediction_horizons_s,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    )
    training = train_gru(
        train,
        validation,
        profile,
        model_config=model_config,
        training_config=training_config,
    )
    learned_test = evaluate_gru(
        training.model,
        test,
        profile,
        batch_size=training_config.batch_size,
        device=training_config.device,
    )
    analytical_mean, analytical_std, analytical_mask, possible_mask = (
        constant_velocity_predictions(test, profile)
    )
    analytical_test = _metrics_from_predictions(
        mean=analytical_mean,
        std=analytical_std,
        targets=test.targets,
        prediction_mask=analytical_mask,
        possible_mask=possible_mask,
        horizons_s=test.manifest.prediction_horizons_s,
    )
    learned_test_matched = evaluate_gru(
        training.model,
        test,
        profile,
        batch_size=training_config.batch_size,
        device=training_config.device,
        evaluation_mask=analytical_mask,
    )
    result: dict[str, object] = {
        "experiment": "gimbal_causal_gru_smoke",
        "profile": profile.value,
        "torch_version": torch.__version__,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "parameter_count": gru_parameter_count(training.model),
        "dataset_hashes": {
            "train": train.manifest.configuration_hash,
            "validation": validation.manifest.configuration_hash,
            "test": test.manifest.configuration_hash,
        },
        "dataset_episodes": {
            "train": train.episode_count,
            "validation": validation.episode_count,
            "test": test.episode_count,
        },
        "best_epoch": training.best_epoch,
        "initial_validation": asdict(training.initial_validation),
        "best_validation": asdict(training.best_validation),
        "learned_test": asdict(learned_test),
        "learned_test_on_analytical_support": asdict(learned_test_matched),
        "constant_velocity_test": asdict(analytical_test),
        "history": [asdict(record) for record in training.history],
        "interpretation": (
            "Pipeline smoke test on fixed development trajectories; not an "
            "independent-motion generalization result."
        ),
    }
    save_gru_checkpoint(
        checkpoint_path,
        training.model,
        metadata={
            "profile": profile.value,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "dataset_hashes": result["dataset_hashes"],
            "training_config": result["training_config"],
            "best_epoch": training.best_epoch,
            "best_validation": result["best_validation"],
            "learned_test": result["learned_test"],
            "learned_test_on_analytical_support": result[
                "learned_test_on_analytical_support"
            ],
        },
    )
    result["checkpoint"] = str(checkpoint_path)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the causal gimbal target-state GRU."
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in ObservationProfile],
        default=ObservationProfile.SERVO_AWARE.value,
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    training_config = GRUTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip,
        seed=args.seed,
        device=args.device,
    )
    result = run_gru_experiment(
        train_path=args.train_data,
        validation_path=args.validation_data,
        test_path=args.test_data,
        checkpoint_path=args.checkpoint,
        profile=ObservationProfile(args.profile),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        training_config=training_config,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
