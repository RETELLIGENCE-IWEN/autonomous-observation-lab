"""Causal GRU target-state predictor and streaming estimator adapter.

This module requires the optional ``learning`` dependency.  It is deliberately
not imported by :mod:`gimbal_servoing.__init__`, so the simulator remains usable
without PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import GimbalServoingConfig, ObservationProfile
from .controllers import AdaptivePositionControllerConfig
from .dataset import FEATURE_NAMES, encode_deployable_observation
from .estimators import TargetStateEstimate
from .types import GimbalObservation, MaskedScalar


GRU_CHECKPOINT_SCHEMA_VERSION = "gimbal_gru_v1"
GRU_POSITION_RESIDUAL_CHECKPOINT_SCHEMA_VERSION = (
    "gimbal_gru_position_residual_v1"
)


class UncertaintyScaleProvider(Protocol):
    """Deployable post-hoc uncertainty scaling contract."""

    def scales_for_observation(
        self,
        horizon_index: int,
        observation: GimbalObservation,
        detection_gap_s: float | None,
    ) -> tuple[float, float]: ...


@dataclass(frozen=True)
class GRUTargetStateModelConfig:
    input_dim: int
    prediction_horizons_s: tuple[float, ...]
    hidden_dim: int = 64
    embedding_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    mean_parameterization: str = "independent"
    minimum_bearing_std_rad: float = math.radians(0.05)
    minimum_rate_std_rad_s: float = math.radians(0.50)

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.prediction_horizons_s:
            raise ValueError("at least one prediction horizon is required")
        if any(
            not math.isfinite(horizon) or horizon < 0.0
            for horizon in self.prediction_horizons_s
        ):
            raise ValueError("prediction horizons must be finite and non-negative")
        if (
            tuple(sorted(self.prediction_horizons_s))
            != self.prediction_horizons_s
        ):
            raise ValueError("prediction horizons must be in ascending order")
        if self.hidden_dim <= 0 or self.embedding_dim <= 0:
            raise ValueError("hidden and embedding dimensions must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mean_parameterization not in {
            "independent",
            "integrated_rate",
            "integrated_midpoint",
        }:
            raise ValueError("unsupported mean parameterization")
        if self.mean_parameterization in {
            "integrated_rate",
            "integrated_midpoint",
        } and any(
            right <= left
            for left, right in zip(
                self.prediction_horizons_s,
                self.prediction_horizons_s[1:],
            )
        ):
            raise ValueError(
                "integrated-rate horizons must be strictly increasing"
            )
        if self.minimum_bearing_std_rad <= 0.0:
            raise ValueError("minimum bearing standard deviation must be positive")
        if self.minimum_rate_std_rad_s <= 0.0:
            raise ValueError("minimum rate standard deviation must be positive")


@dataclass
class GRUTargetStateOutput:
    mean: torch.Tensor
    std: torch.Tensor
    hidden: torch.Tensor
    interval_rate_rad_s: torch.Tensor | None = None
    position_target_residual_fov_fraction: torch.Tensor | None = None


class CausalTargetStateGRU(nn.Module):
    """Unidirectional recurrent predictor of bearing/rate distributions."""

    def __init__(self, config: GRUTargetStateModelConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        horizon_count = len(config.prediction_horizons_s)
        if config.mean_parameterization == "independent":
            head_output_dim = horizon_count * 4
        elif config.mean_parameterization == "integrated_rate":
            head_output_dim = 1 + 3 * horizon_count
        else:
            head_output_dim = 4 * horizon_count
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(
                config.hidden_dim,
                head_output_dim,
            ),
        )

    @property
    def horizon_count(self) -> int:
        return len(self.config.prediction_horizons_s)

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> GRUTargetStateOutput:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError(
                "features must have shape [batch, time, input_dim]"
            )
        encoded = self.encoder(features)
        recurrent, hidden = self.recurrent(encoded, hidden)
        raw = self.head(recurrent)
        interval_rate = None
        if self.config.mean_parameterization == "independent":
            raw = raw.view(
                *recurrent.shape[:2], self.horizon_count, 4
            )
            bearing = math.pi * torch.tanh(raw[..., 0])
            rate = raw[..., 1]
            bearing_std_raw = raw[..., 2]
            rate_std_raw = raw[..., 3]
        else:
            horizon_count = self.horizon_count
            current_bearing = math.pi * torch.tanh(raw[..., 0])
            rate = raw[..., 1 : 1 + horizon_count]
            offset = 1 + horizon_count
            if self.config.mean_parameterization == "integrated_midpoint":
                interval_rate = raw[..., offset : offset + horizon_count - 1]
                offset += horizon_count - 1
            bearing_std_raw = raw[
                ..., offset : offset + horizon_count
            ]
            rate_std_raw = raw[..., offset + horizon_count :]
            if horizon_count == 1:
                bearing = current_bearing.unsqueeze(-1)
            else:
                intervals = raw.new_tensor(
                    [
                        right - left
                        for left, right in zip(
                            self.config.prediction_horizons_s,
                            self.config.prediction_horizons_s[1:],
                        )
                    ]
                )
                if interval_rate is None:
                    increments = 0.5 * (
                        rate[..., 1:] + rate[..., :-1]
                    ) * intervals
                else:
                    increments = (
                        rate[..., :-1]
                        + 4.0 * interval_rate
                        + rate[..., 1:]
                    ) * intervals / 6.0
                future_bearing = current_bearing.unsqueeze(-1) + torch.cumsum(
                    increments,
                    dim=-1,
                )
                future_bearing = torch.atan2(
                    torch.sin(future_bearing),
                    torch.cos(future_bearing),
                )
                bearing = torch.cat(
                    (current_bearing.unsqueeze(-1), future_bearing),
                    dim=-1,
                )
        mean = torch.stack((bearing, rate), dim=-1)
        bearing_std = (
            F.softplus(bearing_std_raw)
            + self.config.minimum_bearing_std_rad
        )
        rate_std = (
            F.softplus(rate_std_raw)
            + self.config.minimum_rate_std_rad_s
        )
        std = torch.stack((bearing_std, rate_std), dim=-1)
        return GRUTargetStateOutput(
            mean=mean,
            std=std,
            hidden=hidden,
            interval_rate_rad_s=interval_rate,
        )

    def forward_step(
        self,
        feature: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> GRUTargetStateOutput:
        if feature.ndim != 2:
            raise ValueError("feature must have shape [batch, input_dim]")
        output = self.forward(feature[:, None, :], hidden)
        return GRUTargetStateOutput(
            mean=output.mean[:, 0],
            std=output.std[:, 0],
            hidden=output.hidden,
            interval_rate_rad_s=(
                output.interval_rate_rad_s[:, 0]
                if output.interval_rate_rad_s is not None
                else None
            ),
            position_target_residual_fov_fraction=(
                output.position_target_residual_fov_fraction[:, 0]
                if output.position_target_residual_fov_fraction is not None
                else None
            ),
        )


@dataclass(frozen=True)
class GRUPositionResidualConfig:
    """Small causal correction head for the adaptive position target.

    The correction is dimensionless and converted to radians using the
    serialized camera field of view inside the adapter. This keeps its bound
    independent of a particular camera or servo.
    """

    hidden_dim: int = 32
    maximum_half_fov_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("residual hidden dimension must be positive")
        if not math.isfinite(self.maximum_half_fov_fraction) or not (
            0.0 < self.maximum_half_fov_fraction <= 1.0
        ):
            raise ValueError(
                "maximum residual half-FOV fraction must be in (0, 1]"
            )


class CausalTargetStateGRUWithPositionResidual(nn.Module):
    """Frozen target-state GRU plus a bounded causal position-target head."""

    def __init__(
        self,
        base_model: CausalTargetStateGRU,
        residual_config: GRUPositionResidualConfig | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.residual_config = residual_config or GRUPositionResidualConfig()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        input_dim = self.config.input_dim + 2 * len(
            self.config.prediction_horizons_s
        )
        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, self.residual_config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.residual_config.hidden_dim, 1),
        )
        final = self.residual_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> GRUTargetStateOutput:
        with torch.no_grad():
            base = self.base_model(features, hidden)
        head_features = torch.cat(
            (
                features,
                base.mean.detach().flatten(start_dim=-2),
            ),
            dim=-1,
        )
        residual = self.residual_config.maximum_half_fov_fraction * torch.tanh(
            self.residual_head(head_features).squeeze(-1)
        )
        return GRUTargetStateOutput(
            mean=base.mean,
            std=base.std,
            hidden=base.hidden,
            interval_rate_rad_s=base.interval_rate_rad_s,
            position_target_residual_fov_fraction=residual,
        )

    def forward_step(
        self,
        feature: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> GRUTargetStateOutput:
        if feature.ndim != 2:
            raise ValueError("feature must have shape [batch, input_dim]")
        output = self.forward(feature[:, None, :], hidden)
        return GRUTargetStateOutput(
            mean=output.mean[:, 0],
            std=output.std[:, 0],
            hidden=output.hidden,
            interval_rate_rad_s=(
                output.interval_rate_rad_s[:, 0]
                if output.interval_rate_rad_s is not None
                else None
            ),
            position_target_residual_fov_fraction=(
                output.position_target_residual_fov_fraction[:, 0]
                if output.position_target_residual_fov_fraction is not None
                else None
            ),
        )


class CausalTargetStateGRUEnsemble(nn.Module):
    """Causal moment-matched ensemble of compatible GRU predictors."""

    def __init__(self, models: tuple[CausalTargetStateGRU, ...]):
        super().__init__()
        if not models:
            raise ValueError("at least one GRU ensemble member is required")
        if any(model.config != models[0].config for model in models[1:]):
            raise ValueError("GRU ensemble members must share one config")
        self.members = nn.ModuleList(models)
        self.config = models[0].config

    @property
    def member_count(self) -> int:
        return len(self.members)

    @staticmethod
    def _combine(
        outputs: tuple[GRUTargetStateOutput, ...],
    ) -> GRUTargetStateOutput:
        means = torch.stack([output.mean for output in outputs], dim=0)
        stds = torch.stack([output.std for output in outputs], dim=0)
        bearing_mean = torch.atan2(
            torch.mean(torch.sin(means[..., 0]), dim=0),
            torch.mean(torch.cos(means[..., 0]), dim=0),
        )
        rate_mean = torch.mean(means[..., 1], dim=0)
        combined_mean = torch.stack((bearing_mean, rate_mean), dim=-1)
        bearing_residual = angular_residual_rad(
            means[..., 0],
            bearing_mean.unsqueeze(0),
        )
        bearing_variance = torch.mean(
            stds[..., 0].square() + bearing_residual.square(),
            dim=0,
        )
        rate_variance = torch.mean(
            stds[..., 1].square()
            + (means[..., 1] - rate_mean.unsqueeze(0)).square(),
            dim=0,
        )
        combined_std = torch.stack(
            (
                torch.sqrt(bearing_variance.clamp_min(1e-12)),
                torch.sqrt(rate_variance.clamp_min(1e-12)),
            ),
            dim=-1,
        )
        interval_values = [
            output.interval_rate_rad_s for output in outputs
        ]
        if all(value is None for value in interval_values):
            interval_rate = None
        elif all(value is not None for value in interval_values):
            interval_rate = torch.mean(
                torch.stack(
                    [value for value in interval_values if value is not None],
                    dim=0,
                ),
                dim=0,
            )
        else:
            raise ValueError("GRU ensemble interval-rate heads are incompatible")
        return GRUTargetStateOutput(
            mean=combined_mean,
            std=combined_std,
            hidden=torch.stack([output.hidden for output in outputs], dim=0),
            interval_rate_rad_s=interval_rate,
        )

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> GRUTargetStateOutput:
        member_hidden = (
            (None,) * self.member_count
            if hidden is None
            else tuple(hidden[index] for index in range(self.member_count))
        )
        if len(member_hidden) != self.member_count:
            raise ValueError("GRU ensemble hidden state is invalid")
        return self._combine(
            tuple(
                member(features, member_hidden[index])
                for index, member in enumerate(self.members)
            )
        )

    def forward_step(
        self,
        feature: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> GRUTargetStateOutput:
        member_hidden = (
            (None,) * self.member_count
            if hidden is None
            else tuple(hidden[index] for index in range(self.member_count))
        )
        if len(member_hidden) != self.member_count:
            raise ValueError("GRU ensemble hidden state is invalid")
        return self._combine(
            tuple(
                member.forward_step(feature, member_hidden[index])
                for index, member in enumerate(self.members)
            )
        )

@dataclass(frozen=True)
class GRUPositionPlantRolloutConfig:
    """Numerical and scoring choices for a local causal servo rollout.

    Hardware properties come from each serialized episode.  The optional
    integration-period override is only a training-time numerical speed knob;
    leaving it unset replays the simulator's configured integration period.
    """

    horizon_index: int = 1
    visibility_margin_fraction: float = 0.85
    integration_period_override_s: float | None = None

    def __post_init__(self) -> None:
        if self.horizon_index <= 0:
            raise ValueError("plant rollout horizon index must be positive")
        if not 0.0 < self.visibility_margin_fraction <= 1.0:
            raise ValueError("visibility margin fraction must be in (0, 1]")
        if self.integration_period_override_s is not None and (
            not math.isfinite(self.integration_period_override_s)
            or self.integration_period_override_s <= 0.0
        ):
            raise ValueError(
                "plant integration-period override must be finite and positive"
            )


@dataclass(frozen=True)
class GRULossConfig:
    bearing_weight: float = 1.0
    rate_weight: float = 1.0
    mean_error_weight: float = 0.05
    bearing_mean_error_weight: float = 0.0
    rate_mean_error_weight: float = 0.0
    dynamic_consistency_weight: float = 0.0
    rate_action_weight: float = 0.0
    position_action_weight: float = 0.0
    adaptive_position_action_weight: float = 0.0
    adaptive_position_config: AdaptivePositionControllerConfig | None = None
    position_plant_tracking_weight: float = 0.0
    position_plant_response_weight: float = 0.0
    position_plant_regret_weight: float = 0.0
    position_plant_visibility_weight: float = 0.0
    position_plant_smoothness_weight: float = 0.0
    position_plant_saturation_weight: float = 0.0
    position_plant_config: GRUPositionPlantRolloutConfig | None = None
    horizon_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "bearing_weight",
            "rate_weight",
            "mean_error_weight",
            "bearing_mean_error_weight",
            "rate_mean_error_weight",
            "dynamic_consistency_weight",
            "rate_action_weight",
            "position_action_weight",
            "adaptive_position_action_weight",
            "position_plant_tracking_weight",
            "position_plant_response_weight",
            "position_plant_regret_weight",
            "position_plant_visibility_weight",
            "position_plant_smoothness_weight",
            "position_plant_saturation_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if any(
            not math.isfinite(weight) or weight <= 0.0
            for weight in self.horizon_weights
        ):
            raise ValueError("horizon weights must be finite and positive")


@dataclass
class GRULoss:
    total: torch.Tensor
    bearing_nll: torch.Tensor
    rate_nll: torch.Tensor
    bearing_rmse_rad: torch.Tensor
    rate_rmse_rad_s: torch.Tensor
    dynamic_consistency_rmse_rad: torch.Tensor
    rate_action_rmse_normalized: torch.Tensor
    position_action_rmse_normalized: torch.Tensor
    adaptive_position_action_rmse_normalized: torch.Tensor
    position_plant_tracking_rmse_normalized: torch.Tensor
    position_plant_response_rmse_normalized: torch.Tensor
    position_plant_regret_rmse_normalized: torch.Tensor
    position_plant_visibility_rmse_normalized: torch.Tensor
    position_plant_smoothness_rmse_normalized: torch.Tensor
    position_plant_saturation_rmse_normalized: torch.Tensor


@dataclass(frozen=True)
class GRUControlLossContext:
    """Batch-local privileged context used only by the training loss."""

    oracle_actions: torch.Tensor
    gimbal_angle_rad: torch.Tensor
    servo_max_rate_rad_s: torch.Tensor
    servo_min_angle_rad: torch.Tensor
    servo_max_angle_rad: torch.Tensor
    rate_feedback_gain_s_inv: torch.Tensor
    position_preview_s: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class GRUAdaptivePositionLossContext:
    teacher_action_normalized: torch.Tensor
    mask: torch.Tensor
    gimbal_angle_rad: torch.Tensor
    gimbal_rate_rad_s: torch.Tensor
    control_dt_s: torch.Tensor
    selected_axis_fov_rad: torch.Tensor
    servo_min_angle_rad: torch.Tensor
    servo_max_angle_rad: torch.Tensor
    servo_max_rate_rad_s: torch.Tensor
    servo_max_acceleration_rad_s2: torch.Tensor
    servo_position_gain_s_inv: torch.Tensor
    servo_position_tolerance_rad: torch.Tensor
    servo_position_quantization_rad: torch.Tensor
    servo_command_polarity: torch.Tensor
    servo_command_latency_s: torch.Tensor
    servo_rate_time_constant_s: torch.Tensor
    control_period_s: torch.Tensor
    integration_period_s: torch.Tensor
    camera_frame_period_s: torch.Tensor


@dataclass
class GRUPositionPlantRollout:
    angle_rad: torch.Tensor
    rate_rad_s: torch.Tensor
    saturation_fraction: torch.Tensor


@dataclass
class GRUPositionPlantSequenceRollout:
    """State after every command in a latency-preserving servo sequence."""

    angle_rad: torch.Tensor
    rate_rad_s: torch.Tensor
    applied_position_rad: torch.Tensor
    saturation_fraction: torch.Tensor


@dataclass
class GRUPositionPlantState:
    """Persistent differentiable state for one causal position servo."""

    angle_rad: torch.Tensor
    rate_rad_s: torch.Tensor
    applied_position_rad: torch.Tensor
    elapsed_s: torch.Tensor
    next_capture_s: torch.Tensor
    pending_positions_rad: tuple[torch.Tensor, ...] = ()
    pending_arrivals_s: tuple[torch.Tensor, ...] = ()
    pending_active: tuple[torch.Tensor, ...] = ()


def angular_residual_rad(
    prediction_rad: torch.Tensor, target_rad: torch.Tensor
) -> torch.Tensor:
    difference = prediction_rad - target_rad
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def _interpolate_gru_output(
    output: GRUTargetStateOutput,
    requested_horizon_s: torch.Tensor,
    prediction_horizons_s: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    requested = torch.clamp(
        requested_horizon_s,
        prediction_horizons_s[0],
        prediction_horizons_s[-1],
    )
    bearing = output.mean[..., 0, 0]
    rate = output.mean[..., 0, 1]
    bearing_std = output.std[..., 0, 0]
    rate_std = output.std[..., 0, 1]
    for index, (left_horizon, right_horizon) in enumerate(
        zip(prediction_horizons_s, prediction_horizons_s[1:])
    ):
        fraction = torch.clamp(
            (requested - left_horizon) / (right_horizon - left_horizon),
            0.0,
            1.0,
        )
        candidate_bearing = output.mean[..., index, 0] + fraction * (
            angular_residual_rad(
                output.mean[..., index + 1, 0],
                output.mean[..., index, 0],
            )
        )
        candidate_bearing = angular_residual_rad(
            candidate_bearing,
            torch.zeros_like(candidate_bearing),
        )
        candidate_rate = output.mean[..., index, 1] + fraction * (
            output.mean[..., index + 1, 1]
            - output.mean[..., index, 1]
        )
        candidate_bearing_std = output.std[..., index, 0] + fraction * (
            output.std[..., index + 1, 0] - output.std[..., index, 0]
        )
        candidate_rate_std = output.std[..., index, 1] + fraction * (
            output.std[..., index + 1, 1] - output.std[..., index, 1]
        )
        selected = (requested >= left_horizon) & (
            requested <= right_horizon
        )
        bearing = torch.where(selected, candidate_bearing, bearing)
        rate = torch.where(selected, candidate_rate, rate)
        bearing_std = torch.where(
            selected,
            candidate_bearing_std,
            bearing_std,
        )
        rate_std = torch.where(selected, candidate_rate_std, rate_std)
    return bearing, rate, bearing_std, rate_std


def _adaptive_prediction_weight(
    current_std: torch.Tensor,
    forecast_std: torch.Tensor,
    config: AdaptivePositionControllerConfig,
) -> torch.Tensor:
    ratio = forecast_std / current_std.clamp_min(1e-9)
    fraction = torch.clamp(
        (ratio - config.full_trust_std_ratio)
        / (config.zero_trust_std_ratio - config.full_trust_std_ratio),
        0.0,
        1.0,
    )
    return 1.0 - fraction * (1.0 - config.minimum_prediction_weight)


def adaptive_position_surrogate_actions(
    output: GRUTargetStateOutput,
    context: GRUAdaptivePositionLossContext,
    prediction_horizons_s: tuple[float, ...],
    config: AdaptivePositionControllerConfig,
    sequence_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiably replay the selected adaptive position adapter."""

    expected_shape = output.mean.shape[:2]
    if sequence_mask.shape != expected_shape:
        raise ValueError("adaptive position sequence mask shape is invalid")
    for value in (
        context.teacher_action_normalized,
        context.mask,
        context.gimbal_angle_rad,
        context.gimbal_rate_rad_s,
        context.control_dt_s,
        context.selected_axis_fov_rad,
        context.servo_min_angle_rad,
        context.servo_max_angle_rad,
        context.servo_max_rate_rad_s,
        context.servo_max_acceleration_rad_s2,
        context.servo_position_gain_s_inv,
        context.servo_position_tolerance_rad,
        context.servo_position_quantization_rad,
        context.servo_command_polarity,
        context.servo_command_latency_s,
        context.servo_rate_time_constant_s,
        context.control_period_s,
        context.integration_period_s,
        context.camera_frame_period_s,
    ):
        if value.shape != expected_shape:
            raise ValueError("adaptive position context shape is invalid")
    if len(prediction_horizons_s) != output.mean.shape[-2]:
        raise ValueError("adaptive position horizons do not match output")
    if any(
        right <= left
        for left, right in zip(
            prediction_horizons_s,
            prediction_horizons_s[1:],
        )
    ):
        raise ValueError(
            "adaptive position horizons must be strictly increasing"
        )

    arrival = config.actuator_arrival_time_scale * (
        context.servo_command_latency_s
        + context.servo_rate_time_constant_s
        + config.position_response_fraction
        / context.servo_position_gain_s_inv
    ) + config.additional_preview_s
    arrival = torch.clamp(
        arrival,
        prediction_horizons_s[0],
        prediction_horizons_s[-1],
    )
    current_bearing = output.mean[..., 0, 0]
    current_rate = output.mean[..., 0, 1]
    current_bearing_std = output.std[..., 0, 0]
    base_bearing, base_rate, base_std, _base_rate_std = (
        _interpolate_gru_output(output, arrival, prediction_horizons_s)
    )
    base_weight = _adaptive_prediction_weight(
        current_bearing_std,
        base_std,
        config,
    )
    blended_base_bearing = current_bearing + base_weight * (
        angular_residual_rad(base_bearing, current_bearing)
    )
    blended_base_bearing = angular_residual_rad(
        blended_base_bearing,
        torch.zeros_like(blended_base_bearing),
    )
    blended_base_rate = current_rate + base_weight * (
        base_rate - current_rate
    )
    blended_base_std = current_bearing_std + base_weight * (
        base_std - current_bearing_std
    )
    image_error = angular_residual_rad(
        blended_base_bearing,
        context.gimbal_angle_rad,
    )
    fov_fraction = (
        torch.abs(image_error)
        + config.visibility_uncertainty_sigma * blended_base_std
    ) / (0.5 * context.selected_axis_fov_rad)
    risk = torch.clamp(
        (fov_fraction - config.visibility_risk_onset_fraction)
        / (
            config.visibility_risk_full_fraction
            - config.visibility_risk_onset_fraction
        ),
        0.0,
        1.0,
    )
    if config.risk_requires_outward_motion:
        outward = image_error * (
            blended_base_rate - context.gimbal_rate_rad_s
        ) > 0.0
        risk = torch.where(outward, risk, torch.zeros_like(risk))
    requested = torch.clamp(
        arrival + risk * config.risk_horizon_boost_s,
        prediction_horizons_s[0],
        prediction_horizons_s[-1],
    )
    forecast_bearing, _forecast_rate, forecast_std, _forecast_rate_std = (
        _interpolate_gru_output(output, requested, prediction_horizons_s)
    )
    prediction_weight = _adaptive_prediction_weight(
        current_bearing_std,
        forecast_std,
        config,
    )
    raw_target = current_bearing + prediction_weight * (
        angular_residual_rad(forecast_bearing, current_bearing)
    )
    if output.position_target_residual_fov_fraction is not None:
        if (
            output.position_target_residual_fov_fraction.shape
            != expected_shape
        ):
            raise ValueError(
                "position-target residual shape is invalid"
            )
        raw_target = raw_target + (
            output.position_target_residual_fov_fraction
            * 0.5
            * context.selected_axis_fov_rad
        )
    raw_target = angular_residual_rad(
        raw_target,
        torch.zeros_like(raw_target),
    )
    raw_target = torch.clamp(
        raw_target,
        context.servo_min_angle_rad,
        context.servo_max_angle_rad,
    )

    batch_size, time_count = expected_shape
    setpoint_angle = context.gimbal_angle_rad[:, 0]
    setpoint_rate = output.mean.new_zeros(batch_size)
    setpoint_acceleration = output.mean.new_zeros(batch_size)
    commands = []
    for time_index in range(time_count):
        active = sequence_mask[:, time_index].bool()
        target = raw_target[:, time_index]
        dt_s = context.control_dt_s[:, time_index].clamp_min(1e-9)
        step_risk = risk[:, time_index]
        maximum_rate = (
            config.setpoint_rate_limit_scale
            * context.servo_max_rate_rad_s[:, time_index]
            * (
                1.0
                + step_risk
                * (config.risk_rate_limit_multiplier - 1.0)
            )
        )
        base_acceleration = (
            config.setpoint_acceleration_limit_scale
            * context.servo_max_acceleration_rad_s2[:, time_index]
        )
        maximum_acceleration = base_acceleration * (
            1.0
            + step_risk
            * (config.risk_acceleration_limit_multiplier - 1.0)
        )
        maximum_jerk = (
            base_acceleration
            / config.setpoint_jerk_rise_time_s
            * (
                1.0
                + step_risk
                * (config.risk_jerk_limit_multiplier - 1.0)
            )
        )
        error = target - setpoint_angle
        stopping_speed = torch.sqrt(
            2.0 * maximum_acceleration * torch.abs(error) + 1e-12
        )
        desired_rate = torch.sign(error) * torch.minimum(
            maximum_rate,
            stopping_speed,
        )
        raw_acceleration = (desired_rate - setpoint_rate) / dt_s
        desired_acceleration = torch.clamp(
            raw_acceleration,
            -maximum_acceleration,
            maximum_acceleration,
        )
        acceleration = setpoint_acceleration + torch.clamp(
            desired_acceleration - setpoint_acceleration,
            -maximum_jerk * dt_s,
            maximum_jerk * dt_s,
        )
        rate = torch.clamp(
            setpoint_rate + acceleration * dt_s,
            -maximum_rate,
            maximum_rate,
        )
        step = rate * dt_s
        overshoot = (step * error > 0.0) & (
            torch.abs(step) >= torch.abs(error)
        )
        next_angle = torch.where(
            overshoot,
            target,
            torch.clamp(
                setpoint_angle + step,
                context.servo_min_angle_rad[:, time_index],
                context.servo_max_angle_rad[:, time_index],
            ),
        )
        rate = torch.where(overshoot, torch.zeros_like(rate), rate)
        acceleration = torch.where(
            overshoot,
            torch.zeros_like(acceleration),
            acceleration,
        )
        setpoint_angle = torch.where(active, next_angle, setpoint_angle)
        setpoint_rate = torch.where(active, rate, setpoint_rate)
        setpoint_acceleration = torch.where(
            active,
            acceleration,
            setpoint_acceleration,
        )
        normalized = torch.where(
            setpoint_angle >= 0.0,
            setpoint_angle / context.servo_max_angle_rad[:, time_index],
            setpoint_angle / (-context.servo_min_angle_rad[:, time_index]),
        )
        commands.append(torch.clamp(normalized, -1.0, 1.0))
    return torch.stack(commands, dim=1)


def differentiable_position_servo_rollout(
    command_normalized: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    *,
    duration_s: float,
    integration_period_override_s: float | None = None,
) -> GRUPositionPlantRollout:
    """Roll one causal position command through the configured servo plant.

    The current command is repeated at each control tick, matching a
    zero-order hold when no future observation is available.  Each sample
    begins from logged gimbal angle/rate and holds its initial angle until the
    first command arrives.  Command, control, camera-frame, and integration
    event boundaries mirror :class:`GimbalServoEnv`. Quantization uses a
    straight-through gradient while retaining the exact forward value.
    """

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("plant rollout duration must be finite and positive")
    if integration_period_override_s is not None and (
        not math.isfinite(integration_period_override_s)
        or integration_period_override_s <= 0.0
    ):
        raise ValueError(
            "plant integration-period override must be finite and positive"
        )
    shape = command_normalized.shape
    for value in (
        context.gimbal_angle_rad,
        context.gimbal_rate_rad_s,
        context.servo_min_angle_rad,
        context.servo_max_angle_rad,
        context.servo_max_rate_rad_s,
        context.servo_max_acceleration_rad_s2,
        context.servo_position_gain_s_inv,
        context.servo_position_tolerance_rad,
        context.servo_position_quantization_rad,
        context.servo_command_polarity,
        context.servo_command_latency_s,
        context.servo_rate_time_constant_s,
        context.control_period_s,
        context.integration_period_s,
        context.camera_frame_period_s,
    ):
        if value.shape != shape:
            raise ValueError("position plant context shape is invalid")
    for value, name in (
        (context.servo_max_rate_rad_s, "maximum rate"),
        (context.servo_max_acceleration_rad_s2, "maximum acceleration"),
        (context.servo_position_gain_s_inv, "position gain"),
        (context.control_period_s, "control period"),
        (context.integration_period_s, "integration period"),
        (context.camera_frame_period_s, "camera frame period"),
    ):
        if torch.any(value <= 0.0):
            raise ValueError(f"servo {name} must be positive")

    command_normalized = torch.clamp(command_normalized, -1.0, 1.0)
    travel = torch.where(
        command_normalized >= 0.0,
        context.servo_max_angle_rad,
        -context.servo_min_angle_rad,
    )
    requested_position = command_normalized * travel
    adapted_position = requested_position * context.servo_command_polarity
    quantum = context.servo_position_quantization_rad
    safe_quantum = quantum.clamp_min(torch.finfo(adapted_position.dtype).eps)
    quantized = torch.round(adapted_position / safe_quantum) * safe_quantum
    exact_adapted_position = torch.where(
        quantum > 0.0,
        quantized,
        adapted_position,
    )
    adapted_position = adapted_position + (
        exact_adapted_position - adapted_position
    ).detach()
    adapted_position = torch.maximum(
        adapted_position,
        context.servo_min_angle_rad,
    )
    adapted_position = torch.minimum(
        adapted_position,
        context.servo_max_angle_rad,
    )

    integration_period = (
        torch.full_like(
            context.integration_period_s,
            integration_period_override_s,
        )
        if integration_period_override_s is not None
        else context.integration_period_s
    )
    minimum_integration = float(
        torch.min(integration_period.detach()).cpu().item()
    )
    minimum_control = float(
        torch.min(context.control_period_s.detach()).cpu().item()
    )
    minimum_frame = float(
        torch.min(context.camera_frame_period_s.detach()).cpu().item()
    )
    maximum_events = (
        math.ceil(duration_s / minimum_integration)
        + 2 * math.ceil(duration_s / minimum_control)
        + math.ceil(duration_s / minimum_frame)
        + 8
    )

    angle = context.gimbal_angle_rad
    rate = context.gimbal_rate_rad_s
    applied_position = context.gimbal_angle_rad
    elapsed = torch.zeros_like(angle)
    next_arrival = context.servo_command_latency_s
    next_control = context.control_period_s
    next_capture = context.camera_frame_period_s
    saturation_integral = torch.zeros_like(angle)
    duration = torch.full_like(angle, duration_s)
    epsilon = 8.0 * torch.finfo(angle.dtype).eps

    for _ in range(maximum_events):
        active = elapsed < duration - epsilon
        arrival_due = active & (next_arrival <= elapsed + epsilon)
        applied_position = torch.where(
            arrival_due,
            adapted_position,
            applied_position,
        )
        next_arrival = torch.where(
            arrival_due,
            next_arrival + context.control_period_s,
            next_arrival,
        )

        next_time = torch.minimum(duration, elapsed + integration_period)
        next_time = torch.minimum(next_time, next_arrival)
        next_time = torch.minimum(next_time, next_control)
        next_time = torch.minimum(next_time, next_capture)
        dt_s = torch.where(
            active,
            torch.clamp(next_time - elapsed, min=0.0),
            torch.zeros_like(elapsed),
        )

        position_error = applied_position - angle
        desired_rate_unclipped = context.servo_position_gain_s_inv * position_error
        desired_rate_unclipped = torch.where(
            torch.abs(position_error) <= context.servo_position_tolerance_rad,
            torch.zeros_like(desired_rate_unclipped),
            desired_rate_unclipped,
        )
        desired_rate = torch.clamp(
            desired_rate_unclipped,
            -context.servo_max_rate_rad_s,
            context.servo_max_rate_rad_s,
        )
        acceleration_denominator = torch.where(
            context.servo_rate_time_constant_s > 0.0,
            context.servo_rate_time_constant_s,
            dt_s.clamp_min(torch.finfo(dt_s.dtype).eps),
        )
        acceleration_unclipped = (desired_rate - rate) / acceleration_denominator
        acceleration = torch.clamp(
            acceleration_unclipped,
            -context.servo_max_acceleration_rad_s2,
            context.servo_max_acceleration_rad_s2,
        )
        rate_unclipped = rate + acceleration * dt_s
        next_rate = torch.clamp(
            rate_unclipped,
            -context.servo_max_rate_rad_s,
            context.servo_max_rate_rad_s,
        )
        angle_unclipped = angle + next_rate * dt_s
        next_angle = torch.maximum(
            angle_unclipped,
            context.servo_min_angle_rad,
        )
        next_angle = torch.minimum(
            next_angle,
            context.servo_max_angle_rad,
        )
        pushing_lower = (
            next_angle <= context.servo_min_angle_rad + epsilon
        ) & (next_rate < 0.0) & (angle_unclipped < context.servo_min_angle_rad)
        pushing_upper = (
            next_angle >= context.servo_max_angle_rad - epsilon
        ) & (next_rate > 0.0) & (angle_unclipped > context.servo_max_angle_rad)
        next_rate = torch.where(
            pushing_lower | pushing_upper,
            torch.zeros_like(next_rate),
            next_rate,
        )

        travel_scale = torch.where(
            angle_unclipped >= 0.0,
            context.servo_max_angle_rad,
            -context.servo_min_angle_rad,
        )
        saturation = (
            torch.relu(
                torch.abs(desired_rate_unclipped)
                / context.servo_max_rate_rad_s
                - 1.0
            ).square()
            + torch.relu(
                torch.abs(acceleration_unclipped)
                / context.servo_max_acceleration_rad_s2
                - 1.0
            ).square()
            + torch.relu(
                torch.abs(rate_unclipped) / context.servo_max_rate_rad_s
                - 1.0
            ).square()
            + torch.relu(torch.abs(angle_unclipped) / travel_scale - 1.0).square()
        )
        saturation_integral = saturation_integral + saturation * dt_s
        angle = torch.where(active, next_angle, angle)
        rate = torch.where(active, next_rate, rate)
        elapsed = torch.where(active, next_time, elapsed)

        control_due = next_control <= elapsed + epsilon
        capture_due = next_capture <= elapsed + epsilon
        next_control = torch.where(
            control_due,
            next_control + context.control_period_s,
            next_control,
        )
        next_capture = torch.where(
            capture_due,
            next_capture + context.camera_frame_period_s,
            next_capture,
        )

    return GRUPositionPlantRollout(
        angle_rad=angle,
        rate_rad_s=rate,
        saturation_fraction=saturation_integral / duration,
    )


def initialize_differentiable_position_servo_state(
    context: GRUAdaptivePositionLossContext,
    *,
    initial_time_s: torch.Tensor | None = None,
    initial_applied_position_rad: torch.Tensor | None = None,
) -> GRUPositionPlantState:
    """Initialize a latency-preserving servo state from a rollout context."""

    if context.gimbal_angle_rad.ndim != 2:
        raise ValueError("position servo context must have batch and time axes")
    batch_size, time_count = context.gimbal_angle_rad.shape
    if time_count == 0:
        raise ValueError("position servo context must contain a time step")
    angle = context.gimbal_angle_rad[:, 0]
    rate = context.gimbal_rate_rad_s[:, 0]
    applied_position = (
        angle
        if initial_applied_position_rad is None
        else initial_applied_position_rad
    )
    if applied_position.shape != (batch_size,):
        raise ValueError("initial applied position shape is invalid")
    elapsed = (
        angle.new_zeros(batch_size)
        if initial_time_s is None
        else initial_time_s
    )
    if elapsed.shape != (batch_size,):
        raise ValueError("initial sequence time shape is invalid")
    frame_period = context.camera_frame_period_s[:, 0]
    next_capture = (
        torch.floor(elapsed / frame_period + 1e-7) + 1.0
    ) * frame_period
    return GRUPositionPlantState(
        angle_rad=angle,
        rate_rad_s=rate,
        applied_position_rad=applied_position,
        elapsed_s=elapsed,
        next_capture_s=next_capture,
    )


def differentiable_position_servo_step(
    command_normalized: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    active_step: torch.Tensor,
    time_index: int,
    state: GRUPositionPlantState,
    *,
    integration_period_override_s: float | None = None,
) -> tuple[GRUPositionPlantState, torch.Tensor]:
    """Advance one command while retaining latency and camera event state."""

    if command_normalized.ndim != 1:
        raise ValueError("position servo step command must have shape [batch]")
    batch_size = command_normalized.shape[0]
    if active_step.shape != (batch_size,):
        raise ValueError("position servo step mask shape is invalid")
    if not 0 <= time_index < context.gimbal_angle_rad.shape[1]:
        raise ValueError("position servo step time index is invalid")
    if integration_period_override_s is not None and (
        not math.isfinite(integration_period_override_s)
        or integration_period_override_s <= 0.0
    ):
        raise ValueError(
            "sequence integration-period override must be finite and positive"
        )

    angle = state.angle_rad
    rate = state.rate_rad_s
    applied_position = state.applied_position_rad
    elapsed = state.elapsed_s
    next_capture = state.next_capture_s
    for value, name in (
        (angle, "angle"),
        (rate, "rate"),
        (applied_position, "applied position"),
        (elapsed, "elapsed time"),
        (next_capture, "next capture"),
    ):
        if value.shape != (batch_size,):
            raise ValueError(f"position servo state {name} shape is invalid")
    if not (
        len(state.pending_positions_rad)
        == len(state.pending_arrivals_s)
        == len(state.pending_active)
    ):
        raise ValueError("position servo pending state is inconsistent")

    command = torch.clamp(command_normalized, -1.0, 1.0)
    travel = torch.where(
        command >= 0.0,
        context.servo_max_angle_rad[:, time_index],
        -context.servo_min_angle_rad[:, time_index],
    )
    adapted = (
        command
        * travel
        * context.servo_command_polarity[:, time_index]
    )
    quantum = context.servo_position_quantization_rad[:, time_index]
    safe_quantum = quantum.clamp_min(torch.finfo(adapted.dtype).eps)
    quantized = torch.round(adapted / safe_quantum) * safe_quantum
    exact_adapted = torch.where(quantum > 0.0, quantized, adapted)
    adapted = adapted + (exact_adapted - adapted).detach()
    adapted = torch.maximum(
        adapted,
        context.servo_min_angle_rad[:, time_index],
    )
    adapted = torch.minimum(
        adapted,
        context.servo_max_angle_rad[:, time_index],
    )
    arrival = torch.where(
        active_step,
        elapsed + context.servo_command_latency_s[:, time_index],
        torch.full_like(elapsed, math.inf),
    )
    pending_positions = [*state.pending_positions_rad, adapted]
    pending_arrivals = [*state.pending_arrivals_s, arrival]
    pending_active = [*state.pending_active, active_step]
    duration = context.control_period_s[:, time_index]
    end_time = torch.where(active_step, elapsed + duration, elapsed)
    integration_period = (
        torch.full_like(
            context.integration_period_s[:, time_index],
            integration_period_override_s,
        )
        if integration_period_override_s is not None
        else context.integration_period_s[:, time_index]
    )
    minimum_integration = float(
        torch.min(integration_period.detach()).cpu().item()
    )
    maximum_duration = float(torch.max(duration.detach()).cpu().item())
    maximum_events = (
        math.ceil(maximum_duration / minimum_integration)
        + len(pending_arrivals)
        + 4
    )
    saturation_integral = torch.zeros_like(angle)
    epsilon = 8.0 * torch.finfo(angle.dtype).eps

    for _ in range(maximum_events):
        active = active_step & (elapsed < end_time - epsilon)
        for pending_position, pending_arrival, issued in zip(
            pending_positions,
            pending_arrivals,
            pending_active,
        ):
            due = active & issued & (pending_arrival <= elapsed + epsilon)
            applied_position = torch.where(
                due,
                pending_position,
                applied_position,
            )

        next_time = torch.minimum(end_time, elapsed + integration_period)
        for pending_arrival in pending_arrivals:
            future_arrival = torch.where(
                pending_arrival > elapsed + epsilon,
                pending_arrival,
                torch.full_like(pending_arrival, math.inf),
            )
            next_time = torch.minimum(next_time, future_arrival)
        next_time = torch.minimum(next_time, next_capture)
        dt_s = torch.where(
            active,
            torch.clamp(next_time - elapsed, min=0.0),
            torch.zeros_like(elapsed),
        )

        position_error = applied_position - angle
        position_gain = context.servo_position_gain_s_inv[:, time_index]
        desired_rate_unclipped = position_gain * position_error
        desired_rate_unclipped = torch.where(
            torch.abs(position_error)
            <= context.servo_position_tolerance_rad[:, time_index],
            torch.zeros_like(desired_rate_unclipped),
            desired_rate_unclipped,
        )
        maximum_rate = context.servo_max_rate_rad_s[:, time_index]
        desired_rate = torch.clamp(
            desired_rate_unclipped,
            -maximum_rate,
            maximum_rate,
        )
        time_constant = context.servo_rate_time_constant_s[:, time_index]
        acceleration_denominator = torch.where(
            time_constant > 0.0,
            time_constant,
            dt_s.clamp_min(torch.finfo(dt_s.dtype).eps),
        )
        acceleration_unclipped = (
            desired_rate - rate
        ) / acceleration_denominator
        maximum_acceleration = (
            context.servo_max_acceleration_rad_s2[:, time_index]
        )
        acceleration = torch.clamp(
            acceleration_unclipped,
            -maximum_acceleration,
            maximum_acceleration,
        )
        rate_unclipped = rate + acceleration * dt_s
        next_rate = torch.clamp(
            rate_unclipped,
            -maximum_rate,
            maximum_rate,
        )
        angle_unclipped = angle + next_rate * dt_s
        minimum_angle = context.servo_min_angle_rad[:, time_index]
        maximum_angle = context.servo_max_angle_rad[:, time_index]
        next_angle = torch.maximum(angle_unclipped, minimum_angle)
        next_angle = torch.minimum(next_angle, maximum_angle)
        pushing_lower = (
            (next_angle <= minimum_angle + epsilon)
            & (next_rate < 0.0)
            & (angle_unclipped < minimum_angle)
        )
        pushing_upper = (
            (next_angle >= maximum_angle - epsilon)
            & (next_rate > 0.0)
            & (angle_unclipped > maximum_angle)
        )
        next_rate = torch.where(
            pushing_lower | pushing_upper,
            torch.zeros_like(next_rate),
            next_rate,
        )
        travel_scale = torch.where(
            angle_unclipped >= 0.0,
            maximum_angle,
            -minimum_angle,
        )
        saturation = (
            torch.relu(
                torch.abs(desired_rate_unclipped) / maximum_rate - 1.0
            ).square()
            + torch.relu(
                torch.abs(acceleration_unclipped)
                / maximum_acceleration
                - 1.0
            ).square()
            + torch.relu(
                torch.abs(rate_unclipped) / maximum_rate - 1.0
            ).square()
            + torch.relu(
                torch.abs(angle_unclipped) / travel_scale - 1.0
            ).square()
        )
        saturation_integral = saturation_integral + saturation * dt_s
        angle = torch.where(active, next_angle, angle)
        rate = torch.where(active, next_rate, rate)
        elapsed = torch.where(active, next_time, elapsed)
        capture_due = active & (next_capture <= elapsed + epsilon)
        next_capture = torch.where(
            capture_due,
            next_capture + context.camera_frame_period_s[:, time_index],
            next_capture,
        )

    for pending_position, pending_arrival, issued in zip(
        pending_positions,
        pending_arrivals,
        pending_active,
    ):
        due = active_step & issued & (pending_arrival <= elapsed + epsilon)
        applied_position = torch.where(
            due,
            pending_position,
            applied_position,
        )
    keep = [
        not bool(
            torch.all(
                (~issued) | (pending_arrival <= elapsed + epsilon)
            ).detach()
        )
        for pending_arrival, issued in zip(
            pending_arrivals,
            pending_active,
        )
    ]
    next_state = GRUPositionPlantState(
        angle_rad=angle,
        rate_rad_s=rate,
        applied_position_rad=applied_position,
        elapsed_s=elapsed,
        next_capture_s=next_capture,
        pending_positions_rad=tuple(
            value
            for value, retain in zip(pending_positions, keep)
            if retain
        ),
        pending_arrivals_s=tuple(
            value
            for value, retain in zip(pending_arrivals, keep)
            if retain
        ),
        pending_active=tuple(
            value
            for value, retain in zip(pending_active, keep)
            if retain
        ),
    )
    saturation_fraction = torch.where(
        active_step,
        saturation_integral / duration,
        torch.zeros_like(duration),
    )
    return next_state, saturation_fraction


def differentiable_position_servo_sequence(
    command_normalized: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    sequence_mask: torch.Tensor,
    *,
    initial_time_s: torch.Tensor | None = None,
    initial_applied_position_rad: torch.Tensor | None = None,
    integration_period_override_s: float | None = None,
) -> GRUPositionPlantSequenceRollout:
    """Roll a causal command sequence through one persistent servo plant.

    Unlike :func:`differentiable_position_servo_rollout`, each command is
    submitted once and its latency queue persists into subsequent control
    intervals. Camera capture boundaries are retained because the simulator's
    explicit Euler integration is event split at those times. Quantization is
    straight-through in the same way as the single-command rollout.
    """

    if command_normalized.ndim != 2:
        raise ValueError("position command sequence must have shape [batch, time]")
    shape = command_normalized.shape
    if sequence_mask.shape != shape:
        raise ValueError("position command sequence mask shape is invalid")
    for value in (
        context.gimbal_angle_rad,
        context.gimbal_rate_rad_s,
        context.servo_min_angle_rad,
        context.servo_max_angle_rad,
        context.servo_max_rate_rad_s,
        context.servo_max_acceleration_rad_s2,
        context.servo_position_gain_s_inv,
        context.servo_position_tolerance_rad,
        context.servo_position_quantization_rad,
        context.servo_command_polarity,
        context.servo_command_latency_s,
        context.servo_rate_time_constant_s,
        context.control_period_s,
        context.integration_period_s,
        context.camera_frame_period_s,
    ):
        if value.shape != shape:
            raise ValueError("position sequence context shape is invalid")
    if integration_period_override_s is not None and (
        not math.isfinite(integration_period_override_s)
        or integration_period_override_s <= 0.0
    ):
        raise ValueError(
            "sequence integration-period override must be finite and positive"
        )
    for value, name in (
        (context.servo_max_rate_rad_s, "maximum rate"),
        (context.servo_max_acceleration_rad_s2, "maximum acceleration"),
        (context.servo_position_gain_s_inv, "position gain"),
        (context.control_period_s, "control period"),
        (context.integration_period_s, "integration period"),
        (context.camera_frame_period_s, "camera frame period"),
    ):
        if torch.any(value <= 0.0):
            raise ValueError(f"servo {name} must be positive")

    batch_size, time_count = shape
    angle = context.gimbal_angle_rad[:, 0]
    rate = context.gimbal_rate_rad_s[:, 0]
    applied_position = (
        angle
        if initial_applied_position_rad is None
        else initial_applied_position_rad
    )
    if applied_position.shape != (batch_size,):
        raise ValueError("initial applied position shape is invalid")
    elapsed = (
        angle.new_zeros(batch_size)
        if initial_time_s is None
        else initial_time_s
    )
    if elapsed.shape != (batch_size,):
        raise ValueError("initial sequence time shape is invalid")
    frame_period = context.camera_frame_period_s[:, 0]
    next_capture = (
        torch.floor(elapsed / frame_period + 1e-7) + 1.0
    ) * frame_period
    pending_positions: list[torch.Tensor] = []
    pending_arrivals: list[torch.Tensor] = []
    pending_active: list[torch.Tensor] = []
    output_angles = []
    output_rates = []
    output_applied = []
    output_saturation = []
    epsilon = 8.0 * torch.finfo(angle.dtype).eps

    for time_index in range(time_count):
        active_step = sequence_mask[:, time_index].bool()
        command = torch.clamp(
            command_normalized[:, time_index],
            -1.0,
            1.0,
        )
        travel = torch.where(
            command >= 0.0,
            context.servo_max_angle_rad[:, time_index],
            -context.servo_min_angle_rad[:, time_index],
        )
        adapted = (
            command
            * travel
            * context.servo_command_polarity[:, time_index]
        )
        quantum = context.servo_position_quantization_rad[:, time_index]
        safe_quantum = quantum.clamp_min(torch.finfo(adapted.dtype).eps)
        quantized = torch.round(adapted / safe_quantum) * safe_quantum
        exact_adapted = torch.where(quantum > 0.0, quantized, adapted)
        adapted = adapted + (exact_adapted - adapted).detach()
        adapted = torch.maximum(
            adapted,
            context.servo_min_angle_rad[:, time_index],
        )
        adapted = torch.minimum(
            adapted,
            context.servo_max_angle_rad[:, time_index],
        )
        arrival = torch.where(
            active_step,
            elapsed + context.servo_command_latency_s[:, time_index],
            torch.full_like(elapsed, math.inf),
        )
        pending_positions.append(adapted)
        pending_arrivals.append(arrival)
        pending_active.append(active_step)
        duration = context.control_period_s[:, time_index]
        end_time = torch.where(active_step, elapsed + duration, elapsed)
        integration_period = (
            torch.full_like(
                context.integration_period_s[:, time_index],
                integration_period_override_s,
            )
            if integration_period_override_s is not None
            else context.integration_period_s[:, time_index]
        )
        minimum_integration = float(
            torch.min(integration_period.detach()).cpu().item()
        )
        maximum_duration = float(torch.max(duration.detach()).cpu().item())
        maximum_events = (
            math.ceil(maximum_duration / minimum_integration)
            + len(pending_arrivals)
            + 4
        )
        saturation_integral = torch.zeros_like(angle)

        for _ in range(maximum_events):
            active = active_step & (elapsed < end_time - epsilon)
            for pending_position, pending_arrival, issued in zip(
                pending_positions,
                pending_arrivals,
                pending_active,
            ):
                due = (
                    active
                    & issued
                    & (pending_arrival <= elapsed + epsilon)
                )
                applied_position = torch.where(
                    due,
                    pending_position,
                    applied_position,
                )

            next_time = torch.minimum(end_time, elapsed + integration_period)
            for pending_arrival in pending_arrivals:
                future_arrival = torch.where(
                    pending_arrival > elapsed + epsilon,
                    pending_arrival,
                    torch.full_like(pending_arrival, math.inf),
                )
                next_time = torch.minimum(next_time, future_arrival)
            next_time = torch.minimum(next_time, next_capture)
            dt_s = torch.where(
                active,
                torch.clamp(next_time - elapsed, min=0.0),
                torch.zeros_like(elapsed),
            )

            position_error = applied_position - angle
            position_gain = context.servo_position_gain_s_inv[:, time_index]
            desired_rate_unclipped = position_gain * position_error
            desired_rate_unclipped = torch.where(
                torch.abs(position_error)
                <= context.servo_position_tolerance_rad[:, time_index],
                torch.zeros_like(desired_rate_unclipped),
                desired_rate_unclipped,
            )
            maximum_rate = context.servo_max_rate_rad_s[:, time_index]
            desired_rate = torch.clamp(
                desired_rate_unclipped,
                -maximum_rate,
                maximum_rate,
            )
            time_constant = context.servo_rate_time_constant_s[:, time_index]
            acceleration_denominator = torch.where(
                time_constant > 0.0,
                time_constant,
                dt_s.clamp_min(torch.finfo(dt_s.dtype).eps),
            )
            acceleration_unclipped = (
                desired_rate - rate
            ) / acceleration_denominator
            maximum_acceleration = (
                context.servo_max_acceleration_rad_s2[:, time_index]
            )
            acceleration = torch.clamp(
                acceleration_unclipped,
                -maximum_acceleration,
                maximum_acceleration,
            )
            rate_unclipped = rate + acceleration * dt_s
            next_rate = torch.clamp(
                rate_unclipped,
                -maximum_rate,
                maximum_rate,
            )
            angle_unclipped = angle + next_rate * dt_s
            minimum_angle = context.servo_min_angle_rad[:, time_index]
            maximum_angle = context.servo_max_angle_rad[:, time_index]
            next_angle = torch.maximum(angle_unclipped, minimum_angle)
            next_angle = torch.minimum(next_angle, maximum_angle)
            pushing_lower = (
                (next_angle <= minimum_angle + epsilon)
                & (next_rate < 0.0)
                & (angle_unclipped < minimum_angle)
            )
            pushing_upper = (
                (next_angle >= maximum_angle - epsilon)
                & (next_rate > 0.0)
                & (angle_unclipped > maximum_angle)
            )
            next_rate = torch.where(
                pushing_lower | pushing_upper,
                torch.zeros_like(next_rate),
                next_rate,
            )
            travel_scale = torch.where(
                angle_unclipped >= 0.0,
                maximum_angle,
                -minimum_angle,
            )
            saturation = (
                torch.relu(
                    torch.abs(desired_rate_unclipped) / maximum_rate - 1.0
                ).square()
                + torch.relu(
                    torch.abs(acceleration_unclipped)
                    / maximum_acceleration
                    - 1.0
                ).square()
                + torch.relu(
                    torch.abs(rate_unclipped) / maximum_rate - 1.0
                ).square()
                + torch.relu(
                    torch.abs(angle_unclipped) / travel_scale - 1.0
                ).square()
            )
            saturation_integral = saturation_integral + saturation * dt_s
            angle = torch.where(active, next_angle, angle)
            rate = torch.where(active, next_rate, rate)
            elapsed = torch.where(active, next_time, elapsed)
            capture_due = active & (next_capture <= elapsed + epsilon)
            next_capture = torch.where(
                capture_due,
                next_capture + context.camera_frame_period_s[:, time_index],
                next_capture,
            )

        for pending_position, pending_arrival, issued in zip(
            pending_positions,
            pending_arrivals,
            pending_active,
        ):
            due = (
                active_step
                & issued
                & (pending_arrival <= elapsed + epsilon)
            )
            applied_position = torch.where(
                due,
                pending_position,
                applied_position,
            )
        keep = [
            not bool(
                torch.all(
                    (~issued) | (pending_arrival <= elapsed + epsilon)
                ).detach()
            )
            for pending_arrival, issued in zip(
                pending_arrivals,
                pending_active,
            )
        ]
        pending_positions = [
            value for value, retain in zip(pending_positions, keep) if retain
        ]
        pending_arrivals = [
            value for value, retain in zip(pending_arrivals, keep) if retain
        ]
        pending_active = [
            value for value, retain in zip(pending_active, keep) if retain
        ]
        output_angles.append(angle)
        output_rates.append(rate)
        output_applied.append(applied_position)
        output_saturation.append(
            torch.where(
                active_step,
                saturation_integral / duration,
                torch.zeros_like(duration),
            )
        )

    return GRUPositionPlantSequenceRollout(
        angle_rad=torch.stack(output_angles, dim=1),
        rate_rad_s=torch.stack(output_rates, dim=1),
        applied_position_rad=torch.stack(output_applied, dim=1),
        saturation_fraction=torch.stack(output_saturation, dim=1),
    )


def target_state_nll(
    output: GRUTargetStateOutput,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    sequence_mask: torch.Tensor,
    config: GRULossConfig | None = None,
    *,
    label_weights: torch.Tensor | None = None,
    prediction_horizons_s: tuple[float, ...] | None = None,
    control_context: GRUControlLossContext | None = None,
    adaptive_position_context: GRUAdaptivePositionLossContext | None = None,
) -> GRULoss:
    """Masked heteroscedastic Gaussian loss with circular bearing residuals."""
    config = config or GRULossConfig()
    if output.mean.shape != targets.shape or output.std.shape != targets.shape:
        raise ValueError("prediction and target shapes must match")
    expected_mask_shape = targets.shape[:-1]
    if target_mask.shape != expected_mask_shape:
        raise ValueError("target_mask shape is invalid")
    if sequence_mask.shape != targets.shape[:2]:
        raise ValueError("sequence_mask shape is invalid")
    mask = target_mask.bool() & sequence_mask.bool().unsqueeze(-1)
    if label_weights is None:
        weights = torch.ones_like(target_mask, dtype=targets.dtype)
    else:
        if label_weights.shape != target_mask.shape:
            raise ValueError("label_weights shape is invalid")
        if torch.any(~torch.isfinite(label_weights)) or torch.any(
            label_weights < 0.0
        ):
            raise ValueError("label_weights must be finite and non-negative")
        weights = label_weights.to(
            device=targets.device,
            dtype=targets.dtype,
        )
    if config.horizon_weights:
        if len(config.horizon_weights) != targets.shape[-2]:
            raise ValueError("horizon weights do not match model outputs")
        horizon_weights = targets.new_tensor(config.horizon_weights)
        weights = weights * horizon_weights.view(1, 1, -1)

    def weighted_mean(value: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        selected_weights = weights * selected.to(dtype=weights.dtype)
        return (value * selected_weights).sum() / selected_weights.sum().clamp_min(
            1.0
        )

    bearing_error = angular_residual_rad(
        output.mean[..., 0], targets[..., 0]
    )
    rate_error = output.mean[..., 1] - targets[..., 1]
    bearing_std = output.std[..., 0]
    rate_std = output.std[..., 1]
    bearing_nll_values = (
        torch.log(bearing_std)
        + 0.5 * (bearing_error / bearing_std).square()
    )
    rate_nll_values = (
        torch.log(rate_std) + 0.5 * (rate_error / rate_std).square()
    )
    bearing_nll = weighted_mean(bearing_nll_values, mask)
    rate_nll = weighted_mean(rate_nll_values, mask)
    bearing_mse = weighted_mean(bearing_error.square(), mask)
    rate_mse = weighted_mean(rate_error.square(), mask)
    consistency_mse = targets.new_zeros(())
    if config.dynamic_consistency_weight > 0.0:
        if prediction_horizons_s is None:
            raise ValueError(
                "prediction horizons are required for dynamic consistency"
            )
        if len(prediction_horizons_s) != targets.shape[-2]:
            raise ValueError("prediction horizons do not match model outputs")
        if any(
            right <= left
            for left, right in zip(
                prediction_horizons_s,
                prediction_horizons_s[1:],
            )
        ):
            raise ValueError("prediction horizons must be strictly increasing")
        intervals = targets.new_tensor(
            [
                right - left
                for left, right in zip(
                    prediction_horizons_s,
                    prediction_horizons_s[1:],
                )
            ]
        )
        bearing_step = angular_residual_rad(
            output.mean[..., 1:, 0],
            output.mean[..., :-1, 0],
        )
        if output.interval_rate_rad_s is None:
            integrated_rate = 0.5 * (
                output.mean[..., 1:, 1] + output.mean[..., :-1, 1]
            ) * intervals
        else:
            if output.interval_rate_rad_s.shape != targets.shape[:-2] + (
                targets.shape[-2] - 1,
            ):
                raise ValueError("interval rate shape is invalid")
            integrated_rate = (
                output.mean[..., :-1, 1]
                + 4.0 * output.interval_rate_rad_s
                + output.mean[..., 1:, 1]
            ) * intervals / 6.0
        consistency_error = angular_residual_rad(
            bearing_step,
            integrated_rate,
        )
        consistency_mask = mask[..., 1:] & mask[..., :-1]
        pair_weights = 0.5 * (weights[..., 1:] + weights[..., :-1])
        selected_weights = pair_weights * consistency_mask.to(
            dtype=pair_weights.dtype
        )
        consistency_mse = (
            consistency_error.square() * selected_weights
        ).sum() / selected_weights.sum().clamp_min(1.0)
    rate_action_mse = targets.new_zeros(())
    position_action_mse = targets.new_zeros(())
    adaptive_position_action_mse = targets.new_zeros(())
    position_plant_tracking_mse = targets.new_zeros(())
    position_plant_response_mse = targets.new_zeros(())
    position_plant_regret_mse = targets.new_zeros(())
    position_plant_visibility_mse = targets.new_zeros(())
    position_plant_smoothness_mse = targets.new_zeros(())
    position_plant_saturation_mse = targets.new_zeros(())
    if config.rate_action_weight > 0.0 or config.position_action_weight > 0.0:
        if control_context is None:
            raise ValueError(
                "control context is required for action-aware supervision"
            )
        expected_time_shape = targets.shape[:2]
        if control_context.oracle_actions.shape != (*expected_time_shape, 2):
            raise ValueError("oracle action context shape is invalid")
        for value in (
            control_context.gimbal_angle_rad,
            control_context.servo_max_rate_rad_s,
            control_context.servo_min_angle_rad,
            control_context.servo_max_angle_rad,
            control_context.rate_feedback_gain_s_inv,
            control_context.position_preview_s,
            control_context.mask,
        ):
            if value.shape != expected_time_shape:
                raise ValueError("control context time shape is invalid")
        if torch.any(control_context.servo_max_rate_rad_s <= 0.0):
            raise ValueError("servo maximum rate must be positive")
        if torch.any(control_context.servo_min_angle_rad >= 0.0) or torch.any(
            control_context.servo_max_angle_rad <= 0.0
        ):
            raise ValueError("servo position limits must straddle zero")

        predicted_bearing = output.mean[..., 0, 0]
        predicted_rate = output.mean[..., 0, 1]
        bearing_error = angular_residual_rad(
            predicted_bearing,
            control_context.gimbal_angle_rad,
        )
        predicted_rate_action = torch.clamp(
            (
                predicted_rate
                + control_context.rate_feedback_gain_s_inv * bearing_error
            )
            / control_context.servo_max_rate_rad_s,
            -1.0,
            1.0,
        )
        predicted_position = angular_residual_rad(
            predicted_bearing
            + control_context.position_preview_s * predicted_rate,
            torch.zeros_like(predicted_bearing),
        )
        predicted_position = torch.maximum(
            predicted_position,
            control_context.servo_min_angle_rad,
        )
        predicted_position = torch.minimum(
            predicted_position,
            control_context.servo_max_angle_rad,
        )
        predicted_position_action = torch.where(
            predicted_position >= 0.0,
            predicted_position / control_context.servo_max_angle_rad,
            predicted_position / (-control_context.servo_min_angle_rad),
        )
        action_mask = (
            mask[..., 0]
            & control_context.mask.bool()
        )
        action_weights = weights[..., 0] * action_mask.to(weights.dtype)
        action_weight_sum = action_weights.sum().clamp_min(1.0)
        rate_action_mse = (
            (
                predicted_rate_action
                - control_context.oracle_actions[..., 0]
            ).square()
            * action_weights
        ).sum() / action_weight_sum
        position_action_mse = (
            (
                predicted_position_action
                - control_context.oracle_actions[..., 1]
            ).square()
            * action_weights
        ).sum() / action_weight_sum
    plant_aware = any(
        weight > 0.0
        for weight in (
            config.position_plant_tracking_weight,
            config.position_plant_response_weight,
            config.position_plant_regret_weight,
            config.position_plant_visibility_weight,
            config.position_plant_smoothness_weight,
            config.position_plant_saturation_weight,
        )
    )
    predicted_adaptive_action = None
    if config.adaptive_position_action_weight > 0.0 or plant_aware:
        if prediction_horizons_s is None:
            raise ValueError(
                "prediction horizons are required for adaptive position loss"
            )
        if config.adaptive_position_config is None:
            raise ValueError(
                "adaptive position config is required for adaptive position loss"
            )
        if adaptive_position_context is None:
            raise ValueError(
                "adaptive position context is required for adaptive position loss"
            )
        predicted_adaptive_action = adaptive_position_surrogate_actions(
            output,
            adaptive_position_context,
            prediction_horizons_s,
            config.adaptive_position_config,
            sequence_mask,
        )
        if config.adaptive_position_action_weight > 0.0:
            adaptive_mask = (
                mask[..., 0] & adaptive_position_context.mask.bool()
            )
            adaptive_weights = (
                weights[..., 0] * adaptive_mask.to(dtype=weights.dtype)
            )
            adaptive_position_action_mse = (
                (
                    predicted_adaptive_action
                    - adaptive_position_context.teacher_action_normalized
                ).square()
                * adaptive_weights
            ).sum() / adaptive_weights.sum().clamp_min(1.0)
    if plant_aware:
        if config.position_plant_config is None:
            raise ValueError(
                "position plant config is required for plant-aware loss"
            )
        if adaptive_position_context is None or predicted_adaptive_action is None:
            raise ValueError(
                "adaptive position context is required for plant-aware loss"
            )
        horizon_index = config.position_plant_config.horizon_index
        if horizon_index >= targets.shape[-2]:
            raise ValueError("position plant horizon index is out of range")
        assert prediction_horizons_s is not None
        rollout = differentiable_position_servo_rollout(
            predicted_adaptive_action,
            adaptive_position_context,
            duration_s=prediction_horizons_s[horizon_index],
            integration_period_override_s=(
                config.position_plant_config.integration_period_override_s
            ),
        )
        rollout_mask = (
            mask[..., horizon_index]
            & adaptive_position_context.mask.bool()
        )
        rollout_weights = weights[..., horizon_index] * rollout_mask.to(
            dtype=weights.dtype
        )
        rollout_weight_sum = rollout_weights.sum().clamp_min(1.0)
        image_error_normalized = angular_residual_rad(
            targets[..., horizon_index, 0],
            rollout.angle_rad,
        ) / (0.5 * adaptive_position_context.selected_axis_fov_rad)
        position_plant_tracking_mse = (
            image_error_normalized.square() * rollout_weights
        ).sum() / rollout_weight_sum
        if (
            config.position_plant_response_weight > 0.0
            or config.position_plant_regret_weight > 0.0
        ):
            teacher_rollout = differentiable_position_servo_rollout(
                adaptive_position_context.teacher_action_normalized.detach(),
                adaptive_position_context,
                duration_s=prediction_horizons_s[horizon_index],
                integration_period_override_s=(
                    config.position_plant_config.
                    integration_period_override_s
                ),
            )
            response_error_normalized = angular_residual_rad(
                rollout.angle_rad,
                teacher_rollout.angle_rad.detach(),
            ) / (0.5 * adaptive_position_context.selected_axis_fov_rad)
            position_plant_response_mse = (
                response_error_normalized.square() * rollout_weights
            ).sum() / rollout_weight_sum
            teacher_error_normalized = angular_residual_rad(
                targets[..., horizon_index, 0],
                teacher_rollout.angle_rad.detach(),
            ) / (0.5 * adaptive_position_context.selected_axis_fov_rad)
            regret = torch.relu(
                image_error_normalized.square()
                - teacher_error_normalized.square()
            )
            position_plant_regret_mse = (
                regret * rollout_weights
            ).sum() / rollout_weight_sum
        visibility_violation = torch.relu(
            torch.abs(image_error_normalized)
            - config.position_plant_config.visibility_margin_fraction
        )
        position_plant_visibility_mse = (
            visibility_violation.square() * rollout_weights
        ).sum() / rollout_weight_sum
        position_plant_saturation_mse = (
            rollout.saturation_fraction * rollout_weights
        ).sum() / rollout_weight_sum

        smooth_mask = rollout_mask[..., 1:] & rollout_mask[..., :-1]
        # ``weights`` is [batch, time, horizon]; keep the time-axis operation
        # explicit to prevent accidentally smoothing across episodes.
        smooth_weights = 0.5 * (
            weights[:, 1:, horizon_index]
            + weights[:, :-1, horizon_index]
        ) * smooth_mask.to(dtype=weights.dtype)
        position_plant_smoothness_mse = (
            (
                predicted_adaptive_action[:, 1:]
                - predicted_adaptive_action[:, :-1]
            ).square()
            * smooth_weights
        ).sum() / smooth_weights.sum().clamp_min(1.0)
    mean_error = bearing_mse + rate_mse
    total = (
        config.bearing_weight * bearing_nll
        + config.rate_weight * rate_nll
        + config.mean_error_weight * mean_error
        + config.bearing_mean_error_weight * bearing_mse
        + config.rate_mean_error_weight * rate_mse
        + config.dynamic_consistency_weight * consistency_mse
        + config.rate_action_weight * rate_action_mse
        + config.position_action_weight * position_action_mse
        + config.adaptive_position_action_weight
        * adaptive_position_action_mse
        + config.position_plant_tracking_weight
        * position_plant_tracking_mse
        + config.position_plant_response_weight
        * position_plant_response_mse
        + config.position_plant_regret_weight
        * position_plant_regret_mse
        + config.position_plant_visibility_weight
        * position_plant_visibility_mse
        + config.position_plant_smoothness_weight
        * position_plant_smoothness_mse
        + config.position_plant_saturation_weight
        * position_plant_saturation_mse
    )
    return GRULoss(
        total=total,
        bearing_nll=bearing_nll,
        rate_nll=rate_nll,
        bearing_rmse_rad=torch.sqrt(bearing_mse),
        rate_rmse_rad_s=torch.sqrt(rate_mse),
        dynamic_consistency_rmse_rad=torch.sqrt(consistency_mse),
        rate_action_rmse_normalized=torch.sqrt(rate_action_mse),
        position_action_rmse_normalized=torch.sqrt(position_action_mse),
        adaptive_position_action_rmse_normalized=torch.sqrt(
            adaptive_position_action_mse
        ),
        position_plant_tracking_rmse_normalized=torch.sqrt(
            position_plant_tracking_mse
        ),
        position_plant_response_rmse_normalized=torch.sqrt(
            position_plant_response_mse
        ),
        position_plant_regret_rmse_normalized=torch.sqrt(
            position_plant_regret_mse
        ),
        position_plant_visibility_rmse_normalized=torch.sqrt(
            position_plant_visibility_mse
        ),
        position_plant_smoothness_rmse_normalized=torch.sqrt(
            position_plant_smoothness_mse
        ),
        position_plant_saturation_rmse_normalized=torch.sqrt(
            position_plant_saturation_mse
        ),
    )


def gru_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@dataclass(frozen=True)
class GRUInferenceConfig:
    observation_profile: ObservationProfile
    horizon_index: int = 0
    maximum_staleness_s: float = 0.50
    bearing_std_scale: float = 1.0
    rate_std_scale: float = 1.0
    uncertainty_calibration: UncertaintyScaleProvider | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.horizon_index < 0:
            raise ValueError("horizon_index must be non-negative")
        if self.maximum_staleness_s < 0.0:
            raise ValueError("maximum staleness must be non-negative")
        for name in ("bearing_std_scale", "rate_std_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.uncertainty_calibration is not None and (
            self.bearing_std_scale != 1.0 or self.rate_std_scale != 1.0
        ):
            raise ValueError(
                "choose either fixed scales or an uncertainty calibration"
            )


class GRUTargetStateEstimator:
    """Stateful online adapter from a trained GRU to controller estimates."""

    name = "gru_target_state"

    def __init__(
        self,
        model: CausalTargetStateGRU,
        gimbal_config: GimbalServoingConfig,
        inference: GRUInferenceConfig,
    ):
        if model.config.input_dim != len(FEATURE_NAMES):
            raise ValueError("model input schema does not match gimbal features")
        if inference.horizon_index >= model.horizon_count:
            raise ValueError("horizon_index is outside the model output")
        self.model = model
        self.gimbal_config = gimbal_config
        self.inference = inference
        self.device = torch.device(inference.device)
        self.model.to(self.device)
        self.model.eval()
        self._hidden: torch.Tensor | None = None
        self._last_measurement_time_s: float | None = None
        self._last_valid_detection_arrival_s: float | None = None
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self.last_estimates = tuple(
            TargetStateEstimate.missing(0.0)
            for _ in self.model.config.prediction_horizons_s
        )

    @property
    def prediction_horizons_s(self) -> tuple[float, ...]:
        return self.model.config.prediction_horizons_s

    def reset(self) -> None:
        self._hidden = None
        self._last_measurement_time_s = None
        self._last_valid_detection_arrival_s = None
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self.last_estimates = tuple(
            TargetStateEstimate.missing(0.0)
            for _ in self.model.config.prediction_horizons_s
        )

    @torch.no_grad()
    def update_all(
        self,
        observation: GimbalObservation,
        *,
        _calibration_horizon_index: int | None = None,
    ) -> tuple[TargetStateEstimate, ...]:
        vector = encode_deployable_observation(
            observation,
            profile=self.inference.observation_profile,
            config=self.gimbal_config,
        )
        feature = torch.from_numpy(vector).to(self.device)[None, :]
        output = self.model.forward_step(feature, self._hidden)
        self._hidden = output.hidden.detach()

        if observation.frame_updated and observation.detection_valid:
            self._last_valid_detection_arrival_s = observation.time_s
            if observation.measurement_age_s.valid:
                self._last_measurement_time_s = (
                    observation.time_s - observation.measurement_age_s.value
                )
        measurement_time_s = self._last_measurement_time_s
        if measurement_time_s is None or (
            observation.time_s - measurement_time_s
            > self.inference.maximum_staleness_s
        ):
            self.last_estimate = TargetStateEstimate.missing(observation.time_s)
            self.last_estimates = tuple(
                TargetStateEstimate.missing(observation.time_s)
                for _ in self.model.config.prediction_horizons_s
            )
            return self.last_estimates

        detection_gap_s = (
            observation.time_s - self._last_valid_detection_arrival_s
            if self._last_valid_detection_arrival_s is not None
            else None
        )
        estimates = []
        for horizon_index, horizon_s in enumerate(
            self.model.config.prediction_horizons_s
        ):
            mean = output.mean[0, horizon_index].cpu().numpy()
            std = output.std[0, horizon_index].cpu().numpy()
            bearing_scale = self.inference.bearing_std_scale
            rate_scale = self.inference.rate_std_scale
            if self.inference.uncertainty_calibration is not None and (
                _calibration_horizon_index is None
                or horizon_index == _calibration_horizon_index
            ):
                contextual_scales = (
                    self.inference.uncertainty_calibration.scales_for_observation(
                        horizon_index,
                        observation,
                        detection_gap_s,
                    )
                )
                bearing_scale *= contextual_scales[0]
                rate_scale *= contextual_scales[1]
            estimate_time_s = observation.time_s + horizon_s
            estimates.append(
                TargetStateEstimate(
                    time_s=estimate_time_s,
                    measurement_time_s=MaskedScalar(measurement_time_s, True),
                    body_relative_bearing_rad=MaskedScalar(float(mean[0]), True),
                    body_relative_rate_rad_s=MaskedScalar(float(mean[1]), True),
                    bearing_std_rad=MaskedScalar(
                        float(std[0]) * bearing_scale,
                        True,
                    ),
                    rate_std_rad_s=MaskedScalar(
                        float(std[1]) * rate_scale,
                        True,
                    ),
                    prediction_horizon_s=MaskedScalar(
                        estimate_time_s - measurement_time_s,
                        True,
                    ),
                )
            )
        self.last_estimates = tuple(estimates)
        self.last_estimate = self.last_estimates[self.inference.horizon_index]
        return self.last_estimates

    def update(self, observation: GimbalObservation) -> TargetStateEstimate:
        self.update_all(
            observation,
            _calibration_horizon_index=self.inference.horizon_index,
        )
        return self.last_estimate


def save_gru_checkpoint(
    path: str | Path,
    model: CausalTargetStateGRU,
    metadata: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".pt":
        raise ValueError("GRU checkpoint path must have a .pt suffix")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": GRU_CHECKPOINT_SCHEMA_VERSION,
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_gru_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CausalTargetStateGRU, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != GRU_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported GRU checkpoint schema")
    raw_config = dict(payload["model_config"])
    raw_config["prediction_horizons_s"] = tuple(
        raw_config["prediction_horizons_s"]
    )
    config = GRUTargetStateModelConfig(**raw_config)
    model = CausalTargetStateGRU(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload["metadata"])


def save_gru_position_residual_checkpoint(
    path: str | Path,
    model: CausalTargetStateGRUWithPositionResidual,
    metadata: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".pt":
        raise ValueError("GRU residual checkpoint path must have a .pt suffix")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": GRU_POSITION_RESIDUAL_CHECKPOINT_SCHEMA_VERSION,
            "model_config": asdict(model.config),
            "residual_config": asdict(model.residual_config),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_gru_position_residual_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CausalTargetStateGRUWithPositionResidual, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != (
        GRU_POSITION_RESIDUAL_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported GRU residual checkpoint schema")
    raw_config = dict(payload["model_config"])
    raw_config["prediction_horizons_s"] = tuple(
        raw_config["prediction_horizons_s"]
    )
    base_model = CausalTargetStateGRU(GRUTargetStateModelConfig(**raw_config))
    residual_config = GRUPositionResidualConfig(**payload["residual_config"])
    model = CausalTargetStateGRUWithPositionResidual(
        base_model,
        residual_config,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload["metadata"])
