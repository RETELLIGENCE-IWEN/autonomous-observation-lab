"""Causal GRU target-state predictor and streaming estimator adapter.

This module requires the optional ``learning`` dependency.  It is deliberately
not imported by :mod:`gimbal_servoing.__init__`, so the simulator remains usable
without PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import GimbalServoingConfig, ObservationProfile
from .dataset import FEATURE_NAMES, encode_deployable_observation
from .estimators import TargetStateEstimate
from .types import GimbalObservation, MaskedScalar


GRU_CHECKPOINT_SCHEMA_VERSION = "gimbal_gru_v1"


@dataclass(frozen=True)
class GRUTargetStateModelConfig:
    input_dim: int
    prediction_horizons_s: tuple[float, ...]
    hidden_dim: int = 64
    embedding_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
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
        if self.minimum_bearing_std_rad <= 0.0:
            raise ValueError("minimum bearing standard deviation must be positive")
        if self.minimum_rate_std_rad_s <= 0.0:
            raise ValueError("minimum rate standard deviation must be positive")


@dataclass
class GRUTargetStateOutput:
    mean: torch.Tensor
    std: torch.Tensor
    hidden: torch.Tensor


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
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(
                config.hidden_dim,
                len(config.prediction_horizons_s) * 4,
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
        raw = self.head(recurrent).view(
            *recurrent.shape[:2], self.horizon_count, 4
        )
        bearing = math.pi * torch.tanh(raw[..., 0])
        rate = raw[..., 1]
        mean = torch.stack((bearing, rate), dim=-1)
        bearing_std = (
            F.softplus(raw[..., 2])
            + self.config.minimum_bearing_std_rad
        )
        rate_std = (
            F.softplus(raw[..., 3])
            + self.config.minimum_rate_std_rad_s
        )
        std = torch.stack((bearing_std, rate_std), dim=-1)
        return GRUTargetStateOutput(mean=mean, std=std, hidden=hidden)

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
        )


@dataclass(frozen=True)
class GRULossConfig:
    bearing_weight: float = 1.0
    rate_weight: float = 1.0
    mean_error_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.bearing_weight < 0.0 or self.rate_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.mean_error_weight < 0.0:
            raise ValueError("mean error weight must be non-negative")


@dataclass
class GRULoss:
    total: torch.Tensor
    bearing_nll: torch.Tensor
    rate_nll: torch.Tensor
    bearing_rmse_rad: torch.Tensor
    rate_rmse_rad_s: torch.Tensor


def angular_residual_rad(
    prediction_rad: torch.Tensor, target_rad: torch.Tensor
) -> torch.Tensor:
    difference = prediction_rad - target_rad
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def target_state_nll(
    output: GRUTargetStateOutput,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    sequence_mask: torch.Tensor,
    config: GRULossConfig | None = None,
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
    bearing_nll = _masked_mean(bearing_nll_values, mask)
    rate_nll = _masked_mean(rate_nll_values, mask)
    bearing_mse = _masked_mean(bearing_error.square(), mask)
    rate_mse = _masked_mean(rate_error.square(), mask)
    mean_error = bearing_mse + rate_mse
    total = (
        config.bearing_weight * bearing_nll
        + config.rate_weight * rate_nll
        + config.mean_error_weight * mean_error
    )
    return GRULoss(
        total=total,
        bearing_nll=bearing_nll,
        rate_nll=rate_nll,
        bearing_rmse_rad=torch.sqrt(bearing_mse),
        rate_rmse_rad_s=torch.sqrt(rate_mse),
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
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.horizon_index < 0:
            raise ValueError("horizon_index must be non-negative")
        if self.maximum_staleness_s < 0.0:
            raise ValueError("maximum staleness must be non-negative")


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
        self.last_estimate = TargetStateEstimate.missing(0.0)

    def reset(self) -> None:
        self._hidden = None
        self._last_measurement_time_s = None
        self.last_estimate = TargetStateEstimate.missing(0.0)

    @torch.no_grad()
    def update(self, observation: GimbalObservation) -> TargetStateEstimate:
        vector = encode_deployable_observation(
            observation,
            profile=self.inference.observation_profile,
            config=self.gimbal_config,
        )
        feature = torch.from_numpy(vector).to(self.device)[None, :]
        output = self.model.forward_step(feature, self._hidden)
        self._hidden = output.hidden.detach()

        if (
            observation.frame_updated
            and observation.detection_valid
            and observation.measurement_age_s.valid
        ):
            self._last_measurement_time_s = (
                observation.time_s - observation.measurement_age_s.value
            )
        measurement_time_s = self._last_measurement_time_s
        if measurement_time_s is None or (
            observation.time_s - measurement_time_s
            > self.inference.maximum_staleness_s
        ):
            self.last_estimate = TargetStateEstimate.missing(observation.time_s)
            return self.last_estimate

        horizon_index = self.inference.horizon_index
        horizon_s = self.model.config.prediction_horizons_s[horizon_index]
        mean = output.mean[0, horizon_index].cpu().numpy()
        std = output.std[0, horizon_index].cpu().numpy()
        estimate_time_s = observation.time_s + horizon_s
        self.last_estimate = TargetStateEstimate(
            time_s=estimate_time_s,
            measurement_time_s=MaskedScalar(measurement_time_s, True),
            body_relative_bearing_rad=MaskedScalar(float(mean[0]), True),
            body_relative_rate_rad_s=MaskedScalar(float(mean[1]), True),
            bearing_std_rad=MaskedScalar(float(std[0]), True),
            rate_std_rad_s=MaskedScalar(float(std[1]), True),
            prediction_horizon_s=MaskedScalar(
                estimate_time_s - measurement_time_s, True
            ),
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
