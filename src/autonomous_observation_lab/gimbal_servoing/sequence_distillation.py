"""Causal hardware-conditioned policy for V11 sequence distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .dataset import FEATURE_NAMES
from .gru import GRUAdaptivePositionLossContext


_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
HARDWARE_FEATURE_COUNT = 10


@dataclass(frozen=True)
class SequenceDistillationPolicyConfig:
    hidden_dim: int = 48
    embedding_dim: int = 48

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.embedding_dim <= 0:
            raise ValueError("distillation policy dimensions must be positive")


def normalized_hardware_features(
    context: GRUAdaptivePositionLossContext,
) -> torch.Tensor:
    """Expose configured plant values in stable, dimensionless units."""

    return torch.stack(
        (
            context.selected_axis_fov_rad / math.pi,
            context.servo_min_angle_rad / math.pi,
            context.servo_max_angle_rad / math.pi,
            context.servo_max_rate_rad_s / math.pi,
            context.servo_max_acceleration_rad_s2 / (4.0 * math.pi),
            context.servo_position_gain_s_inv / 12.0,
            context.servo_command_latency_s / 0.20,
            context.servo_rate_time_constant_s / 0.20,
            context.control_period_s / 0.05,
            context.camera_frame_period_s / 0.05,
        ),
        dim=-1,
    )


class CausalHardwareConditionedPositionPolicy(nn.Module):
    """Absolute position actor with recurrent self-action feedback."""

    def __init__(self, config: SequenceDistillationPolicyConfig | None = None):
        super().__init__()
        self.config = config or SequenceDistillationPolicyConfig()
        input_dim = len(FEATURE_NAMES) + HARDWARE_FEATURE_COUNT
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(
            self.config.embedding_dim,
            self.config.hidden_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(
        self,
        logged_features: torch.Tensor,
        context: GRUAdaptivePositionLossContext,
        *,
        initial_previous_command_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logged_features.ndim != 3 or logged_features.shape[-1] != len(
            FEATURE_NAMES
        ):
            raise ValueError(
                "distillation features must have shape [batch, time, feature]"
            )
        batch_size, time_count, _ = logged_features.shape
        hardware = normalized_hardware_features(context)
        if hardware.shape != (batch_size, time_count, HARDWARE_FEATURE_COUNT):
            raise ValueError("distillation hardware context shape is invalid")
        previous = (
            logged_features[:, 0, _FEATURE_INDEX["previous_action_normalized"]]
            if initial_previous_command_normalized is None
            else initial_previous_command_normalized
        )
        if previous.shape != (batch_size,):
            raise ValueError("initial distillation command shape is invalid")
        hidden = logged_features.new_zeros(batch_size, self.config.hidden_dim)
        commands = []
        for time_index in range(time_count):
            feature = logged_features[:, time_index].clone()
            feature[:, _FEATURE_INDEX["previous_action_normalized"]] = previous
            minimum = context.servo_min_angle_rad[:, time_index]
            maximum = context.servo_max_angle_rad[:, time_index]
            previous_position = previous * torch.where(
                previous >= 0.0,
                maximum,
                -minimum,
            )
            feature[:, _FEATURE_INDEX["previous_position_command_rad"]] = (
                previous_position
            )
            inputs = torch.cat((feature, hardware[:, time_index]), dim=-1)
            hidden = self.recurrent(self.encoder(inputs), hidden)
            command = torch.tanh(self.head(hidden).squeeze(-1))
            commands.append(command)
            previous = command
        return torch.stack(commands, dim=1)
