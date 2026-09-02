"""Causal command inference for the frozen GRU/V2.1 position reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .controllers import AdaptivePositionControllerConfig
from .dataset import FEATURE_NAMES
from .gru import CausalTargetStateGRU, GRUAdaptivePositionLossContext
from .multi_command_policy import _normalized_position, _policy_target


_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass
class DeployableReferenceSequence:
    command_normalized: torch.Tensor
    risk: torch.Tensor
    arrival_s: torch.Tensor
    requested_horizon_s: torch.Tensor


def deployable_position_commands_from_features(
    base_model: CausalTargetStateGRU,
    features: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    sequence_mask: torch.Tensor,
    *,
    prediction_horizons_s: tuple[float, ...],
    adapter: AdaptivePositionControllerConfig,
    initial_hidden: torch.Tensor | None = None,
) -> DeployableReferenceSequence:
    """Run the midpoint-GRU and V2.1 setpoint filter on supplied causal states."""

    if features.ndim != 3 or features.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(
            "deployable reference features must have shape [batch, time, feature]"
        )
    batch_size, time_count, _ = features.shape
    if sequence_mask.shape != (batch_size, time_count):
        raise ValueError("deployable reference sequence-mask shape is invalid")
    if len(prediction_horizons_s) != base_model.horizon_count:
        raise ValueError("deployable reference horizons do not match model")

    hidden = initial_hidden
    if hidden is not None and hidden.shape != (
        base_model.config.num_layers,
        batch_size,
        base_model.config.hidden_dim,
    ):
        raise ValueError("deployable reference initial hidden shape is invalid")
    angle = features[:, 0, _FEATURE_INDEX["gimbal_angle_rad"]]
    previous_position = features[
        :, 0, _FEATURE_INDEX["previous_position_command_rad"]
    ]
    position_mode = (
        features[:, 0, _FEATURE_INDEX["command_mode_position"]] > 0.5
    )
    setpoint_angle = torch.where(position_mode, previous_position, angle)
    setpoint_rate = torch.zeros_like(angle)
    setpoint_acceleration = torch.zeros_like(angle)
    commands = []
    risks = []
    arrivals = []
    requested_horizons = []

    for time_index in range(time_count):
        active = sequence_mask[:, time_index].bool()
        output = base_model.forward_step(features[:, time_index], hidden)
        hidden = output.hidden
        target, risk, arrival, requested = _policy_target(
            output,
            context,
            time_index,
            features[:, time_index, _FEATURE_INDEX["gimbal_angle_rad"]],
            features[:, time_index, _FEATURE_INDEX["gimbal_rate_rad_s"]],
            prediction_horizons_s,
            adapter,
        )
        dt_s = context.control_period_s[:, time_index]
        maximum_rate = (
            adapter.setpoint_rate_limit_scale
            * context.servo_max_rate_rad_s[:, time_index]
            * (1.0 + risk * (adapter.risk_rate_limit_multiplier - 1.0))
        )
        base_acceleration = (
            adapter.setpoint_acceleration_limit_scale
            * context.servo_max_acceleration_rad_s2[:, time_index]
        )
        maximum_acceleration = base_acceleration * (
            1.0
            + risk * (adapter.risk_acceleration_limit_multiplier - 1.0)
        )
        maximum_jerk = base_acceleration / adapter.setpoint_jerk_rise_time_s
        maximum_jerk = maximum_jerk * (
            1.0 + risk * (adapter.risk_jerk_limit_multiplier - 1.0)
        )
        error = target - setpoint_angle
        stopping_speed = torch.sqrt(
            2.0 * maximum_acceleration * torch.abs(error) + 1e-12
        )
        desired_rate = torch.sign(error) * torch.minimum(
            maximum_rate,
            stopping_speed,
        )
        desired_acceleration = torch.clamp(
            (desired_rate - setpoint_rate) / dt_s,
            -maximum_acceleration,
            maximum_acceleration,
        )
        next_acceleration = setpoint_acceleration + torch.clamp(
            desired_acceleration - setpoint_acceleration,
            -maximum_jerk * dt_s,
            maximum_jerk * dt_s,
        )
        next_rate = torch.clamp(
            setpoint_rate + next_acceleration * dt_s,
            -maximum_rate,
            maximum_rate,
        )
        step = next_rate * dt_s
        overshoot = (step * error > 0.0) & (
            torch.abs(step) >= torch.abs(error)
        )
        minimum = context.servo_min_angle_rad[:, time_index]
        maximum = context.servo_max_angle_rad[:, time_index]
        next_angle = torch.where(
            overshoot,
            target,
            torch.clamp(setpoint_angle + step, minimum, maximum),
        )
        next_rate = torch.where(
            overshoot,
            torch.zeros_like(next_rate),
            next_rate,
        )
        next_acceleration = torch.where(
            overshoot,
            torch.zeros_like(next_acceleration),
            next_acceleration,
        )
        setpoint_angle = torch.where(active, next_angle, setpoint_angle)
        setpoint_rate = torch.where(active, next_rate, setpoint_rate)
        setpoint_acceleration = torch.where(
            active,
            next_acceleration,
            setpoint_acceleration,
        )
        commands.append(_normalized_position(setpoint_angle, minimum, maximum))
        risks.append(risk)
        arrivals.append(arrival)
        requested_horizons.append(requested)

    return DeployableReferenceSequence(
        command_normalized=torch.stack(commands, dim=1),
        risk=torch.stack(risks, dim=1),
        arrival_s=torch.stack(arrivals, dim=1),
        requested_horizon_s=torch.stack(requested_horizons, dim=1),
    )
