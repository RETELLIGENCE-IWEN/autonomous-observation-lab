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


@dataclass
class DeployableReferenceState:
    """Recurrent predictor and V2.1 setpoint-filter state."""

    hidden: torch.Tensor | None
    setpoint_angle_rad: torch.Tensor
    setpoint_rate_rad_s: torch.Tensor
    setpoint_acceleration_rad_s2: torch.Tensor


def initialize_deployable_reference_state(
    feature: torch.Tensor,
    *,
    initial_hidden: torch.Tensor | None = None,
) -> DeployableReferenceState:
    """Initialize the deployable controller from its causal feature state."""

    if feature.ndim != 2 or feature.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("deployable reference feature shape is invalid")
    angle = feature[:, _FEATURE_INDEX["gimbal_angle_rad"]]
    previous_position = feature[
        :, _FEATURE_INDEX["previous_position_command_rad"]
    ]
    position_mode = (
        feature[:, _FEATURE_INDEX["command_mode_position"]] > 0.5
    )
    return DeployableReferenceState(
        hidden=initial_hidden,
        setpoint_angle_rad=torch.where(
            position_mode,
            previous_position,
            angle,
        ),
        setpoint_rate_rad_s=torch.zeros_like(angle),
        setpoint_acceleration_rad_s2=torch.zeros_like(angle),
    )


def deployable_position_command_step(
    base_model: CausalTargetStateGRU,
    feature: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    time_index: int,
    active: torch.Tensor,
    state: DeployableReferenceState,
    *,
    prediction_horizons_s: tuple[float, ...],
    adapter: AdaptivePositionControllerConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    DeployableReferenceState,
]:
    """Advance one exact midpoint-GRU/V2.1 controller step."""

    if feature.ndim != 2 or feature.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("deployable reference feature shape is invalid")
    batch_size = feature.shape[0]
    if active.shape != (batch_size,):
        raise ValueError("deployable reference active-mask shape is invalid")
    if not 0 <= time_index < context.control_period_s.shape[1]:
        raise ValueError("deployable reference time index is invalid")
    output = base_model.forward_step(feature, state.hidden)
    target, risk, arrival, requested = _policy_target(
        output,
        context,
        time_index,
        feature[:, _FEATURE_INDEX["gimbal_angle_rad"]],
        feature[:, _FEATURE_INDEX["gimbal_rate_rad_s"]],
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
        1.0 + risk * (adapter.risk_acceleration_limit_multiplier - 1.0)
    )
    maximum_jerk = base_acceleration / adapter.setpoint_jerk_rise_time_s
    maximum_jerk = maximum_jerk * (
        1.0 + risk * (adapter.risk_jerk_limit_multiplier - 1.0)
    )
    error = target - state.setpoint_angle_rad
    stopping_speed = torch.sqrt(
        2.0 * maximum_acceleration * torch.abs(error) + 1e-12
    )
    desired_rate = torch.sign(error) * torch.minimum(
        maximum_rate,
        stopping_speed,
    )
    desired_acceleration = torch.clamp(
        (desired_rate - state.setpoint_rate_rad_s) / dt_s,
        -maximum_acceleration,
        maximum_acceleration,
    )
    next_acceleration = state.setpoint_acceleration_rad_s2 + torch.clamp(
        desired_acceleration - state.setpoint_acceleration_rad_s2,
        -maximum_jerk * dt_s,
        maximum_jerk * dt_s,
    )
    next_rate = torch.clamp(
        state.setpoint_rate_rad_s + next_acceleration * dt_s,
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
        torch.clamp(state.setpoint_angle_rad + step, minimum, maximum),
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
    next_state = DeployableReferenceState(
        hidden=output.hidden,
        setpoint_angle_rad=torch.where(
            active,
            next_angle,
            state.setpoint_angle_rad,
        ),
        setpoint_rate_rad_s=torch.where(
            active,
            next_rate,
            state.setpoint_rate_rad_s,
        ),
        setpoint_acceleration_rad_s2=torch.where(
            active,
            next_acceleration,
            state.setpoint_acceleration_rad_s2,
        ),
    )
    command = _normalized_position(
        next_state.setpoint_angle_rad,
        minimum,
        maximum,
    )
    return command, risk, arrival, requested, next_state


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

    if initial_hidden is not None and initial_hidden.shape != (
        base_model.config.num_layers,
        batch_size,
        base_model.config.hidden_dim,
    ):
        raise ValueError("deployable reference initial hidden shape is invalid")
    state = initialize_deployable_reference_state(
        features[:, 0],
        initial_hidden=initial_hidden,
    )
    commands = []
    risks = []
    arrivals = []
    requested_horizons = []

    for time_index in range(time_count):
        active = sequence_mask[:, time_index].bool()
        command, risk, arrival, requested, state = (
            deployable_position_command_step(
                base_model,
                features[:, time_index],
                context,
                time_index,
                active,
                state,
                prediction_horizons_s=prediction_horizons_s,
                adapter=adapter,
            )
        )
        commands.append(command)
        risks.append(risk)
        arrivals.append(arrival)
        requested_horizons.append(requested)

    return DeployableReferenceSequence(
        command_normalized=torch.stack(commands, dim=1),
        risk=torch.stack(risks, dim=1),
        arrival_s=torch.stack(arrivals, dim=1),
        requested_horizon_s=torch.stack(requested_horizons, dim=1),
    )
