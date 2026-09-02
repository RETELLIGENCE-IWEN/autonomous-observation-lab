"""Closed-loop causal rollout utilities for on-policy oracle distillation."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .dataset import FEATURE_NAMES
from .gru import (
    GRUAdaptivePositionLossContext,
    angular_residual_rad,
    differentiable_position_servo_sequence,
)
from .multi_command_policy import _counterfactual_feature, _slice_context
from .sequence_distillation import (
    CausalHardwareConditionedPositionPolicy,
    normalized_hardware_features,
)


_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass
class CounterfactualPositionPolicyRollout:
    """Commands, plant effects, and regenerated causal observations."""

    command_normalized: torch.Tensor
    gimbal_angle_rad: torch.Tensor
    gimbal_rate_rad_s: torch.Tensor
    tracking_error_normalized: torch.Tensor
    visibility_violation_normalized: torch.Tensor
    saturation_fraction: torch.Tensor
    synthetic_features: torch.Tensor


def rollout_counterfactual_position_commands(
    command_normalized: torch.Tensor,
    logged_features: torch.Tensor,
    target_bearing_rad: torch.Tensor,
    time_s: torch.Tensor,
    capture_source_index: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    sequence_mask: torch.Tensor,
    *,
    integration_period_override_s: float | None = None,
    visibility_margin_fraction: float = 0.85,
) -> CounterfactualPositionPolicyRollout:
    """Replay a prescribed command sequence with regenerated observations."""

    if logged_features.ndim != 3 or logged_features.shape[-1] != len(
        FEATURE_NAMES
    ):
        raise ValueError(
            "counterfactual command features must have shape [batch, time, feature]"
        )
    batch_size, step_count, _ = logged_features.shape
    expected_sequence = (batch_size, step_count)
    if command_normalized.shape != expected_sequence:
        raise ValueError("counterfactual command sequence shape is invalid")
    if target_bearing_rad.shape != (batch_size, step_count + 1):
        raise ValueError("counterfactual command target-bearing shape is invalid")
    if time_s.shape != (batch_size, step_count + 1):
        raise ValueError("counterfactual command time shape is invalid")
    if capture_source_index.shape != expected_sequence:
        raise ValueError("counterfactual command capture-source shape is invalid")
    if sequence_mask.shape != expected_sequence:
        raise ValueError("counterfactual command sequence-mask shape is invalid")
    for field in fields(context):
        if getattr(context, field.name).shape != expected_sequence:
            raise ValueError(
                f"counterfactual command context {field.name} shape is invalid"
            )
    if not 0.0 < visibility_margin_fraction <= 1.0:
        raise ValueError("visibility margin fraction must be in (0, 1]")

    commands = torch.clamp(command_normalized, -1.0, 1.0)
    angle = context.gimbal_angle_rad[:, 0]
    rate = context.gimbal_rate_rad_s[:, 0]
    previous_command = logged_features[
        :, 0, _FEATURE_INDEX["previous_action_normalized"]
    ]
    previous_position = logged_features[
        :, 0, _FEATURE_INDEX["previous_position_command_rad"]
    ]
    position_mode = (
        logged_features[:, 0, _FEATURE_INDEX["command_mode_position"]] > 0.5
    )
    previous_position = torch.where(position_mode, previous_position, angle)
    initial_applied_position = angle
    previous_synthetic = None
    angles_before: list[torch.Tensor] = []
    output_angles: list[torch.Tensor] = []
    output_rates: list[torch.Tensor] = []
    output_saturation: list[torch.Tensor] = []
    synthetic_features: list[torch.Tensor] = []

    for time_index in range(step_count):
        angles_before.append(angle)
        feature = _counterfactual_feature(
            logged=logged_features[:, time_index],
            previous_synthetic=previous_synthetic,
            time_index=time_index,
            angle_rad=angle,
            rate_rad_s=rate,
            previous_command_normalized=previous_command,
            previous_position_command_rad=previous_position,
            target_bearing_rad=target_bearing_rad,
            simulated_angle_history=torch.stack(angles_before, dim=1),
            capture_source_index=capture_source_index,
            context=context,
        )
        synthetic_features.append(feature)
        previous_synthetic = feature
        command = commands[:, time_index]
        plant = differentiable_position_servo_sequence(
            commands[:, : time_index + 1],
            _slice_context(context, time_index + 1),
            sequence_mask[:, : time_index + 1],
            initial_time_s=time_s[:, 0],
            initial_applied_position_rad=initial_applied_position,
            integration_period_override_s=integration_period_override_s,
        )
        angle = plant.angle_rad[:, -1]
        rate = plant.rate_rad_s[:, -1]
        minimum = context.servo_min_angle_rad[:, time_index]
        maximum = context.servo_max_angle_rad[:, time_index]
        previous_command = command
        previous_position = command * torch.where(
            command >= 0.0,
            maximum,
            -minimum,
        )
        output_angles.append(angle)
        output_rates.append(rate)
        output_saturation.append(plant.saturation_fraction[:, -1])

    angle_tensor = torch.stack(output_angles, dim=1)
    half_fov = 0.5 * context.selected_axis_fov_rad
    tracking_error = angular_residual_rad(
        target_bearing_rad[:, 1:],
        angle_tensor,
    ) / half_fov
    visibility = torch.relu(
        torch.abs(tracking_error) - visibility_margin_fraction
    )
    return CounterfactualPositionPolicyRollout(
        command_normalized=commands,
        gimbal_angle_rad=angle_tensor,
        gimbal_rate_rad_s=torch.stack(output_rates, dim=1),
        tracking_error_normalized=tracking_error,
        visibility_violation_normalized=visibility,
        saturation_fraction=torch.stack(output_saturation, dim=1),
        synthetic_features=torch.stack(synthetic_features, dim=1),
    )


def rollout_counterfactual_position_policy(
    policy: CausalHardwareConditionedPositionPolicy,
    logged_features: torch.Tensor,
    target_bearing_rad: torch.Tensor,
    time_s: torch.Tensor,
    capture_source_index: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    sequence_mask: torch.Tensor,
    *,
    integration_period_override_s: float | None = None,
    visibility_margin_fraction: float = 0.85,
) -> CounterfactualPositionPolicyRollout:
    """Run an absolute actor on its own image, servo, and action history.

    Exogenous detector timing and non-geometric detector attributes are kept
    from the logged episode. Geometry-dependent image measurements are
    regenerated from target truth and the actor-induced gimbal trajectory.
    Privileged target bearing is therefore part of the simulator, never an
    actor input.
    """

    if logged_features.ndim != 3 or logged_features.shape[-1] != len(
        FEATURE_NAMES
    ):
        raise ValueError(
            "counterfactual actor features must have shape [batch, time, feature]"
        )
    batch_size, step_count, _ = logged_features.shape
    if target_bearing_rad.shape != (batch_size, step_count + 1):
        raise ValueError("counterfactual actor target-bearing shape is invalid")
    if time_s.shape != (batch_size, step_count + 1):
        raise ValueError("counterfactual actor time shape is invalid")
    if capture_source_index.shape != (batch_size, step_count):
        raise ValueError("counterfactual actor capture-source shape is invalid")
    if sequence_mask.shape != (batch_size, step_count):
        raise ValueError("counterfactual actor sequence-mask shape is invalid")
    for field in fields(context):
        if getattr(context, field.name).shape != (batch_size, step_count):
            raise ValueError(
                f"counterfactual actor context {field.name} shape is invalid"
            )
    if not 0.0 < visibility_margin_fraction <= 1.0:
        raise ValueError("visibility margin fraction must be in (0, 1]")

    hardware = normalized_hardware_features(context)
    hidden = None
    angle = context.gimbal_angle_rad[:, 0]
    rate = context.gimbal_rate_rad_s[:, 0]
    previous_command = logged_features[
        :, 0, _FEATURE_INDEX["previous_action_normalized"]
    ]
    previous_position = logged_features[
        :, 0, _FEATURE_INDEX["previous_position_command_rad"]
    ]
    position_mode = (
        logged_features[:, 0, _FEATURE_INDEX["command_mode_position"]] > 0.5
    )
    previous_position = torch.where(position_mode, previous_position, angle)
    initial_applied_position = angle
    previous_synthetic = None
    angles_before: list[torch.Tensor] = []
    commands: list[torch.Tensor] = []
    output_angles: list[torch.Tensor] = []
    output_rates: list[torch.Tensor] = []
    output_saturation: list[torch.Tensor] = []
    synthetic_features: list[torch.Tensor] = []

    for time_index in range(step_count):
        angles_before.append(angle)
        feature = _counterfactual_feature(
            logged=logged_features[:, time_index],
            previous_synthetic=previous_synthetic,
            time_index=time_index,
            angle_rad=angle,
            rate_rad_s=rate,
            previous_command_normalized=previous_command,
            previous_position_command_rad=previous_position,
            target_bearing_rad=target_bearing_rad,
            simulated_angle_history=torch.stack(angles_before, dim=1),
            capture_source_index=capture_source_index,
            context=context,
        )
        synthetic_features.append(feature)
        previous_synthetic = feature
        minimum = context.servo_min_angle_rad[:, time_index]
        maximum = context.servo_max_angle_rad[:, time_index]
        command, hidden = policy.forward_step(
            feature,
            hardware[:, time_index],
            hidden,
            previous_command_normalized=previous_command,
            minimum_angle_rad=minimum,
            maximum_angle_rad=maximum,
        )
        commands.append(command)
        command_sequence = torch.stack(commands, dim=1)
        plant = differentiable_position_servo_sequence(
            command_sequence,
            _slice_context(context, time_index + 1),
            sequence_mask[:, : time_index + 1],
            initial_time_s=time_s[:, 0],
            initial_applied_position_rad=initial_applied_position,
            integration_period_override_s=integration_period_override_s,
        )
        angle = plant.angle_rad[:, -1]
        rate = plant.rate_rad_s[:, -1]
        previous_command = command
        previous_position = command * torch.where(
            command >= 0.0,
            maximum,
            -minimum,
        )
        output_angles.append(angle)
        output_rates.append(rate)
        output_saturation.append(plant.saturation_fraction[:, -1])

    angle_tensor = torch.stack(output_angles, dim=1)
    half_fov = 0.5 * context.selected_axis_fov_rad
    tracking_error = angular_residual_rad(
        target_bearing_rad[:, 1:],
        angle_tensor,
    ) / half_fov
    visibility = torch.relu(
        torch.abs(tracking_error) - visibility_margin_fraction
    )
    return CounterfactualPositionPolicyRollout(
        command_normalized=torch.stack(commands, dim=1),
        gimbal_angle_rad=angle_tensor,
        gimbal_rate_rad_s=torch.stack(output_rates, dim=1),
        tracking_error_normalized=tracking_error,
        visibility_violation_normalized=visibility,
        saturation_fraction=torch.stack(output_saturation, dim=1),
        synthetic_features=torch.stack(synthetic_features, dim=1),
    )
