"""Counterfactual rollout for gated corrections around the deployable reference."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from .controllers import AdaptivePositionControllerConfig
from .dataset import FEATURE_NAMES
from .deployable_reference import (
    deployable_position_command_step,
    initialize_deployable_reference_state,
)
from .failure_gated_policy import FailureGatedCommandResidualPolicy
from .gru import (
    CausalTargetStateGRU,
    GRUAdaptivePositionLossContext,
    angular_residual_rad,
    differentiable_position_servo_step,
    initialize_differentiable_position_servo_state,
)
from .multi_command_policy import _counterfactual_feature
from .sequence_distillation import normalized_hardware_features


_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass
class DeployableFailureGatedRollout:
    command_normalized: torch.Tensor
    base_command_normalized: torch.Tensor
    residual_normalized: torch.Tensor
    gate_probability: torch.Tensor
    failure_evidence: torch.Tensor
    gimbal_angle_rad: torch.Tensor
    gimbal_rate_rad_s: torch.Tensor
    tracking_error_normalized: torch.Tensor
    visibility_violation_normalized: torch.Tensor
    saturation_fraction: torch.Tensor
    synthetic_features: torch.Tensor


def rollout_deployable_failure_gated_policy(
    base_model: CausalTargetStateGRU,
    policy: FailureGatedCommandResidualPolicy,
    logged_features: torch.Tensor,
    target_bearing_rad: torch.Tensor,
    time_s: torch.Tensor,
    capture_source_index: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    sequence_mask: torch.Tensor,
    *,
    prediction_horizons_s: tuple[float, ...],
    adapter: AdaptivePositionControllerConfig,
    warmup_features: torch.Tensor | None = None,
    residual_scale: float = 1.0,
    visibility_shield_strength: float = 0.0,
    integration_period_override_s: float | None = None,
    visibility_margin_fraction: float = 0.85,
) -> DeployableFailureGatedRollout:
    """Run the frozen GRU/adapter and gated residual on their own history."""

    if logged_features.ndim != 3 or logged_features.shape[-1] != len(
        FEATURE_NAMES
    ):
        raise ValueError(
            "deployable gated features must have shape [batch, time, feature]"
        )
    batch_size, step_count, _ = logged_features.shape
    expected = (batch_size, step_count)
    if target_bearing_rad.shape != (batch_size, step_count + 1):
        raise ValueError("deployable gated target-bearing shape is invalid")
    if time_s.shape != (batch_size, step_count + 1):
        raise ValueError("deployable gated time shape is invalid")
    if capture_source_index.shape != expected:
        raise ValueError("deployable gated capture-source shape is invalid")
    if sequence_mask.shape != expected:
        raise ValueError("deployable gated sequence-mask shape is invalid")
    for field in fields(context):
        if getattr(context, field.name).shape != expected:
            raise ValueError(
                f"deployable gated context {field.name} shape is invalid"
            )
    if not 0.0 < visibility_margin_fraction <= 1.0:
        raise ValueError("visibility margin fraction must be in (0, 1]")
    if not math.isfinite(residual_scale) or not 0.0 <= residual_scale <= 1.0:
        raise ValueError("deployable gated residual scale must be in [0, 1]")
    if not math.isfinite(visibility_shield_strength) or not (
        0.0 <= visibility_shield_strength <= 1.0
    ):
        raise ValueError("visibility shield strength must be in [0, 1]")
    base_initial_hidden = None
    if warmup_features is not None:
        if warmup_features.ndim != 3 or (
            warmup_features.shape[0] != batch_size
            or warmup_features.shape[2] != len(FEATURE_NAMES)
        ):
            raise ValueError("deployable gated warmup shape is invalid")
        if warmup_features.shape[1] > 0:
            base_initial_hidden = base_model(warmup_features).hidden

    hardware = normalized_hardware_features(context)
    correction_hidden = None
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
    plant_state = initialize_differentiable_position_servo_state(
        context,
        initial_time_s=time_s[:, 0],
        initial_applied_position_rad=initial_applied_position,
    )
    reference_state = None
    previous_synthetic = None
    angles_before: list[torch.Tensor] = []
    synthetic_features = []
    commands = []
    base_commands = []
    residuals = []
    gates = []
    evidence = []
    output_angles = []
    output_rates = []
    output_saturation = []

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
        if reference_state is None:
            reference_state = initialize_deployable_reference_state(
                feature,
                initial_hidden=base_initial_hidden,
            )
        base_command, _risk, _arrival, _requested, reference_state = (
            deployable_position_command_step(
                base_model,
                feature,
                context,
                time_index,
                sequence_mask[:, time_index].bool(),
                reference_state,
                prediction_horizons_s=prediction_horizons_s,
                adapter=adapter,
            )
        )
        _command, residual, gate, step_evidence, correction_hidden = (
            policy.forward_step(
                feature,
                hardware[:, time_index],
                base_command,
                correction_hidden,
            )
        )
        residual = residual_scale * residual
        minimum = context.servo_min_angle_rad[:, time_index]
        maximum = context.servo_max_angle_rad[:, time_index]
        base_position = base_command * torch.where(
            base_command >= 0.0,
            maximum,
            -minimum,
        )
        proposed_command = torch.clamp(base_command + residual, -1.0, 1.0)
        proposed_position = proposed_command * torch.where(
            proposed_command >= 0.0,
            maximum,
            -minimum,
        )
        observed_target = (
            angle + feature[:, _FEATURE_INDEX["image_error_rad"]]
        )
        image_valid = feature[:, _FEATURE_INDEX["image_error_valid"]] > 0.5
        moves_farther = torch.abs(proposed_position - observed_target) > (
            torch.abs(base_position - observed_target)
        )
        shield = (
            visibility_shield_strength
            * step_evidence[:, 0]
            * (image_valid & moves_farther).to(residual.dtype)
        )
        residual = residual * (1.0 - shield)
        command = torch.clamp(base_command + residual, -1.0, 1.0)
        commands.append(command)
        base_commands.append(base_command)
        residuals.append(residual)
        gates.append(gate)
        evidence.append(step_evidence)
        plant_state, step_saturation = differentiable_position_servo_step(
            command,
            context,
            sequence_mask[:, time_index].bool(),
            time_index,
            plant_state,
            integration_period_override_s=integration_period_override_s,
        )
        angle = plant_state.angle_rad
        rate = plant_state.rate_rad_s
        previous_command = command
        previous_position = command * torch.where(
            command >= 0.0,
            maximum,
            -minimum,
        )
        output_angles.append(angle)
        output_rates.append(rate)
        output_saturation.append(step_saturation)

    angle_tensor = torch.stack(output_angles, dim=1)
    half_fov = 0.5 * context.selected_axis_fov_rad
    tracking_error = angular_residual_rad(
        target_bearing_rad[:, 1:],
        angle_tensor,
    ) / half_fov
    visibility = torch.relu(
        torch.abs(tracking_error) - visibility_margin_fraction
    )
    return DeployableFailureGatedRollout(
        command_normalized=torch.stack(commands, dim=1),
        base_command_normalized=torch.stack(base_commands, dim=1),
        residual_normalized=torch.stack(residuals, dim=1),
        gate_probability=torch.stack(gates, dim=1),
        failure_evidence=torch.stack(evidence, dim=1),
        gimbal_angle_rad=angle_tensor,
        gimbal_rate_rad_s=torch.stack(output_rates, dim=1),
        tracking_error_normalized=tracking_error,
        visibility_violation_normalized=visibility,
        saturation_fraction=torch.stack(output_saturation, dim=1),
        synthetic_features=torch.stack(synthetic_features, dim=1),
    )
