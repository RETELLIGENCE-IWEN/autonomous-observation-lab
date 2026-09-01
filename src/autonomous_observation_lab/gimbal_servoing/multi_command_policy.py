"""Differentiable multi-command policy rollout with synthetic observations."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import nn

from .controllers import AdaptivePositionControllerConfig
from .dataset import FEATURE_NAMES
from .gru import (
    CausalTargetStateGRU,
    GRUAdaptivePositionLossContext,
    GRUTargetStateOutput,
    _adaptive_prediction_weight,
    _interpolate_gru_output,
    angular_residual_rad,
    differentiable_position_servo_sequence,
)


_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class RecurrentPositionResidualPolicyConfig:
    input_dim: int
    hidden_dim: int = 32
    embedding_dim: int = 32
    maximum_residual_magnitude: float = 0.25

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("recurrent policy input dimension must be positive")
        if self.hidden_dim <= 0 or self.embedding_dim <= 0:
            raise ValueError("recurrent policy dimensions must be positive")
        if not math.isfinite(self.maximum_residual_magnitude) or not (
            0.0 < self.maximum_residual_magnitude <= 1.0
        ):
            raise ValueError(
                "maximum recurrent residual magnitude must be in (0, 1]"
            )


class CausalRecurrentPositionResidualPolicy(nn.Module):
    """A small recurrent correction head with a zero-behavior initialization."""

    def __init__(self, config: RecurrentPositionResidualPolicyConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(config.embedding_dim, config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward_step(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.config.input_dim:
            raise ValueError(
                "recurrent policy input must have shape [batch, input_dim]"
            )
        if hidden is None:
            hidden = inputs.new_zeros(inputs.shape[0], self.config.hidden_dim)
        if hidden.shape != (inputs.shape[0], self.config.hidden_dim):
            raise ValueError("recurrent policy hidden state shape is invalid")
        hidden = self.recurrent(self.encoder(inputs), hidden)
        residual = self.config.maximum_residual_magnitude * torch.tanh(
            self.head(hidden).squeeze(-1)
        )
        return residual, hidden


@dataclass(frozen=True)
class CounterfactualWindowBatch:
    logged_features: torch.Tensor
    warmup_features: torch.Tensor
    target_bearing_rad: torch.Tensor
    time_s: torch.Tensor
    capture_source_index: torch.Tensor
    context: GRUAdaptivePositionLossContext
    sequence_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.logged_features.ndim != 3:
            raise ValueError("logged window features must be three-dimensional")
        batch_size, step_count, feature_count = self.logged_features.shape
        if feature_count != len(FEATURE_NAMES):
            raise ValueError("logged window feature dimension is invalid")
        if self.warmup_features.ndim != 3 or (
            self.warmup_features.shape[0] != batch_size
            or self.warmup_features.shape[2] != feature_count
        ):
            raise ValueError("warmup feature shape is invalid")
        if self.target_bearing_rad.shape != (batch_size, step_count + 1):
            raise ValueError("window target-bearing shape is invalid")
        if self.time_s.shape != (batch_size, step_count + 1):
            raise ValueError("window time shape is invalid")
        if self.capture_source_index.shape != (batch_size, step_count):
            raise ValueError("capture-source shape is invalid")
        if self.sequence_mask.shape != (batch_size, step_count):
            raise ValueError("window sequence mask shape is invalid")
        for field in fields(self.context):
            value = getattr(self.context, field.name)
            if value.shape != (batch_size, step_count):
                raise ValueError(
                    f"window context {field.name} shape is invalid"
                )


@dataclass
class CounterfactualWindowRollout:
    command_normalized: torch.Tensor
    policy_residual_normalized: torch.Tensor
    gimbal_angle_rad: torch.Tensor
    gimbal_rate_rad_s: torch.Tensor
    tracking_error_normalized: torch.Tensor
    visibility_violation_normalized: torch.Tensor
    saturation_fraction: torch.Tensor
    synthetic_features: torch.Tensor


def recurrent_policy_input_dim(horizon_count: int) -> int:
    if horizon_count <= 0:
        raise ValueError("policy horizon count must be positive")
    return len(FEATURE_NAMES) + 4 * horizon_count + 2


def counterfactual_capture_source_indices(
    time_s: torch.Tensor,
    logged_features: torch.Tensor,
) -> torch.Tensor:
    """Map released measurements to causal simulated states in one window.

    ``-1`` denotes a capture before the window, for which the logged released
    measurement is retained. The mapping depends only on exogenous timestamps
    and detector release metadata, never on policy actions.
    """

    if time_s.ndim != 2 or logged_features.ndim != 3:
        raise ValueError("capture-source inputs have invalid rank")
    batch_size, step_count, feature_count = logged_features.shape
    if feature_count != len(FEATURE_NAMES) or time_s.shape != (
        batch_size,
        step_count + 1,
    ):
        raise ValueError("capture-source input shapes are inconsistent")
    result = torch.full(
        (batch_size, step_count),
        -1,
        dtype=torch.long,
        device=time_s.device,
    )
    age_index = _FEATURE_INDEX["measurement_age_s"]
    age_valid_index = _FEATURE_INDEX["measurement_age_valid"]
    frame_index = _FEATURE_INDEX["frame_updated"]
    for time_index in range(step_count):
        released = logged_features[:, time_index, frame_index] > 0.5
        age_valid = (
            logged_features[:, time_index, age_valid_index] > 0.5
        )
        capture_time = (
            time_s[:, time_index]
            - logged_features[:, time_index, age_index]
        )
        candidate_times = time_s[:, : time_index + 1]
        causal = candidate_times <= capture_time[:, None] + 1e-7
        causal_count = causal.sum(dim=1)
        source = torch.clamp(causal_count - 1, min=0)
        inside = capture_time >= time_s[:, 0] - 1e-7
        use_source = released & age_valid & inside & (causal_count > 0)
        result[:, time_index] = torch.where(
            use_source,
            source,
            result[:, time_index],
        )
    return result


def _slice_context(
    context: GRUAdaptivePositionLossContext,
    end: int,
) -> GRUAdaptivePositionLossContext:
    return GRUAdaptivePositionLossContext(
        **{
            field.name: getattr(context, field.name)[:, :end]
            for field in fields(context)
        }
    )


def _normalized_position(
    position_rad: torch.Tensor,
    minimum_angle_rad: torch.Tensor,
    maximum_angle_rad: torch.Tensor,
) -> torch.Tensor:
    travel = torch.where(
        position_rad >= 0.0,
        maximum_angle_rad,
        -minimum_angle_rad,
    )
    return torch.clamp(position_rad / travel, -1.0, 1.0)


def _policy_target(
    output: GRUTargetStateOutput,
    context: GRUAdaptivePositionLossContext,
    time_index: int,
    gimbal_angle: torch.Tensor,
    gimbal_rate: torch.Tensor,
    prediction_horizons_s: tuple[float, ...],
    adapter: AdaptivePositionControllerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latency = context.servo_command_latency_s[:, time_index]
    time_constant = context.servo_rate_time_constant_s[:, time_index]
    position_gain = context.servo_position_gain_s_inv[:, time_index]
    arrival = adapter.actuator_arrival_time_scale * (
        latency
        + time_constant
        + adapter.position_response_fraction / position_gain
    ) + adapter.additional_preview_s
    arrival = torch.clamp(
        arrival,
        prediction_horizons_s[0],
        prediction_horizons_s[-1],
    )
    current_bearing = output.mean[..., 0, 0]
    current_rate = output.mean[..., 0, 1]
    current_std = output.std[..., 0, 0]
    base_bearing, base_rate, base_std, _ = _interpolate_gru_output(
        output,
        arrival,
        prediction_horizons_s,
    )
    base_weight = _adaptive_prediction_weight(current_std, base_std, adapter)
    blended_bearing = current_bearing + base_weight * angular_residual_rad(
        base_bearing,
        current_bearing,
    )
    blended_rate = current_rate + base_weight * (base_rate - current_rate)
    image_error = angular_residual_rad(blended_bearing, gimbal_angle)
    fov_fraction = (
        torch.abs(image_error)
        + adapter.visibility_uncertainty_sigma
        * (current_std + base_weight * (base_std - current_std))
    ) / (0.5 * context.selected_axis_fov_rad[:, time_index])
    risk = torch.clamp(
        (fov_fraction - adapter.visibility_risk_onset_fraction)
        / (
            adapter.visibility_risk_full_fraction
            - adapter.visibility_risk_onset_fraction
        ),
        0.0,
        1.0,
    )
    if adapter.risk_requires_outward_motion:
        outward = image_error * (blended_rate - gimbal_rate) > 0.0
        risk = torch.where(outward, risk, torch.zeros_like(risk))
    requested = torch.clamp(
        arrival + risk * adapter.risk_horizon_boost_s,
        prediction_horizons_s[0],
        prediction_horizons_s[-1],
    )
    forecast_bearing, _, forecast_std, _ = _interpolate_gru_output(
        output,
        requested,
        prediction_horizons_s,
    )
    prediction_weight = _adaptive_prediction_weight(
        current_std,
        forecast_std,
        adapter,
    )
    target = current_bearing + prediction_weight * angular_residual_rad(
        forecast_bearing,
        current_bearing,
    )
    return target, risk, arrival, requested


def _counterfactual_feature(
    *,
    logged: torch.Tensor,
    previous_synthetic: torch.Tensor | None,
    time_index: int,
    angle_rad: torch.Tensor,
    rate_rad_s: torch.Tensor,
    previous_command_normalized: torch.Tensor,
    previous_position_command_rad: torch.Tensor,
    target_bearing_rad: torch.Tensor,
    simulated_angle_history: torch.Tensor,
    capture_source_index: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
) -> torch.Tensor:
    feature = logged.clone()
    minimum_angle = context.servo_min_angle_rad[:, time_index]
    maximum_angle = context.servo_max_angle_rad[:, time_index]
    maximum_rate = context.servo_max_rate_rad_s[:, time_index]
    feature[:, _FEATURE_INDEX["gimbal_position_normalized"]] = (
        _normalized_position(angle_rad, minimum_angle, maximum_angle)
    )
    feature[:, _FEATURE_INDEX["gimbal_angle_rad"]] = angle_rad
    feature[:, _FEATURE_INDEX["gimbal_rate_normalized"]] = (
        rate_rad_s / maximum_rate
    )
    feature[:, _FEATURE_INDEX["gimbal_rate_rad_s"]] = rate_rad_s
    feature[:, _FEATURE_INDEX["previous_action_normalized"]] = (
        previous_command_normalized
    )
    feature[:, _FEATURE_INDEX["previous_rate_command_rad_s"]] = 0.0
    feature[:, _FEATURE_INDEX["previous_position_command_rad"]] = (
        previous_position_command_rad
    )

    if time_index == 0:
        return feature
    frame_updated = logged[:, _FEATURE_INDEX["frame_updated"]] > 0.5
    if previous_synthetic is not None:
        held_names = (
            "image_error_normalized",
            "image_error_rad",
            "image_error_valid",
            "bbox_width_fraction",
            "bbox_width_valid",
            "bbox_height_fraction",
            "bbox_height_valid",
            "confidence",
            "confidence_valid",
        )
        for name in held_names:
            index = _FEATURE_INDEX[name]
            feature[:, index] = torch.where(
                frame_updated,
                feature[:, index],
                previous_synthetic[:, index],
            )

    source = capture_source_index[:, time_index]
    local_capture = source >= 0
    safe_source = torch.clamp(source, min=0, max=time_index)
    captured_angle = torch.gather(
        simulated_angle_history,
        1,
        safe_source[:, None],
    ).squeeze(1)
    captured_target = torch.gather(
        target_bearing_rad,
        1,
        safe_source[:, None],
    ).squeeze(1)
    counterfactual_error = angular_residual_rad(
        captured_target,
        captured_angle,
    )
    logged_error = logged[:, _FEATURE_INDEX["image_error_rad"]]
    counterfactual_error = torch.where(
        local_capture,
        counterfactual_error,
        logged_error,
    )
    half_fov = 0.5 * context.selected_axis_fov_rad[:, time_index]
    logged_valid = logged[:, _FEATURE_INDEX["image_error_valid"]] > 0.5
    geometrically_visible = torch.abs(counterfactual_error) <= half_fov
    valid = frame_updated & logged_valid & geometrically_visible
    error_rad = torch.where(
        valid,
        counterfactual_error,
        torch.zeros_like(counterfactual_error),
    )
    feature[:, _FEATURE_INDEX["image_error_rad"]] = torch.where(
        frame_updated,
        error_rad,
        feature[:, _FEATURE_INDEX["image_error_rad"]],
    )
    feature[:, _FEATURE_INDEX["image_error_normalized"]] = torch.where(
        frame_updated,
        error_rad / half_fov,
        feature[:, _FEATURE_INDEX["image_error_normalized"]],
    )
    feature[:, _FEATURE_INDEX["image_error_valid"]] = torch.where(
        frame_updated,
        valid.to(feature.dtype),
        feature[:, _FEATURE_INDEX["image_error_valid"]],
    )
    for value_name, valid_name in (
        ("bbox_width_fraction", "bbox_width_valid"),
        ("bbox_height_fraction", "bbox_height_valid"),
        ("confidence", "confidence_valid"),
    ):
        value_index = _FEATURE_INDEX[value_name]
        valid_index = _FEATURE_INDEX[valid_name]
        feature[:, value_index] = torch.where(
            frame_updated & ~valid,
            torch.zeros_like(feature[:, value_index]),
            feature[:, value_index],
        )
        feature[:, valid_index] = torch.where(
            frame_updated,
            valid.to(feature.dtype),
            feature[:, valid_index],
        )
    return feature


def rollout_counterfactual_window(
    base_model: CausalTargetStateGRU,
    policy: CausalRecurrentPositionResidualPolicy,
    batch: CounterfactualWindowBatch,
    *,
    prediction_horizons_s: tuple[float, ...],
    adapter: AdaptivePositionControllerConfig,
    integration_period_override_s: float | None = None,
    visibility_margin_fraction: float = 0.85,
    residual_application: str = "target_half_fov",
) -> CounterfactualWindowRollout:
    """Unroll recurrent commands and regenerate geometry-dependent inputs."""

    if not 0.0 < visibility_margin_fraction <= 1.0:
        raise ValueError("visibility margin fraction must be in (0, 1]")
    if residual_application not in {
        "target_half_fov",
        "command_normalized",
    }:
        raise ValueError("unsupported recurrent residual application")
    batch_size, step_count, _ = batch.logged_features.shape
    if len(prediction_horizons_s) != base_model.horizon_count:
        raise ValueError("counterfactual rollout horizons do not match model")
    expected_policy_input = recurrent_policy_input_dim(
        base_model.horizon_count
    )
    if policy.config.input_dim != expected_policy_input:
        raise ValueError("counterfactual policy input dimension is invalid")

    base_hidden = None
    if batch.warmup_features.shape[1] > 0:
        with torch.no_grad():
            warmup = base_model(batch.warmup_features)
        base_hidden = warmup.hidden.detach()
    policy_hidden = None
    angle = batch.context.gimbal_angle_rad[:, 0]
    rate = batch.context.gimbal_rate_rad_s[:, 0]
    previous_command = batch.logged_features[
        :, 0, _FEATURE_INDEX["previous_action_normalized"]
    ]
    previous_position = batch.logged_features[
        :, 0, _FEATURE_INDEX["previous_position_command_rad"]
    ]
    position_mode = (
        batch.logged_features[:, 0, _FEATURE_INDEX["command_mode_position"]]
        > 0.5
    )
    previous_position = torch.where(position_mode, previous_position, angle)
    setpoint_angle = previous_position
    previous_issued_position = previous_position
    setpoint_rate = torch.zeros_like(angle)
    setpoint_acceleration = torch.zeros_like(angle)
    initial_applied_position = angle
    commands = []
    residuals = []
    angles_before = []
    output_angles = []
    output_rates = []
    output_saturation = []
    synthetic_features = []
    previous_synthetic = None

    for time_index in range(step_count):
        angles_before.append(angle)
        angle_history = torch.stack(angles_before, dim=1)
        feature = _counterfactual_feature(
            logged=batch.logged_features[:, time_index],
            previous_synthetic=previous_synthetic,
            time_index=time_index,
            angle_rad=angle,
            rate_rad_s=rate,
            previous_command_normalized=previous_command,
            previous_position_command_rad=previous_issued_position,
            target_bearing_rad=batch.target_bearing_rad,
            simulated_angle_history=angle_history,
            capture_source_index=batch.capture_source_index,
            context=batch.context,
        )
        synthetic_features.append(feature)
        previous_synthetic = feature
        output = base_model.forward_step(feature, base_hidden)
        base_hidden = output.hidden
        base_target, risk, arrival, requested = _policy_target(
            output,
            batch.context,
            time_index,
            angle,
            rate,
            prediction_horizons_s,
            adapter,
        )
        policy_inputs = torch.cat(
            (
                feature,
                output.mean.flatten(start_dim=1),
                output.std.flatten(start_dim=1),
                (arrival / prediction_horizons_s[-1]).unsqueeze(-1),
                (requested / prediction_horizons_s[-1]).unsqueeze(-1),
            ),
            dim=-1,
        )
        residual, policy_hidden = policy.forward_step(
            policy_inputs,
            policy_hidden,
        )
        residuals.append(residual)
        target = base_target
        if residual_application == "target_half_fov":
            target = target + (
                residual
                * 0.5
                * batch.context.selected_axis_fov_rad[:, time_index]
            )
        target = angular_residual_rad(target, torch.zeros_like(target))
        minimum_angle = batch.context.servo_min_angle_rad[:, time_index]
        maximum_angle = batch.context.servo_max_angle_rad[:, time_index]
        target = torch.maximum(target, minimum_angle)
        target = torch.minimum(target, maximum_angle)

        dt_s = batch.context.control_period_s[:, time_index]
        maximum_rate = (
            adapter.setpoint_rate_limit_scale
            * batch.context.servo_max_rate_rad_s[:, time_index]
            * (1.0 + risk * (adapter.risk_rate_limit_multiplier - 1.0))
        )
        base_acceleration = (
            adapter.setpoint_acceleration_limit_scale
            * batch.context.servo_max_acceleration_rad_s2[:, time_index]
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
        setpoint_acceleration = setpoint_acceleration + torch.clamp(
            desired_acceleration - setpoint_acceleration,
            -maximum_jerk * dt_s,
            maximum_jerk * dt_s,
        )
        setpoint_rate = torch.clamp(
            setpoint_rate + setpoint_acceleration * dt_s,
            -maximum_rate,
            maximum_rate,
        )
        step = setpoint_rate * dt_s
        overshoot = (step * error > 0.0) & (
            torch.abs(step) >= torch.abs(error)
        )
        setpoint_angle = torch.where(
            overshoot,
            target,
            torch.clamp(
                setpoint_angle + step,
                minimum_angle,
                maximum_angle,
            ),
        )
        setpoint_rate = torch.where(
            overshoot,
            torch.zeros_like(setpoint_rate),
            setpoint_rate,
        )
        setpoint_acceleration = torch.where(
            overshoot,
            torch.zeros_like(setpoint_acceleration),
            setpoint_acceleration,
        )
        command = _normalized_position(
            setpoint_angle,
            minimum_angle,
            maximum_angle,
        )
        if residual_application == "command_normalized":
            command = torch.clamp(command + residual, -1.0, 1.0)
        command_travel = torch.where(
            command >= 0.0,
            maximum_angle,
            -minimum_angle,
        )
        issued_position = command * command_travel
        commands.append(command)
        command_sequence = torch.stack(commands, dim=1)
        plant = differentiable_position_servo_sequence(
            command_sequence,
            _slice_context(batch.context, time_index + 1),
            batch.sequence_mask[:, : time_index + 1],
            initial_time_s=batch.time_s[:, 0],
            initial_applied_position_rad=initial_applied_position,
            integration_period_override_s=integration_period_override_s,
        )
        angle = plant.angle_rad[:, -1]
        rate = plant.rate_rad_s[:, -1]
        previous_command = command
        previous_issued_position = issued_position
        output_angles.append(angle)
        output_rates.append(rate)
        output_saturation.append(plant.saturation_fraction[:, -1])

    commands_tensor = torch.stack(commands, dim=1)
    angle_tensor = torch.stack(output_angles, dim=1)
    rate_tensor = torch.stack(output_rates, dim=1)
    half_fov = 0.5 * batch.context.selected_axis_fov_rad
    tracking_error = angular_residual_rad(
        batch.target_bearing_rad[:, 1:],
        angle_tensor,
    ) / half_fov
    visibility = torch.relu(
        torch.abs(tracking_error) - visibility_margin_fraction
    )
    return CounterfactualWindowRollout(
        command_normalized=commands_tensor,
        policy_residual_normalized=torch.stack(residuals, dim=1),
        gimbal_angle_rad=angle_tensor,
        gimbal_rate_rad_s=rate_tensor,
        tracking_error_normalized=tracking_error,
        visibility_violation_normalized=visibility,
        saturation_fraction=torch.stack(output_saturation, dim=1),
        synthetic_features=torch.stack(synthetic_features, dim=1),
    )
