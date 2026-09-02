"""Privileged constrained shooting oracle for position-command sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from .gru import (
    GRUAdaptivePositionLossContext,
    angular_residual_rad,
    differentiable_position_servo_sequence,
)


@dataclass(frozen=True)
class PrivilegedSequenceOracleConfig:
    """Optimization and feasibility contract for one focus segment."""

    focus_start_index: int = 48
    focus_steps: int = 16
    optimization_iterations: int = 24
    learning_rate: float = 0.05
    gradient_clip_norm: float = 5.0
    maximum_command_residual: float = 0.40
    training_integration_period_s: float = 0.010
    tracking_weight: float = 1.0
    visibility_weight: float = 2.0
    smoothness_weight: float = 0.10
    saturation_weight: float = 0.01
    residual_weight: float = 0.005
    terminal_tracking_multiplier: float = 2.0
    visibility_margin_fraction: float = 0.85
    maximum_visibility_regression_fraction: float = 0.0
    maximum_smoothness_regression_fraction: float = 0.0
    maximum_saturation_regression_fraction: float = 0.05
    maximum_visibility_regression_absolute_mse: float = 0.0
    maximum_smoothness_regression_absolute_mse: float = 0.0
    maximum_saturation_regression_absolute_mean: float = 0.0
    blend_fractions: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    )

    def __post_init__(self) -> None:
        if self.focus_start_index < 0:
            raise ValueError("oracle focus start must be non-negative")
        for name in ("focus_steps", "optimization_iterations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"oracle {name} must be positive")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "training_integration_period_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"oracle {name} must be finite and positive")
        for name in (
            "tracking_weight",
            "visibility_weight",
            "smoothness_weight",
            "saturation_weight",
            "residual_weight",
            "maximum_visibility_regression_fraction",
            "maximum_smoothness_regression_fraction",
            "maximum_saturation_regression_fraction",
            "maximum_visibility_regression_absolute_mse",
            "maximum_smoothness_regression_absolute_mse",
            "maximum_saturation_regression_absolute_mean",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"oracle {name} must be finite and non-negative"
                )
        if not 0.0 < self.maximum_command_residual <= 2.0:
            raise ValueError("oracle command residual must be in (0, 2]")
        if self.terminal_tracking_multiplier < 1.0:
            raise ValueError(
                "oracle terminal tracking multiplier must be at least one"
            )
        if not 0.0 < self.visibility_margin_fraction <= 1.0:
            raise ValueError("oracle visibility margin must be in (0, 1]")
        if not self.blend_fractions:
            raise ValueError("oracle blend fractions must not be empty")
        if self.blend_fractions[0] != 0.0:
            raise ValueError("oracle blend fractions must begin with zero")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.blend_fractions
        ):
            raise ValueError("oracle blend fractions must be in [0, 1]")
        if any(
            right <= left
            for left, right in zip(
                self.blend_fractions,
                self.blend_fractions[1:],
            )
        ):
            raise ValueError("oracle blend fractions must be increasing")


@dataclass
class PrivilegedSequenceOracleResult:
    base_command_normalized: torch.Tensor
    proposal_command_normalized: torch.Tensor
    selected_command_normalized: torch.Tensor
    selected_blend_fraction: torch.Tensor
    base_angle_rad: torch.Tensor
    selected_angle_rad: torch.Tensor
    base_rate_rad_s: torch.Tensor
    selected_rate_rad_s: torch.Tensor
    base_saturation_fraction: torch.Tensor
    selected_saturation_fraction: torch.Tensor
    target_bearing_after_command_rad: torch.Tensor
    sequence_mask: torch.Tensor
    base_metrics: dict[str, torch.Tensor]
    selected_metrics: dict[str, torch.Tensor]
    optimization_history: list[dict[str, float]]


def _repeat_context(
    context: GRUAdaptivePositionLossContext,
    count: int,
) -> GRUAdaptivePositionLossContext:
    return GRUAdaptivePositionLossContext(
        **{
            field.name: getattr(context, field.name).repeat_interleave(
                count,
                dim=0,
            )
            for field in fields(context)
        }
    )


def _sequence_metrics(
    command_normalized: torch.Tensor,
    angle_rad: torch.Tensor,
    saturation_fraction: torch.Tensor,
    target_bearing_after_command_rad: torch.Tensor,
    selected_axis_fov_rad: torch.Tensor,
    sequence_mask: torch.Tensor,
    previous_command_normalized: torch.Tensor,
    *,
    focus_start_index: int,
    focus_steps: int,
    visibility_margin_fraction: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    focus_end = focus_start_index + focus_steps
    focus_mask = sequence_mask[:, focus_start_index:focus_end]
    weights = focus_mask.to(angle_rad.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    half_fov = 0.5 * selected_axis_fov_rad[
        :, focus_start_index:focus_end
    ]
    tracking_error = angular_residual_rad(
        target_bearing_after_command_rad[:, focus_start_index:focus_end],
        angle_rad[:, focus_start_index:focus_end],
    ) / half_fov
    visibility = torch.relu(
        torch.abs(tracking_error) - visibility_margin_fraction
    )
    focus_command = command_normalized[:, focus_start_index:focus_end]
    preceding = (
        previous_command_normalized
        if focus_start_index == 0
        else command_normalized[:, focus_start_index - 1]
    )
    command_difference = torch.diff(
        torch.cat((preceding[:, None], focus_command), dim=1),
        dim=1,
    )
    metrics = {
        "tracking_mse": (
            tracking_error.square() * weights
        ).sum(dim=1)
        / denominator,
        "visibility_mse": (
            visibility.square() * weights
        ).sum(dim=1)
        / denominator,
        "smoothness_mse": (
            command_difference.square() * weights
        ).sum(dim=1)
        / denominator,
        "saturation_mean": (
            saturation_fraction[:, focus_start_index:focus_end] * weights
        ).sum(dim=1)
        / denominator,
    }
    return metrics, tracking_error, visibility


def _gather_candidate(
    values: torch.Tensor,
    candidate_index: torch.Tensor,
) -> torch.Tensor:
    batch_size = values.shape[0]
    return values[torch.arange(batch_size, device=values.device), candidate_index]


def optimize_privileged_command_sequence(
    base_command_normalized: torch.Tensor,
    target_bearing_after_command_rad: torch.Tensor,
    context: GRUAdaptivePositionLossContext,
    sequence_mask: torch.Tensor,
    initial_time_s: torch.Tensor,
    previous_command_normalized: torch.Tensor,
    *,
    config: PrivilegedSequenceOracleConfig | None = None,
) -> PrivilegedSequenceOracleResult:
    """Optimize a privileged focus segment and enforce exact constraints.

    Commands before the focus segment are frozen to the supplied baseline and
    replayed from episode start. The proposal is optimized with an approximate
    integration period, then blended back toward the baseline. Blend selection
    uses exact serialized integration independently for every episode and
    always includes the unchanged baseline as a feasible fallback.
    """

    config = config or PrivilegedSequenceOracleConfig()
    if base_command_normalized.ndim != 2:
        raise ValueError("oracle base commands must have shape [batch, time]")
    shape = base_command_normalized.shape
    batch_size, time_count = shape
    if target_bearing_after_command_rad.shape != shape:
        raise ValueError("oracle target-bearing shape is invalid")
    if sequence_mask.shape != shape:
        raise ValueError("oracle sequence-mask shape is invalid")
    if initial_time_s.shape != (batch_size,):
        raise ValueError("oracle initial-time shape is invalid")
    if previous_command_normalized.shape != (batch_size,):
        raise ValueError("oracle previous-command shape is invalid")
    focus_end = config.focus_start_index + config.focus_steps
    if focus_end > time_count:
        raise ValueError("oracle focus segment exceeds the supplied sequence")
    for field in fields(context):
        if getattr(context, field.name).shape != shape:
            raise ValueError(f"oracle context {field.name} shape is invalid")

    base_command = torch.clamp(base_command_normalized, -1.0, 1.0)
    raw_residual = torch.nn.Parameter(
        base_command.new_zeros(batch_size, config.focus_steps)
    )
    optimizer = torch.optim.Adam((raw_residual,), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    focus_mask = sequence_mask[:, config.focus_start_index:focus_end]

    for iteration in range(1, config.optimization_iterations + 1):
        residual = config.maximum_command_residual * torch.tanh(raw_residual)
        focus_command = torch.clamp(
            base_command[:, config.focus_start_index:focus_end] + residual,
            -1.0,
            1.0,
        )
        command = torch.cat(
            (
                base_command[:, : config.focus_start_index],
                focus_command,
                base_command[:, focus_end:],
            ),
            dim=1,
        )
        rollout = differentiable_position_servo_sequence(
            command,
            context,
            sequence_mask,
            initial_time_s=initial_time_s,
            integration_period_override_s=(
                config.training_integration_period_s
            ),
        )
        metrics, tracking_error, _ = _sequence_metrics(
            command,
            rollout.angle_rad,
            rollout.saturation_fraction,
            target_bearing_after_command_rad,
            context.selected_axis_fov_rad,
            sequence_mask,
            previous_command_normalized,
            focus_start_index=config.focus_start_index,
            focus_steps=config.focus_steps,
            visibility_margin_fraction=config.visibility_margin_fraction,
        )
        terminal_mask = focus_mask[:, -1].to(tracking_error.dtype)
        terminal_tracking = (
            tracking_error[:, -1].square() * terminal_mask
        ).sum() / terminal_mask.sum().clamp_min(1.0)
        residual_mse = (
            residual.square() * focus_mask.to(residual.dtype)
        ).sum() / focus_mask.sum().clamp_min(1)
        loss = (
            config.tracking_weight * metrics["tracking_mse"].mean()
            + config.visibility_weight * metrics["visibility_mse"].mean()
            + config.smoothness_weight * metrics["smoothness_mse"].mean()
            + config.saturation_weight * metrics["saturation_mean"].mean()
            + config.residual_weight * residual_mse
            + config.tracking_weight
            * (config.terminal_tracking_multiplier - 1.0)
            * terminal_tracking
            / config.focus_steps
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((raw_residual,), config.gradient_clip_norm)
        optimizer.step()
        history.append(
            {
                "iteration": iteration,
                "loss": float(loss.detach()),
                **{
                    name: float(value.mean().detach())
                    for name, value in metrics.items()
                },
                "residual_rms": float(torch.sqrt(residual_mse).detach()),
            }
        )

    with torch.no_grad():
        proposal_focus = torch.clamp(
            base_command[:, config.focus_start_index:focus_end]
            + config.maximum_command_residual * torch.tanh(raw_residual),
            -1.0,
            1.0,
        )
        proposal = torch.cat(
            (
                base_command[:, : config.focus_start_index],
                proposal_focus,
                base_command[:, focus_end:],
            ),
            dim=1,
        )
        blend = base_command.new_tensor(config.blend_fractions)
        candidate_count = blend.numel()
        candidates = base_command[:, None, :] + blend[None, :, None] * (
            proposal[:, None, :] - base_command[:, None, :]
        )
        flat_command = candidates.reshape(batch_size * candidate_count, time_count)
        flat_context = _repeat_context(context, candidate_count)
        flat_mask = sequence_mask.repeat_interleave(candidate_count, dim=0)
        flat_target = target_bearing_after_command_rad.repeat_interleave(
            candidate_count,
            dim=0,
        )
        flat_previous = previous_command_normalized.repeat_interleave(
            candidate_count,
            dim=0,
        )
        exact = differentiable_position_servo_sequence(
            flat_command,
            flat_context,
            flat_mask,
            initial_time_s=initial_time_s.repeat_interleave(
                candidate_count,
                dim=0,
            ),
        )
        flat_metrics, _, _ = _sequence_metrics(
            flat_command,
            exact.angle_rad,
            exact.saturation_fraction,
            flat_target,
            flat_context.selected_axis_fov_rad,
            flat_mask,
            flat_previous,
            focus_start_index=config.focus_start_index,
            focus_steps=config.focus_steps,
            visibility_margin_fraction=config.visibility_margin_fraction,
        )
        candidate_metrics = {
            name: value.reshape(batch_size, candidate_count)
            for name, value in flat_metrics.items()
        }
        base_metrics = {
            name: value[:, 0] for name, value in candidate_metrics.items()
        }
        visibility_limit = base_metrics["visibility_mse"] * (
            1.0 + config.maximum_visibility_regression_fraction
        ) + config.maximum_visibility_regression_absolute_mse + 1e-9
        smoothness_limit = base_metrics["smoothness_mse"] * (
            1.0 + config.maximum_smoothness_regression_fraction
        ) + config.maximum_smoothness_regression_absolute_mse + 1e-9
        saturation_limit = base_metrics["saturation_mean"] * (
            1.0 + config.maximum_saturation_regression_fraction
        ) + config.maximum_saturation_regression_absolute_mean + 1e-9
        feasible = (
            (candidate_metrics["visibility_mse"] <= visibility_limit[:, None])
            & (
                candidate_metrics["smoothness_mse"]
                <= smoothness_limit[:, None]
            )
            & (
                candidate_metrics["saturation_mean"]
                <= saturation_limit[:, None]
            )
        )
        feasible[:, 0] = True
        ranked_tracking = torch.where(
            feasible,
            candidate_metrics["tracking_mse"],
            torch.full_like(candidate_metrics["tracking_mse"], math.inf),
        )
        selected_index = torch.argmin(ranked_tracking, dim=1)
        selected_metrics = {
            name: _gather_candidate(value, selected_index)
            for name, value in candidate_metrics.items()
        }
        candidate_angle = exact.angle_rad.reshape(
            batch_size,
            candidate_count,
            time_count,
        )
        candidate_rate = exact.rate_rad_s.reshape(
            batch_size,
            candidate_count,
            time_count,
        )
        candidate_saturation = exact.saturation_fraction.reshape(
            batch_size,
            candidate_count,
            time_count,
        )

    return PrivilegedSequenceOracleResult(
        base_command_normalized=base_command.detach(),
        proposal_command_normalized=proposal.detach(),
        selected_command_normalized=_gather_candidate(
            candidates,
            selected_index,
        ).detach(),
        selected_blend_fraction=blend[selected_index].detach(),
        base_angle_rad=candidate_angle[:, 0].detach(),
        selected_angle_rad=_gather_candidate(
            candidate_angle,
            selected_index,
        ).detach(),
        base_rate_rad_s=candidate_rate[:, 0].detach(),
        selected_rate_rad_s=_gather_candidate(
            candidate_rate,
            selected_index,
        ).detach(),
        base_saturation_fraction=candidate_saturation[:, 0].detach(),
        selected_saturation_fraction=_gather_candidate(
            candidate_saturation,
            selected_index,
        ).detach(),
        target_bearing_after_command_rad=(
            target_bearing_after_command_rad.detach()
        ),
        sequence_mask=sequence_mask.detach(),
        base_metrics={name: value.detach() for name, value in base_metrics.items()},
        selected_metrics={
            name: value.detach() for name, value in selected_metrics.items()
        },
        optimization_history=history,
    )
