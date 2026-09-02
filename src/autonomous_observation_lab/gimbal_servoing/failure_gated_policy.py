"""Baseline-retaining failure-gated position correction policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .dataset import FEATURE_NAMES
from .gru import GRUAdaptivePositionLossContext
from .sequence_distillation import (
    HARDWARE_FEATURE_COUNT,
    CausalHardwareConditionedPositionPolicy,
    normalized_hardware_features,
)


_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FAILURE_EVIDENCE_COUNT = 6


def _ramp(
    value: torch.Tensor,
    onset: float,
    full: float,
) -> torch.Tensor:
    return torch.clamp((value - onset) / (full - onset), 0.0, 1.0)


@dataclass(frozen=True)
class FailureEvidenceConfig:
    visibility_onset_fraction: float = 0.45
    visibility_full_fraction: float = 0.90
    servo_rate_onset_fraction: float = 0.35
    servo_rate_full_fraction: float = 0.90
    travel_onset_fraction: float = 0.75
    travel_full_fraction: float = 0.98
    measurement_age_onset_frames: float = 1.5
    measurement_age_full_frames: float = 5.0
    body_rate_onset_fraction: float = 0.35
    body_rate_full_fraction: float = 0.90

    def __post_init__(self) -> None:
        for onset_name, full_name in (
            ("visibility_onset_fraction", "visibility_full_fraction"),
            ("servo_rate_onset_fraction", "servo_rate_full_fraction"),
            ("travel_onset_fraction", "travel_full_fraction"),
            ("measurement_age_onset_frames", "measurement_age_full_frames"),
            ("body_rate_onset_fraction", "body_rate_full_fraction"),
        ):
            onset = getattr(self, onset_name)
            full = getattr(self, full_name)
            if not math.isfinite(onset) or onset < 0.0:
                raise ValueError(f"{onset_name} must be finite and non-negative")
            if not math.isfinite(full) or full <= onset:
                raise ValueError(f"{full_name} must exceed its onset")


@dataclass(frozen=True)
class FailureGatedPositionPolicyConfig:
    hidden_dim: int = 48
    embedding_dim: int = 48
    maximum_residual_magnitude: float = 0.40
    initial_gate_bias: float = -2.0
    evidence: FailureEvidenceConfig = FailureEvidenceConfig()

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.embedding_dim <= 0:
            raise ValueError("failure-gated policy dimensions must be positive")
        if not math.isfinite(self.maximum_residual_magnitude) or not (
            0.0 < self.maximum_residual_magnitude <= 1.0
        ):
            raise ValueError("maximum failure-gated residual must be in (0, 1]")
        if not math.isfinite(self.initial_gate_bias):
            raise ValueError("initial gate bias must be finite")


@dataclass
class FailureGatedPositionStep:
    command_normalized: torch.Tensor
    base_command_normalized: torch.Tensor
    residual_normalized: torch.Tensor
    gate_probability: torch.Tensor
    failure_evidence: torch.Tensor
    base_hidden: torch.Tensor
    correction_hidden: torch.Tensor


@dataclass
class FailureGatedPositionSequence:
    command_normalized: torch.Tensor
    base_command_normalized: torch.Tensor
    residual_normalized: torch.Tensor
    gate_probability: torch.Tensor
    failure_evidence: torch.Tensor


def deployable_failure_evidence(
    feature: torch.Tensor,
    hardware: torch.Tensor,
    config: FailureEvidenceConfig | None = None,
) -> torch.Tensor:
    """Convert deployable image/servo fields into hardware-relative risks."""

    config = config or FailureEvidenceConfig()
    if feature.ndim != 2 or feature.shape[-1] != len(FEATURE_NAMES):
        raise ValueError("failure evidence feature shape is invalid")
    if hardware.shape != (feature.shape[0], HARDWARE_FEATURE_COUNT):
        raise ValueError("failure evidence hardware shape is invalid")
    image_valid = feature[:, _FEATURE_INDEX["image_error_valid"]].clamp(0.0, 1.0)
    visibility = _ramp(
        torch.abs(feature[:, _FEATURE_INDEX["image_error_normalized"]]),
        config.visibility_onset_fraction,
        config.visibility_full_fraction,
    ) * image_valid
    detector_gap = 1.0 - image_valid
    servo_rate = _ramp(
        torch.abs(feature[:, _FEATURE_INDEX["gimbal_rate_normalized"]]),
        config.servo_rate_onset_fraction,
        config.servo_rate_full_fraction,
    )
    travel = _ramp(
        torch.abs(feature[:, _FEATURE_INDEX["gimbal_position_normalized"]]),
        config.travel_onset_fraction,
        config.travel_full_fraction,
    )
    camera_period_s = (hardware[:, 9] * 0.05).clamp_min(1e-6)
    measurement_age_frames = (
        feature[:, _FEATURE_INDEX["measurement_age_s"]] / camera_period_s
    )
    age_valid = feature[
        :, _FEATURE_INDEX["measurement_age_valid"]
    ].clamp(0.0, 1.0)
    age = _ramp(
        measurement_age_frames,
        config.measurement_age_onset_frames,
        config.measurement_age_full_frames,
    ) * age_valid
    body_rate = _ramp(
        torch.abs(feature[:, _FEATURE_INDEX["body_rate_normalized"]]),
        config.body_rate_onset_fraction,
        config.body_rate_full_fraction,
    ) * feature[:, _FEATURE_INDEX["body_rate_valid"]].clamp(0.0, 1.0)
    return torch.stack(
        (visibility, detector_gap, servo_rate, travel, age, body_rate),
        dim=-1,
    )


class FailureGatedPositionCorrectionPolicy(nn.Module):
    """Freeze a causal base actor and learn only bounded gated corrections."""

    def __init__(
        self,
        base_policy: CausalHardwareConditionedPositionPolicy,
        config: FailureGatedPositionPolicyConfig | None = None,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.config = config or FailureGatedPositionPolicyConfig()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        input_dim = (
            len(FEATURE_NAMES)
            + HARDWARE_FEATURE_COUNT
            + 1
            + FAILURE_EVIDENCE_COUNT
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(
            self.config.embedding_dim,
            self.config.hidden_dim,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        residual_final = self.residual_head[-1]
        gate_final = self.gate_head[-1]
        assert isinstance(residual_final, nn.Linear)
        assert isinstance(gate_final, nn.Linear)
        nn.init.zeros_(residual_final.weight)
        nn.init.zeros_(residual_final.bias)
        nn.init.zeros_(gate_final.weight)
        nn.init.constant_(gate_final.bias, self.config.initial_gate_bias)

    def forward_step(
        self,
        feature: torch.Tensor,
        hardware: torch.Tensor,
        base_hidden: torch.Tensor | None = None,
        correction_hidden: torch.Tensor | None = None,
        *,
        previous_command_normalized: torch.Tensor,
        minimum_angle_rad: torch.Tensor,
        maximum_angle_rad: torch.Tensor,
    ) -> FailureGatedPositionStep:
        base_command, base_hidden = self.base_policy.forward_step(
            feature,
            hardware,
            base_hidden,
            previous_command_normalized=previous_command_normalized,
            minimum_angle_rad=minimum_angle_rad,
            maximum_angle_rad=maximum_angle_rad,
        )
        evidence = deployable_failure_evidence(
            feature,
            hardware,
            self.config.evidence,
        )
        inputs = torch.cat(
            (feature, hardware, base_command.unsqueeze(-1), evidence),
            dim=-1,
        )
        if correction_hidden is None:
            correction_hidden = feature.new_zeros(
                feature.shape[0],
                self.config.hidden_dim,
            )
        if correction_hidden.shape != (feature.shape[0], self.config.hidden_dim):
            raise ValueError("failure-gated correction hidden shape is invalid")
        correction_hidden = self.recurrent(
            self.encoder(inputs),
            correction_hidden,
        )
        gate = torch.sigmoid(self.gate_head(correction_hidden).squeeze(-1))
        raw_residual = self.config.maximum_residual_magnitude * torch.tanh(
            self.residual_head(correction_hidden).squeeze(-1)
        )
        residual = gate * raw_residual
        command = torch.clamp(base_command + residual, -1.0, 1.0)
        return FailureGatedPositionStep(
            command_normalized=command,
            base_command_normalized=base_command,
            residual_normalized=residual,
            gate_probability=gate,
            failure_evidence=evidence,
            base_hidden=base_hidden,
            correction_hidden=correction_hidden,
        )

    def forward(
        self,
        features: torch.Tensor,
        context: GRUAdaptivePositionLossContext,
        *,
        use_recorded_previous_command: bool = True,
    ) -> FailureGatedPositionSequence:
        if features.ndim != 3 or features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError(
                "failure-gated features must have shape [batch, time, feature]"
            )
        batch_size, time_count, _ = features.shape
        hardware = normalized_hardware_features(context)
        if hardware.shape != (batch_size, time_count, HARDWARE_FEATURE_COUNT):
            raise ValueError("failure-gated hardware context shape is invalid")
        previous = features[
            :, 0, _FEATURE_INDEX["previous_action_normalized"]
        ]
        base_hidden = None
        correction_hidden = None
        commands = []
        base_commands = []
        residuals = []
        gates = []
        evidence = []
        for time_index in range(time_count):
            if use_recorded_previous_command:
                previous = features[
                    :, time_index, _FEATURE_INDEX["previous_action_normalized"]
                ]
            step = self.forward_step(
                features[:, time_index],
                hardware[:, time_index],
                base_hidden,
                correction_hidden,
                previous_command_normalized=previous,
                minimum_angle_rad=context.servo_min_angle_rad[:, time_index],
                maximum_angle_rad=context.servo_max_angle_rad[:, time_index],
            )
            base_hidden = step.base_hidden
            correction_hidden = step.correction_hidden
            commands.append(step.command_normalized)
            base_commands.append(step.base_command_normalized)
            residuals.append(step.residual_normalized)
            gates.append(step.gate_probability)
            evidence.append(step.failure_evidence)
            previous = step.command_normalized
        return FailureGatedPositionSequence(
            command_normalized=torch.stack(commands, dim=1),
            base_command_normalized=torch.stack(base_commands, dim=1),
            residual_normalized=torch.stack(residuals, dim=1),
            gate_probability=torch.stack(gates, dim=1),
            failure_evidence=torch.stack(evidence, dim=1),
        )


class FailureGatedCommandResidualPolicy(nn.Module):
    """Bounded gated correction around externally supplied base commands."""

    def __init__(
        self,
        config: FailureGatedPositionPolicyConfig | None = None,
    ):
        super().__init__()
        self.config = config or FailureGatedPositionPolicyConfig()
        input_dim = (
            len(FEATURE_NAMES)
            + HARDWARE_FEATURE_COUNT
            + 1
            + FAILURE_EVIDENCE_COUNT
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
            nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(
            self.config.embedding_dim,
            self.config.hidden_dim,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        residual_final = self.residual_head[-1]
        gate_final = self.gate_head[-1]
        assert isinstance(residual_final, nn.Linear)
        assert isinstance(gate_final, nn.Linear)
        nn.init.zeros_(residual_final.weight)
        nn.init.zeros_(residual_final.bias)
        nn.init.zeros_(gate_final.weight)
        nn.init.constant_(gate_final.bias, self.config.initial_gate_bias)

    def forward_step(
        self,
        feature: torch.Tensor,
        hardware: torch.Tensor,
        base_command_normalized: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if feature.ndim != 2 or feature.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("command-residual feature shape is invalid")
        batch_size = feature.shape[0]
        if hardware.shape != (batch_size, HARDWARE_FEATURE_COUNT):
            raise ValueError("command-residual hardware shape is invalid")
        if base_command_normalized.shape != (batch_size,):
            raise ValueError("command-residual base-command shape is invalid")
        if hidden is None:
            hidden = feature.new_zeros(batch_size, self.config.hidden_dim)
        if hidden.shape != (batch_size, self.config.hidden_dim):
            raise ValueError("command-residual hidden shape is invalid")
        evidence = deployable_failure_evidence(
            feature,
            hardware,
            self.config.evidence,
        )
        inputs = torch.cat(
            (
                feature,
                hardware,
                base_command_normalized.unsqueeze(-1),
                evidence,
            ),
            dim=-1,
        )
        hidden = self.recurrent(self.encoder(inputs), hidden)
        gate = torch.sigmoid(self.gate_head(hidden).squeeze(-1))
        raw_residual = self.config.maximum_residual_magnitude * torch.tanh(
            self.residual_head(hidden).squeeze(-1)
        )
        residual = gate * raw_residual
        command = torch.clamp(
            base_command_normalized + residual,
            -1.0,
            1.0,
        )
        return command, residual, gate, evidence, hidden

    def forward(
        self,
        features: torch.Tensor,
        context: GRUAdaptivePositionLossContext,
        base_command_normalized: torch.Tensor,
    ) -> FailureGatedPositionSequence:
        if features.ndim != 3 or features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError(
                "command-residual features must have shape [batch, time, feature]"
            )
        batch_size, time_count, _ = features.shape
        if base_command_normalized.shape != (batch_size, time_count):
            raise ValueError("command-residual base sequence shape is invalid")
        hardware = normalized_hardware_features(context)
        hidden = None
        commands = []
        residuals = []
        gates = []
        evidence = []
        for time_index in range(time_count):
            command, residual, gate, step_evidence, hidden = self.forward_step(
                features[:, time_index],
                hardware[:, time_index],
                base_command_normalized[:, time_index],
                hidden,
            )
            commands.append(command)
            residuals.append(residual)
            gates.append(gate)
            evidence.append(step_evidence)
        return FailureGatedPositionSequence(
            command_normalized=torch.stack(commands, dim=1),
            base_command_normalized=base_command_normalized,
            residual_normalized=torch.stack(residuals, dim=1),
            gate_probability=torch.stack(gates, dim=1),
            failure_evidence=torch.stack(evidence, dim=1),
        )


@dataclass(frozen=True)
class ResidualAuthorityCalibratorConfig:
    hidden_dim: int = 16
    initial_authority: float = 1.0
    maximum_authority: float = 1.5

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("authority calibrator hidden dimension must be positive")
        for name in ("initial_authority", "maximum_authority"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"authority calibrator {name} must be finite")
        if not 0.0 < self.initial_authority < self.maximum_authority:
            raise ValueError(
                "initial authority must be between zero and maximum authority"
            )


class HardwareConditionedResidualAuthorityCalibrator(nn.Module):
    """Calibrate a frozen recurrent residual using deployable failure context."""

    def __init__(
        self,
        base_policy: FailureGatedCommandResidualPolicy,
        config: ResidualAuthorityCalibratorConfig | None = None,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.config = config or ResidualAuthorityCalibratorConfig()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.authority_head = nn.Sequential(
            nn.Linear(
                HARDWARE_FEATURE_COUNT + FAILURE_EVIDENCE_COUNT,
                self.config.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        final = self.authority_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        initial_fraction = (
            self.config.initial_authority / self.config.maximum_authority
        )
        nn.init.constant_(
            final.bias,
            math.log(initial_fraction / (1.0 - initial_fraction)),
        )

    def authority(
        self,
        hardware: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        if hardware.ndim < 2 or hardware.shape[-1] != HARDWARE_FEATURE_COUNT:
            raise ValueError("authority calibrator hardware shape is invalid")
        if evidence.shape != hardware.shape[:-1] + (FAILURE_EVIDENCE_COUNT,):
            raise ValueError("authority calibrator evidence shape is invalid")
        return self.config.maximum_authority * torch.sigmoid(
            self.authority_head(
                torch.cat((hardware, evidence), dim=-1)
            ).squeeze(-1)
        )

    def forward_step(
        self,
        feature: torch.Tensor,
        hardware: torch.Tensor,
        base_command_normalized: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            _base_corrected_command,
            base_residual,
            base_gate,
            evidence,
            hidden,
        ) = self.base_policy.forward_step(
            feature,
            hardware,
            base_command_normalized,
            hidden,
        )
        authority = self.authority(hardware, evidence)
        residual = authority * base_residual
        command = torch.clamp(
            base_command_normalized + residual,
            -1.0,
            1.0,
        )
        effective_gate = torch.clamp(base_gate * authority, 0.0, 1.0)
        return command, residual, effective_gate, evidence, hidden
