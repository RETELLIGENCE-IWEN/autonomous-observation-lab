from dataclasses import dataclass

from .config import GimbalCommandMode


@dataclass(frozen=True)
class MaskedScalar:
    """A scalar whose validity is explicit rather than encoded as a false zero."""

    value: float
    valid: bool

    @classmethod
    def missing(cls) -> "MaskedScalar":
        return cls(value=0.0, valid=False)


@dataclass(frozen=True)
class GimbalAction:
    """Exactly one normalized logical rate or position command."""

    desired_rate_normalized: float | None = None
    desired_position_normalized: float | None = None

    def __post_init__(self) -> None:
        values = (self.desired_rate_normalized, self.desired_position_normalized)
        if sum(value is not None for value in values) != 1:
            raise ValueError("exactly one of desired rate or position must be provided")
        if any(value is not None and not -1.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized commands must be in [-1, 1]")

    @classmethod
    def rate(cls, value: float) -> "GimbalAction":
        return cls(desired_rate_normalized=value)

    @classmethod
    def position(cls, value: float) -> "GimbalAction":
        return cls(desired_position_normalized=value)

    @property
    def mode(self) -> GimbalCommandMode:
        if self.desired_rate_normalized is not None:
            return GimbalCommandMode.RATE
        return GimbalCommandMode.POSITION

    @property
    def command_normalized(self) -> float:
        value = (
            self.desired_rate_normalized
            if self.desired_rate_normalized is not None
            else self.desired_position_normalized
        )
        assert value is not None
        return value


@dataclass(frozen=True)
class GimbalObservation:
    time_s: float
    control_dt_s: float
    frame_updated: bool
    measurement_age_s: MaskedScalar
    image_error_normalized: MaskedScalar
    bbox_width_fraction: MaskedScalar
    bbox_height_fraction: MaskedScalar
    confidence: MaskedScalar
    gimbal_angle_rad: MaskedScalar
    gimbal_rate_rad_s: MaskedScalar
    body_rate_rad_s: MaskedScalar
    command_mode: GimbalCommandMode
    previous_action_normalized: float

    @property
    def detection_valid(self) -> bool:
        return self.image_error_normalized.valid


@dataclass(frozen=True)
class GimbalDiagnostics:
    """Simulator-only state for evaluation; never part of actor observations."""

    time_s: float
    target_bearing_rad: float
    target_rate_rad_s: float
    body_bearing_rad: float
    body_rate_rad_s: float
    gimbal_angle_rad: float
    gimbal_rate_rad_s: float
    optical_axis_bearing_rad: float
    true_image_error_normalized: float
    command_mode: GimbalCommandMode
    requested_command_normalized: float
    requested_rate_rad_s: float | None
    requested_position_rad: float | None
    applied_rate_command_rad_s: float | None
    applied_position_command_rad: float | None
    inner_rate_target_rad_s: float
    target_in_view: bool
    rate_saturated: bool
    angle_saturated: bool


@dataclass(frozen=True)
class GimbalStepResult:
    observation: GimbalObservation
    reward: float
    terminated: bool
    truncated: bool
    diagnostics: GimbalDiagnostics
