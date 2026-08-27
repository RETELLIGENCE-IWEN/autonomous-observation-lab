import math
from dataclasses import dataclass, field
from enum import Enum


class ObservationProfile(str, Enum):
    """Signals exposed to the deployable controller."""

    VISION_ONLY = "o0_vision_only"
    SERVO_AWARE = "o1_servo_aware"
    DISTURBANCE_AWARE = "o2_disturbance_aware"


class GimbalCommandMode(str, Enum):
    """Logical outer-loop command accepted by a configured environment."""

    RATE = "desired_rate"
    POSITION = "desired_position"


@dataclass(frozen=True)
class ServoConfig:
    """Custom-servo plant and logical desired-rate adapter.

    Gimbal angle is body-relative. Zero radians always points the camera along
    the vehicle's body-forward axis. Hardware polarity is configurable without
    changing that logical coordinate convention.
    """

    min_angle_rad: float = math.radians(-90.0)
    max_angle_rad: float = math.radians(90.0)
    max_rate_rad_s: float = math.radians(120.0)
    max_acceleration_rad_s2: float = math.radians(720.0)
    rate_time_constant_s: float = 0.030
    command_latency_s: float = 0.010
    rate_deadband_rad_s: float = 0.0
    rate_quantization_rad_s: float = 0.0
    position_gain_s_inv: float = 12.0
    position_tolerance_rad: float = math.radians(0.10)
    position_quantization_rad: float = 0.0
    command_polarity: int = 1

    def __post_init__(self) -> None:
        if self.min_angle_rad >= 0.0 or self.max_angle_rad <= 0.0:
            raise ValueError("servo travel must include body-forward angle zero")
        if self.min_angle_rad >= self.max_angle_rad:
            raise ValueError("min_angle_rad must be less than max_angle_rad")
        if self.max_rate_rad_s <= 0.0:
            raise ValueError("max_rate_rad_s must be positive")
        if self.max_acceleration_rad_s2 <= 0.0:
            raise ValueError("max_acceleration_rad_s2 must be positive")
        if self.rate_time_constant_s < 0.0 or self.command_latency_s < 0.0:
            raise ValueError("servo time constant and latency must be non-negative")
        if self.rate_deadband_rad_s < 0.0 or self.rate_quantization_rad_s < 0.0:
            raise ValueError("servo deadband and quantization must be non-negative")
        if self.position_gain_s_inv <= 0.0:
            raise ValueError("position_gain_s_inv must be positive")
        if self.position_tolerance_rad < 0.0 or self.position_quantization_rad < 0.0:
            raise ValueError("position tolerance and quantization must be non-negative")
        if self.command_polarity not in {-1, 1}:
            raise ValueError("command_polarity must be -1 or 1")

    def position_from_normalized(self, value: float) -> float:
        """Map [-1, 1] onto asymmetric travel while preserving zero-forward."""
        if not -1.0 <= value <= 1.0:
            raise ValueError("normalized position must be in [-1, 1]")
        limit = self.max_angle_rad if value >= 0.0 else -self.min_angle_rad
        return value * limit

    def normalized_from_position(self, angle_rad: float) -> float:
        if not self.min_angle_rad <= angle_rad <= self.max_angle_rad:
            raise ValueError("angle is outside servo travel")
        limit = self.max_angle_rad if angle_rad >= 0.0 else -self.min_angle_rad
        return angle_rad / limit


@dataclass(frozen=True)
class CameraConfig:
    """Camera, detector, and observation-pipeline parameters."""

    selected_axis_fov_rad: float = math.radians(60.0)
    orthogonal_fov_rad: float = math.radians(45.0)
    frame_rate_hz: float = 30.0
    detection_latency_s: float = 0.040
    detection_latency_jitter_s: float = 0.0
    center_noise_std_normalized: float = 0.0
    size_noise_std_fraction: float = 0.0
    confidence_mean: float = 0.95
    confidence_noise_std: float = 0.0
    miss_probability: float = 0.0
    require_full_bbox_in_view: bool = False

    def __post_init__(self) -> None:
        for name in ("selected_axis_fov_rad", "orthogonal_fov_rad"):
            value = getattr(self, name)
            if not 0.0 < value < 2.0 * math.pi:
                raise ValueError(f"{name} must be in (0, 2*pi)")
        if self.frame_rate_hz <= 0.0:
            raise ValueError("frame_rate_hz must be positive")
        if self.detection_latency_s < 0.0 or self.detection_latency_jitter_s < 0.0:
            raise ValueError("camera latency and jitter must be non-negative")
        if self.center_noise_std_normalized < 0.0:
            raise ValueError("center noise must be non-negative")
        if self.size_noise_std_fraction < 0.0 or self.confidence_noise_std < 0.0:
            raise ValueError("size and confidence noise must be non-negative")
        if not 0.0 <= self.confidence_mean <= 1.0:
            raise ValueError("confidence_mean must be in [0, 1]")
        if not 0.0 <= self.miss_probability <= 1.0:
            raise ValueError("miss_probability must be in [0, 1]")

    @property
    def frame_period_s(self) -> float:
        return 1.0 / self.frame_rate_hz


@dataclass(frozen=True)
class TimingConfig:
    """Outer-loop, numerical-integration, and episode timing."""

    control_rate_hz: float = 30.0
    integration_rate_hz: float = 1000.0
    episode_duration_s: float = 5.0

    def __post_init__(self) -> None:
        if self.control_rate_hz <= 0.0 or self.integration_rate_hz <= 0.0:
            raise ValueError("control and integration rates must be positive")
        if self.integration_rate_hz < self.control_rate_hz:
            raise ValueError("integration rate must not be below control rate")
        if self.episode_duration_s <= 0.0:
            raise ValueError("episode_duration_s must be positive")

    @property
    def control_period_s(self) -> float:
        return 1.0 / self.control_rate_hz

    @property
    def integration_period_s(self) -> float:
        return 1.0 / self.integration_rate_hz


@dataclass(frozen=True)
class ScenarioConfig:
    """Initial geometry and target extent for a one-axis episode."""

    initial_gimbal_angle_rad: float = 0.0
    initial_gimbal_rate_rad_s: float = 0.0
    target_angular_width_rad: float = math.radians(4.0)
    target_angular_height_rad: float = math.radians(4.0)

    def __post_init__(self) -> None:
        if self.target_angular_width_rad <= 0.0:
            raise ValueError("target_angular_width_rad must be positive")
        if self.target_angular_height_rad <= 0.0:
            raise ValueError("target_angular_height_rad must be positive")


@dataclass(frozen=True)
class ObjectiveConfig:
    error_weight: float = 1.0
    loss_of_view_penalty: float = 2.0
    action_effort_weight: float = 0.01
    action_change_weight: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "error_weight",
            "loss_of_view_penalty",
            "action_effort_weight",
            "action_change_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class GimbalServoingConfig:
    servo: ServoConfig = field(default_factory=ServoConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    observation_profile: ObservationProfile = ObservationProfile.VISION_ONLY
    command_mode: GimbalCommandMode = GimbalCommandMode.RATE

    def __post_init__(self) -> None:
        initial_angle = self.scenario.initial_gimbal_angle_rad
        if not self.servo.min_angle_rad <= initial_angle <= self.servo.max_angle_rad:
            raise ValueError("initial gimbal angle is outside servo travel")
        if abs(self.scenario.initial_gimbal_rate_rad_s) > self.servo.max_rate_rad_s:
            raise ValueError("initial gimbal rate exceeds servo rate limit")
