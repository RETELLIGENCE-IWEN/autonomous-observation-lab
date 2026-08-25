from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the minimal Gate-1 benchmark."""

    num_objects: int = 5
    signature_bits: int = 3
    horizon: int = 16
    target_present_probability: float = 0.85

    discovery_end: int = 3
    occlusion_start: int = 7
    occlusion_end: int = 10

    wide_bit_accuracy: float = 0.62
    focus_bit_accuracy: float = 0.88
    dwell_bit_accuracy: float = 0.95
    motion_accuracy: float = 0.90
    miss_probability: float = 0.05

    wide_cost: float = 0.010
    focus_cost: float = 0.020
    dwell_cost: float = 0.035
    hold_cost: float = 0.005
    wrong_decision_utility: float = -1.0
    correct_decision_utility: float = 1.0
    abstain_utility: float = -0.15
    timeout_utility: float = -0.40

    def __post_init__(self) -> None:
        if self.num_objects < 3:
            raise ValueError("num_objects must be at least 3")
        if self.signature_bits < 2:
            raise ValueError("signature_bits must be at least 2")
        if not 0 <= self.occlusion_start < self.occlusion_end <= self.horizon:
            raise ValueError("invalid occlusion interval")
        for name in (
            "target_present_probability",
            "wide_bit_accuracy",
            "focus_bit_accuracy",
            "dwell_bit_accuracy",
            "motion_accuracy",
            "miss_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

