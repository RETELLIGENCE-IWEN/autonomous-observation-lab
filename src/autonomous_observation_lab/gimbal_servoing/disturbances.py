import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class AngularMotion(Protocol):
    """Exogenous one-axis angular trajectory."""

    def state_at(self, time_s: float) -> tuple[float, float]:
        """Return angular position in radians and rate in radians/second."""


@dataclass(frozen=True)
class StaticAngularMotion:
    angle_rad: float = 0.0

    def state_at(self, time_s: float) -> tuple[float, float]:
        del time_s
        return self.angle_rad, 0.0


@dataclass(frozen=True)
class ConstantRateAngularMotion:
    initial_angle_rad: float = 0.0
    rate_rad_s: float = 0.0

    def state_at(self, time_s: float) -> tuple[float, float]:
        return self.initial_angle_rad + self.rate_rad_s * time_s, self.rate_rad_s


@dataclass(frozen=True)
class SinusoidalAngularMotion:
    bias_rad: float = 0.0
    amplitude_rad: float = math.radians(8.0)
    frequency_hz: float = 1.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency_hz < 0.0:
            raise ValueError("frequency_hz must be non-negative")

    def state_at(self, time_s: float) -> tuple[float, float]:
        omega = 2.0 * math.pi * self.frequency_hz
        phase = omega * time_s + self.phase_rad
        return (
            self.bias_rad + self.amplitude_rad * math.sin(phase),
            self.amplitude_rad * omega * math.cos(phase),
        )


@dataclass(frozen=True)
class RatePulseAngularMotion:
    """A finite angular-rate pulse used as a constructed impulse-like case."""

    initial_angle_rad: float = 0.0
    onset_s: float = 1.0
    duration_s: float = 0.050
    rate_rad_s: float = math.radians(120.0)

    def __post_init__(self) -> None:
        if self.onset_s < 0.0 or self.duration_s <= 0.0:
            raise ValueError("pulse onset must be non-negative and duration positive")

    def state_at(self, time_s: float) -> tuple[float, float]:
        elapsed = min(max(time_s - self.onset_s, 0.0), self.duration_s)
        rate = self.rate_rad_s if self.onset_s <= time_s < self.onset_s + self.duration_s else 0.0
        return self.initial_angle_rad + self.rate_rad_s * elapsed, rate


@dataclass(frozen=True)
class SumAngularMotion:
    components: tuple[AngularMotion, ...]

    def state_at(self, time_s: float) -> tuple[float, float]:
        states = [component.state_at(time_s) for component in self.components]
        return sum(state[0] for state in states), sum(state[1] for state in states)


@dataclass(frozen=True)
class SampledAngularMotion:
    """Piecewise-linear motion reconstructed from deterministic samples."""

    times_s: tuple[float, ...]
    angles_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.angles_rad) or len(self.times_s) < 2:
            raise ValueError("sampled motion needs equally sized arrays with 2+ samples")
        if any(right <= left for left, right in zip(self.times_s, self.times_s[1:])):
            raise ValueError("sample times must be strictly increasing")

    def state_at(self, time_s: float) -> tuple[float, float]:
        index = int(np.searchsorted(self.times_s, time_s, side="right") - 1)
        index = max(0, min(index, len(self.times_s) - 2))
        left_time = self.times_s[index]
        right_time = self.times_s[index + 1]
        left_angle = self.angles_rad[index]
        right_angle = self.angles_rad[index + 1]
        rate = (right_angle - left_angle) / (right_time - left_time)
        if time_s <= self.times_s[0]:
            return self.angles_rad[0], rate
        if time_s >= self.times_s[-1]:
            return self.angles_rad[-1], rate
        angle = left_angle + rate * (time_s - left_time)
        return angle, rate
