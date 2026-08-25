from dataclasses import dataclass
from enum import Enum

import numpy as np


class ActionKind(str, Enum):
    WIDE_SCAN = "wide_scan"
    LOOK = "look"
    DWELL = "dwell"
    HOLD = "hold"
    COMMIT = "commit"
    DECLARE_ABSENT = "declare_absent"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    object_id: int | None = None

    def __post_init__(self) -> None:
        needs_object = self.kind in {
            ActionKind.LOOK,
            ActionKind.DWELL,
            ActionKind.COMMIT,
        }
        if needs_object != (self.object_id is not None):
            raise ValueError(f"{self.kind.value} object_id validity mismatch")


@dataclass(frozen=True)
class Detection:
    handle: int
    bbox: np.ndarray
    confidence: float
    appearance: np.ndarray
    appearance_valid: np.ndarray
    motion_cue: int
    motion_valid: bool
    quality: float


@dataclass(frozen=True)
class Observation:
    step: int
    detections: tuple[Detection, ...]
    remaining_steps: int
    last_action: Action | None


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]

