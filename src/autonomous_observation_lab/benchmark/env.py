from dataclasses import dataclass

import numpy as np

from .config import BenchmarkConfig
from .types import Action, ActionKind, Detection, Observation, StepResult


@dataclass(frozen=True)
class HiddenObject:
    object_id: int
    signature: np.ndarray
    motion_class: int
    position: np.ndarray
    velocity: np.ndarray

    @property
    def is_target(self) -> bool:
        return bool(self.signature[0] == 1 and self.motion_class == 1)


class StagedEvidenceEnv:
    """Deterministic-by-seed feature-level active-observation environment.

    The environment exposes integer handles equal to object IDs in Gate 1.
    They are handles, not target labels, and are randomized by scenario
    permutation. Handle corruption is reserved for the next benchmark version.
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config or BenchmarkConfig()
        self._rng = np.random.default_rng()
        self._objects: tuple[HiddenObject, ...] = ()
        self._target_id: int | None = None
        self._step = 0
        self._done = False
        self._last_action: Action | None = None
        self._handle_to_object: dict[int, int] = {}

    @property
    def target_id(self) -> int | None:
        """Privileged diagnostic state. Never include in actor observations."""
        return self._target_id

    @property
    def objects(self) -> tuple[HiddenObject, ...]:
        """Privileged diagnostic state for oracle policies and tests."""
        return self._objects

    def reset(self, seed: int) -> tuple[Observation, dict[str, object]]:
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._done = False
        self._last_action = None
        self._handle_to_object = {}
        self._objects, self._target_id = self._generate_scenario()
        obs = self._observe(Action(ActionKind.WIDE_SCAN), charge_step=False)
        return obs, {"seed": seed}

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("step called after episode completion")
        self._validate_action(action)

        reward = -self._action_cost(action)
        terminated = False
        truncated = False
        outcome = None

        resolved_object = self._resolve_handle(action.object_id)
        if action.kind is ActionKind.COMMIT:
            correct = resolved_object == self._target_id
            reward += (
                self.config.correct_decision_utility
                if correct
                else self.config.wrong_decision_utility
            )
            outcome = "correct_commit" if correct else "wrong_commit"
            terminated = True
        elif action.kind is ActionKind.DECLARE_ABSENT:
            correct = self._target_id is None
            reward += (
                self.config.correct_decision_utility
                if correct
                else self.config.wrong_decision_utility
            )
            outcome = "correct_absent" if correct else "wrong_absent"
            terminated = True
        elif action.kind is ActionKind.ABSTAIN:
            reward += self.config.abstain_utility
            outcome = "abstain"
            terminated = True

        self._last_action = action
        self._step += 1
        if not terminated and self._step >= self.config.horizon:
            reward += self.config.timeout_utility
            outcome = "timeout"
            truncated = True

        self._done = terminated or truncated
        obs = self._observe(action)
        return StepResult(
            observation=obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                "outcome": outcome,
                "target_id": self._target_id,
                "stage": self._stage_name(),
            },
        )

    def _generate_scenario(self) -> tuple[tuple[HiddenObject, ...], int | None]:
        cfg = self.config
        target_present = self._rng.random() < cfg.target_present_probability
        target_slot = int(self._rng.integers(cfg.num_objects)) if target_present else None
        objects: list[HiddenObject] = []

        for object_id in range(cfg.num_objects):
            signature = self._rng.integers(0, 2, size=cfg.signature_bits, dtype=np.int8)
            motion_class = int(self._rng.integers(0, 2))
            if object_id == target_slot:
                signature[0] = 1
                motion_class = 1
            elif signature[0] == 1 and motion_class == 1:
                # Guarantee a unique target predicate.
                if self._rng.random() < 0.5:
                    signature[0] = 0
                else:
                    motion_class = 0

            position = self._rng.uniform(-0.8, 0.8, size=2)
            velocity = self._rng.uniform(-0.035, 0.035, size=2)
            objects.append(
                HiddenObject(
                    object_id=object_id,
                    signature=signature,
                    motion_class=motion_class,
                    position=position,
                    velocity=velocity,
                )
            )
        return tuple(objects), target_slot

    def _observe(self, action: Action, charge_step: bool = True) -> Observation:
        del charge_step
        detections: list[Detection] = []
        # Preserve last-known associations so a policy can revisit an object
        # that is no longer in the current detection set. New observations
        # overwrite the mapping, including deliberate corruption/collisions.
        next_handle_to_object: dict[int, int] = dict(self._handle_to_object)
        focused_object = self._resolve_handle(action.object_id)
        for obj in self._objects:
            if self._is_occluded(obj.object_id):
                continue
            focused = focused_object == obj.object_id
            included = action.kind is ActionKind.WIDE_SCAN or focused
            if action.kind is ActionKind.HOLD:
                included = False
            if not included or self._rng.random() < self.config.miss_probability:
                continue

            if action.kind is ActionKind.DWELL:
                bit_accuracy = self.config.dwell_bit_accuracy
                quality = 1.0
            elif focused:
                bit_accuracy = self.config.focus_bit_accuracy
                quality = 0.8
            else:
                bit_accuracy = self.config.wide_bit_accuracy
                quality = 0.4

            appearance = self._noisy_bits(obj.signature, bit_accuracy)
            motion_cue = int(
                obj.motion_class
                if self._rng.random() < self.config.motion_accuracy
                else 1 - obj.motion_class
            )
            position = obj.position + self._step * obj.velocity
            bbox = np.array(
                [position[0], position[1], 0.08 + 0.02 * quality, 0.06 + 0.02 * quality],
                dtype=np.float32,
            )
            observed_handle = self._observed_handle(obj.object_id)
            next_handle_to_object[observed_handle] = obj.object_id
            detections.append(
                Detection(
                    handle=observed_handle,
                    bbox=bbox,
                    confidence=float(0.55 + 0.4 * quality),
                    appearance=appearance,
                    appearance_valid=np.ones_like(appearance, dtype=bool),
                    motion_cue=motion_cue,
                    motion_valid=True,
                    quality=quality,
                )
            )

        self._handle_to_object = next_handle_to_object
        return Observation(
            step=self._step,
            detections=tuple(detections),
            remaining_steps=max(0, self.config.horizon - self._step),
            last_action=self._last_action,
        )

    def _is_occluded(self, object_id: int) -> bool:
        return (
            object_id == self._target_id
            and self.config.occlusion_start <= self._step < self.config.occlusion_end
        )

    def _noisy_bits(self, truth: np.ndarray, accuracy: float) -> np.ndarray:
        keep = self._rng.random(size=truth.shape) < accuracy
        return np.where(keep, truth, 1 - truth).astype(np.int8)

    def _observed_handle(self, object_id: int) -> int:
        probability = self.config.handle_corruption_probability
        if probability <= 0.0:
            # Disabled features must not consume RNG state; this preserves
            # seeded trajectories across compatible benchmark revisions.
            return object_id
        if self._rng.random() < probability:
            # Independent reassignment naturally creates resets, switches, and
            # collisions. It is an observed association hint, not identity.
            return int(self._rng.integers(self.config.num_objects))
        return object_id

    def _resolve_handle(self, handle: int | None) -> int | None:
        if handle is None:
            return None
        return self._handle_to_object.get(handle)

    def _action_cost(self, action: Action) -> float:
        return {
            ActionKind.WIDE_SCAN: self.config.wide_cost,
            ActionKind.LOOK: self.config.focus_cost,
            ActionKind.DWELL: self.config.dwell_cost,
            ActionKind.HOLD: self.config.hold_cost,
            ActionKind.COMMIT: 0.0,
            ActionKind.DECLARE_ABSENT: 0.0,
            ActionKind.ABSTAIN: 0.0,
        }[action.kind]

    def _validate_action(self, action: Action) -> None:
        if action.object_id is not None and not 0 <= action.object_id < self.config.num_objects:
            raise ValueError("object_id outside configured object range")

    def _stage_name(self) -> str:
        if self._step < self.config.discovery_end:
            return "discovery"
        if self._step < self.config.occlusion_start:
            return "disambiguation"
        if self._step < self.config.occlusion_end:
            return "interruption"
        return "reacquisition"
