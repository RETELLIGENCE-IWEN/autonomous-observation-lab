from dataclasses import dataclass

import numpy as np

from .config import BenchmarkConfig
from .types import ActionKind, Detection, Observation


def binary_entropy(p: np.ndarray | float) -> np.ndarray | float:
    values = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))


def binary_update(prior: float, observation: int, accuracy: float) -> float:
    likelihood_one = accuracy if observation == 1 else 1.0 - accuracy
    likelihood_zero = 1.0 - accuracy if observation == 1 else accuracy
    numerator = likelihood_one * prior
    denominator = numerator + likelihood_zero * (1.0 - prior)
    return float(numerator / denominator)


def expected_binary_entropy(prior: float, accuracy: float) -> float:
    p_y1 = accuracy * prior + (1.0 - accuracy) * (1.0 - prior)
    post_y1 = binary_update(prior, 1, accuracy)
    post_y0 = binary_update(prior, 0, accuracy)
    return float(
        p_y1 * binary_entropy(post_y1)
        + (1.0 - p_y1) * binary_entropy(post_y0)
    )


@dataclass
class FactorizedBelief:
    """Exact binary-channel filter used as a transparent Gate-1 oracle.

    It intentionally factorizes objects and attributes. It is not the final
    learned belief and serves as a benchmark validator and oracle baseline.
    """

    config: BenchmarkConfig
    signature_probability: np.ndarray
    motion_probability: np.ndarray
    seen_count: np.ndarray
    last_seen_step: np.ndarray

    @classmethod
    def create(cls, config: BenchmarkConfig) -> "FactorizedBelief":
        return cls(
            config=config,
            signature_probability=np.full(
                (config.num_objects, config.signature_bits), 0.5, dtype=np.float64
            ),
            motion_probability=np.full(config.num_objects, 0.5, dtype=np.float64),
            seen_count=np.zeros(config.num_objects, dtype=np.int64),
            last_seen_step=np.full(config.num_objects, -1, dtype=np.int64),
        )

    def update(self, observation: Observation) -> None:
        for detection in observation.detections:
            self._update_detection(detection, observation.step)

    def _update_detection(self, detection: Detection, step: int) -> None:
        object_id = detection.handle
        accuracy = self._appearance_accuracy(detection.quality)
        for bit, valid in enumerate(detection.appearance_valid):
            if valid:
                prior = self.signature_probability[object_id, bit]
                self.signature_probability[object_id, bit] = binary_update(
                    float(prior), int(detection.appearance[bit]), accuracy
                )
        if detection.motion_valid:
            prior_motion = self.motion_probability[object_id]
            self.motion_probability[object_id] = binary_update(
                float(prior_motion),
                detection.motion_cue,
                self.config.motion_accuracy,
            )
        self.seen_count[object_id] += 1
        self.last_seen_step[object_id] = step

    def target_probability(self) -> np.ndarray:
        return self.signature_probability[:, 0] * self.motion_probability

    def absent_probability(self) -> float:
        return float(np.prod(1.0 - self.target_probability()))

    def decision_confidence(self) -> tuple[int | None, float]:
        target_p = self.target_probability()
        best_id = int(np.argmax(target_p))
        best_p = float(target_p[best_id])
        absent_p = self.absent_probability()
        if absent_p > best_p:
            return None, absent_p
        return best_id, best_p

    def entropy_score(self, object_id: int, kind: ActionKind = ActionKind.LOOK) -> float:
        accuracy = self._accuracy_for_action(kind)
        priors = self.signature_probability[object_id]
        gain = np.sum(
            [
                binary_entropy(float(p)) - expected_binary_entropy(float(p), accuracy)
                for p in priors
            ]
        )
        motion_prior = float(self.motion_probability[object_id])
        gain += binary_entropy(motion_prior) - expected_binary_entropy(
            motion_prior, self.config.motion_accuracy
        )
        return float(gain)

    def decision_voi(self, object_id: int, kind: ActionKind = ActionKind.LOOK) -> float:
        """One-step expected improvement in approximate decision confidence."""
        accuracy = self._accuracy_for_action(kind)
        base_confidence = self.decision_confidence()[1]
        p_sig = float(self.signature_probability[object_id, 0])
        p_motion = float(self.motion_probability[object_id])
        expected = 0.0

        for sig_obs in (0, 1):
            p_sig_obs = (
                accuracy * p_sig + (1.0 - accuracy) * (1.0 - p_sig)
                if sig_obs == 1
                else (1.0 - accuracy) * p_sig + accuracy * (1.0 - p_sig)
            )
            for mot_obs in (0, 1):
                ma = self.config.motion_accuracy
                p_mot_obs = (
                    ma * p_motion + (1.0 - ma) * (1.0 - p_motion)
                    if mot_obs == 1
                    else (1.0 - ma) * p_motion + ma * (1.0 - p_motion)
                )
                old_sig = self.signature_probability[object_id, 0]
                old_motion = self.motion_probability[object_id]
                self.signature_probability[object_id, 0] = binary_update(
                    p_sig, sig_obs, accuracy
                )
                self.motion_probability[object_id] = binary_update(
                    p_motion, mot_obs, ma
                )
                confidence = self.decision_confidence()[1]
                self.signature_probability[object_id, 0] = old_sig
                self.motion_probability[object_id] = old_motion
                expected += p_sig_obs * p_mot_obs * confidence

        cost = (
            self.config.focus_cost
            if kind is ActionKind.LOOK
            else self.config.dwell_cost
        )
        return float(expected - base_confidence - cost)

    def _appearance_accuracy(self, quality: float) -> float:
        if quality >= 0.99:
            return self.config.dwell_bit_accuracy
        if quality >= 0.79:
            return self.config.focus_bit_accuracy
        return self.config.wide_bit_accuracy

    def _accuracy_for_action(self, kind: ActionKind) -> float:
        if kind is ActionKind.DWELL:
            return self.config.dwell_bit_accuracy
        return self.config.focus_bit_accuracy

