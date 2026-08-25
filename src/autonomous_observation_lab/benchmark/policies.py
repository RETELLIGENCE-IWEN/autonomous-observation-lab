from abc import ABC, abstractmethod

import numpy as np

from .belief import FactorizedBelief
from .config import BenchmarkConfig
from .types import Action, ActionKind, Observation


class Policy(ABC):
    name = "policy"

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def observe(self, observation: Observation) -> None: ...

    @abstractmethod
    def act(self, observation: Observation) -> Action: ...


class BeliefPolicy(Policy):
    def __init__(self, config: BenchmarkConfig, commit_threshold: float = 0.90):
        self.config = config
        self.commit_threshold = commit_threshold
        self.belief = FactorizedBelief.create(config)

    def reset(self) -> None:
        self.belief = FactorizedBelief.create(self.config)

    def observe(self, observation: Observation) -> None:
        self.belief.update(observation)

    def _commit_if_ready(self, observation: Observation) -> Action | None:
        decision, confidence = self.belief.decision_confidence()
        if confidence >= self.commit_threshold:
            if decision is None:
                return Action(ActionKind.DECLARE_ABSENT)
            return Action(ActionKind.COMMIT, decision)
        if observation.remaining_steps <= 1:
            if confidence >= 0.55:
                if decision is None:
                    return Action(ActionKind.DECLARE_ABSENT)
                return Action(ActionKind.COMMIT, decision)
            return Action(ActionKind.ABSTAIN)
        return None

    @staticmethod
    def _visible_ids(observation: Observation) -> list[int]:
        return sorted({d.handle for d in observation.detections})


class FixedScanPolicy(BeliefPolicy):
    name = "fixed_scan"

    def act(self, observation: Observation) -> Action:
        decision = self._commit_if_ready(observation)
        if decision is not None:
            return decision
        object_id = observation.step % self.config.num_objects
        return Action(ActionKind.LOOK, object_id)


class EntropyGreedyPolicy(BeliefPolicy):
    name = "entropy_greedy"

    def act(self, observation: Observation) -> Action:
        decision = self._commit_if_ready(observation)
        if decision is not None:
            return decision
        visible = self._visible_ids(observation)
        if not visible:
            return Action(ActionKind.WIDE_SCAN)
        object_id = max(visible, key=self.belief.entropy_score)
        return Action(ActionKind.LOOK, object_id)


class DecisionAwareVoIPolicy(BeliefPolicy):
    name = "decision_voi"

    def act(self, observation: Observation) -> Action:
        decision = self._commit_if_ready(observation)
        if decision is not None:
            return decision
        visible = self._visible_ids(observation)
        if not visible:
            return Action(ActionKind.WIDE_SCAN)
        scores = {object_id: self.belief.decision_voi(object_id) for object_id in visible}
        best_id = max(scores, key=scores.get)
        if scores[best_id] <= 0 and observation.step >= self.config.occlusion_start:
            return Action(ActionKind.WIDE_SCAN)
        return Action(ActionKind.LOOK, best_id)


class RandomPolicy(BeliefPolicy):
    name = "random"

    def __init__(self, config: BenchmarkConfig, seed: int = 0):
        super().__init__(config)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        super().reset()
        self.rng = np.random.default_rng(self.seed)

    def act(self, observation: Observation) -> Action:
        decision = self._commit_if_ready(observation)
        if decision is not None:
            return decision
        visible = self._visible_ids(observation)
        if not visible:
            return Action(ActionKind.WIDE_SCAN)
        return Action(ActionKind.LOOK, int(self.rng.choice(visible)))


def make_policy(name: str, config: BenchmarkConfig) -> Policy:
    factories = {
        "random": lambda: RandomPolicy(config),
        "fixed_scan": lambda: FixedScanPolicy(config),
        "entropy_greedy": lambda: EntropyGreedyPolicy(config),
        "decision_voi": lambda: DecisionAwareVoIPolicy(config),
    }
    try:
        return factories[name]()
    except KeyError as error:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(factories)}") from error

