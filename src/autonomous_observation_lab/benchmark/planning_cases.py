from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningSeparationResult:
    one_step_choice: str
    two_step_choice: str
    one_step_scores: dict[str, float]
    two_step_scores: dict[str, float]


def constructed_multistep_case() -> PlanningSeparationResult:
    """A minimal staged-evidence decision tree.

    DIRECT produces useful evidence immediately but saturates. SCOUT has no
    immediate decision benefit and unlocks REVEAL on the next step. The case
    establishes why a benchmark for latent imagination needs delayed evidence.
    """
    current_confidence = 0.55
    direct_confidence = 0.70
    scout_confidence = 0.55
    reveal_after_scout = 0.95
    action_cost = 0.02

    one_step = {
        "direct": direct_confidence - current_confidence - action_cost,
        "scout": scout_confidence - current_confidence - action_cost,
    }
    two_step = {
        "direct": direct_confidence - current_confidence - 2.0 * action_cost,
        "scout": reveal_after_scout - current_confidence - 2.0 * action_cost,
    }
    return PlanningSeparationResult(
        one_step_choice=max(one_step, key=one_step.get),
        two_step_choice=max(two_step, key=two_step.get),
        one_step_scores=one_step,
        two_step_scores=two_step,
    )

