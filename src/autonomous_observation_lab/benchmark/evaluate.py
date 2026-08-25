import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from .belief import FactorizedBelief
from .config import BenchmarkConfig
from .env import StagedEvidenceEnv
from .policies import make_policy


@dataclass(frozen=True)
class EvaluationSummary:
    policy: str
    episodes: int
    mean_return: float
    correct_rate: float
    wrong_rate: float
    abstain_rate: float
    timeout_rate: float
    mean_steps: float


def run_episode(policy_name: str, seed: int, config: BenchmarkConfig) -> dict[str, object]:
    env = StagedEvidenceEnv(config)
    policy = make_policy(policy_name, config)
    observation, _ = env.reset(seed)
    policy.reset()
    policy.observe(observation)
    total_reward = 0.0
    trajectory: list[dict[str, object]] = []

    while True:
        action = policy.act(observation)
        result = env.step(action)
        total_reward += result.reward
        trajectory.append(
            {
                "step": observation.step,
                "action": action.kind.value,
                "object_id": action.object_id,
                "reward": result.reward,
                "stage": result.info["stage"],
            }
        )
        observation = result.observation
        policy.observe(observation)
        if result.terminated or result.truncated:
            return {
                "seed": seed,
                "return": total_reward,
                "steps": len(trajectory),
                "outcome": result.info["outcome"],
                "target_id": result.info["target_id"],
                "trajectory": trajectory,
            }


def evaluate(
    policy_name: str, seeds: range, config: BenchmarkConfig
) -> EvaluationSummary:
    episodes = [run_episode(policy_name, seed, config) for seed in seeds]
    outcomes = [str(ep["outcome"]) for ep in episodes]
    correct = sum(outcome.startswith("correct") for outcome in outcomes)
    wrong = sum(outcome.startswith("wrong") for outcome in outcomes)
    return EvaluationSummary(
        policy=policy_name,
        episodes=len(episodes),
        mean_return=float(np.mean([ep["return"] for ep in episodes])),
        correct_rate=correct / len(episodes),
        wrong_rate=wrong / len(episodes),
        abstain_rate=outcomes.count("abstain") / len(episodes),
        timeout_rate=outcomes.count("timeout") / len(episodes),
        mean_steps=float(np.mean([ep["steps"] for ep in episodes])),
    )


def constructed_divergence_case(config: BenchmarkConfig) -> dict[str, object]:
    """Analytic case proving that generic entropy and decision VoI can diverge."""
    belief = FactorizedBelief.create(config)
    belief.signature_probability[:] = 0.02
    belief.motion_probability[:] = 0.02

    # Object 0 has three uncertain appearance bits but is known not to satisfy
    # the motion predicate. Object 1 is the only decision-relevant candidate.
    belief.signature_probability[0, :] = 0.5
    belief.motion_probability[0] = 0.01
    belief.signature_probability[1, :] = np.array([0.5, 0.02, 0.02])
    belief.motion_probability[1] = 0.95

    entropy_scores = [belief.entropy_score(i) for i in range(config.num_objects)]
    voi_scores = [belief.decision_voi(i) for i in range(config.num_objects)]
    return {
        "entropy_choice": int(np.argmax(entropy_scores)),
        "voi_choice": int(np.argmax(voi_scores)),
        "entropy_scores": entropy_scores,
        "voi_scores": voi_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "fixed_scan", "entropy_greedy", "decision_voi"],
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig()
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    summaries = [asdict(evaluate(name, seeds, config)) for name in args.policies]
    output = {
        "config": asdict(config),
        "constructed_divergence_case": constructed_divergence_case(config),
        "summaries": summaries,
    }
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

