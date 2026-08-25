from autonomous_observation_lab.benchmark.config import BenchmarkConfig
from autonomous_observation_lab.benchmark.evaluate import (
    constructed_divergence_case,
    run_episode,
)


def test_entropy_and_decision_value_diverge_on_irrelevant_uncertainty():
    result = constructed_divergence_case(BenchmarkConfig())
    assert result["entropy_choice"] == 0
    assert result["voi_choice"] == 1


def test_episode_terminates_and_is_reproducible():
    config = BenchmarkConfig()
    first = run_episode("decision_voi", seed=44, config=config)
    second = run_episode("decision_voi", seed=44, config=config)
    assert first == second
    assert first["outcome"] in {
        "correct_commit",
        "wrong_commit",
        "correct_absent",
        "wrong_absent",
        "abstain",
        "timeout",
    }

