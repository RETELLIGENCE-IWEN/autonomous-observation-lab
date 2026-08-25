from autonomous_observation_lab.benchmark.config import BenchmarkConfig
from autonomous_observation_lab.benchmark.evaluate import (
    constructed_divergence_case,
    run_episode,
)
from autonomous_observation_lab.benchmark.leakage import run_leakage_probe
from autonomous_observation_lab.benchmark.planning_cases import (
    constructed_multistep_case,
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


def test_nuisance_features_fail_to_predict_target_above_null_tolerance():
    result = run_leakage_probe(episodes=800, permutations=8)
    assert result.samples > 3_000
    assert result.passed


def test_multistep_planning_selects_delayed_evidence():
    result = constructed_multistep_case()
    assert result.one_step_choice == "direct"
    assert result.two_step_choice == "scout"
    assert result.two_step_scores["scout"] > result.two_step_scores["direct"]
