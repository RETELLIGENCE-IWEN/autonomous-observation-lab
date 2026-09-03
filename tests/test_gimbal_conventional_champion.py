from dataclasses import replace

import pytest

from autonomous_observation_lab.gimbal_servoing.adaptive_position_v21 import (
    adaptive_position_v2_config,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    closed_loop_scenarios,
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.conventional_champion import (
    ConventionalChampionProtocolConfig,
    FeedbackScheduleCandidate,
    conventional_champion_run,
    locked_feedback_schedule,
    practical_feedback_run,
)


def test_feedback_schedule_reduces_gain_for_high_latency_hardware() -> None:
    scenarios = {item.name: item for item in closed_loop_scenarios()}
    schedule = locked_feedback_schedule()

    nominal_gain = schedule.gain_for(scenarios["nominal_combined"])
    high_latency_gain = schedule.gain_for(scenarios["high_latency"])

    assert nominal_gain <= schedule.maximum_gain
    assert high_latency_gain >= schedule.minimum_gain
    assert high_latency_gain < nominal_gain


def test_conventional_champion_and_feedback_expose_expected_forecasts() -> None:
    scenario = nominal_scenario()
    scenario = replace(
        scenario,
        config=replace(
            scenario.config,
            timing=replace(scenario.config.timing, episode_duration_s=0.5),
        ),
    )
    feedback = practical_feedback_run(scenario=scenario, seed=4)
    champion = conventional_champion_run(
        scenario=scenario,
        seed=4,
        adapter=adaptive_position_v2_config(),
    )

    assert not any(feedback.forecasts)
    assert any(len(values) == 4 for values in champion.forecasts)
    assert champion.episode.config.servo == feedback.episode.config.servo


def test_conventional_protocol_validates_unique_candidates() -> None:
    schedule = FeedbackScheduleCandidate("same", 0.09)
    with pytest.raises(ValueError, match="schedule names"):
        ConventionalChampionProtocolConfig(
            feedback_schedules=(schedule, schedule)
        )
