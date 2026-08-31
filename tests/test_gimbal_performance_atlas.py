import json
import math
from dataclasses import replace

import pytest

from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    ClosedLoopScenario,
    closed_loop_config,
    run_closed_loop_controller,
)
from autonomous_observation_lab.gimbal_servoing.config import (
    GimbalCommandMode,
    ObservationProfile,
)
from autonomous_observation_lab.gimbal_servoing.disturbances import (
    StaticAngularMotion,
)
from autonomous_observation_lab.gimbal_servoing.performance_atlas import (
    FailureAtlasConfig,
    PerformanceContract,
    _aggregate_records,
    analyze_controller_run,
    evaluate_contract,
    load_performance_contract,
)
from autonomous_observation_lab.gimbal_servoing.types import GimbalAction


class _HoldPositionController:
    name = "hold_position"

    def reset(self) -> None:
        pass

    def act(self, _observation):
        return GimbalAction.position(0.0)


def _static_run(target_deg: float):
    config = closed_loop_config(GimbalCommandMode.POSITION)
    config = replace(
        config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        timing=replace(config.timing, episode_duration_s=0.50),
    )
    scenario = ClosedLoopScenario(
        name="static_failure",
        description="Static target used to verify failure attribution.",
        config=config,
        target_motion=StaticAngularMotion(angle_rad=math.radians(target_deg)),
        body_motion=StaticAngularMotion(),
    )
    run = run_closed_loop_controller(
        name="hold_position",
        description="Hold body-forward.",
        scenario=scenario,
        config=config,
        controller=_HoldPositionController(),
        seed=7,
    )
    return scenario, run


def test_failure_atlas_separates_physical_and_controller_induced_loss():
    unreachable_scenario, unreachable_run = _static_run(82.0)
    unreachable = analyze_controller_run(
        unreachable_run,
        unreachable_scenario,
        controller_name="hold",
        world_seed=1,
        training_seed=2,
    )
    assert unreachable["physical_unreachable_fraction"] == pytest.approx(1.0)
    assert unreachable["avoidable_loss_fraction"] == pytest.approx(0.0)
    assert unreachable["loss_events"][0]["cause"] == "physical_envelope"

    reachable_scenario, reachable_run = _static_run(40.0)
    reachable = analyze_controller_run(
        reachable_run,
        reachable_scenario,
        controller_name="hold",
        world_seed=1,
        training_seed=2,
        analysis=FailureAtlasConfig(
            minimum_recent_detection_valid_fraction=0.0
        ),
    )
    assert reachable["physical_unreachable_fraction"] == pytest.approx(0.0)
    assert reachable["avoidable_loss_fraction"] > 0.0
    assert reachable["loss_events"][0]["cause"] != "physical_envelope"
    assert reachable["loss_events"][0]["cause"] != "detector_gap"


def test_performance_contract_is_configurable_and_absolute(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "name": "test_contract",
                "tracked_scenarios": ["static_failure"],
                "maximum_loss_of_view_fraction": 0.0,
            }
        ),
        encoding="utf-8",
    )
    contract = load_performance_contract(path)
    assert contract.name == "test_contract"
    assert contract.tracked_scenarios == ("static_failure",)

    scenario, run = _static_run(40.0)
    record = analyze_controller_run(
        run,
        scenario,
        controller_name="hold",
        world_seed=1,
        training_seed=2,
    )
    summary = _aggregate_records([record])
    verdict = evaluate_contract(summary, contract)
    assert not verdict["passed"]
    assert not verdict["checks"]["loss_of_view"]
    assert verdict["check_count"] == 10


def test_contract_rejects_duplicate_scenarios_and_invalid_thresholds():
    with pytest.raises(ValueError, match="unique"):
        PerformanceContract(tracked_scenarios=("nominal", "nominal"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        PerformanceContract(maximum_loss_of_view_fraction=-0.1)
