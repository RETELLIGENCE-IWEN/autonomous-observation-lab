from dataclasses import asdict, replace

import pytest

from autonomous_observation_lab.gimbal_servoing.adaptive_position_v21 import (
    adaptive_position_v2_config,
)
from autonomous_observation_lab.gimbal_servoing.baseline_benchmark import (
    BASELINE_BENCHMARK_SCHEMA_VERSION,
    BaselineBenchmarkProtocolConfig,
    IdentifiedMPCCandidate,
    RobustPIDCandidate,
    baseline_method_registry,
    default_robust_pid_candidates,
    default_identified_mpc_candidates,
    develop_baseline_benchmark_v1,
    locked_identified_mpc_candidate,
    locked_robust_pid_config,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.config import GimbalCommandMode
from autonomous_observation_lab.gimbal_servoing.controllers import (
    RobustPIDControllerConfig,
)
from autonomous_observation_lab.gimbal_servoing.model_predictive_control import (
    IdentifiedMPCControllerConfig,
)


def test_baseline_registry_preserves_concept_locked_ids() -> None:
    registry = baseline_method_registry()

    assert [item.identifier for item in registry] == [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "L0",
        "L1",
        "L2",
        "P",
        "UB",
    ]
    assert len(registry) == 10
    assert sum(item.implementation_status == "ready" for item in registry) == 7
    assert sum(item.implementation_status == "partial" for item in registry) == 1
    assert sum(item.implementation_status == "missing" for item in registry) == 2
    assert not registry[-1].deployable


def test_robust_pid_development_lock_matches_selected_candidate() -> None:
    candidates = default_robust_pid_candidates()
    selected = next(
        item.controller for item in candidates if item.name == "imu_p_low_050"
    )

    assert len(candidates) == 12
    assert locked_robust_pid_config(GimbalCommandMode.POSITION) == selected
    assert locked_robust_pid_config(GimbalCommandMode.RATE) == selected


def test_identified_mpc_development_locks_match_selected_candidates() -> None:
    candidates = {
        item.name: item for item in default_identified_mpc_candidates()
    }

    assert len(candidates) == 6
    assert locked_identified_mpc_candidate(
        GimbalCommandMode.POSITION
    ) == candidates["mpc_h020_w5_s075"]
    assert locked_identified_mpc_candidate(
        GimbalCommandMode.RATE
    ) == candidates["mpc_h030_w10_s100"]


def test_baseline_protocol_develops_without_opening_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = nominal_scenario()
    scenario = replace(
        scenario,
        config=replace(
            scenario.config,
            timing=replace(scenario.config.timing, episode_duration_s=0.2),
        ),
    )
    monkeypatch.setattr(
        "autonomous_observation_lab.gimbal_servoing.baseline_benchmark._fresh_variants",
        lambda seeds: ((seeds[0], 0, scenario),),
    )
    monkeypatch.setattr(
        "autonomous_observation_lab.gimbal_servoing.baseline_benchmark._dream_to_center_reference",
        lambda **_kwargs: {
            "training_seeds": [7],
            "summary": {},
            "by_scenario": {},
        },
    )
    candidate = RobustPIDCandidate(
        "smoke",
        RobustPIDControllerConfig(
            proportional_gain_s_inv=2.0,
            integral_gain_s_inv2=0.0,
            derivative_gain=0.0,
        ),
    )
    protocol = BaselineBenchmarkProtocolConfig(
        selection_scenarios=("nominal_combined",),
        command_modes=(GimbalCommandMode.POSITION,),
        candidates=(candidate,),
        mpc_candidates=(
            IdentifiedMPCCandidate(
                "mpc_smoke",
                0.10,
                0.70,
                IdentifiedMPCControllerConfig(optimization_iterations=2),
            ),
        ),
    )
    visibility_result = {
        "protocol": {"maximum_staleness_s": 0.5},
        "development": {
            "selected_candidate": "adapter",
            "candidates": [
                {
                    "name": "adapter",
                    "controller_config": asdict(adaptive_position_v2_config()),
                }
            ],
        },
    }

    result = develop_baseline_benchmark_v1(
        visibility_risk_result=visibility_result,
        protocol=protocol,
        development_seeds=(120000,),
        sealed_test_seeds=(121000,),
    )

    assert result["experiment"] == BASELINE_BENCHMARK_SCHEMA_VERSION
    assert result["protocol_hash"]
    assert result["development"]["opened"]
    position = result["development"]["robust_pid"]["desired_position"]
    assert position["selected_candidate"] == "smoke"
    mpc = result["development"]["identified_mpc"]["desired_position"]
    assert mpc["selected_candidate"] == "mpc_smoke"
    assert not result["test"]["opened"]
    assert result["test"]["sealed"]
    assert result["test"]["world_seeds"] == [121000]


def test_baseline_protocol_rejects_overlapping_seed_blocks() -> None:
    visibility_result = {
        "development": {
            "selected_candidate": "unused",
            "candidates": [],
        }
    }
    with pytest.raises(ValueError, match="overlap"):
        develop_baseline_benchmark_v1(
            visibility_risk_result=visibility_result,
            development_seeds=(10,),
            sealed_test_seeds=(10,),
        )
