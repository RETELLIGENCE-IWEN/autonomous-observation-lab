import json

import pytest

from autonomous_observation_lab.gimbal_servoing.performance_atlas import (
    PerformanceContract,
)
from autonomous_observation_lab.gimbal_servoing.predictive_position_v3 import (
    PredictivePositionV3ProtocolConfig,
    _gate,
    default_v3_candidates,
    evaluate_predictive_position_v3,
)


def _summary(*, mean, p95, loss, avoidable, variation, events):
    return {
        "mean_absolute_error_fov_fraction": mean,
        "p95_absolute_error_fov_fraction": p95,
        "loss_of_view_fraction": loss,
        "avoidable_loss_fraction": avoidable,
        "command_variation_per_s": variation,
        "actuator_acceleration_rms_normalized": 0.5,
        "unrecovered_loss_events": events,
    }


def _scenario(summary):
    return {
        "aggressive_motion": {
            "summary": summary,
            "contract_applicable": True,
            "contract": None,
        }
    }


def test_v3_gate_requires_tracking_loss_smoothness_and_event_safety():
    reference = _summary(
        mean=0.34,
        p95=0.78,
        loss=0.05,
        avoidable=0.04,
        variation=1.2,
        events=4,
    )
    candidate = _summary(
        mean=0.32,
        p95=0.74,
        loss=0.04,
        avoidable=0.03,
        variation=1.0,
        events=3,
    )

    gate = _gate(
        candidate=candidate,
        reference=reference,
        candidate_by_scenario=_scenario(candidate),
        reference_by_scenario=_scenario(reference),
        minimum_avoidable_reduction=0.02,
        minimum_variation_reduction=0.03,
        maximum_mean_regression=0.0,
        maximum_p95_regression=0.0,
        maximum_scenario_p95_regression=0.03,
        maximum_scenario_avoidable_regression=0.005,
    )

    assert gate["passed"]
    assert gate["comparison"][
        "avoidable_loss_reduction_fraction"
    ] == pytest.approx(0.25)

    regressed = dict(candidate, unrecovered_loss_events=5)
    rejected = _gate(
        candidate=regressed,
        reference=reference,
        candidate_by_scenario=_scenario(regressed),
        reference_by_scenario=_scenario(reference),
        minimum_avoidable_reduction=0.02,
        minimum_variation_reduction=0.03,
        maximum_mean_regression=0.0,
        maximum_p95_regression=0.0,
        maximum_scenario_p95_regression=0.03,
        maximum_scenario_avoidable_regression=0.005,
    )
    assert not rejected["passed"]
    assert not rejected["checks"]["unrecovered_events"]


def test_v3_candidates_are_predeclared_and_hardware_relative():
    candidates = default_v3_candidates()
    assert len(candidates) == 4
    assert len({candidate.name for candidate in candidates}) == 4
    assert all(
        candidate.controller.minimum_optimizer_position_gain_s_inv > 0.0
        for candidate in candidates
    )
    assert all(
        candidate.controller.maximum_optimization_horizon_s == 0.1
        for candidate in candidates
    )


def test_v3_rejects_historical_world_seed_before_loading_models(tmp_path):
    source = tmp_path / "v21.json"
    source.write_text(
        json.dumps(
            {"experiment": "gimbal_adaptive_position_v21_protocol_v1"}
        ),
        encoding="utf-8",
    )

    try:
        evaluate_predictive_position_v3(
            visibility_risk_results=source,
            protocol=PredictivePositionV3ProtocolConfig(),
            contract=PerformanceContract(),
            development_seeds=(82000,),
            confirmation_seeds=(84000,),
        )
    except ValueError as error:
        assert "historical evaluation seeds" in str(error)
    else:
        raise AssertionError("historical seed reuse should be rejected")
