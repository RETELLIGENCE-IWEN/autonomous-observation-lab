from copy import deepcopy

import pytest

from autonomous_observation_lab.gimbal_servoing.adaptive_position_v21 import (
    VisibilityRiskProtocolConfig,
    _confirmation_gate,
    evaluate_visibility_risk_v21,
)


def _summary(mean, p95, loss, variation, cost, events):
    return {
        "mean_absolute_error_deg": mean,
        "p95_absolute_error_deg": p95,
        "loss_of_view_fraction": loss,
        "command_variation_per_s": variation,
        "mean_control_cost": cost,
        "total_unrecovered_loss_events": events,
    }


def test_visibility_risk_protocol_requires_a_development_event_reduction():
    with pytest.raises(ValueError, match="at least one fewer"):
        VisibilityRiskProtocolConfig(
            development_minimum_unrecovered_event_reduction=0
        )


def test_visibility_risk_protocol_rejects_historical_v2_fresh_seeds():
    with pytest.raises(ValueError, match="historical V2 fresh seeds"):
        evaluate_visibility_risk_v21(
            validation_data="unused-validation.npz",
            test_data="unused-test.npz",
            checkpoints={17: "unused-17.pt", 29: "unused-29.pt"},
            development_seeds=(80000,),
            confirmation_seeds=(82000,),
        )


def test_confirmation_gate_checks_v2_and_fixed_terminal_events():
    fixed = _summary(13.29, 30.03, 0.108, 1.106, 0.802, 24)
    v2 = _summary(13.34, 29.97, 0.112, 0.979, 0.804, 24)
    candidate = _summary(13.31, 29.89, 0.111, 1.039, 0.800, 24)
    seed_summaries = {
        "17": {
            "fixed": v2,
            "v2": candidate,
            "deltas": {
                "mean_absolute_error_deg": -0.03,
                "p95_absolute_error_deg": -0.08,
                "loss_of_view_fraction": -0.001,
            },
        },
        "distributions": {},
    }
    versus_v2 = {
        "aggressive_motion": {
            "fixed": v2,
            "v2": candidate,
            "deltas": {
                "p95_absolute_error_deg": -0.08,
                "loss_of_view_fraction": -0.001,
            },
        }
    }
    versus_fixed = {
        "aggressive_motion": {
            "fixed": fixed,
            "v2": candidate,
            "deltas": {
                "p95_absolute_error_deg": -0.14,
                "loss_of_view_fraction": 0.003,
            },
        }
    }

    passed = _confirmation_gate(
        candidate=candidate,
        v2=v2,
        fixed=fixed,
        seed_summaries=seed_summaries,
        scenario_summaries=versus_v2,
        fixed_scenario_summaries=versus_fixed,
        protocol=VisibilityRiskProtocolConfig(),
    )

    assert passed["passed"]

    regressed_fixed = deepcopy(versus_fixed)
    regressed_fixed["aggressive_motion"]["fixed"] = _summary(
        13.29, 30.03, 0.108, 1.106, 0.802, 23
    )
    rejected = _confirmation_gate(
        candidate=candidate,
        v2=v2,
        fixed=fixed,
        seed_summaries=seed_summaries,
        scenario_summaries=versus_v2,
        fixed_scenario_summaries=regressed_fixed,
        protocol=VisibilityRiskProtocolConfig(),
    )

    assert not rejected["passed"]
    assert not rejected["per_scenario_checks"]["aggressive_motion"]
