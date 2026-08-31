import json

import pytest

from autonomous_observation_lab.gimbal_servoing.visualization import (
    _adaptive_position_markdown,
    _load_adaptive_position_result,
)


def _summary(mean, p95, loss, variation, cost, unrecovered):
    return {
        "mean_absolute_error_deg": mean,
        "p95_absolute_error_deg": p95,
        "loss_of_view_fraction": loss,
        "command_variation_per_s": variation,
        "mean_control_cost": cost,
        "total_unrecovered_loss_events": unrecovered,
    }


def _result():
    return {
        "experiment": "gimbal_adaptive_position_v2_protocol_v1",
        "validation": {
            "variant_count_per_training_seed": 12,
            "selected_candidate": "light_smoothing_calibrated",
        },
        "test": {
            "opened": True,
            "world_seeds": [80000, 80001, 80002, 80003],
            "fixed_summary": _summary(12.68, 30.06, 0.0975, 1.138, 0.744, 8),
            "v2_summary": _summary(12.65, 29.48, 0.0972, 0.984, 0.737, 9),
            "adapter_diagnostics": {
                "mean_requested_horizon_s": 0.133,
                "mean_effective_horizon_s": 0.122,
                "mean_prediction_weight": 0.923,
            },
            "by_scenario": {
                "aggressive_motion": {
                    "deltas": {
                        "p95_absolute_error_deg": -0.15,
                        "command_variation_per_s": -0.15,
                        "loss_of_view_fraction": 0.0042,
                    }
                }
            },
            "acceptance_gate": {
                "passed": False,
                "aggregate_checks": {
                    "mean_error": True,
                    "p95_error": True,
                    "loss_of_view": True,
                    "command_variation": True,
                    "unrecovered_events": False,
                },
                "per_training_seed_core_checks": {
                    "17": True,
                    "29": True,
                    "43": True,
                },
                "per_scenario_tail_visibility_checks": {
                    "nominal": True,
                    "aggressive_motion": True,
                },
                "command_variation_reduction_fraction": 0.1357,
                "unrecovered_event_delta": 1,
            },
            "recommendation": "retain_fixed_horizon_position",
        },
        "representative_trace": {
            "scenario_name": "aggressive_motion",
            "world_seed": 80000,
            "training_seed": 17,
            "records": [
                {
                    "time_s": 0.0,
                    "target_body_bearing_deg": 1.0,
                    "fixed_gimbal_angle_deg": 0.0,
                    "v2_gimbal_angle_deg": 0.0,
                    "fixed_command_deg": 0.0,
                    "v2_shaped_command_deg": 0.0,
                    "v2_raw_target_deg": 0.0,
                    "requested_horizon_s": 0.0,
                    "effective_horizon_s": 0.0,
                    "prediction_weight": 0.0,
                    "uncertainty_ratio": 0.0,
                }
            ],
        },
    }


def test_adaptive_position_result_surfaces_fresh_gate_failure(tmp_path):
    artifact = tmp_path / "adaptive.json"
    artifact.write_text(json.dumps(_result()), encoding="utf-8")

    loaded = _load_adaptive_position_result(artifact)
    markdown = _adaptive_position_markdown(loaded)

    assert "Fresh safety gate: REJECT" in markdown
    assert "13.6%" in markdown
    assert "Unrecovered-event delta: +1" in markdown
    assert "extra terminal loss occurs under aggressive motion" in markdown


def test_adaptive_position_result_rejects_a_closed_fresh_test(tmp_path):
    result = _result()
    result["test"]["opened"] = False
    artifact = tmp_path / "closed.json"
    artifact.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="fresh test was not opened"):
        _load_adaptive_position_result(artifact)
