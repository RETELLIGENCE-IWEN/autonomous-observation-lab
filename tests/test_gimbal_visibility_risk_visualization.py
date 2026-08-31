import json

import pytest

from autonomous_observation_lab.gimbal_servoing.visualization import (
    _load_visibility_risk_result,
    _visibility_risk_markdown,
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


def _result():
    fixed = _summary(13.29, 30.03, 0.1076, 1.106, 0.802, 24)
    v2 = _summary(13.34, 29.97, 0.1116, 0.979, 0.804, 24)
    v21 = _summary(13.31, 29.89, 0.1109, 1.039, 0.800, 24)
    return {
        "experiment": "gimbal_adaptive_position_v21_protocol_v1",
        "training_seeds": [17, 29, 43],
        "development": {
            "world_seeds": list(range(81000, 81008)),
            "selected_candidate": "preview_125",
        },
        "confirmation": {
            "opened": True,
            "world_seeds": list(range(82000, 82008)),
            "fixed_summary": fixed,
            "v2_summary": v2,
            "v21_summary": v21,
            "adapter_diagnostics": {
                "visibility_guard_active_fraction": 0.426,
                "mean_horizon_boost_s": 0.035,
                "mean_effective_horizon_s": 0.152,
            },
            "by_scenario_vs_v2": {
                "aggressive_motion": {
                    "deltas": {
                        "p95_absolute_error_deg": -0.488,
                        "loss_of_view_fraction": -0.00487,
                        "command_variation_per_s": 0.169,
                    }
                }
            },
            "acceptance_gate": {
                "passed": True,
                "per_training_seed_checks": {
                    "17": True,
                    "29": True,
                    "43": True,
                },
                "per_scenario_checks": {"aggressive_motion": True},
                "vs_v2": {
                    "unrecovered_event_delta": 0,
                    "command_variation_reduction_fraction": -0.0619,
                },
                "vs_fixed": {
                    "unrecovered_event_delta": 0,
                    "command_variation_reduction_fraction": 0.0603,
                },
            },
            "recommendation": "adaptive_position_v21",
        },
        "representative_trace": {
            "world_seed": 82000,
            "training_seed": 17,
            "scenario_name": "aggressive_motion",
            "records": [
                {
                    "time_s": 0.0,
                    "target_body_bearing_deg": 1.0,
                    "v2_gimbal_angle_deg": 0.0,
                    "v21_gimbal_angle_deg": 0.0,
                    "v2_command_deg": 0.0,
                    "v21_command_deg": 0.0,
                    "v21_raw_target_deg": 0.0,
                    "visibility_risk": 0.0,
                    "predicted_fov_fraction": 0.0,
                    "horizon_boost_s": 0.0,
                    "effective_horizon_s": 0.0,
                }
            ],
        },
    }


def test_visibility_risk_markdown_reports_confirmation_tradeoff(tmp_path):
    artifact = tmp_path / "v21.json"
    artifact.write_text(json.dumps(_result()), encoding="utf-8")

    loaded = _load_visibility_risk_result(artifact)
    markdown = _visibility_risk_markdown(loaded)

    assert "Untouched confirmation gate: PASS" in markdown
    assert "preview_125" in markdown
    assert "+6.2%" in markdown
    assert "6.0%" in markdown
    assert "3/3 pass" in markdown


def test_visibility_risk_loader_rejects_closed_confirmation(tmp_path):
    result = _result()
    result["confirmation"]["opened"] = False
    artifact = tmp_path / "closed.json"
    artifact.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="confirmation block was not opened"):
        _load_visibility_risk_result(artifact)
