import json

from autonomous_observation_lab.gimbal_servoing.visualization import (
    _load_predictive_position_result,
    _load_predictive_position_v31_result,
    _predictive_position_markdown,
)


def _summary(*, mean, p95, loss, avoidable, variation, acceleration, events):
    return {
        "mean_absolute_error_deg": mean,
        "p95_absolute_error_deg": p95,
        "loss_of_view_fraction": loss,
        "avoidable_loss_fraction": avoidable,
        "command_variation_per_s": variation,
        "actuator_acceleration_rms_normalized": acceleration,
        "unrecovered_loss_events": events,
    }


def test_predictive_position_dashboard_reports_rejection_and_sealed_v31(tmp_path):
    reference = _summary(
        mean=8.867,
        p95=20.707,
        loss=0.02342,
        avoidable=0.01379,
        variation=1.135,
        acceleration=0.612,
        events=4,
    )
    candidate = _summary(
        mean=9.007,
        p95=21.123,
        loss=0.02134,
        avoidable=0.01171,
        variation=1.020,
        acceleration=0.591,
        events=4,
    )
    v3 = {
        "experiment": "gimbal_constrained_predictive_position_v3_protocol_v1",
        "development": {"world_seeds": [83000, 83007]},
        "confirmation": {
            "opened": True,
            "world_seeds": [84000, 84007],
            "selected_candidate": "capacity_smooth",
            "recommendation": "retain_visibility_risk_v21",
            "visibility_risk_v21": {"tracked_summary": reference},
            "predictive_position_v3": {"tracked_summary": candidate},
            "acceptance_gate": {
                "passed": False,
                "checks": {"mean_error": False, "p95_error": False},
                "comparison": {
                    "avoidable_loss_reduction_fraction": 0.151,
                    "command_variation_reduction_fraction": 0.101,
                },
            },
            "diagnostics": {"optimizer_active_fraction": 0.197},
        },
        "representative_trace": {"records": []},
    }
    v31 = {
        "experiment": "gimbal_dual_risk_predictive_position_v31_protocol_v1",
        "development": {
            "eligible_candidate_count": 0,
            "candidates": [
                {
                    "name": "dual_risk_early",
                    "gate": {
                        "comparison": {
                            "avoidable_loss_reduction_fraction": -0.0235,
                            "command_variation_reduction_fraction": 0.0343,
                        }
                    },
                    "diagnostics": {"optimizer_active_fraction": 0.0875},
                }
            ],
        },
    }
    v3_path = tmp_path / "v3.json"
    v31_path = tmp_path / "v31.json"
    v3_path.write_text(json.dumps(v3), encoding="utf-8")
    v31_path.write_text(json.dumps(v31), encoding="utf-8")

    markdown = _predictive_position_markdown(
        _load_predictive_position_result(v3_path),
        _load_predictive_position_v31_result(v31_path),
    )

    assert "confirmation gate: REJECT" in markdown
    assert "15.1%" in markdown
    assert "V3.1 development" in markdown
    assert "confirmation block remains unopened" in markdown
