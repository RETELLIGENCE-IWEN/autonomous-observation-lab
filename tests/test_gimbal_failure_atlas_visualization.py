import json
from dataclasses import asdict

import pytest

from autonomous_observation_lab.gimbal_servoing.performance_atlas import (
    PerformanceContract,
)
from autonomous_observation_lab.gimbal_servoing.visualization import (
    _failure_atlas_markdown,
    _load_failure_atlas_result,
)


def _summary(mean, p95, loss, avoidable, variation, events, causes):
    return {
        "mean_absolute_error_fov_fraction": mean,
        "p95_absolute_error_fov_fraction": p95,
        "loss_of_view_fraction": loss,
        "avoidable_loss_fraction": avoidable,
        "command_variation_per_s": variation,
        "unrecovered_loss_events": events,
        "loss_event_causes": causes,
    }


def _controller(summary, checks, *, applicable=True):
    verdict = {
        "passed": checks == 10,
        "passed_check_count": checks,
        "check_count": 10,
        "checks": {},
    }
    return {
        "tracked_summary": summary,
        "contract": verdict,
        "by_scenario": {
            "aggressive_motion": {
                "summary": summary,
                "contract_applicable": applicable,
                "contract": verdict if applicable else None,
            }
        },
    }


def _result():
    return {
        "experiment": "gimbal_position_failure_atlas_v1",
        "scenario_names": ["aggressive_motion"],
        "contract": asdict(
            PerformanceContract(tracked_scenarios=("aggressive_motion",))
        ),
        "controllers": {
            "fixed_horizon": _controller(
                _summary(0.40, 0.95, 0.10, 0.10, 1.1, 2, {"forecast_error": 2}),
                5,
            ),
            "adaptive_v2": _controller(
                _summary(0.39, 0.93, 0.09, 0.09, 0.9, 2, {"forecast_error": 2}),
                6,
            ),
            "visibility_risk_v21": _controller(
                _summary(
                    0.38,
                    0.90,
                    0.08,
                    0.08,
                    1.0,
                    2,
                    {"command_timing_or_shaping": 2},
                ),
                6,
            ),
        },
        "v3_priorities": [
            {
                "rank": 1,
                "title": "Servo-aware constrained predictive control",
                "event_count": 2,
                "recommended_change": "Optimize the position trajectory.",
            }
        ],
    }


def test_failure_atlas_markdown_surfaces_absolute_failure_and_priority(tmp_path):
    artifact = tmp_path / "atlas.json"
    artifact.write_text(json.dumps(_result()), encoding="utf-8")

    loaded = _load_failure_atlas_result(artifact)
    markdown = _failure_atlas_markdown(loaded)

    assert "V2.1 absolute contract: FAIL (6/10 checks)" in markdown
    assert "8.00%" in markdown
    assert "command timing or shaping" in markdown
    assert "Servo-aware constrained predictive control" in markdown


def test_failure_atlas_loader_requires_three_controllers(tmp_path):
    result = _result()
    del result["controllers"]["fixed_horizon"]
    artifact = tmp_path / "bad.json"
    artifact.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="all three controllers"):
        _load_failure_atlas_result(artifact)
