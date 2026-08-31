import json

import pytest

from autonomous_observation_lab.gimbal_servoing.visualization import (
    _load_gru_control_result,
    _paired_performance_records,
    _performance_verification_markdown,
)


def _summary(mean_error, p95, loss, cost, variation, recovery):
    return {
        "mean_absolute_error_deg": mean_error,
        "p95_absolute_error_deg": p95,
        "loss_of_view_fraction": loss,
        "mean_control_cost": cost,
        "command_variation_per_s": variation,
        "event_weighted_mean_recovery_time_s": recovery,
    }


def _aggregate(summary):
    return {
        "mean_control_cost": summary["mean_control_cost"],
        "mean_metrics": {
            "mean_absolute_error_deg": summary["mean_absolute_error_deg"],
            "p95_absolute_error_deg": summary["p95_absolute_error_deg"],
            "loss_of_view_fraction": summary["loss_of_view_fraction"],
        },
    }


def _run(controller, summary):
    return {
        "seed": 30000,
        "scenario_index": 0,
        "scenario_name": "nominal",
        "controller": controller,
        "tracking_metrics": {
            "mean_absolute_error_deg": summary["mean_absolute_error_deg"],
            "p95_absolute_error_deg": summary["p95_absolute_error_deg"],
            "loss_of_view_fraction": summary["loss_of_view_fraction"],
        },
        "control_cost": summary["mean_control_cost"],
    }


def _control_result():
    summaries = {
        "proportional_rate": _summary(26.0, 54.0, 0.33, 2.1, 2.0, 1.4),
        "analytical_rate": _summary(23.0, 47.0, 0.27, 1.7, 2.3, 1.1),
        "gru_o2_rate": _summary(18.0, 37.0, 0.20, 1.3, 2.6, 1.6),
        "proportional_position": _summary(
            25.0, 50.0, 0.29, 2.0, 1.0, 1.3
        ),
        "analytical_position": _summary(
            19.0, 39.0, 0.21, 1.3, 1.0, 1.0
        ),
        "gru_o2_position": _summary(17.0, 36.0, 0.18, 1.1, 1.2, 1.4),
    }
    return {
        "experiment": "gimbal_gru_closed_loop_comparison_v1",
        "test_variant_count": 1,
        "summary": summaries,
        "scenario_aggregates": {
            "nominal": {
                name: _aggregate(summary)
                for name, summary in summaries.items()
            }
        },
        "paired_comparisons": {},
        "runs": [
            _run(name, summaries[name])
            for name in (
                "analytical_rate",
                "gru_o2_rate",
                "analytical_position",
                "gru_o2_position",
            )
        ],
    }


def _replication_result(control):
    summaries = control["summary"]
    replication = {}
    for mode in ("rate", "position"):
        reference = summaries[f"analytical_{mode}"]
        learned = summaries[f"gru_o2_{mode}"]
        distributions = {}
        deltas = {}
        for metric in (
            "mean_absolute_error_deg",
            "p95_absolute_error_deg",
            "loss_of_view_fraction",
            "mean_control_cost",
        ):
            value = learned[metric]
            distributions[metric] = {
                "mean": value,
                "sample_std": 0.1,
                "minimum": value - 0.1,
                "maximum": value + 0.1,
            }
            deltas[metric] = {
                "mean": value - reference[metric],
                "all_training_seeds_improve": True,
            }
        replication[mode] = {
            "analytical_reference": reference,
            "learned_metric_distribution": distributions,
            "delta_vs_analytical_distribution": deltas,
        }
    return {
        "experiment": "gimbal_gru_o2_replication_v1",
        "training_seed_results": [
            {
                "closed_loop_summary": {
                    "gru_o2_rate": summaries["gru_o2_rate"],
                    "gru_o2_position": summaries["gru_o2_position"],
                }
            }
        ],
        "replication_summary": replication,
    }


def test_performance_verification_pairs_runs_and_surfaces_weaknesses(tmp_path):
    control = _control_result()
    artifact = tmp_path / "control.json"
    artifact.write_text(json.dumps(control), encoding="utf-8")

    loaded = _load_gru_control_result(artifact)
    paired = _paired_performance_records(loaded, "rate")
    markdown = _performance_verification_markdown(
        loaded,
        _replication_result(loaded),
    )

    assert len(paired) == 1
    assert paired[0]["mean_error_delta_deg"] == pytest.approx(-5.0)
    assert paired[0]["control_cost_delta"] == pytest.approx(-0.4)
    assert "**Core synthetic gate: PASS.**" in markdown
    assert "## Rate scenario deltas" in markdown
    assert "## Weakness audit" in markdown
    assert "less smooth" in markdown


def test_performance_verification_rejects_unpaired_runs(tmp_path):
    control = _control_result()
    control["runs"] = [
        record
        for record in control["runs"]
        if record["controller"] != "gru_o2_rate"
    ]
    artifact = tmp_path / "unpaired.json"
    artifact.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(ValueError, match="unpaired analytical/O2 rate"):
        _load_gru_control_result(artifact)
