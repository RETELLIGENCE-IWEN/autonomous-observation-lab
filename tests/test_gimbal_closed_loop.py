from dataclasses import replace

import pytest

from autonomous_observation_lab.gimbal_servoing import (
    GimbalAction,
    GimbalCommandMode,
    GimbalServoEnv,
    SearchFallbackController,
    TargetStateEstimate,
)
from autonomous_observation_lab.gimbal_servoing.controllers import (
    TargetStateRateController,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    closed_loop_benchmark_suite,
    closed_loop_comparison,
    nominal_scenario,
)


@pytest.fixture(scope="module")
def comparison():
    return closed_loop_comparison(seed=31)


def test_comparison_uses_native_command_modes_and_identical_world_motion(
    comparison,
) -> None:
    runs = {run.episode.name: run for run in comparison.runs}
    assert runs["proportional_rate"].episode.config.command_mode is GimbalCommandMode.RATE
    assert (
        runs["proportional_position"].episode.config.command_mode
        is GimbalCommandMode.POSITION
    )
    assert runs["predictive_rate"].episode.config.command_mode is GimbalCommandMode.RATE
    assert (
        runs["predictive_position"].episode.config.command_mode
        is GimbalCommandMode.POSITION
    )

    reference = runs["proportional_rate"].episode.frames
    for run in comparison.runs[1:]:
        assert len(run.episode.frames) == len(reference)
        for expected, actual in zip(reference, run.episode.frames, strict=True):
            assert actual.diagnostics.time_s == pytest.approx(
                expected.diagnostics.time_s
            )
            assert actual.diagnostics.target_bearing_rad == pytest.approx(
                expected.diagnostics.target_bearing_rad
            )
            assert actual.diagnostics.body_bearing_rad == pytest.approx(
                expected.diagnostics.body_bearing_rad
            )


def test_predictive_rate_reduces_error_and_lag_with_visible_effort_tradeoff(
    comparison,
) -> None:
    metrics = {run.episode.name: run.metrics for run in comparison.runs}
    predictive = metrics["predictive_rate"]
    proportional_rate = metrics["proportional_rate"]
    proportional_position = metrics["proportional_position"]

    assert predictive.rms_error_normalized < 0.85 * proportional_rate.rms_error_normalized
    assert (
        predictive.rms_error_normalized
        < 0.85 * proportional_position.rms_error_normalized
    )
    assert predictive.tracking_lag_s < proportional_rate.tracking_lag_s
    assert predictive.tracking_lag_s < proportional_position.tracking_lag_s
    assert predictive.loss_of_view_events == 0

    assert (
        predictive.actuator_rate_rms_normalized
        > proportional_rate.actuator_rate_rms_normalized
    )
    assert (
        predictive.rate_saturation_fraction
        > proportional_rate.rate_saturation_fraction
    )

    predictive_position = metrics["predictive_position"]
    assert predictive_position.rms_error_normalized < predictive.rms_error_normalized
    assert predictive_position.rate_saturation_fraction == 0.0


def test_rate_and_position_adapters_share_the_same_target_estimate(
    comparison,
) -> None:
    runs = {run.episode.name: run for run in comparison.runs}
    rate_run = runs["predictive_rate"]
    position_run = runs["predictive_position"]

    assert rate_run.estimator_metrics is not None
    assert rate_run.estimator_metrics.valid_fraction > 0.95
    assert rate_run.estimator_metrics.bearing_rmse_deg < 2.0
    assert rate_run.estimator_metrics.rate_rmse_deg_s < 20.0
    assert rate_run.estimator_metrics.two_sigma_bearing_coverage > 0.95
    assert position_run.estimator_metrics is not None
    assert position_run.estimator_metrics.valid_fraction == pytest.approx(
        rate_run.estimator_metrics.valid_fraction
    )
    assert position_run.estimator_metrics.bearing_rmse_deg == pytest.approx(
        rate_run.estimator_metrics.bearing_rmse_deg,
        abs=1e-10,
    )
    assert position_run.estimator_metrics.rate_rmse_deg_s == pytest.approx(
        rate_run.estimator_metrics.rate_rmse_deg_s,
        abs=1e-10,
    )

    for rate_estimate, position_estimate in zip(
        rate_run.estimates,
        position_run.estimates,
        strict=True,
    ):
        assert rate_estimate.valid is position_estimate.valid
        if rate_estimate.valid:
            assert position_estimate.body_relative_bearing_rad.value == pytest.approx(
                rate_estimate.body_relative_bearing_rad.value,
                abs=1e-12,
            )
            assert position_estimate.body_relative_rate_rad_s.value == pytest.approx(
                rate_estimate.body_relative_rate_rad_s.value,
                abs=1e-12,
            )


def test_comparison_is_deterministic_by_seed(comparison) -> None:
    replay = closed_loop_comparison(seed=31)
    assert tuple(run.metrics for run in replay.runs) == tuple(
        run.metrics for run in comparison.runs
    )


def test_stress_suite_spans_configured_failure_modes() -> None:
    suite = closed_loop_benchmark_suite(seed=41)
    comparisons = {
        comparison.scenario_name: comparison
        for comparison in suite.comparisons
    }
    assert tuple(comparisons) == (
        "nominal_combined",
        "high_latency",
        "dropout_noise",
        "slow_servo",
        "aggressive_motion",
        "travel_limit_recovery",
    )

    nominal_config = comparisons["nominal_combined"].runs[0].episode.config
    latency_config = comparisons["high_latency"].runs[0].episode.config
    dropout_config = comparisons["dropout_noise"].runs[0].episode.config
    slow_config = comparisons["slow_servo"].runs[0].episode.config
    assert (
        latency_config.camera.detection_latency_s
        > nominal_config.camera.detection_latency_s
    )
    assert dropout_config.camera.miss_probability > 0.10
    assert slow_config.servo.max_rate_rad_s < nominal_config.servo.max_rate_rad_s

    recovery = comparisons["travel_limit_recovery"]
    assert all(run.metrics.loss_of_view_events >= 2 for run in recovery.runs)
    assert all(run.metrics.loss_of_view_fraction > 0.25 for run in recovery.runs)

    position_rms = {
        name: sum(
            next(
                run.metrics.rms_error_normalized
                for run in comparison.runs
                if run.episode.name == name
            )
            for comparison in suite.comparisons
        )
        for name in ("proportional_position", "predictive_position")
    }
    assert position_rms["predictive_position"] < position_rms["proportional_position"]

    for comparison in suite.comparisons:
        for run in comparison.runs:
            assert run.metrics.mean_recovery_time_s >= 0.0
            assert run.metrics.max_recovery_time_s >= 0.0
            assert (
                run.metrics.unrecovered_loss_events
                <= run.metrics.loss_of_view_events
            )


def test_search_fallback_reverses_at_configured_travel_limits() -> None:
    config = nominal_scenario().config

    class MissingEstimator:
        name = "missing"

        def __init__(self):
            self.last_estimate = TargetStateEstimate.missing(0.0)

        def reset(self):
            self.last_estimate = TargetStateEstimate.missing(0.0)

        def update(self, observation):
            self.last_estimate = TargetStateEstimate.missing(observation.time_s)
            return self.last_estimate

    delegate = TargetStateRateController(
        estimator=MissingEstimator(),
        max_rate_rad_s=config.servo.max_rate_rad_s,
    )
    controller = SearchFallbackController(
        delegate=delegate,
        servo=config.servo,
        command_mode=GimbalCommandMode.RATE,
        search_rate_normalized=0.2,
    )
    observation, _ = GimbalServoEnv(config).reset(seed=4)
    assert controller.act(observation) == GimbalAction.rate(0.2)
    at_upper_limit = replace(
        observation,
        gimbal_angle_rad=replace(
            observation.gimbal_angle_rad,
            value=config.servo.max_angle_rad,
        ),
    )
    assert controller.act(at_upper_limit) == GimbalAction.rate(-0.2)
