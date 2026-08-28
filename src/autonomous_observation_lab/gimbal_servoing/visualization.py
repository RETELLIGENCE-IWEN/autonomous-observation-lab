"""Rerun dashboards for gimbal causality, control, and recovery."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .closed_loop import (
    ClosedLoopBenchmarkSuite,
    ClosedLoopComparison,
    ControllerRun,
    closed_loop_benchmark_suite,
    closed_loop_comparison,
)
from .config import GimbalCommandMode
from .demos import DemoEpisode, DemoFrame, paired_cause_demo
from .estimators import TargetStateEstimate
from .recovery import RecoveryState

if TYPE_CHECKING:
    from .recovery_evaluation import RecoveryReplay, RecoveryReplayRun


_APP_ID = "autonomous_observation_lab_gimbal_demo"
_TIMELINE = "sim_time"
_RECOVERY_STATE_CODE = {
    RecoveryState.TRACK: 0.0,
    RecoveryState.COAST: 1.0,
    RecoveryState.SEARCH: 2.0,
    RecoveryState.REACQUIRE: 3.0,
}


def _point_at(angle_rad: float, distance: float) -> list[float]:
    return [
        distance * math.cos(angle_rad),
        distance * math.sin(angle_rad),
        0.0,
    ]


def _blueprint(rrb: Any, episodes: Sequence[DemoEpisode]) -> Any:
    rows = []
    for episode in episodes:
        root = f"/{episode.name}"
        signal_views = rrb.Vertical(
            rrb.TimeSeriesView(
                origin=f"{root}/signals/angle_deg",
                name="Position loop (deg)",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/signals/rate_deg_s",
                name="Inner rate loop (deg/s)",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/signals/bbox_error",
                name="BBox center error",
            ),
            row_shares=[1, 1, 1],
        )
        rows.append(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=f"{root}/world",
                    name=f"{episode.name}: world",
                ),
                rrb.Spatial2DView(
                    origin=f"{root}/image",
                    name="Normalized image frame",
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[-1.15, 1.15],
                        y_range=[-1.15, 1.15],
                    ),
                ),
                signal_views,
                column_shares=[1.35, 1.0, 1.65],
                name=episode.description,
            )
        )
    return rrb.Blueprint(
        rrb.Vertical(*rows, row_shares=[1.0] * len(rows)),
        collapse_panels=True,
    )


def _closed_loop_blueprint(
    rrb: Any, comparison: ClosedLoopComparison
) -> Any:
    rows = []
    for run in comparison.runs:
        episode = run.episode
        root = f"/{episode.name}"
        mode_unit = (
            "deg/s"
            if episode.config.command_mode is GimbalCommandMode.RATE
            else "deg"
        )
        signal_views = rrb.Vertical(
            rrb.TimeSeriesView(
                origin=f"{root}/comparison/tracking_deg",
                name="True / estimated bearing / gimbal (deg)",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/comparison/target_rate_deg_s",
                name="True vs estimated target rate (deg/s)",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/signals/bbox_error",
                name="BBox center error",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/comparison/actuator",
                name=f"Requested / applied / actual ({mode_unit})",
            ),
            row_shares=[1, 1, 1, 1],
        )
        rows.append(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=f"{root}/world",
                    name=f"{episode.name}: world",
                ),
                rrb.Spatial2DView(
                    origin=f"{root}/image",
                    name="Normalized image frame",
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[-1.15, 1.15],
                        y_range=[-1.15, 1.15],
                    ),
                ),
                signal_views,
                column_shares=[1.25, 0.90, 1.85],
                name=episode.description,
            )
        )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(
                origin="/comparison/summary",
                name="Closed-loop benchmark summary",
            ),
            *rows,
            row_shares=[0.75] + [1.0] * len(rows),
        ),
        collapse_panels=True,
    )


def _recovery_blueprint(rrb: Any, replay: RecoveryReplay) -> Any:
    rows = []
    for replay_run in replay.runs:
        run = replay_run.run
        episode = run.episode
        root = f"/{episode.name}"
        mode_unit = (
            "deg/s"
            if episode.config.command_mode is GimbalCommandMode.RATE
            else "deg"
        )
        signals = rrb.Vertical(
            rrb.TimeSeriesView(
                origin=f"{root}/comparison/tracking_deg",
                name="True / belief / gimbal (deg)",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/recovery/visibility",
                name="Target / detector / frame update",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/recovery/state",
                name="Phase: TRACK 0, COAST 1, SEARCH 2, REACQUIRE 3",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/signals/bbox_error",
                name="BBox center error",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/comparison/actuator",
                name=f"Requested / applied / actual ({mode_unit})",
            ),
            row_shares=[1.25, 0.7, 0.7, 0.8, 1.0],
        )
        rows.append(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=f"{root}/world",
                    name=f"{episode.name}: world",
                ),
                rrb.Spatial2DView(
                    origin=f"{root}/image",
                    name="Normalized image frame",
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[-1.15, 1.15],
                        y_range=[-1.15, 1.15],
                    ),
                ),
                signals,
                column_shares=[1.15, 0.85, 2.0],
                name=episode.description,
            )
        )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(
                origin="/recovery/summary",
                name="Recovery replay summary",
            ),
            *rows,
            row_shares=[0.8] + [1.0] * len(rows),
        ),
        collapse_panels=True,
    )


def _log_static(rr: Any, episode: DemoEpisode) -> None:
    root = f"{episode.name}"
    rr.set_time(_TIMELINE, duration=0.0)
    rr.log(
        f"{root}/world/body",
        rr.Boxes3D(
            centers=[[0.0, 0.0, 0.0]],
            half_sizes=[[0.30, 0.20, 0.08]],
            colors=[[90, 100, 120]],
            labels=["quadcopter body"],
        ),
    )
    rr.log(
        f"{root}/world/body_forward",
        rr.Arrows3D(
            origins=[[0.0, 0.0, 0.10]],
            vectors=[[1.0, 0.0, 0.0]],
            colors=[[150, 150, 160]],
            labels=["0 deg / body forward"],
        ),
    )
    rr.log(
        f"{root}/image/frame",
        rr.Boxes2D(
            centers=[[0.0, 0.0]],
            half_sizes=[[1.0, 1.0]],
            colors=[[120, 130, 145]],
            labels=["camera FOV"],
        ),
    )
    rr.log(
        f"{root}/image/crosshair",
        rr.LineStrips2D(
            [
                [[-0.08, 0.0], [0.08, 0.0]],
                [[0.0, -0.08], [0.0, 0.08]],
            ],
            colors=[[130, 130, 140], [130, 130, 140]],
        ),
    )

    styles = (
        ("angle_deg/requested", [255, 170, 40], "requested position"),
        ("angle_deg/applied", [190, 100, 255], "latency-delayed position"),
        ("angle_deg/actual", [40, 190, 255], "actual gimbal position"),
        ("rate_deg_s/inner_target", [255, 170, 40], "inner-loop rate target"),
        ("rate_deg_s/actual", [40, 190, 255], "actual gimbal rate"),
        ("bbox_error/true", [50, 220, 100], "true bbox center"),
        ("bbox_error/observed", [255, 150, 40], "delayed observed bbox"),
    )
    for suffix, color, label in styles:
        rr.log(
            f"{root}/signals/{suffix}",
            rr.SeriesLines(colors=color, names=label),
            static=True,
        )


def _log_frame(rr: Any, episode: DemoEpisode, frame: DemoFrame) -> None:
    root = episode.name
    diagnostics = frame.diagnostics
    config = episode.config
    rr.set_time(_TIMELINE, duration=diagnostics.time_s)

    display_range = 4.0
    target = _point_at(diagnostics.target_bearing_rad, display_range)
    optical_end = _point_at(diagnostics.optical_axis_bearing_rad, 1.7)
    half_fov = 0.5 * config.camera.selected_axis_fov_rad
    lower_fov = _point_at(diagnostics.optical_axis_bearing_rad - half_fov, 2.4)
    upper_fov = _point_at(diagnostics.optical_axis_bearing_rad + half_fov, 2.4)

    rr.log(
        f"{root}/world/camera_axis",
        rr.Arrows3D(
            origins=[[0.0, 0.0, 0.12]],
            vectors=[[optical_end[0], optical_end[1], 0.0]],
            colors=[[40, 190, 255]],
            labels=["camera optical axis"],
        ),
    )
    rr.log(
        f"{root}/world/camera_fov",
        rr.LineStrips3D(
            [
                [[0.0, 0.0, 0.0], lower_fov],
                [[0.0, 0.0, 0.0], upper_fov],
            ],
            colors=[[60, 120, 180], [60, 120, 180]],
        ),
    )
    rr.log(
        f"{root}/world/target",
        rr.Points3D(
            positions=[target],
            colors=[[255, 210, 30]],
            radii=[0.13],
            labels=["target"],
        ),
    )
    rr.log(
        f"{root}/world/line_of_sight",
        rr.LineStrips3D(
            [[[0.0, 0.0, 0.0], target]],
            colors=[[255, 210, 30]],
        ),
    )

    true_half_width = (
        config.scenario.target_angular_width_rad
        / config.camera.selected_axis_fov_rad
    )
    true_half_height = (
        config.scenario.target_angular_height_rad
        / config.camera.orthogonal_fov_rad
    )
    rr.log(
        f"{root}/image/true_bbox",
        rr.Boxes2D(
            centers=[[diagnostics.true_image_error_normalized, 0.0]],
            half_sizes=[[true_half_width, true_half_height]],
            colors=[[50, 220, 100]],
            labels=["true bbox"],
        ),
    )
    observation = frame.observation
    if observation.detection_valid:
        rr.log(
            f"{root}/image/observed_bbox",
            rr.Boxes2D(
                centers=[[observation.image_error_normalized.value, 0.0]],
                half_sizes=[
                    [
                        0.5 * observation.bbox_width_fraction.value * 2.0,
                        0.5 * observation.bbox_height_fraction.value * 2.0,
                    ]
                ],
                colors=[[255, 150, 40]],
                labels=["delayed detector bbox"],
            ),
        )
    else:
        rr.log(f"{root}/image/observed_bbox", rr.Clear(recursive=False))

    radians_to_degrees = 180.0 / math.pi
    requested_position = diagnostics.requested_position_rad or 0.0
    applied_position = diagnostics.applied_position_command_rad or 0.0
    values = (
        ("angle_deg/requested", requested_position * radians_to_degrees),
        ("angle_deg/applied", applied_position * radians_to_degrees),
        ("angle_deg/actual", diagnostics.gimbal_angle_rad * radians_to_degrees),
        (
            "rate_deg_s/inner_target",
            diagnostics.inner_rate_target_rad_s * radians_to_degrees,
        ),
        ("rate_deg_s/actual", diagnostics.gimbal_rate_rad_s * radians_to_degrees),
        ("bbox_error/true", diagnostics.true_image_error_normalized),
    )
    for suffix, value in values:
        rr.log(f"{root}/signals/{suffix}", rr.Scalars(value))
    if observation.detection_valid:
        rr.log(
            f"{root}/signals/bbox_error/observed",
            rr.Scalars(observation.image_error_normalized.value),
        )


def _metrics_markdown(comparison: ClosedLoopComparison) -> str:
    lines = [
        f"# Closed-loop gimbal benchmark: {comparison.scenario_name}",
        "",
        comparison.description,
        "",
        "| Controller | RMS error | Mean error | P95 error | Lag | Lost view | Rate saturation | Rate effort | Accel effort |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in comparison.runs:
        metrics = run.metrics
        lines.append(
            "| "
            f"{run.episode.name} | "
            f"{metrics.rms_error_normalized:.3f} | "
            f"{metrics.mean_absolute_error_deg:.2f}° | "
            f"{metrics.p95_absolute_error_deg:.2f}° | "
            f"{1000.0 * metrics.tracking_lag_s:.0f} ms | "
            f"{100.0 * metrics.loss_of_view_fraction:.1f}% "
            f"({metrics.loss_of_view_events}) | "
            f"{100.0 * metrics.rate_saturation_fraction:.1f}% | "
            f"{metrics.actuator_rate_rms_normalized:.3f} | "
            f"{metrics.actuator_acceleration_rms_normalized:.3f} |"
        )
    lines.extend(
        (
            "",
            "All runs use identical target motion, body rotation, detector "
            "randomness, and configured hardware model.",
            "",
            "## Estimator quality",
            "",
            "| Controller | Valid | Bearing RMSE | Rate RMSE | 2σ bearing coverage |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for run in comparison.runs:
        metrics = run.estimator_metrics
        if metrics is None:
            continue
        lines.append(
            f"| {run.episode.name} | {100.0 * metrics.valid_fraction:.1f}% | "
            f"{metrics.bearing_rmse_deg:.2f}° | "
            f"{metrics.rate_rmse_deg_s:.2f}°/s | "
            f"{100.0 * metrics.two_sigma_bearing_coverage:.1f}% |"
        )
    return "\n".join(lines)


def _recovery_markdown(replay: RecoveryReplay) -> str:
    lines = [
        f"# Recovery replay: {replay.scenario.name}",
        "",
        f"Exact recorded hardware variant at seed **{replay.seed}**; "
        f"estimator **{replay.estimator_kind}**, command mode "
        f"**{replay.command_mode.value}**.",
        "",
        replay.scenario.description,
        "",
        "Phase code: `0 TRACK`, `1 COAST/hold`, `2 SEARCH`, "
        "`3 REACQUIRE`.",
        "",
        "## Selected-variant metrics",
        "",
        "| Strategy | Mean error | P95 error | Lost view | Recovery | "
        "Unrecovered | Control cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    from .gru_control import _control_cost

    for replay_run in replay.runs:
        run = replay_run.run
        metrics = run.metrics
        strategy = run.episode.name.rsplit("_", 1)[-1]
        lines.append(
            f"| {strategy} | {metrics.mean_absolute_error_deg:.2f}° | "
            f"{metrics.p95_absolute_error_deg:.2f}° | "
            f"{100.0 * metrics.loss_of_view_fraction:.1f}% | "
            f"{metrics.mean_recovery_time_s:.2f} s | "
            f"{metrics.unrecovered_loss_events} | {_control_cost(run):.3f} |"
        )
    if replay.aggregate_summary:
        lines.extend(
            (
                "",
                "## Full 24-variant aggregate",
                "",
                "| Strategy | Mean error | P95 error | Lost view | Cost | "
                "Unrecovered |",
                "|---|---:|---:|---:|---:|---:|",
            )
        )
        for replay_run in replay.runs:
            name = replay_run.run.episode.name
            aggregate = replay.aggregate_summary.get(name)
            if aggregate is None:
                continue
            strategy = name.rsplit("_", 1)[-1]
            lines.append(
                f"| {strategy} | "
                f"{aggregate['mean_absolute_error_deg']:.2f}° | "
                f"{aggregate['p95_absolute_error_deg']:.2f}° | "
                f"{100.0 * aggregate['loss_of_view_fraction']:.1f}% | "
                f"{aggregate['mean_control_cost']:.3f} | "
                f"{aggregate['total_unrecovered_loss_events']} |"
            )
    belief_run = next(
        (
            replay_run
            for replay_run in replay.runs
            if replay_run.transitions
        ),
        None,
    )
    if belief_run is not None:
        lines.extend(
            (
                "",
                "## Belief-recovery transitions",
                "",
                "| Time | From | To | Reason |",
                "|---:|---|---|---|",
            )
        )
        for transition in belief_run.transitions:
            lines.append(
                f"| {transition.time_s:.2f} s | "
                f"{transition.previous.value} | {transition.current.value} | "
                f"{transition.reason} |"
            )
    lines.extend(
        (
            "",
            "All three rows replay identical target motion, body rotation, "
            "detector randomness, initial state, and configured hardware.",
        )
    )
    return "\n".join(lines)


def _suite_metric_table(
    suite: ClosedLoopBenchmarkSuite,
    *,
    title: str,
    value_at: Any,
) -> list[str]:
    controller_names = tuple(
        run.episode.name for run in suite.comparisons[0].runs
    )
    lines = [
        f"## {title}",
        "",
        "| Scenario | " + " | ".join(controller_names) + " |",
        "|---|" + "---:|" * len(controller_names),
    ]
    for comparison in suite.comparisons:
        values = [value_at(run) for run in comparison.runs]
        lines.append(
            f"| {comparison.scenario_name} | " + " | ".join(values) + " |"
        )
    lines.append("")
    return lines


def _suite_markdown(suite: ClosedLoopBenchmarkSuite) -> str:
    lines = [
        "# Predictive gimbal stress-suite",
        "",
        suite.description,
        "",
        "Each row uses identical exogenous motion and detector randomness across "
        "controllers. Lower is better unless noted.",
        "",
    ]
    lines.extend(
        _suite_metric_table(
            suite,
            title="Normalized RMS tracking error",
            value_at=lambda run: f"{run.metrics.rms_error_normalized:.3f}",
        )
    )
    lines.extend(
        _suite_metric_table(
            suite,
            title="Estimated tracking lag",
            value_at=lambda run: f"{1000.0 * run.metrics.tracking_lag_s:.0f} ms",
        )
    )
    lines.extend(
        _suite_metric_table(
            suite,
            title="Loss of view (fraction and events)",
            value_at=lambda run: (
                f"{100.0 * run.metrics.loss_of_view_fraction:.1f}% "
                f"({run.metrics.loss_of_view_events})"
            ),
        )
    )
    lines.extend(
        _suite_metric_table(
            suite,
            title="Rate saturation fraction",
            value_at=lambda run: (
                f"{100.0 * run.metrics.rate_saturation_fraction:.1f}%"
            ),
        )
    )
    lines.extend(
        (
            "## Explicit scenario configuration",
            "",
            "| Scenario | Detector latency | Misses | Servo max rate | "
            "Servo max acceleration |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for comparison in suite.comparisons:
        config = comparison.runs[0].episode.config
        lines.append(
            f"| {comparison.scenario_name} | "
            f"{1000.0 * config.camera.detection_latency_s:.0f} ms | "
            f"{100.0 * config.camera.miss_probability:.1f}% | "
            f"{math.degrees(config.servo.max_rate_rad_s):.0f}°/s | "
            f"{math.degrees(config.servo.max_acceleration_rad_s2):.0f}°/s² |"
        )
    return "\n".join(lines)


def _log_closed_loop_static(rr: Any, run: ControllerRun) -> None:
    episode = run.episode
    root = episode.name
    _log_static(rr, episode)
    styles = (
        ("tracking_deg/desired", [255, 210, 30], "target bearing in body frame"),
        ("tracking_deg/estimated", [210, 90, 255], "estimated target bearing"),
        ("tracking_deg/lower", [110, 80, 130], "estimate - 2 sigma"),
        ("tracking_deg/upper", [110, 80, 130], "estimate + 2 sigma"),
        ("tracking_deg/actual", [40, 190, 255], "actual gimbal angle"),
        ("target_rate_deg_s/true", [255, 210, 30], "true target rate"),
        ("target_rate_deg_s/estimated", [210, 90, 255], "estimated target rate"),
        ("actuator/requested", [255, 170, 40], "requested"),
        ("actuator/applied", [190, 100, 255], "latency-delayed"),
        ("actuator/actual", [40, 190, 255], "actual"),
    )
    for suffix, color, label in styles:
        rr.log(
            f"{root}/comparison/{suffix}",
            rr.SeriesLines(colors=color, names=label),
            static=True,
        )


def _log_closed_loop_frame(
    rr: Any,
    run: ControllerRun,
    frame: DemoFrame,
    estimate: TargetStateEstimate | None,
) -> None:
    episode = run.episode
    diagnostics = frame.diagnostics
    root = episode.name
    _log_frame(rr, episode, frame)
    rr.set_time(_TIMELINE, duration=diagnostics.time_s)

    body_forward = _point_at(diagnostics.body_bearing_rad, 1.0)
    rr.log(
        f"{root}/world/body_forward",
        rr.Arrows3D(
            origins=[[0.0, 0.0, 0.10]],
            vectors=[[body_forward[0], body_forward[1], 0.0]],
            colors=[[150, 150, 160]],
            labels=["0 deg / rotating body forward"],
        ),
    )

    radians_to_degrees = 180.0 / math.pi
    desired_body_relative_angle = math.atan2(
        math.sin(diagnostics.target_bearing_rad - diagnostics.body_bearing_rad),
        math.cos(diagnostics.target_bearing_rad - diagnostics.body_bearing_rad),
    )
    rr.log(
        f"{root}/comparison/tracking_deg/desired",
        rr.Scalars(desired_body_relative_angle * radians_to_degrees),
    )
    rr.log(
        f"{root}/comparison/tracking_deg/actual",
        rr.Scalars(diagnostics.gimbal_angle_rad * radians_to_degrees),
    )
    true_target_rate = diagnostics.target_rate_rad_s - diagnostics.body_rate_rad_s
    rr.log(
        f"{root}/comparison/target_rate_deg_s/true",
        rr.Scalars(true_target_rate * radians_to_degrees),
    )
    if estimate is not None and estimate.valid:
        estimated_bearing = estimate.body_relative_bearing_rad.value
        bearing_interval = 2.0 * estimate.bearing_std_rad.value
        rr.log(
            f"{root}/comparison/tracking_deg/estimated",
            rr.Scalars(estimated_bearing * radians_to_degrees),
        )
        rr.log(
            f"{root}/comparison/tracking_deg/lower",
            rr.Scalars(
                (estimated_bearing - bearing_interval) * radians_to_degrees
            ),
        )
        rr.log(
            f"{root}/comparison/tracking_deg/upper",
            rr.Scalars(
                (estimated_bearing + bearing_interval) * radians_to_degrees
            ),
        )
        rr.log(
            f"{root}/comparison/target_rate_deg_s/estimated",
            rr.Scalars(
                estimate.body_relative_rate_rad_s.value * radians_to_degrees
            ),
        )

    if episode.config.command_mode is GimbalCommandMode.RATE:
        requested = diagnostics.requested_rate_rad_s or 0.0
        applied = diagnostics.applied_rate_command_rad_s or 0.0
        actual = diagnostics.gimbal_rate_rad_s
    else:
        requested = diagnostics.requested_position_rad or 0.0
        applied = diagnostics.applied_position_command_rad or 0.0
        actual = diagnostics.gimbal_angle_rad
    for suffix, value in (
        ("requested", requested),
        ("applied", applied),
        ("actual", actual),
    ):
        rr.log(
            f"{root}/comparison/actuator/{suffix}",
            rr.Scalars(value * radians_to_degrees),
        )


def _recovery_state_at(
    replay_run: RecoveryReplayRun, frame_index: int
) -> RecoveryState:
    if replay_run.state_trace:
        index = min(frame_index, len(replay_run.state_trace) - 1)
        return replay_run.state_trace[index][1]
    run = replay_run.run
    estimate_valid = (
        frame_index < len(run.estimates) and run.estimates[frame_index].valid
    )
    if estimate_valid:
        return RecoveryState.TRACK
    strategy = run.episode.name.rsplit("_", 1)[-1]
    return (
        RecoveryState.SEARCH
        if strategy == "blind"
        else RecoveryState.COAST
    )


def _log_recovery_static(rr: Any, replay_run: RecoveryReplayRun) -> None:
    run = replay_run.run
    root = run.episode.name
    _log_closed_loop_static(rr, run)
    styles = (
        ("visibility/target_in_view", [50, 220, 100], "target in camera FOV"),
        ("visibility/detection_valid", [255, 150, 40], "detector valid"),
        ("visibility/frame_updated", [130, 150, 255], "new detector frame"),
        (
            "state/code",
            [230, 100, 255],
            "phase: TRACK 0 / COAST 1 / SEARCH 2 / REACQUIRE 3",
        ),
    )
    for suffix, color, label in styles:
        rr.log(
            f"{root}/recovery/{suffix}",
            rr.SeriesLines(colors=color, names=label),
            static=True,
        )


def _log_recovery_frame(
    rr: Any,
    replay_run: RecoveryReplayRun,
    frame: DemoFrame,
    frame_index: int,
) -> None:
    run = replay_run.run
    estimate = (
        run.estimates[frame_index]
        if frame_index < len(run.estimates)
        else None
    )
    _log_closed_loop_frame(rr, run, frame, estimate)
    rr.set_time(_TIMELINE, duration=frame.diagnostics.time_s)
    root = run.episode.name
    observation = frame.observation
    state = _recovery_state_at(replay_run, frame_index)
    for suffix, value in (
        ("visibility/target_in_view", frame.diagnostics.target_in_view),
        ("visibility/detection_valid", observation.detection_valid),
        ("visibility/frame_updated", observation.frame_updated),
        ("state/code", _RECOVERY_STATE_CODE[state]),
    ):
        rr.log(f"{root}/recovery/{suffix}", rr.Scalars(float(value)))


def write_rerun_demo(
    episodes: Sequence[DemoEpisode],
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write an RRD recording or open the interactive viewer.

    File and interactive sinks are intentionally exclusive here so the CLI has
    simple, predictable lifetime semantics.
    """
    if not episodes:
        raise ValueError("at least one demo episode is required")
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _blueprint(rrb, episodes)
    rr.init(_APP_ID)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)

    for episode in episodes:
        _log_static(rr, episode)
    for frames in zip(*(episode.frames for episode in episodes), strict=True):
        for episode, frame in zip(episodes, frames, strict=True):
            _log_frame(rr, episode, frame)


def write_closed_loop_comparison(
    comparison: ClosedLoopComparison,
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or interactively show the synchronized controller benchmark."""
    if not comparison.runs:
        raise ValueError("at least one controller run is required")
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _closed_loop_blueprint(rrb, comparison)
    rr.init(f"{_APP_ID}_closed_loop_estimator_v2")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)

    rr.set_time(_TIMELINE, duration=0.0)
    rr.log(
        "comparison/summary",
        rr.TextDocument(
            _metrics_markdown(comparison),
            media_type=rr.MediaType.MARKDOWN,
        ),
    )
    for run in comparison.runs:
        _log_closed_loop_static(rr, run)
    frame_sequences = tuple(run.episode.frames for run in comparison.runs)
    for frame_index, frames in enumerate(zip(*frame_sequences, strict=True)):
        for run, frame in zip(comparison.runs, frames, strict=True):
            estimate = (
                run.estimates[frame_index]
                if frame_index < len(run.estimates)
                else None
            )
            _log_closed_loop_frame(rr, run, frame, estimate)


def write_recovery_replay(
    replay: RecoveryReplay,
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show an exact hold/blind/belief recovery replay."""
    if not replay.runs:
        raise ValueError("at least one recovery run is required")
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _recovery_blueprint(rrb, replay)
    rr.init(f"{_APP_ID}_belief_recovery_v1")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.set_time(_TIMELINE, duration=0.0)
    rr.log(
        "recovery/summary",
        rr.TextDocument(
            _recovery_markdown(replay),
            media_type=rr.MediaType.MARKDOWN,
        ),
    )
    for replay_run in replay.runs:
        _log_recovery_static(rr, replay_run)
    frame_sequences = tuple(
        replay_run.run.episode.frames for replay_run in replay.runs
    )
    for frame_index, frames in enumerate(
        zip(*frame_sequences, strict=True)
    ):
        for replay_run, frame in zip(replay.runs, frames, strict=True):
            _log_recovery_frame(rr, replay_run, frame, frame_index)


def write_benchmark_suite(
    suite: ClosedLoopBenchmarkSuite,
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show the complete stress-suite metric matrix."""
    if not suite.comparisons:
        raise ValueError("at least one scenario comparison is required")
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = rrb.Blueprint(
        rrb.TextDocumentView(
            origin="/benchmark_suite/summary",
            name="Predictive gimbal stress-suite",
        ),
        collapse_panels=True,
    )
    rr.init(f"{_APP_ID}_benchmark_suite")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "benchmark_suite/summary",
        rr.TextDocument(
            _suite_markdown(suite),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize gimbal causality and closed-loop control experiments."
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help="write a Rerun .rrd recording instead of opening a viewer",
    )
    destination.add_argument(
        "--spawn",
        action="store_true",
        help="open the interactive Rerun viewer (the default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="deterministic replay seed (defaults depend on the selected demo)",
    )
    parser.add_argument(
        "--demo",
        choices=("causality", "closed-loop", "benchmark-suite", "recovery"),
        default="causality",
        help="select a causality, controller, stress, or recovery dashboard",
    )
    parser.add_argument(
        "--recovery-results",
        type=Path,
        default=Path("artifacts/gimbal_belief_recovery_comparison.json"),
        help="recorded belief-recovery JSON used for exact replay",
    )
    parser.add_argument(
        "--recovery-scenario",
        choices=(
            "detector_burst_recovery",
            "travel_limit_reentry",
            "physically_unreachable",
        ),
        default="detector_burst_recovery",
        help="recorded recovery scenario to replay",
    )
    parser.add_argument(
        "--recovery-estimator",
        choices=("analytical", "gru_o2"),
        default="gru_o2",
        help="target-state estimator for hold/blind/belief replay",
    )
    parser.add_argument(
        "--recovery-command-mode",
        choices=("rate", "position"),
        default="rate",
        help="hardware command adapter for recovery replay",
    )
    parser.add_argument(
        "--o2-checkpoint",
        type=Path,
        help="override the O2 checkpoint recorded in recovery results",
    )
    parser.add_argument(
        "--control-results",
        type=Path,
        help="override the prior control result recorded in recovery results",
    )
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    spawn = args.spawn or args.output is None
    if args.demo == "recovery":
        from .recovery_evaluation import replay_recovery_variant

        seed = 41000 if args.seed is None else args.seed
        command_mode = (
            GimbalCommandMode.RATE
            if args.recovery_command_mode == "rate"
            else GimbalCommandMode.POSITION
        )
        replay = replay_recovery_variant(
            results=args.recovery_results,
            scenario_name=args.recovery_scenario,
            seed=seed,
            estimator_kind=args.recovery_estimator,
            command_mode=command_mode,
            o2_checkpoint=args.o2_checkpoint,
            control_results=args.control_results,
            device=args.device,
        )
        write_recovery_replay(
            replay,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "benchmark-suite":
        seed = 41 if args.seed is None else args.seed
        suite = closed_loop_benchmark_suite(seed=seed)
        write_benchmark_suite(
            suite,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "closed-loop":
        seed = 31 if args.seed is None else args.seed
        comparison = closed_loop_comparison(seed=seed)
        write_closed_loop_comparison(
            comparison,
            output=args.output,
            spawn=spawn,
        )
    else:
        seed = 21 if args.seed is None else args.seed
        episodes = paired_cause_demo(seed=seed)
        write_rerun_demo(
            episodes,
            output=args.output,
            spawn=spawn,
        )


if __name__ == "__main__":
    main()
