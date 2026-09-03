"""Rerun dashboards for gimbal causality, control, and recovery."""

from __future__ import annotations

import argparse
import json
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
    from .controller_arena import ControllerArena
    from .recovery_evaluation import RecoveryReplay, RecoveryReplayRun


_APP_ID = "autonomous_observation_lab_gimbal_demo"
_TIMELINE = "sim_time"
_CALIBRATION_TIMELINE = "calibration_axis"
_REPLICATION_TIMELINE = "training_seed_index"
_PERFORMANCE_TIMELINE = "comparison_index"
_ADAPTIVE_TIMELINE = "adaptive_comparison_index"
_VISIBILITY_RISK_TIMELINE = "visibility_risk_comparison_index"
_FAILURE_ATLAS_TIMELINE = "failure_atlas_index"
_PREDICTIVE_POSITION_TIMELINE = "predictive_position_comparison_index"
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


def _controller_arena_blueprint(rrb: Any, arena: ControllerArena) -> Any:
    controller_views = []
    for run in arena.comparison.runs:
        root = f"/{run.episode.name}"
        controller_views.append(
            rrb.Vertical(
                rrb.TextDocumentView(
                    origin=f"{root}/hud",
                    name="Live controller HUD",
                ),
                rrb.Spatial3DView(
                    origin=f"{root}/world",
                    name=f"{run.episode.description}: 3D gimbal",
                ),
                rrb.Spatial2DView(
                    origin=f"{root}/image",
                    name="Camera frame and bounding boxes",
                    visual_bounds=rrb.VisualBounds2D(
                        x_range=[-1.15, 1.15],
                        y_range=[-1.15, 1.15],
                    ),
                ),
                row_shares=[0.75, 1.15, 0.85],
            )
        )
    if arena.kind == "challenge":
        diagnostics = rrb.Horizontal(
            rrb.TimeSeriesView(
                origin="/arena/signals/fov_margin",
                name="FOV margin: positive is visible",
            ),
            rrb.TimeSeriesView(
                origin="/arena/signals/body_rate_deg_s",
                name="Shared body yaw rate",
            ),
            rrb.TimeSeriesView(
                origin="/arena/dream/risk_trust",
                name="Dream-to-Center risk and forecast trust",
            ),
            rrb.TimeSeriesView(
                origin="/arena/dream/horizon_ms",
                name="Dream-to-Center prediction horizon",
            ),
        )
    else:
        diagnostics = rrb.Horizontal(
            rrb.TimeSeriesView(
                origin="/arena/v21/risk",
                name="V2.1 visibility risk and predicted FOV fraction",
            ),
            rrb.TimeSeriesView(
                origin="/arena/v21/horizon_ms",
                name="V2.1 horizon boost and effective horizon",
            ),
        )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(
                origin="/arena/summary",
                name=(
                    "Predictive Gimbal Challenge Arena"
                    if arena.kind == "challenge"
                    else "Controller arena instructions and metrics"
                ),
            ),
            rrb.Horizontal(
                *controller_views,
                column_shares=[1.0] * len(controller_views),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="/arena/signals/tracking_deg",
                    name="Target and gimbal angles",
                ),
                rrb.TimeSeriesView(
                    origin="/arena/signals/error_deg",
                    name="Absolute tracking error",
                ),
                rrb.TimeSeriesView(
                    origin="/arena/signals/command_deg",
                    name="Requested position commands",
                ),
                rrb.TimeSeriesView(
                    origin="/arena/signals/in_view",
                    name="Target visible: 1 yes, 0 no",
                ),
            ),
            diagnostics,
            row_shares=[0.45, 2.65, 0.95, 0.70],
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
                name="Target / detector / frame update / edge evidence",
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


def _uncertainty_calibration_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(
                origin="/calibration/summary",
                name="Uncertainty calibration summary",
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/calibration/test/bearing/reliability",
                        name="Test bearing reliability (50–99% grid)",
                    ),
                    rrb.TimeSeriesView(
                        origin="/calibration/test/rate/reliability",
                        name="Test rate reliability (50–99% grid)",
                    ),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/calibration/test/bearing/two_sigma_by_horizon",
                        name="Test bearing 2σ (0/100/200/300 ms)",
                    ),
                    rrb.TimeSeriesView(
                        origin="/calibration/test/rate/two_sigma_by_horizon",
                        name="Test rate 2σ (0/100/200/300 ms)",
                    ),
                ),
                row_shares=[1.0, 1.0],
            ),
            column_shares=[1.15, 1.85],
        ),
        collapse_panels=True,
    )


def _gru_replication_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(
                origin="/replication/summary",
                name="O2 multi-seed replication summary",
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/replication/rate/mean_error_deg",
                        name="Rate control: mean error by training seed",
                    ),
                    rrb.TimeSeriesView(
                        origin="/replication/rate/loss_of_view_percent",
                        name="Rate control: loss of view by training seed",
                    ),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/replication/position/mean_error_deg",
                        name="Position control: mean error by training seed",
                    ),
                    rrb.TimeSeriesView(
                        origin="/replication/position/loss_of_view_percent",
                        name="Position control: loss of view by training seed",
                    ),
                ),
                row_shares=[1.0, 1.0],
            ),
            column_shares=[1.25, 1.75],
        ),
        collapse_panels=True,
    )


def _performance_verification_blueprint(rrb: Any) -> Any:
    def scenario_views(mode: str, label: str) -> Any:
        root = f"/performance/scenario/{mode}"
        return rrb.Horizontal(
            rrb.TimeSeriesView(
                origin=f"{root}/mean_error_deg",
                name=f"{label}: mean error by scenario",
            ),
            rrb.TimeSeriesView(
                origin=f"{root}/loss_of_view_percent",
                name=f"{label}: loss of view by scenario",
            ),
        )

    def paired_views(mode: str, label: str) -> Any:
        root = f"/performance/paired/{mode}"
        return rrb.Vertical(
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin=f"{root}/mean_error_delta_deg",
                    name=f"{label}: paired mean-error delta",
                ),
                rrb.TimeSeriesView(
                    origin=f"{root}/p95_error_delta_deg",
                    name=f"{label}: paired P95-error delta",
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin=f"{root}/loss_of_view_delta_percent",
                    name=f"{label}: paired lost-view delta (pp)",
                ),
                rrb.TimeSeriesView(
                    origin=f"{root}/control_cost_delta",
                    name=f"{label}: paired control-cost delta",
                ),
            ),
            row_shares=[1.0, 1.0],
        )

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(
                origin="/performance/summary",
                name="Baseline vs learned verification",
            ),
            rrb.Vertical(
                scenario_views("rate", "Rate command"),
                scenario_views("position", "Position command"),
                paired_views("rate", "Rate command; below zero is better"),
                paired_views(
                    "position",
                    "Position command; below zero is better",
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/performance/stability/rate/mean_error_deg",
                        name="Rate: mean error by training seed",
                    ),
                    rrb.TimeSeriesView(
                        origin=(
                            "/performance/stability/position/mean_error_deg"
                        ),
                        name="Position: mean error by training seed",
                    ),
                ),
                row_shares=[1.0, 1.0, 2.0, 2.0, 1.0],
            ),
            column_shares=[1.25, 1.75],
        ),
        collapse_panels=True,
    )


def _adaptive_position_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(
                origin="/adaptive_position/summary",
                name="Adaptive position V2 safety-gate result",
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/adaptive_position/aggregate/p95_error_deg",
                        name="Fresh aggregate P95 error",
                    ),
                    rrb.TimeSeriesView(
                        origin="/adaptive_position/aggregate/command_variation",
                        name="Fresh aggregate command variation",
                    ),
                    rrb.TimeSeriesView(
                        origin="/adaptive_position/aggregate/loss_of_view_percent",
                        name="Fresh aggregate loss of view",
                    ),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/adaptive_position/scenario/p95_delta_deg",
                        name="V2 − fixed P95 by scenario",
                    ),
                    rrb.TimeSeriesView(
                        origin="/adaptive_position/scenario/variation_delta",
                        name="V2 − fixed variation by scenario",
                    ),
                    rrb.TimeSeriesView(
                        origin="/adaptive_position/scenario/loss_delta_percent",
                        name="V2 − fixed lost-view points",
                    ),
                ),
                rrb.TimeSeriesView(
                    origin="/adaptive_position/trace/tracking_deg",
                    name="Representative target and fixed/V2 gimbal angles",
                ),
                rrb.TimeSeriesView(
                    origin="/adaptive_position/trace/command_deg",
                    name="Fixed, raw adaptive, and shaped position commands",
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/adaptive_position/trace/horizon_s",
                        name="Requested vs uncertainty-adjusted horizon",
                    ),
                    rrb.TimeSeriesView(
                        origin="/adaptive_position/trace/trust",
                        name="Prediction weight and uncertainty ratio",
                    ),
                ),
                row_shares=[0.8, 0.8, 1.1, 1.1, 0.9],
            ),
            column_shares=[1.15, 1.85],
        ),
        collapse_panels=True,
    )


def _visibility_risk_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(
                origin="/visibility_risk/summary",
                name="Visibility-risk position V2.1 confirmation",
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/aggregate/p95_error_deg",
                        name="Confirmation P95 error",
                    ),
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/aggregate/command_variation",
                        name="Confirmation command variation",
                    ),
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/aggregate/loss_of_view_percent",
                        name="Confirmation loss of view",
                    ),
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/aggregate/unrecovered_events",
                        name="Unrecovered events",
                    ),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/scenario/p95_delta_deg",
                        name="V2.1 − V2 P95 by scenario",
                    ),
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/scenario/variation_delta",
                        name="V2.1 − V2 variation by scenario",
                    ),
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/scenario/loss_delta_percent",
                        name="V2.1 − V2 lost-view points",
                    ),
                ),
                rrb.TimeSeriesView(
                    origin="/visibility_risk/trace/tracking_deg",
                    name="Aggressive-motion target and V2/V2.1 gimbal angles",
                ),
                rrb.TimeSeriesView(
                    origin="/visibility_risk/trace/command_deg",
                    name="V2, risk-preview, and raw V2.1 commands",
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/trace/risk",
                        name="Visibility risk and predicted FOV fraction",
                    ),
                    rrb.TimeSeriesView(
                        origin="/visibility_risk/trace/horizon_s",
                        name="Risk boost and effective horizon",
                    ),
                ),
                row_shares=[0.8, 0.8, 1.1, 1.1, 0.9],
            ),
            column_shares=[1.15, 1.85],
        ),
        collapse_panels=True,
    )


def _failure_atlas_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(
                origin="/failure_atlas/summary",
                name="Absolute contract, failure causes, and V3 priorities",
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/failure_atlas/scenario/mean_error_fov_percent",
                        name="Mean error (% camera half-FOV)",
                    ),
                    rrb.TimeSeriesView(
                        origin="/failure_atlas/scenario/p95_error_fov_percent",
                        name="P95 error (% camera half-FOV)",
                    ),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/failure_atlas/scenario/loss_percent",
                        name="Total loss of view",
                    ),
                    rrb.TimeSeriesView(
                        origin="/failure_atlas/scenario/avoidable_loss_percent",
                        name="Loss while mechanically reachable",
                    ),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/failure_atlas/scenario/command_variation",
                        name="Position-command variation per second",
                    ),
                    rrb.TimeSeriesView(
                        origin="/failure_atlas/scenario/contract_checks",
                        name="Absolute checks passed (of 10)",
                    ),
                ),
                rrb.TimeSeriesView(
                    origin="/failure_atlas/causes/events",
                    name="Loss-event attribution by cause index",
                ),
                row_shares=[1.0, 1.0, 1.0, 0.8],
            ),
            column_shares=[1.15, 1.85],
        ),
        collapse_panels=True,
    )


def _predictive_position_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.TextDocumentView(
                origin="/predictive_position/summary",
                name="Constrained predictive position V3/V3.1 verdict",
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/predictive_position/aggregate/mean_error_deg",
                        name="Confirmation mean error",
                    ),
                    rrb.TimeSeriesView(
                        origin="/predictive_position/aggregate/p95_error_deg",
                        name="Confirmation P95 error",
                    ),
                    rrb.TimeSeriesView(
                        origin=(
                            "/predictive_position/aggregate/"
                            "avoidable_loss_percent"
                        ),
                        name="Mechanically avoidable loss",
                    ),
                    rrb.TimeSeriesView(
                        origin=(
                            "/predictive_position/aggregate/command_variation"
                        ),
                        name="Position-command variation",
                    ),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin=(
                            "/predictive_position/scenario/mean_delta_deg"
                        ),
                        name="V3 − V2.1 mean error",
                    ),
                    rrb.TimeSeriesView(
                        origin=(
                            "/predictive_position/scenario/p95_delta_deg"
                        ),
                        name="V3 − V2.1 P95 error",
                    ),
                    rrb.TimeSeriesView(
                        origin=(
                            "/predictive_position/scenario/"
                            "avoidable_loss_delta_percent"
                        ),
                        name="V3 − V2.1 avoidable-loss points",
                    ),
                    rrb.TimeSeriesView(
                        origin=(
                            "/predictive_position/scenario/variation_delta"
                        ),
                        name="V3 − V2.1 command variation",
                    ),
                ),
                rrb.TimeSeriesView(
                    origin="/predictive_position/trace/tracking_deg",
                    name="Representative target and V2.1/V3 gimbal angles",
                ),
                rrb.TimeSeriesView(
                    origin="/predictive_position/trace/command_deg",
                    name="V2.1, V3, and fallback position commands",
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        origin="/predictive_position/trace/activation",
                        name="Optimizer activation and score",
                    ),
                    rrb.TimeSeriesView(
                        origin="/predictive_position/trace/prediction",
                        name="Predicted terminal error / camera half-FOV",
                    ),
                ),
                row_shares=[0.8, 0.8, 1.1, 1.1, 0.9],
            ),
            column_shares=[1.15, 1.85],
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


def _arena_controller_label(name: str) -> str:
    return {
        "arena_fixed_horizon": "Fixed horizon",
        "arena_adaptive_v2": "Adaptive V2",
        "arena_visibility_risk_v21": "Risk V2.1",
        "challenge_reactive_position": "Reactive position",
        "challenge_classical_predictive": "Classical predictive",
        "challenge_dream_to_center": "Dream-to-Center",
    }.get(name, name)


def _controller_arena_markdown(arena: ControllerArena) -> str:
    challenge = arena.kind == "challenge"
    lines = [
        (
            "# Predictive Gimbal Challenge Arena"
            if challenge
            else "# Three-controller visual arena"
        ),
        "",
        f"Scenario **{arena.scenario_name}**, world seed "
        f"**{arena.world_seed}**, GRU training seed "
        f"**{arena.training_seed}**. Deployable V2.1 candidate: "
        f"**{arena.selected_v21_candidate}**.",
        "",
        "Press Play or drag the shared `sim_time` cursor. Every column uses "
        "the same target motion, body rotation, detector randomness, camera, "
        "servo, and initial state. Green boxes are ground truth; orange boxes "
        "are delayed detector observations.",
        "",
        "| Controller | Mean error | P95 | Lost view | Variation/s | "
        "Unrecovered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in arena.comparison.runs:
        metrics = run.metrics
        lines.append(
            f"| {_arena_controller_label(run.episode.name)} | "
            f"{metrics.mean_absolute_error_deg:.2f}° | "
            f"{metrics.p95_absolute_error_deg:.2f}° | "
            f"{100.0 * metrics.loss_of_view_fraction:.2f}% | "
            f"{metrics.command_variation_per_s:.3f} | "
            f"{metrics.unrecovered_loss_events} |"
        )
    lines.extend(
        (
            "",
            (
                "Color key: **orange reactive**, **blue classical "
                "predictive**, **green Dream-to-Center**, and **yellow "
                "target**. Cyan/purple boxes are causal +100/+200/+300 ms "
                "forecast ghosts. The privileged V16 authority oracle is "
                "deliberately excluded."
                if challenge
                else "Color key in the shared plots: **blue fixed horizon**, "
                "**magenta adaptive V2**, **green risk V2.1**, and "
                "**yellow target**."
            ),
        )
    )
    return "\n".join(lines)


def _forecast_bearing_at(
    forecasts: tuple[TargetStateEstimate, ...],
    *,
    frame_time_s: float,
    horizon_s: float,
) -> float | None:
    valid = sorted(
        (estimate for estimate in forecasts if estimate.valid),
        key=lambda estimate: estimate.time_s,
    )
    if not valid:
        return None
    target_time_s = frame_time_s + horizon_s
    if len(valid) == 1:
        estimate = valid[0]
        delta_s = target_time_s - estimate.time_s
        return math.atan2(
            math.sin(
                estimate.body_relative_bearing_rad.value
                + delta_s * estimate.body_relative_rate_rad_s.value
            ),
            math.cos(
                estimate.body_relative_bearing_rad.value
                + delta_s * estimate.body_relative_rate_rad_s.value
            ),
        )
    if target_time_s <= valid[0].time_s:
        return valid[0].body_relative_bearing_rad.value
    if target_time_s >= valid[-1].time_s:
        return valid[-1].body_relative_bearing_rad.value
    for left, right in zip(valid, valid[1:]):
        if left.time_s <= target_time_s <= right.time_s:
            fraction = (target_time_s - left.time_s) / (
                right.time_s - left.time_s
            )
            delta = math.atan2(
                math.sin(
                    right.body_relative_bearing_rad.value
                    - left.body_relative_bearing_rad.value
                ),
                math.cos(
                    right.body_relative_bearing_rad.value
                    - left.body_relative_bearing_rad.value
                ),
            )
            bearing = left.body_relative_bearing_rad.value + fraction * delta
            return math.atan2(math.sin(bearing), math.cos(bearing))
    return None


def _forecast_image_center(
    run: ControllerRun,
    frame: DemoFrame,
    frame_index: int,
    horizon_s: float,
) -> float | None:
    if frame_index >= len(run.forecasts):
        return None
    bearing = _forecast_bearing_at(
        run.forecasts[frame_index],
        frame_time_s=frame.diagnostics.time_s,
        horizon_s=horizon_s,
    )
    if bearing is None:
        return None
    image_error = math.atan2(
        math.sin(bearing - frame.diagnostics.gimbal_angle_rad),
        math.cos(bearing - frame.diagnostics.gimbal_angle_rad),
    ) / (0.5 * run.episode.config.camera.selected_axis_fov_rad)
    return float(max(-1.12, min(1.12, image_error)))


def _arena_hud_markdown(
    arena: ControllerArena,
    run: ControllerRun,
    frame: DemoFrame,
    frame_index: int,
) -> str:
    diagnostics = frame.diagnostics
    observation = frame.observation
    margin = 1.0 - abs(diagnostics.true_image_error_normalized)
    status = "✅ TARGET VISIBLE" if diagnostics.target_in_view else "🚨 TARGET LOST"
    age_ms = (
        1000.0 * observation.measurement_age_s.value
        if observation.measurement_age_s.valid
        else math.nan
    )
    estimate = (
        run.estimates[frame_index]
        if frame_index < len(run.estimates)
        else None
    )
    horizon_ms = (
        1000.0 * max(0.0, estimate.time_s - diagnostics.time_s)
        if estimate is not None and estimate.valid
        else 0.0
    )
    adapter = (
        run.adapter_diagnostics[frame_index]
        if frame_index < len(run.adapter_diagnostics)
        else {}
    )
    if run.episode.name == "challenge_dream_to_center":
        horizon_ms = 1000.0 * float(
            adapter.get("effective_horizon_s", horizon_ms / 1000.0)
        )
    trust = float(adapter.get("prediction_weight", 0.0))
    risk = float(adapter.get("visibility_risk", 0.0))
    forecast_text = (
        f"{horizon_ms:.0f} ms | trust {100.0 * trust:.0f}% | risk "
        f"{100.0 * risk:.0f}%"
        if run.episode.name == "challenge_dream_to_center"
        else (f"{horizon_ms:.0f} ms" if horizon_ms > 0.0 else "none")
    )
    age_text = f"{age_ms:.0f} ms" if math.isfinite(age_ms) else "unavailable"
    lines = [
        f"## {_arena_controller_label(run.episode.name)}",
        "",
        f"### {status}",
        "",
        f"FOV margin: **{100.0 * margin:+.1f}%**  ",
        f"Body yaw rate: **{math.degrees(diagnostics.body_rate_rad_s):+.1f}°/s**  ",
        f"Measurement age: **{age_text}**  ",
        "Servo limit: "
        f"**{math.degrees(run.episode.config.servo.max_rate_rad_s):.1f}°/s**  ",
        f"Forecast: **{forecast_text}**",
    ]
    if arena.kind == "challenge" and (
        run.episode.name == "challenge_dream_to_center"
    ):
        lines.extend(("", "V16 privileged authority: **not shown / not deployed**"))
    return "\n".join(lines)


def _log_arena_forecast_ghosts(
    rr: Any,
    arena: ControllerArena,
    run: ControllerRun,
    frame: DemoFrame,
    frame_index: int,
) -> None:
    colors = ([80, 225, 255], [80, 165, 255], [165, 105, 255])
    config = run.episode.config
    half_size = [
        config.scenario.target_angular_width_rad
        / config.camera.selected_axis_fov_rad,
        config.scenario.target_angular_height_rad
        / config.camera.orthogonal_fov_rad,
    ]
    for index, horizon_s in enumerate(arena.ghost_horizons_s):
        path = (
            f"{run.episode.name}/image/forecast_"
            f"{int(round(1000.0 * horizon_s))}_ms"
        )
        center = _forecast_image_center(run, frame, frame_index, horizon_s)
        if center is None:
            rr.log(path, rr.Clear(recursive=False))
            continue
        rr.log(
            path,
            rr.Boxes2D(
                centers=[[center, 0.0]],
                half_sizes=[half_size],
                colors=[colors[min(index, len(colors) - 1)]],
                labels=[f"+{1000.0 * horizon_s:.0f} ms"],
            ),
        )


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
                f"## Full {replay.variant_count}-variant aggregate",
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


def _uncertainty_calibration_markdown(result: dict[str, Any]) -> str:
    if result["experiment"] == "gimbal_contextual_uncertainty_calibration_v1":
        return _contextual_uncertainty_calibration_markdown(result)
    calibration = result["calibration"]
    horizons = calibration["prediction_horizons_s"]
    bearing_scales = calibration["bearing_std_scale"]
    rate_scales = calibration["rate_std_scale"]
    lines = [
        "# O2 GRU uncertainty calibration",
        "",
        "Scale factors were fit **only on validation predictions**, frozen, "
        "and then evaluated on the untouched test split. Predicted bearing "
        "and rate means are unchanged.",
        "",
        f"Method: `{result['method']}`",
        "",
        "## Learned standard-deviation scales",
        "",
        "| Horizon | Bearing | Rate |",
        "|---:|---:|---:|",
    ]
    for horizon, bearing, rate in zip(
        horizons, bearing_scales, rate_scales, strict=True
    ):
        lines.append(
            f"| {1000.0 * horizon:.0f} ms | {bearing:.3f} | {rate:.3f} |"
        )

    lines.extend(
        (
            "",
            "## Aggregate results",
            "",
            "| Split / signal | NLL b | NLL a | 1σ b | 1σ a | "
            "2σ b | 2σ a | MACE b | MACE a |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for split_name in ("validation", "test"):
        split = result[split_name]
        for signal in ("bearing", "rate"):
            before = split["uncalibrated"]
            after = split["calibrated"]
            reliability = split["reliability"]
            lines.append(
                f"| {split_name} {signal} | "
                f"{before[f'{signal}_nll']:.4f} | "
                f"{after[f'{signal}_nll']:.4f} | "
                f"{100.0 * before[f'{signal}_one_sigma_coverage']:.2f}% | "
                f"{100.0 * after[f'{signal}_one_sigma_coverage']:.2f}% | "
                f"{100.0 * before[f'{signal}_two_sigma_coverage']:.2f}% | "
                f"{100.0 * after[f'{signal}_two_sigma_coverage']:.2f}% | "
                f"{100.0 * reliability['uncalibrated'][signal]['mean_absolute_calibration_error']:.2f}% | "
                f"{100.0 * reliability['calibrated'][signal]['mean_absolute_calibration_error']:.2f}% |"
            )

    lines.extend(
        (
            "",
            "## Test regimes most relevant to recovery",
            "",
            "| Regime | Samples | Bearing 2σ before → after | "
            "Rate 2σ before → after |",
            "|---|---:|---:|---:|",
        )
    )
    for name in (
        "fresh_valid_detection",
        "detection_gap_150_to_650ms",
        "detection_gap_ge_650ms",
        "target_out_of_view",
    ):
        record = result["test"]["strata"].get(name)
        if record is None:
            continue
        before = record["uncalibrated"]
        after = record["calibrated"]
        lines.append(
            f"| `{name}` | {before['valid_samples']} | "
            f"{100.0 * before['bearing_two_sigma_coverage']:.2f}% → "
            f"{100.0 * after['bearing_two_sigma_coverage']:.2f}% | "
            f"{100.0 * before['rate_two_sigma_coverage']:.2f}% → "
            f"{100.0 * after['rate_two_sigma_coverage']:.2f}% |"
        )
    lines.extend(
        (
            "",
            "The scaling improves held-out Gaussian likelihood and 2σ tail "
            "coverage. The full reliability-curve error does not improve: "
            "residuals are non-Gaussian and their calibration depends on the "
            "measurement/dropout regime. This artifact is therefore an "
            "optional variance correction, not a claim of complete "
            "distribution calibration.",
        )
    )
    return "\n".join(lines)


def _contextual_uncertainty_calibration_markdown(
    result: dict[str, Any],
) -> str:
    calibration = result["calibration"]
    lines = [
        "# Context-aware O2 uncertainty calibration",
        "",
        "Eight deployable measurement regimes use shrinkage toward the "
        "global per-horizon scale. The fit uses only validation predictions; "
        "the test split is evaluated once.",
        "",
        "## Context scale ranges across horizons",
        "",
        "| Context | Bearing scale | Rate scale | Validation labels |",
        "|---|---:|---:|---:|",
    ]
    counts = result["context_fit_sample_count_per_horizon"]
    for name, bearing, rate in zip(
        calibration["context_names"],
        calibration["bearing_std_scale_by_context"],
        calibration["rate_std_scale_by_context"],
        strict=True,
    ):
        lines.append(
            f"| `{name}` | {min(bearing):.3f}–{max(bearing):.3f} | "
            f"{min(rate):.3f}–{max(rate):.3f} | {sum(counts[name])} |"
        )
    lines.extend(
        (
            "",
            "## Aggregate comparison",
            "",
            "| Split / method | Bearing NLL | Rate NLL | Bearing 2σ | "
            "Rate 2σ | Bearing MACE | Rate MACE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for split_name in ("validation", "test"):
        split = result[split_name]
        for variant in ("uncalibrated", "global", "contextual"):
            metrics = split[variant]
            reliability = split["reliability"][variant]
            lines.append(
                f"| {split_name} {variant} | "
                f"{metrics['bearing_nll']:.4f} | {metrics['rate_nll']:.4f} | "
                f"{100.0 * metrics['bearing_two_sigma_coverage']:.2f}% | "
                f"{100.0 * metrics['rate_two_sigma_coverage']:.2f}% | "
                f"{100.0 * reliability['bearing']['mean_absolute_calibration_error']:.2f}% | "
                f"{100.0 * reliability['rate']['mean_absolute_calibration_error']:.2f}% |"
            )
    lines.extend(
        (
            "",
            "## Decision",
            "",
            "The contextual table improves validation NLL, but fails the "
            "held-out generalization gate. On test, bearing NLL worsens from "
            "-1.3813 uncalibrated / -1.3823 global to -1.3561 contextual, "
            "and bearing 2σ coverage falls to 93.88%. The contextual artifact "
            "is retained as a negative experiment and is **not selected for "
            "recovery deployment**.",
        )
    )
    return "\n".join(lines)


def _load_uncertainty_calibration_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("experiment") not in {
        "gimbal_uncertainty_calibration_v1",
        "gimbal_contextual_uncertainty_calibration_v1",
    }:
        raise ValueError("unsupported uncertainty calibration result")
    for key in ("calibration", "validation", "test"):
        if not isinstance(result.get(key), dict):
            raise ValueError(f"calibration result is missing {key}")
    return result


def _gru_replication_markdown(result: dict[str, Any]) -> str:
    seeds = result["training_seed_results"]
    summary = result["replication_summary"]
    lines = [
        "# O2 GRU multi-seed replication",
        "",
        f"{len(seeds)} independently initialized O2 models were trained on "
        "the same frozen train split. Each model selected rate and position "
        "horizons on validation, then ran once on the same paired test "
        "variants. Training seed—not test episode—is the replication unit. "
        "No best training seed was selected.",
        "",
        "## Rate control",
        "",
        "| Plot index | Training seed | Best epoch | Horizon | Mean error | "
        "P95 error | Loss of view | Cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, seed_result in enumerate(seeds):
        metrics = seed_result["closed_loop_summary"]["gru_o2_rate"]
        horizon = seed_result["selected_horizons"]["rate"]["horizon_s"]
        lines.append(
            f"| {index} | {seed_result['training_seed']} | "
            f"{seed_result['best_epoch']} | {1000.0 * horizon:.0f} ms | "
            f"{metrics['mean_absolute_error_deg']:.2f}° | "
            f"{metrics['p95_absolute_error_deg']:.2f}° | "
            f"{100.0 * metrics['loss_of_view_fraction']:.2f}% | "
            f"{metrics['mean_control_cost']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Position control",
            "",
            "| Plot index | Training seed | Best epoch | Horizon | Mean error | "
            "P95 error | Loss of view | Cost |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for index, seed_result in enumerate(seeds):
        metrics = seed_result["closed_loop_summary"]["gru_o2_position"]
        horizon = seed_result["selected_horizons"]["position"]["horizon_s"]
        lines.append(
            f"| {index} | {seed_result['training_seed']} | "
            f"{seed_result['best_epoch']} | {1000.0 * horizon:.0f} ms | "
            f"{metrics['mean_absolute_error_deg']:.2f}° | "
            f"{metrics['p95_absolute_error_deg']:.2f}° | "
            f"{100.0 * metrics['loss_of_view_fraction']:.2f}% | "
            f"{metrics['mean_control_cost']:.3f} |"
        )

    lines.extend(
        (
            "",
            "## Aggregate replication result",
            "",
            "| Mode / metric | Analytical | O2 mean ± seed SD | "
            "O2 range | Mean delta | Every seed better? |",
            "|---|---:|---:|---:|---:|:---:|",
        )
    )
    metric_formats = (
        ("mean_absolute_error_deg", "mean error", 1.0, "°"),
        ("p95_absolute_error_deg", "P95 error", 1.0, "°"),
        ("loss_of_view_fraction", "loss of view", 100.0, "%"),
        ("mean_control_cost", "control cost", 1.0, ""),
    )
    for mode in ("rate", "position"):
        mode_summary = summary[mode]
        for metric, label, scale, unit in metric_formats:
            reference = scale * mode_summary["analytical_reference"][metric]
            learned = mode_summary["learned_metric_distribution"][metric]
            delta = mode_summary["delta_vs_analytical_distribution"][metric]
            lines.append(
                f"| {mode} {label} | {reference:.3f}{unit} | "
                f"{scale * learned['mean']:.3f} ± "
                f"{scale * learned['sample_std']:.3f}{unit} | "
                f"{scale * learned['minimum']:.3f}–"
                f"{scale * learned['maximum']:.3f}{unit} | "
                f"{scale * delta['mean']:+.3f}{unit} | "
                f"{'yes' if delta['all_training_seeds_improve'] else 'no'} |"
            )

    rate_horizons = summary["rate"][
        "selected_horizon_s_by_training_seed"
    ]
    position_horizons = summary["position"][
        "selected_horizon_s_by_training_seed"
    ]
    core_metrics = (
        "mean_absolute_error_deg",
        "p95_absolute_error_deg",
        "loss_of_view_fraction",
        "mean_control_cost",
    )
    core_replicated = all(
        summary[mode]["delta_vs_analytical_distribution"][metric][
            "all_training_seeds_improve"
        ]
        for mode in ("rate", "position")
        for metric in core_metrics
    )
    if core_replicated:
        core_interpretation = (
            "The core claim replicates: every initialization improves mean "
            "error, P95 error, loss-of-view time, and control cost over the "
            "analytical controller in both command modes."
        )
    else:
        core_interpretation = (
            "The full core claim does not replicate across every "
            "initialization; inspect the metric table before selecting a "
            "controller."
        )
    rate_horizon_interpretation = (
        f"Rate horizon selection is stable at "
        f"{1000.0 * rate_horizons[0]:.0f} ms."
        if summary["rate"]["selected_horizon_consistent"]
        else "Rate horizon selection is initialization-sensitive "
        f"({', '.join(f'{1000.0 * value:.0f} ms' for value in rate_horizons)})."
    )
    position_horizon_interpretation = (
        f"Position horizon selection is stable at "
        f"{1000.0 * position_horizons[0]:.0f} ms."
        if summary["position"]["selected_horizon_consistent"]
        else "Position horizon selection is initialization-sensitive "
        f"({', '.join(f'{1000.0 * value:.0f} ms' for value in position_horizons)}), "
        "so a particular predictive position horizon is not yet a stable "
        "finding."
    )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            f"{core_interpretation} {rate_horizon_interpretation} "
            f"{position_horizon_interpretation} Command smoothness and rate "
            "saturation remain tradeoffs unless their rows also improve "
            "across every seed.",
        )
    )
    return "\n".join(lines)


def _load_gru_replication_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("experiment") != "gimbal_gru_o2_replication_v1":
        raise ValueError("unsupported GRU replication result")
    seed_results = result.get("training_seed_results")
    if not isinstance(seed_results, list) or not seed_results:
        raise ValueError("GRU replication result has no training seeds")
    summary = result.get("replication_summary")
    if not isinstance(summary, dict) or not all(
        isinstance(summary.get(mode), dict) for mode in ("rate", "position")
    ):
        raise ValueError("GRU replication result is missing mode summaries")
    return result


def _load_adaptive_position_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("experiment") != "gimbal_adaptive_position_v2_protocol_v1":
        raise ValueError("unsupported adaptive-position V2 result")
    validation = result.get("validation")
    test = result.get("test")
    if not isinstance(validation, dict) or not isinstance(test, dict):
        raise ValueError("adaptive-position result is missing protocol splits")
    if not test.get("opened", False):
        raise ValueError("adaptive-position fresh test was not opened")
    trace = result.get("representative_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("records"), list):
        raise ValueError("adaptive-position result has no representative trace")
    return result


def _adaptive_position_markdown(result: dict[str, Any]) -> str:
    validation = result["validation"]
    test = result["test"]
    fixed = test["fixed_summary"]
    candidate = test["v2_summary"]
    gate = test["acceptance_gate"]
    selected = validation["selected_candidate"]
    training_seed_count = len(
        result.get("training_seeds", gate["per_training_seed_core_checks"])
    )
    verdict = "PASS" if gate["passed"] else "REJECT"
    recommendation = test["recommendation"].replace("_", " ")
    lines = [
        "# Adaptive predictive position V2",
        "",
        f"**Fresh safety gate: {verdict}.** Recommendation: **{recommendation}**.",
        "",
        f"Candidate `{selected}` was selected across {training_seed_count} GRU "
        f"initializations on {validation['variant_count_per_training_seed']} "
        "recorded validation worlds per seed. Only then was the disjoint fresh "
        f"block opened: seeds {', '.join(str(seed) for seed in test['world_seeds'])}.",
        "",
        "## Fresh aggregate result",
        "",
        "| Metric | Fixed horizon | Adaptive V2 | V2 − fixed |",
        "|---|---:|---:|---:|",
    ]
    specifications = (
        ("mean_absolute_error_deg", "Mean error", 1.0, "°"),
        ("p95_absolute_error_deg", "P95 error", 1.0, "°"),
        ("loss_of_view_fraction", "Loss of view", 100.0, "%"),
        ("command_variation_per_s", "Command variation/s", 1.0, ""),
        ("mean_control_cost", "Control cost", 1.0, ""),
    )
    for metric, label, scale, unit in specifications:
        fixed_value = scale * fixed[metric]
        candidate_value = scale * candidate[metric]
        lines.append(
            f"| {label} | {fixed_value:.3f}{unit} | "
            f"{candidate_value:.3f}{unit} | "
            f"{candidate_value - fixed_value:+.3f}{unit} |"
        )
    lines.extend(
        (
            "",
            "## Gate audit",
            "",
            f"- Command variation falls by "
            f"{100.0 * gate['command_variation_reduction_fraction']:.1f}%.",
            f"- Unrecovered-event delta: {gate['unrecovered_event_delta']:+d}.",
            f"- Seed-level core checks: "
            f"{sum(gate['per_training_seed_core_checks'].values())}/"
            f"{len(gate['per_training_seed_core_checks'])} pass.",
            f"- Scenario tail/visibility checks: "
            f"{sum(gate['per_scenario_tail_visibility_checks'].values())}/"
            f"{len(gate['per_scenario_tail_visibility_checks'])} pass.",
            "",
        )
    )
    failed = [
        name
        for name, passed in gate["aggregate_checks"].items()
        if not passed
    ]
    if failed:
        lines.extend(
            (
                "The aggregate gate fails only on: "
                + ", ".join(name.replace("_", " ") for name in failed)
                + ". The extra terminal loss occurs under aggressive motion, "
                "so the fixed-horizon controller remains the accepted default. "
                "No threshold was relaxed after observing the fresh block.",
                "",
            )
        )
    lines.extend(
        (
            "## Scenario deltas",
            "",
            "| Index | Scenario | P95 | Lost view | Command variation |",
            "|---:|---|---:|---:|---:|",
        )
    )
    for index, (name, scenario) in enumerate(test["by_scenario"].items()):
        deltas = scenario["deltas"]
        lines.append(
            f"| {index} | {name.replace('_', ' ')} | "
            f"{deltas['p95_absolute_error_deg']:+.3f}° | "
            f"{100.0 * deltas['loss_of_view_fraction']:+.3f} pp | "
            f"{deltas['command_variation_per_s']:+.3f}/s |"
        )
    lines.append("")
    diagnostics = test["adapter_diagnostics"]
    trace = result["representative_trace"]
    lines.extend(
        (
            "## Adapter behavior",
            "",
            f"Mean requested horizon: "
            f"{1000.0 * diagnostics['mean_requested_horizon_s']:.0f} ms; "
            f"effective horizon: "
            f"{1000.0 * diagnostics['mean_effective_horizon_s']:.0f} ms; "
            f"prediction weight: {diagnostics['mean_prediction_weight']:.3f}.",
            "",
            f"The trace panels replay `{trace['scenario_name']}` at world seed "
            f"{trace['world_seed']} with GRU training seed "
            f"{trace['training_seed']}. They expose the raw target, shaped "
            "command, adaptive horizon, and uncertainty trust directly.",
        )
    )
    return "\n".join(lines)


def _load_visibility_risk_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("experiment") != "gimbal_adaptive_position_v21_protocol_v1":
        raise ValueError("unsupported visibility-risk V2.1 result")
    development = result.get("development")
    confirmation = result.get("confirmation")
    if not isinstance(development, dict) or not isinstance(confirmation, dict):
        raise ValueError("visibility-risk result is missing protocol blocks")
    if not confirmation.get("opened", False):
        raise ValueError("visibility-risk confirmation block was not opened")
    trace = result.get("representative_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("records"), list):
        raise ValueError("visibility-risk result has no representative trace")
    return result


def _load_predictive_position_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("experiment") != (
        "gimbal_constrained_predictive_position_v3_protocol_v1"
    ):
        raise ValueError("unsupported constrained predictive-position result")
    confirmation = result.get("confirmation")
    if not isinstance(confirmation, dict) or not confirmation.get("opened"):
        raise ValueError("predictive-position confirmation was not opened")
    trace = result.get("representative_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("records"), list):
        raise ValueError("predictive-position result has no representative trace")
    return result


def _load_predictive_position_v31_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("experiment") != (
        "gimbal_dual_risk_predictive_position_v31_protocol_v1"
    ):
        raise ValueError("unsupported dual-risk V3.1 result")
    if not isinstance(result.get("development"), dict):
        raise ValueError("dual-risk V3.1 result has no development block")
    return result


def _predictive_position_markdown(
    result: dict[str, Any],
    v31_result: dict[str, Any] | None,
) -> str:
    confirmation = result["confirmation"]
    reference = confirmation["visibility_risk_v21"]["tracked_summary"]
    candidate = confirmation["predictive_position_v3"]["tracked_summary"]
    gate = confirmation["acceptance_gate"]
    comparison = gate["comparison"]
    lines = [
        "# Constrained predictive position V3",
        "",
        f"**Untouched confirmation gate: "
        f"{'PASS' if gate['passed'] else 'REJECT'}.** Recommendation: "
        f"**{confirmation['recommendation'].replace('_', ' ')}**.",
        "",
        f"Candidate `{confirmation['selected_candidate']}` was selected on "
        f"world seeds {result['development']['world_seeds'][0]}–"
        f"{result['development']['world_seeds'][-1]}, then evaluated once on "
        f"seeds {confirmation['world_seeds'][0]}–"
        f"{confirmation['world_seeds'][-1]}. No threshold was relaxed after "
        "confirmation.",
        "",
        "## Confirmation versus accepted V2.1",
        "",
        "| Metric | V2.1 | V3 | V3 − V2.1 |",
        "|---|---:|---:|---:|",
    ]
    for key, label, scale, unit in (
        ("mean_absolute_error_deg", "Mean error", 1.0, "°"),
        ("p95_absolute_error_deg", "P95 error", 1.0, "°"),
        ("loss_of_view_fraction", "Loss of view", 100.0, "%"),
        ("avoidable_loss_fraction", "Avoidable loss", 100.0, "%"),
        ("command_variation_per_s", "Command variation/s", 1.0, ""),
        (
            "actuator_acceleration_rms_normalized",
            "Actuator acceleration RMS",
            1.0,
            "",
        ),
        ("unrecovered_loss_events", "Unrecovered events", 1.0, ""),
    ):
        left = scale * reference[key]
        right = scale * candidate[key]
        lines.append(
            f"| {label} | {left:.3f}{unit} | {right:.3f}{unit} | "
            f"{right - left:+.3f}{unit} |"
        )
    failed = [name for name, passed in gate["checks"].items() if not passed]
    lines.extend(
        (
            "",
            "## Gate audit",
            "",
            f"- Avoidable loss reduction: "
            f"{100.0 * comparison['avoidable_loss_reduction_fraction']:.1f}%.",
            f"- Command-variation reduction: "
            f"{100.0 * comparison['command_variation_reduction_fraction']:.1f}%.",
            f"- Failed aggregate checks: "
            f"{', '.join(name.replace('_', ' ') for name in failed)}.",
            f"- Optimizer active on "
            f"{100.0 * confirmation['diagnostics']['optimizer_active_fraction']:.1f}% "
            "of valid steps.",
            "",
            "V3 improves aggressive-motion visibility and smoothness, but "
            "regresses ordinary tracking. It therefore remains a rejected "
            "research controller; V2.1 stays the deployment candidate.",
        )
    )
    if v31_result is not None:
        development = v31_result["development"]
        lines.extend(
            (
                "",
                "## Corrective V3.1 development",
                "",
                "V3.1 required both predicted rate-capacity risk and "
                "visibility risk before optimization. No candidate passed "
                f"development ({development['eligible_candidate_count']}/"
                f"{len(development['candidates'])}); the fresh 86k "
                "confirmation block remains unopened.",
                "",
                "| Candidate | Avoidable-loss change | Variation reduction | Active |",
                "|---|---:|---:|---:|",
            )
        )
        for item in development["candidates"]:
            comparison = item["gate"]["comparison"]
            lines.append(
                f"| {item['name']} | "
                f"{-100.0 * comparison['avoidable_loss_reduction_fraction']:+.2f}% | "
                f"{100.0 * comparison['command_variation_reduction_fraction']:.2f}% | "
                f"{100.0 * item['diagnostics']['optimizer_active_fraction']:.2f}% |"
            )
    return "\n".join(lines)


def _visibility_risk_markdown(result: dict[str, Any]) -> str:
    development = result["development"]
    confirmation = result["confirmation"]
    fixed = confirmation["fixed_summary"]
    v2 = confirmation["v2_summary"]
    v21 = confirmation["v21_summary"]
    gate = confirmation["acceptance_gate"]
    verdict = "PASS" if gate["passed"] else "REJECT"
    lines = [
        "# Visibility-risk position V2.1",
        "",
        f"**Untouched confirmation gate: {verdict}.** Recommendation: "
        f"**{confirmation['recommendation'].replace('_', ' ')}**.",
        "",
        f"`{development['selected_candidate']}` was selected on world seeds "
        f"{development['world_seeds'][0]}–{development['world_seeds'][-1]}. "
        f"The frozen candidate was then evaluated once on seeds "
        f"{confirmation['world_seeds'][0]}–{confirmation['world_seeds'][-1]} "
        "across all three GRU initializations and six scenario families.",
        "",
        "## Confirmation aggregate",
        "",
        "| Metric | Fixed horizon | Adaptive V2 | Risk V2.1 |",
        "|---|---:|---:|---:|",
    ]
    for key, label, scale, unit in (
        ("mean_absolute_error_deg", "Mean error", 1.0, "°"),
        ("p95_absolute_error_deg", "P95 error", 1.0, "°"),
        ("loss_of_view_fraction", "Loss of view", 100.0, "%"),
        ("command_variation_per_s", "Command variation/s", 1.0, ""),
        ("mean_control_cost", "Control cost", 1.0, ""),
        ("total_unrecovered_loss_events", "Unrecovered events", 1.0, ""),
    ):
        lines.append(
            f"| {label} | {scale * fixed[key]:.3f}{unit} | "
            f"{scale * v2[key]:.3f}{unit} | {scale * v21[key]:.3f}{unit} |"
        )
    vs_v2 = gate["vs_v2"]
    vs_fixed = gate["vs_fixed"]
    lines.extend(
        (
            "",
            "## Gate audit",
            "",
            f"- V2.1 changes command variation by "
            f"{-100.0 * vs_v2['command_variation_reduction_fraction']:+.1f}% "
            "versus V2 and remains "
            f"{100.0 * vs_fixed['command_variation_reduction_fraction']:.1f}% "
            "below fixed horizon.",
            f"- Unrecovered-event deltas: "
            f"{vs_v2['unrecovered_event_delta']:+d} versus V2 and "
            f"{vs_fixed['unrecovered_event_delta']:+d} versus fixed.",
            f"- Training-seed checks: "
            f"{sum(gate['per_training_seed_checks'].values())}/"
            f"{len(gate['per_training_seed_checks'])} pass.",
            f"- Scenario checks: {sum(gate['per_scenario_checks'].values())}/"
            f"{len(gate['per_scenario_checks'])} pass.",
            "",
            "V2.1 recovers some of V2's deliberate smoothing only when the "
            "predicted target approaches the camera boundary. Against V2 it "
            "slightly improves mean error, P95, loss of view, and control cost; "
            "against fixed horizon it preserves the declared smoothness margin.",
            "",
            "## Scenario deltas versus V2",
            "",
            "| Index | Scenario | P95 | Lost view | Command variation |",
            "|---:|---|---:|---:|---:|",
        )
    )
    for index, (name, scenario) in enumerate(
        confirmation["by_scenario_vs_v2"].items()
    ):
        deltas = scenario["deltas"]
        lines.append(
            f"| {index} | {name.replace('_', ' ')} | "
            f"{deltas['p95_absolute_error_deg']:+.3f}° | "
            f"{100.0 * deltas['loss_of_view_fraction']:+.3f} pp | "
            f"{deltas['command_variation_per_s']:+.3f}/s |"
        )
    diagnostics = confirmation["adapter_diagnostics"]
    trace = result["representative_trace"]
    lines.extend(
        (
            "",
            "## Guard behavior",
            "",
            f"The guard is active on "
            f"{100.0 * diagnostics['visibility_guard_active_fraction']:.1f}% "
            f"of valid steps. Mean risk boost is "
            f"{1000.0 * diagnostics['mean_horizon_boost_s']:.0f} ms and mean "
            f"effective horizon is "
            f"{1000.0 * diagnostics['mean_effective_horizon_s']:.0f} ms.",
            "",
            f"The trace replays `{trace['scenario_name']}` at world seed "
            f"{trace['world_seed']} with GRU seed {trace['training_seed']}.",
        )
    )
    return "\n".join(lines)


def _load_failure_atlas_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("experiment") != "gimbal_position_failure_atlas_v1":
        raise ValueError("unsupported position failure-atlas result")
    controllers = result.get("controllers")
    if not isinstance(controllers, dict) or set(controllers) != {
        "fixed_horizon",
        "adaptive_v2",
        "visibility_risk_v21",
    }:
        raise ValueError("failure atlas must contain all three controllers")
    if not isinstance(result.get("v3_priorities"), list):
        raise ValueError("failure atlas has no V3 priorities")
    return result


def _failure_atlas_markdown(result: dict[str, Any]) -> str:
    controllers = result["controllers"]
    contract = result["contract"]
    v21 = controllers["visibility_risk_v21"]
    verdict = v21["contract"]
    lines = [
        "# Position-controller failure atlas",
        "",
        f"**V2.1 absolute contract: {'PASS' if verdict['passed'] else 'FAIL'} "
        f"({verdict['passed_check_count']}/{verdict['check_count']} checks).**",
        "",
        f"Contract: `{contract['name']}`. Error is normalized by each "
        "randomized camera's half-FOV; plant activity is normalized by each "
        "randomized servo's limits. These are provisional research targets, "
        "not hardware acceptance claims.",
        "",
        "## Reachable-scenario aggregate",
        "",
        "| Controller | Mean / half-FOV | P95 / half-FOV | Lost view | "
        "Avoidable loss | Variation/s | Unrecovered | Checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "fixed_horizon": "Fixed learned horizon",
        "adaptive_v2": "Adaptive V2",
        "visibility_risk_v21": "Risk V2.1",
    }
    for name in labels:
        item = controllers[name]
        summary = item["tracked_summary"]
        gate = item["contract"]
        lines.append(
            f"| {labels[name]} | "
            f"{100.0 * summary['mean_absolute_error_fov_fraction']:.1f}% | "
            f"{100.0 * summary['p95_absolute_error_fov_fraction']:.1f}% | "
            f"{100.0 * summary['loss_of_view_fraction']:.2f}% | "
            f"{100.0 * summary['avoidable_loss_fraction']:.2f}% | "
            f"{summary['command_variation_per_s']:.3f} | "
            f"{summary['unrecovered_loss_events']} | "
            f"{gate['passed_check_count']}/{gate['check_count']} |"
        )
    lines.extend(
        (
            "",
            "The `travel_limit_recovery` family remains in the atlas but is "
            "excluded from the reachable tracking contract. Its loss is split "
            "into physical-envelope time and avoidable controller time.",
            "",
            "## V2.1 by scenario",
            "",
            "| Index | Scenario | Mean / half-FOV | P95 / half-FOV | "
            "Lost | Avoidable | Contract |",
            "|---:|---|---:|---:|---:|---:|---:|",
        )
    )
    for index, (name, item) in enumerate(v21["by_scenario"].items()):
        summary = item["summary"]
        if item["contract_applicable"]:
            gate = item["contract"]
            gate_text = (
                "PASS"
                if gate["passed"]
                else f"{gate['passed_check_count']}/{gate['check_count']}"
            )
        else:
            gate_text = "physical-limit audit"
        lines.append(
            f"| {index} | {name.replace('_', ' ')} | "
            f"{100.0 * summary['mean_absolute_error_fov_fraction']:.1f}% | "
            f"{100.0 * summary['p95_absolute_error_fov_fraction']:.1f}% | "
            f"{100.0 * summary['loss_of_view_fraction']:.2f}% | "
            f"{100.0 * summary['avoidable_loss_fraction']:.2f}% | "
            f"{gate_text} |"
        )
    causes = v21["tracked_summary"]["loss_event_causes"]
    lines.extend(
        (
            "",
            "## Reachable loss-event attribution",
            "",
            "| Cause index | Cause | Events |",
            "|---:|---|---:|",
        )
    )
    for index, (cause, count) in enumerate(causes.items()):
        lines.append(f"| {index} | {cause.replace('_', ' ')} | {count} |")
    if not causes:
        lines.append("| 0 | no loss events | 0 |")
    lines.extend(("", "## V3 priorities", ""))
    for priority in result["v3_priorities"]:
        lines.append(
            f"{priority['rank']}. **{priority['title']}** — "
            f"{priority['event_count']} attributed events. "
            f"{priority['recommended_change']}"
        )
    lines.extend(
        (
            "",
            "An event that reaches the episode boundary is marked terminally "
            "censored: the controller did not recover within the observed "
            "episode, but recovery after the cutoff is unknown.",
        )
    )
    return "\n".join(lines)


def _paired_performance_records(
    result: dict[str, Any], mode: str
) -> list[dict[str, Any]]:
    baseline_name = f"analytical_{mode}"
    learned_name = f"gru_o2_{mode}"
    records = result.get("runs")
    if not isinstance(records, list):
        raise ValueError("GRU control result is missing paired runs")

    def indexed(controller: str) -> dict[tuple[int, int], dict[str, Any]]:
        selected = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("controller") == controller
        ]
        values = {
            (int(record["seed"]), int(record["scenario_index"])): record
            for record in selected
        }
        if len(values) != len(selected):
            raise ValueError(f"duplicate paired runs for {controller}")
        return values

    baseline = indexed(baseline_name)
    learned = indexed(learned_name)
    if not baseline or baseline.keys() != learned.keys():
        raise ValueError(f"unpaired analytical/O2 {mode} runs")

    paired = []
    for seed, scenario_index in sorted(baseline):
        reference = baseline[(seed, scenario_index)]
        candidate = learned[(seed, scenario_index)]
        if reference["scenario_name"] != candidate["scenario_name"]:
            raise ValueError("paired run scenario names do not match")
        reference_metrics = reference["tracking_metrics"]
        candidate_metrics = candidate["tracking_metrics"]
        paired.append(
            {
                "seed": seed,
                "scenario_index": scenario_index,
                "scenario_name": reference["scenario_name"],
                "mean_error_delta_deg": (
                    candidate_metrics["mean_absolute_error_deg"]
                    - reference_metrics["mean_absolute_error_deg"]
                ),
                "p95_error_delta_deg": (
                    candidate_metrics["p95_absolute_error_deg"]
                    - reference_metrics["p95_absolute_error_deg"]
                ),
                "loss_of_view_delta_percent": 100.0
                * (
                    candidate_metrics["loss_of_view_fraction"]
                    - reference_metrics["loss_of_view_fraction"]
                ),
                "control_cost_delta": (
                    candidate["control_cost"] - reference["control_cost"]
                ),
            }
        )
    return paired


def _load_gru_control_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("experiment") != "gimbal_gru_closed_loop_comparison_v1":
        raise ValueError("unsupported GRU control result")
    for key in ("summary", "scenario_aggregates", "paired_comparisons"):
        if not isinstance(result.get(key), dict):
            raise ValueError(f"GRU control result is missing {key}")
    for mode in ("rate", "position"):
        for controller in (
            f"proportional_{mode}",
            f"analytical_{mode}",
            f"gru_o2_{mode}",
        ):
            if controller not in result["summary"]:
                raise ValueError(
                    f"GRU control result is missing {controller}"
                )
        _paired_performance_records(result, mode)
    return result


def _performance_verification_markdown(
    control_result: dict[str, Any],
    replication_result: dict[str, Any],
) -> str:
    replication = replication_result["replication_summary"]
    core_metrics = (
        "mean_absolute_error_deg",
        "p95_absolute_error_deg",
        "loss_of_view_fraction",
        "mean_control_cost",
    )
    core_replicated = all(
        replication[mode]["delta_vs_analytical_distribution"][metric][
            "all_training_seeds_improve"
        ]
        for mode in ("rate", "position")
        for metric in core_metrics
    )
    lines = [
        "# Baseline vs learned performance verification",
        "",
        "**Core synthetic gate: "
        + ("PASS.**" if core_replicated else "FAIL.**")
        + " The primary baseline is the analytical constant-velocity target "
        "state controller. O2 is the disturbance-aware causal GRU with the "
        "same configured rate or position adapter.",
        "",
        f"The scenario and paired plots use the frozen "
        f"{control_result['test_variant_count']}-variant test artifact. "
        f"Training stability uses "
        f"{len(replication_result['training_seed_results'])} independent "
        "GRU initializations. Lower is better in every plot; paired deltas "
        "below zero favor O2.",
        "",
        "## Replicated primary result",
        "",
        "| Mode / metric | Analytical | O2 mean ± seed SD | Mean delta | "
        "Every seed better? |",
        "|---|---:|---:|---:|:---:|",
    ]
    metric_formats = (
        ("mean_absolute_error_deg", "mean error", 1.0, "°", "°"),
        ("p95_absolute_error_deg", "P95 error", 1.0, "°", "°"),
        ("loss_of_view_fraction", "loss of view", 100.0, "%", " pp"),
        ("mean_control_cost", "control cost", 1.0, "", ""),
    )
    for mode in ("rate", "position"):
        mode_summary = replication[mode]
        for metric, label, scale, unit, delta_unit in metric_formats:
            reference = scale * mode_summary["analytical_reference"][metric]
            learned = mode_summary["learned_metric_distribution"][metric]
            delta = mode_summary["delta_vs_analytical_distribution"][metric]
            lines.append(
                f"| {mode} {label} | {reference:.2f}{unit} | "
                f"{scale * learned['mean']:.2f} ± "
                f"{scale * learned['sample_std']:.2f}{unit} | "
                f"{scale * delta['mean']:+.2f}{delta_unit} | "
                f"{'yes' if delta['all_training_seeds_improve'] else 'no'} |"
            )

    lines.extend(
        (
            "",
            "## Frozen test aggregate, including secondary proportional baseline",
            "",
            "| Mode | Controller | Mean error | P95 | Lost view | Cost | "
            "Command variation/s |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for mode in ("rate", "position"):
        for controller, label in (
            (f"proportional_{mode}", "proportional"),
            (f"analytical_{mode}", "analytical"),
            (f"gru_o2_{mode}", "O2 GRU"),
        ):
            summary = control_result["summary"][controller]
            lines.append(
                f"| {mode} | {label} | "
                f"{summary['mean_absolute_error_deg']:.2f}° | "
                f"{summary['p95_absolute_error_deg']:.2f}° | "
                f"{100.0 * summary['loss_of_view_fraction']:.2f}% | "
                f"{summary['mean_control_cost']:.3f} | "
                f"{summary['command_variation_per_s']:.3f} |"
            )

    for mode in ("rate", "position"):
        lines.extend(
            (
                "",
                f"## {mode.capitalize()} scenario deltas: O2 − analytical",
                "",
                "| Plot index | Scenario | Mean error | P95 | Lost view | "
                "Cost |",
                "|---:|---|---:|---:|---:|---:|",
            )
        )
        baseline_name = f"analytical_{mode}"
        learned_name = f"gru_o2_{mode}"
        for index, (scenario, records) in enumerate(
            control_result["scenario_aggregates"].items()
        ):
            baseline = records[baseline_name]
            learned = records[learned_name]
            baseline_metrics = baseline["mean_metrics"]
            learned_metrics = learned["mean_metrics"]
            lines.append(
                f"| {index} | `{scenario}` | "
                f"{learned_metrics['mean_absolute_error_deg'] - baseline_metrics['mean_absolute_error_deg']:+.2f}° | "
                f"{learned_metrics['p95_absolute_error_deg'] - baseline_metrics['p95_absolute_error_deg']:+.2f}° | "
                f"{100.0 * (learned_metrics['loss_of_view_fraction'] - baseline_metrics['loss_of_view_fraction']):+.2f} pp | "
                f"{learned['mean_control_cost'] - baseline['mean_control_cost']:+.3f} |"
            )

    rate_pairs = _paired_performance_records(control_result, "rate")
    seed_ranges = []
    for seed in dict.fromkeys(record["seed"] for record in rate_pairs):
        indices = [
            index
            for index, record in enumerate(rate_pairs)
            if record["seed"] == seed
        ]
        seed_ranges.append(f"{indices[0]}–{indices[-1]} = seed {seed}")
    lines.extend(
        (
            "",
            "Paired plot index map: "
            + "; ".join(seed_ranges)
            + ". Within each seed, scenario order matches the scenario-table "
            "plot index.",
            "",
            "## Weakness audit",
            "",
        )
    )
    for mode in ("rate", "position"):
        paired = _paired_performance_records(control_result, mode)
        regressions = sorted(
            (
                record
                for record in paired
                if record["control_cost_delta"] > 0.0
            ),
            key=lambda record: record["control_cost_delta"],
            reverse=True,
        )
        if regressions:
            worst = regressions[0]
            lines.append(
                f"- Worst {mode} cost regression is "
                f"`{worst['scenario_name']}` seed {worst['seed']}: "
                f"{worst['control_cost_delta']:+.3f} cost and "
                f"{worst['mean_error_delta_deg']:+.2f}° mean error."
            )
    rate_reference = control_result["summary"]["analytical_rate"]
    rate_learned = control_result["summary"]["gru_o2_rate"]
    position_reference = control_result["summary"]["analytical_position"]
    position_learned = control_result["summary"]["gru_o2_position"]
    lines.extend(
        (
            f"- O2 commands are less smooth in this frozen run: variation "
            f"changes by "
            f"{rate_learned['command_variation_per_s'] - rate_reference['command_variation_per_s']:+.3f}/s "
            f"for rate and "
            f"{position_learned['command_variation_per_s'] - position_reference['command_variation_per_s']:+.3f}/s "
            "for position.",
            f"- Loss recovery is slower despite fewer losses: event-weighted "
            f"recovery is {rate_reference['event_weighted_mean_recovery_time_s']:.2f} → "
            f"{rate_learned['event_weighted_mean_recovery_time_s']:.2f} s "
            f"for rate and "
            f"{position_reference['event_weighted_mean_recovery_time_s']:.2f} → "
            f"{position_learned['event_weighted_mean_recovery_time_s']:.2f} s "
            "for position.",
            "- Travel-limit recovery remains a physical ceiling; learned and "
            "analytical control are effectively tied there.",
            "- Directed loss-of-view search failed its separate fresh safety "
            "gate. Native hold remains the accepted fallback.",
            "- This passes the synthetic comparison target, not the deployment "
            "target. Recorded flight motion, identified camera/servo values, "
            "and embedded inference timing are still unverified.",
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
            "visibility/edge_search_supported",
            [240, 220, 60],
            "edge-conditioned search evidence",
        ),
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
    edge_supported = (
        replay_run.edge_evidence_trace[
            min(frame_index, len(replay_run.edge_evidence_trace) - 1)
        ][1]
        if replay_run.edge_evidence_trace
        else False
    )
    for suffix, value in (
        ("visibility/target_in_view", frame.diagnostics.target_in_view),
        ("visibility/detection_valid", observation.detection_valid),
        ("visibility/frame_updated", observation.frame_updated),
        ("visibility/edge_search_supported", edge_supported),
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


def write_controller_arena(
    arena: ControllerArena,
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show synchronized 3D/2D fixed, V2, and V2.1 replays."""
    if len(arena.comparison.runs) != 3:
        raise ValueError("controller arena requires exactly three runs")
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _controller_arena_blueprint(rrb, arena)
    rr.init(
        f"{_APP_ID}_"
        f"{'challenge_arena_v1' if arena.kind == 'challenge' else 'controller_arena_v1'}"
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "arena/summary",
        rr.TextDocument(
            _controller_arena_markdown(arena),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    styles = (
        (
            (
                "challenge_reactive_position",
                [255, 145, 45],
                "reactive position",
            ),
            (
                "challenge_classical_predictive",
                [55, 155, 255],
                "classical predictive",
            ),
            (
                "challenge_dream_to_center",
                [65, 215, 125],
                "Dream-to-Center",
            ),
        )
        if arena.kind == "challenge"
        else (
            ("arena_fixed_horizon", [50, 150, 255], "fixed horizon"),
            ("arena_adaptive_v2", [215, 90, 255], "adaptive V2"),
            ("arena_visibility_risk_v21", [70, 210, 130], "risk V2.1"),
        )
    )
    rr.log(
        "arena/signals/tracking_deg/target",
        rr.SeriesLines(colors=[255, 200, 40], names="target bearing"),
        static=True,
    )
    for name, color, label in styles:
        for panel, suffix in (
            ("tracking_deg", "gimbal"),
            ("error_deg", "absolute_error"),
            ("command_deg", "requested"),
            ("in_view", "visible"),
        ):
            rr.log(
                f"arena/signals/{panel}/{name}_{suffix}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )
        if arena.kind == "challenge":
            rr.log(
                f"arena/signals/fov_margin/{name}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )
    if arena.kind == "challenge":
        rr.log(
            "arena/signals/body_rate_deg_s/shared",
            rr.SeriesLines(colors=[235, 210, 70], names="body yaw rate"),
            static=True,
        )
        for path, color, label in (
            ("risk_trust/risk", [255, 100, 70], "visibility risk"),
            ("risk_trust/trust", [70, 190, 255], "forecast trust"),
            ("horizon_ms/effective", [70, 210, 130], "effective horizon"),
        ):
            rr.log(
                f"arena/dream/{path}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )
    else:
        for root, color, label in (
            ("risk/visibility", [255, 100, 70], "visibility risk"),
            ("risk/fov_fraction", [70, 190, 255], "predicted FOV fraction"),
            ("horizon_ms/boost", [255, 130, 40], "risk boost"),
            ("horizon_ms/effective", [70, 210, 130], "effective horizon"),
        ):
            rr.log(
                f"arena/v21/{root}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )

    for run in arena.comparison.runs:
        _log_closed_loop_static(rr, run)
    frames_by_run = tuple(
        run.episode.frames for run in arena.comparison.runs
    )
    for frame_index, frames in enumerate(zip(*frames_by_run, strict=True)):
        desired_body_relative_deg = None
        for run, frame in zip(arena.comparison.runs, frames, strict=True):
            estimate = (
                run.estimates[frame_index]
                if frame_index < len(run.estimates)
                else None
            )
            _log_closed_loop_frame(rr, run, frame, estimate)
            rr.log(
                f"{run.episode.name}/hud",
                rr.TextDocument(
                    _arena_hud_markdown(
                        arena,
                        run,
                        frame,
                        frame_index,
                    ),
                    media_type=rr.MediaType.MARKDOWN,
                ),
            )
            if arena.kind == "challenge":
                _log_arena_forecast_ghosts(
                    rr,
                    arena,
                    run,
                    frame,
                    frame_index,
                )
            diagnostics = frame.diagnostics
            desired_body_relative = math.atan2(
                math.sin(
                    diagnostics.target_bearing_rad
                    - diagnostics.body_bearing_rad
                ),
                math.cos(
                    diagnostics.target_bearing_rad
                    - diagnostics.body_bearing_rad
                ),
            )
            desired_body_relative_deg = math.degrees(desired_body_relative)
            actual_deg = math.degrees(diagnostics.gimbal_angle_rad)
            error_deg = abs(
                math.degrees(
                    math.atan2(
                        math.sin(
                            desired_body_relative
                            - diagnostics.gimbal_angle_rad
                        ),
                        math.cos(
                            desired_body_relative
                            - diagnostics.gimbal_angle_rad
                        ),
                    )
                )
            )
            requested_deg = math.degrees(
                diagnostics.requested_position_rad or 0.0
            )
            name = run.episode.name
            for path, value in (
                (f"tracking_deg/{name}_gimbal", actual_deg),
                (f"error_deg/{name}_absolute_error", error_deg),
                (f"command_deg/{name}_requested", requested_deg),
                (
                    f"in_view/{name}_visible",
                    float(diagnostics.target_in_view),
                ),
            ):
                rr.log(f"arena/signals/{path}", rr.Scalars(value))
            if arena.kind == "challenge":
                rr.log(
                    f"arena/signals/fov_margin/{name}",
                    rr.Scalars(
                        1.0 - abs(diagnostics.true_image_error_normalized)
                    ),
                )
        if desired_body_relative_deg is not None:
            rr.log(
                "arena/signals/tracking_deg/target",
                rr.Scalars(desired_body_relative_deg),
            )

        v21_run = arena.comparison.runs[-1]
        if arena.kind == "challenge":
            rr.log(
                "arena/signals/body_rate_deg_s/shared",
                rr.Scalars(math.degrees(frames[0].diagnostics.body_rate_rad_s)),
            )
        if frame_index < len(v21_run.adapter_diagnostics):
            item = v21_run.adapter_diagnostics[frame_index]
            if arena.kind == "challenge":
                for path, value in (
                    ("risk_trust/risk", item.get("visibility_risk", 0.0)),
                    ("risk_trust/trust", item.get("prediction_weight", 0.0)),
                    (
                        "horizon_ms/effective",
                        1000.0 * float(item.get("effective_horizon_s", 0.0)),
                    ),
                ):
                    rr.log(f"arena/dream/{path}", rr.Scalars(value))
            else:
                for path, value in (
                    ("risk/visibility", item.get("visibility_risk", 0.0)),
                    (
                        "risk/fov_fraction",
                        item.get("predicted_fov_fraction", 0.0),
                    ),
                    (
                        "horizon_ms/boost",
                        1000.0 * float(item.get("horizon_boost_s", 0.0)),
                    ),
                    (
                        "horizon_ms/effective",
                        1000.0 * float(item.get("effective_horizon_s", 0.0)),
                    ),
                ):
                    rr.log(f"arena/v21/{path}", rr.Scalars(value))


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


def write_uncertainty_calibration_dashboard(
    result: dict[str, Any],
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show validation/test uncertainty reliability diagnostics."""
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _uncertainty_calibration_blueprint(rrb)
    rr.init(f"{_APP_ID}_uncertainty_calibration_v1")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "calibration/summary",
        rr.TextDocument(
            _uncertainty_calibration_markdown(result),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    test = result["test"]
    variants = (
        ("uncalibrated", "global", "contextual")
        if result["experiment"]
        == "gimbal_contextual_uncertainty_calibration_v1"
        else ("uncalibrated", "calibrated")
    )
    for signal in ("bearing", "rate"):
        uncalibrated = test["reliability"]["uncalibrated"][signal]
        nominal_values = uncalibrated["nominal_coverage"]
        for index, nominal in enumerate(nominal_values):
            rr.set_time(
                _CALIBRATION_TIMELINE,
                sequence=index,
            )
            root = f"calibration/test/{signal}/reliability"
            rr.log(f"{root}/ideal", rr.Scalars(100.0 * nominal))
            for variant in variants:
                empirical = test["reliability"][variant][signal][
                    "empirical_coverage"
                ]
                rr.log(
                    f"{root}/{variant}",
                    rr.Scalars(100.0 * empirical[index]),
                )

        metric_name = f"{signal}_two_sigma_coverage"
        horizon_count = len(test["uncalibrated"]["per_horizon"])
        for horizon_index in range(horizon_count):
            rr.set_time(
                _CALIBRATION_TIMELINE,
                sequence=horizon_index,
            )
            root = f"calibration/test/{signal}/two_sigma_by_horizon"
            rr.log(f"{root}/nominal", rr.Scalars(95.45))
            for variant in variants:
                horizon = test[variant]["per_horizon"][horizon_index]
                rr.log(
                    f"{root}/{variant}",
                    rr.Scalars(100.0 * horizon[metric_name]),
                )


def write_gru_replication_dashboard(
    result: dict[str, Any],
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show O2 closed-loop variation across training seeds."""
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _gru_replication_blueprint(rrb)
    rr.init(f"{_APP_ID}_gru_o2_replication_v1")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "replication/summary",
        rr.TextDocument(
            _gru_replication_markdown(result),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    summary = result["replication_summary"]
    for index, seed_result in enumerate(result["training_seed_results"]):
        rr.set_time(_REPLICATION_TIMELINE, sequence=index)
        for mode in ("rate", "position"):
            learned = seed_result["closed_loop_summary"][f"gru_o2_{mode}"]
            analytical = summary[mode]["analytical_reference"]
            mean = summary[mode]["learned_metric_distribution"]
            for path, metric, scale in (
                ("mean_error_deg", "mean_absolute_error_deg", 1.0),
                ("loss_of_view_percent", "loss_of_view_fraction", 100.0),
            ):
                root = f"replication/{mode}/{path}"
                rr.log(
                    f"{root}/analytical",
                    rr.Scalars(scale * analytical[metric]),
                )
                rr.log(
                    f"{root}/o2_seed",
                    rr.Scalars(scale * learned[metric]),
                )
                rr.log(
                    f"{root}/o2_seed_mean",
                    rr.Scalars(scale * mean[metric]["mean"]),
                )


def write_adaptive_position_dashboard(
    result: dict[str, Any],
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show the fresh fixed-horizon/adaptive-position comparison."""
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _adaptive_position_blueprint(rrb)
    rr.init(f"{_APP_ID}_adaptive_position_v2")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "adaptive_position/summary",
        rr.TextDocument(
            _adaptive_position_markdown(result),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    line_styles = (
        ("fixed", [50, 150, 255], "fixed horizon"),
        ("v2", [215, 90, 255], "adaptive V2"),
    )
    for metric in (
        "p95_error_deg",
        "command_variation",
        "loss_of_view_percent",
    ):
        for suffix, color, label in line_styles:
            rr.log(
                f"adaptive_position/aggregate/{metric}/{suffix}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )
    for metric in (
        "p95_delta_deg",
        "variation_delta",
        "loss_delta_percent",
    ):
        rr.log(
            f"adaptive_position/scenario/{metric}/zero",
            rr.SeriesLines(colors=[130, 130, 140], names="no change"),
            static=True,
        )
        rr.log(
            f"adaptive_position/scenario/{metric}/v2_minus_fixed",
            rr.SeriesLines(
                colors=[215, 90, 255],
                names="adaptive V2 - fixed horizon",
            ),
            static=True,
        )
    trace_styles = {
        "tracking_deg": (
            ("target", [255, 200, 40], "target bearing"),
            ("fixed", [50, 150, 255], "fixed-horizon gimbal"),
            ("v2", [215, 90, 255], "adaptive-V2 gimbal"),
        ),
        "command_deg": (
            ("fixed", [50, 150, 255], "fixed command"),
            ("raw_v2", [255, 130, 40], "raw adaptive target"),
            ("shaped_v2", [215, 90, 255], "shaped adaptive command"),
        ),
        "horizon_s": (
            ("requested", [70, 190, 255], "actuator-arrival horizon"),
            ("effective", [215, 90, 255], "uncertainty-adjusted horizon"),
        ),
        "trust": (
            ("prediction_weight", [70, 210, 130], "prediction weight"),
            ("uncertainty_ratio", [255, 150, 40], "forecast/current std ratio"),
        ),
    }
    for panel, styles in trace_styles.items():
        for suffix, color, label in styles:
            rr.log(
                f"adaptive_position/trace/{panel}/{suffix}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )

    test = result["test"]
    fixed = test["fixed_summary"]
    candidate = test["v2_summary"]
    rr.set_time(_ADAPTIVE_TIMELINE, sequence=0)
    for suffix, summary in (("fixed", fixed), ("v2", candidate)):
        rr.log(
            f"adaptive_position/aggregate/p95_error_deg/{suffix}",
            rr.Scalars(summary["p95_absolute_error_deg"]),
        )
        rr.log(
            f"adaptive_position/aggregate/command_variation/{suffix}",
            rr.Scalars(summary["command_variation_per_s"]),
        )
        rr.log(
            f"adaptive_position/aggregate/loss_of_view_percent/{suffix}",
            rr.Scalars(100.0 * summary["loss_of_view_fraction"]),
        )

    for index, scenario in enumerate(test["by_scenario"].values()):
        rr.set_time(_ADAPTIVE_TIMELINE, sequence=index)
        deltas = scenario["deltas"]
        for path, value in (
            ("p95_delta_deg", deltas["p95_absolute_error_deg"]),
            ("variation_delta", deltas["command_variation_per_s"]),
            ("loss_delta_percent", 100.0 * deltas["loss_of_view_fraction"]),
        ):
            root = f"adaptive_position/scenario/{path}"
            rr.log(f"{root}/zero", rr.Scalars(0.0))
            rr.log(f"{root}/v2_minus_fixed", rr.Scalars(value))

    for record in result["representative_trace"]["records"]:
        rr.set_time(_TIMELINE, duration=record["time_s"])
        for suffix, key in (
            ("target", "target_body_bearing_deg"),
            ("fixed", "fixed_gimbal_angle_deg"),
            ("v2", "v2_gimbal_angle_deg"),
        ):
            rr.log(
                f"adaptive_position/trace/tracking_deg/{suffix}",
                rr.Scalars(record[key]),
            )
        for suffix, key in (
            ("fixed", "fixed_command_deg"),
            ("raw_v2", "v2_raw_target_deg"),
            ("shaped_v2", "v2_shaped_command_deg"),
        ):
            rr.log(
                f"adaptive_position/trace/command_deg/{suffix}",
                rr.Scalars(record[key]),
            )
        for suffix, key in (
            ("requested", "requested_horizon_s"),
            ("effective", "effective_horizon_s"),
        ):
            rr.log(
                f"adaptive_position/trace/horizon_s/{suffix}",
                rr.Scalars(record[key]),
            )
        for suffix, key in (
            ("prediction_weight", "prediction_weight"),
            ("uncertainty_ratio", "uncertainty_ratio"),
        ):
            rr.log(
                f"adaptive_position/trace/trust/{suffix}",
                rr.Scalars(record[key]),
            )


def write_visibility_risk_dashboard(
    result: dict[str, Any],
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show the fixed/V2/visibility-risk V2.1 confirmation."""
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _visibility_risk_blueprint(rrb)
    rr.init(f"{_APP_ID}_visibility_risk_position_v21")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "visibility_risk/summary",
        rr.TextDocument(
            _visibility_risk_markdown(result),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    aggregate_styles = (
        ("fixed", [50, 150, 255], "fixed horizon"),
        ("v2", [215, 90, 255], "adaptive V2"),
        ("v21", [70, 210, 130], "visibility-risk V2.1"),
    )
    for metric in (
        "p95_error_deg",
        "command_variation",
        "loss_of_view_percent",
        "unrecovered_events",
    ):
        for suffix, color, name in aggregate_styles:
            rr.log(
                f"visibility_risk/aggregate/{metric}/{suffix}",
                rr.SeriesLines(colors=color, names=name),
                static=True,
            )
    for metric in (
        "p95_delta_deg",
        "variation_delta",
        "loss_delta_percent",
    ):
        rr.log(
            f"visibility_risk/scenario/{metric}/zero",
            rr.SeriesLines(colors=[130, 130, 140], names="no change"),
            static=True,
        )
        rr.log(
            f"visibility_risk/scenario/{metric}/v21_minus_v2",
            rr.SeriesLines(colors=[70, 210, 130], names="V2.1 - V2"),
            static=True,
        )
    trace_styles = {
        "tracking_deg": (
            ("target", [255, 200, 40], "target bearing"),
            ("v2", [215, 90, 255], "adaptive V2 gimbal"),
            ("v21", [70, 210, 130], "risk V2.1 gimbal"),
        ),
        "command_deg": (
            ("v2", [215, 90, 255], "adaptive V2 command"),
            ("v21", [70, 210, 130], "risk V2.1 command"),
            ("raw_v21", [255, 130, 40], "raw V2.1 target"),
        ),
        "risk": (
            ("visibility", [255, 100, 70], "visibility risk"),
            ("fov_fraction", [70, 190, 255], "predicted FOV fraction"),
        ),
        "horizon_s": (
            ("boost", [255, 130, 40], "risk horizon boost"),
            ("effective", [70, 210, 130], "effective horizon"),
        ),
    }
    for panel, styles in trace_styles.items():
        for suffix, color, name in styles:
            rr.log(
                f"visibility_risk/trace/{panel}/{suffix}",
                rr.SeriesLines(colors=color, names=name),
                static=True,
            )

    confirmation = result["confirmation"]
    rr.set_time(_VISIBILITY_RISK_TIMELINE, sequence=0)
    for suffix, summary in (
        ("fixed", confirmation["fixed_summary"]),
        ("v2", confirmation["v2_summary"]),
        ("v21", confirmation["v21_summary"]),
    ):
        for path, key, scale in (
            ("p95_error_deg", "p95_absolute_error_deg", 1.0),
            ("command_variation", "command_variation_per_s", 1.0),
            ("loss_of_view_percent", "loss_of_view_fraction", 100.0),
            ("unrecovered_events", "total_unrecovered_loss_events", 1.0),
        ):
            rr.log(
                f"visibility_risk/aggregate/{path}/{suffix}",
                rr.Scalars(scale * summary[key]),
            )

    for index, scenario in enumerate(
        confirmation["by_scenario_vs_v2"].values()
    ):
        rr.set_time(_VISIBILITY_RISK_TIMELINE, sequence=index)
        deltas = scenario["deltas"]
        for path, value in (
            ("p95_delta_deg", deltas["p95_absolute_error_deg"]),
            ("variation_delta", deltas["command_variation_per_s"]),
            ("loss_delta_percent", 100.0 * deltas["loss_of_view_fraction"]),
        ):
            root = f"visibility_risk/scenario/{path}"
            rr.log(f"{root}/zero", rr.Scalars(0.0))
            rr.log(f"{root}/v21_minus_v2", rr.Scalars(value))

    for record in result["representative_trace"]["records"]:
        rr.set_time(_TIMELINE, duration=record["time_s"])
        for suffix, key in (
            ("target", "target_body_bearing_deg"),
            ("v2", "v2_gimbal_angle_deg"),
            ("v21", "v21_gimbal_angle_deg"),
        ):
            rr.log(
                f"visibility_risk/trace/tracking_deg/{suffix}",
                rr.Scalars(record[key]),
            )
        for suffix, key in (
            ("v2", "v2_command_deg"),
            ("v21", "v21_command_deg"),
            ("raw_v21", "v21_raw_target_deg"),
        ):
            rr.log(
                f"visibility_risk/trace/command_deg/{suffix}",
                rr.Scalars(record[key]),
            )
        for suffix, key in (
            ("visibility", "visibility_risk"),
            ("fov_fraction", "predicted_fov_fraction"),
        ):
            rr.log(
                f"visibility_risk/trace/risk/{suffix}",
                rr.Scalars(record[key]),
            )
        for suffix, key in (
            ("boost", "horizon_boost_s"),
            ("effective", "effective_horizon_s"),
        ):
            rr.log(
                f"visibility_risk/trace/horizon_s/{suffix}",
                rr.Scalars(record[key]),
            )


def write_predictive_position_dashboard(
    result: dict[str, Any],
    v31_result: dict[str, Any] | None = None,
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show the V2.1/V3 confirmation and V3.1 development audit."""

    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _predictive_position_blueprint(rrb)
    rr.init(f"{_APP_ID}_predictive_position_v3")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "predictive_position/summary",
        rr.TextDocument(
            _predictive_position_markdown(result, v31_result),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    styles = (
        ("fixed", [50, 150, 255], "fixed horizon"),
        ("v21", [70, 210, 130], "visibility-risk V2.1"),
        ("v3", [255, 120, 70], "constrained V3 (rejected)"),
    )
    for metric in (
        "mean_error_deg",
        "p95_error_deg",
        "avoidable_loss_percent",
        "command_variation",
    ):
        for suffix, color, label in styles:
            rr.log(
                f"predictive_position/aggregate/{metric}/{suffix}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )
    for metric in (
        "mean_delta_deg",
        "p95_delta_deg",
        "avoidable_loss_delta_percent",
        "variation_delta",
    ):
        root = f"predictive_position/scenario/{metric}"
        rr.log(
            f"{root}/zero",
            rr.SeriesLines(colors=[130, 130, 140], names="no change"),
            static=True,
        )
        rr.log(
            f"{root}/v3_minus_v21",
            rr.SeriesLines(
                colors=[255, 120, 70],
                names="constrained V3 - V2.1",
            ),
            static=True,
        )
    for panel, panel_styles in {
        "tracking_deg": (
            ("target", [255, 200, 40], "target bearing"),
            ("v21", [70, 210, 130], "V2.1 gimbal"),
            ("v3", [255, 120, 70], "V3 gimbal"),
        ),
        "command_deg": (
            ("v21", [70, 210, 130], "V2.1 command"),
            ("v3", [255, 120, 70], "V3 command"),
            ("fallback", [100, 170, 255], "V3 fallback target"),
        ),
        "activation": (
            ("score", [255, 120, 70], "activation score"),
            ("active", [255, 220, 70], "optimizer active"),
        ),
        "prediction": (
            ("terminal_error", [215, 90, 255], "terminal error / half-FOV"),
        ),
    }.items():
        for suffix, color, label in panel_styles:
            rr.log(
                f"predictive_position/trace/{panel}/{suffix}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )

    confirmation = result["confirmation"]
    rr.set_time(_PREDICTIVE_POSITION_TIMELINE, sequence=0)
    for suffix, block in (
        ("fixed", confirmation["fixed_horizon"]),
        ("v21", confirmation["visibility_risk_v21"]),
        ("v3", confirmation["predictive_position_v3"]),
    ):
        summary = block["tracked_summary"]
        for path, key, scale in (
            ("mean_error_deg", "mean_absolute_error_deg", 1.0),
            ("p95_error_deg", "p95_absolute_error_deg", 1.0),
            ("avoidable_loss_percent", "avoidable_loss_fraction", 100.0),
            ("command_variation", "command_variation_per_s", 1.0),
        ):
            rr.log(
                f"predictive_position/aggregate/{path}/{suffix}",
                rr.Scalars(scale * summary[key]),
            )

    for index, name in enumerate(confirmation["scenario_names"]):
        rr.set_time(_PREDICTIVE_POSITION_TIMELINE, sequence=index)
        reference = confirmation["visibility_risk_v21"]["by_scenario"][name][
            "summary"
        ]
        candidate = confirmation["predictive_position_v3"]["by_scenario"][
            name
        ]["summary"]
        for path, key, scale in (
            ("mean_delta_deg", "mean_absolute_error_deg", 1.0),
            ("p95_delta_deg", "p95_absolute_error_deg", 1.0),
            (
                "avoidable_loss_delta_percent",
                "avoidable_loss_fraction",
                100.0,
            ),
            ("variation_delta", "command_variation_per_s", 1.0),
        ):
            root = f"predictive_position/scenario/{path}"
            rr.log(f"{root}/zero", rr.Scalars(0.0))
            rr.log(
                f"{root}/v3_minus_v21",
                rr.Scalars(scale * (candidate[key] - reference[key])),
            )

    for record in result["representative_trace"]["records"]:
        rr.set_time(_TIMELINE, duration=record["time_s"])
        for suffix, key in (
            ("target", "target_body_bearing_deg"),
            ("v21", "v21_gimbal_angle_deg"),
            ("v3", "v3_gimbal_angle_deg"),
        ):
            rr.log(
                f"predictive_position/trace/tracking_deg/{suffix}",
                rr.Scalars(record[key]),
            )
        for suffix, key in (
            ("v21", "v21_command_deg"),
            ("v3", "v3_command_deg"),
            ("fallback", "v3_fallback_target_deg"),
        ):
            rr.log(
                f"predictive_position/trace/command_deg/{suffix}",
                rr.Scalars(record[key]),
            )
        rr.log(
            "predictive_position/trace/activation/score",
            rr.Scalars(record["activation_score"]),
        )
        rr.log(
            "predictive_position/trace/activation/active",
            rr.Scalars(float(record["optimizer_active"])),
        )
        rr.log(
            "predictive_position/trace/prediction/terminal_error",
            rr.Scalars(record["predicted_terminal_error_fov_fraction"]),
        )


def write_failure_atlas_dashboard(
    result: dict[str, Any],
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show the absolute contract and loss-attribution dashboard."""

    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _failure_atlas_blueprint(rrb)
    rr.init(f"{_APP_ID}_position_failure_atlas_v1")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "failure_atlas/summary",
        rr.TextDocument(
            _failure_atlas_markdown(result),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    styles = (
        ("fixed_horizon", [50, 150, 255], "fixed learned horizon"),
        ("adaptive_v2", [215, 90, 255], "adaptive V2"),
        ("visibility_risk_v21", [70, 210, 130], "risk V2.1"),
    )
    panels = (
        "mean_error_fov_percent",
        "p95_error_fov_percent",
        "loss_percent",
        "avoidable_loss_percent",
        "command_variation",
        "contract_checks",
    )
    for suffix, color, label in styles:
        for panel in panels:
            rr.log(
                f"failure_atlas/scenario/{panel}/{suffix}",
                rr.SeriesLines(colors=color, names=label),
                static=True,
            )
        rr.log(
            f"failure_atlas/causes/events/{suffix}",
            rr.SeriesLines(colors=color, names=label),
            static=True,
        )
    contract_styles = (
        ("mean_error_fov_percent", [150, 150, 150], "contract limit"),
        ("p95_error_fov_percent", [150, 150, 150], "contract limit"),
        ("loss_percent", [150, 150, 150], "contract limit"),
        ("avoidable_loss_percent", [150, 150, 150], "contract limit"),
        ("command_variation", [150, 150, 150], "contract limit"),
        ("contract_checks", [150, 150, 150], "all checks"),
    )
    for panel, color, label in contract_styles:
        rr.log(
            f"failure_atlas/scenario/{panel}/contract_limit",
            rr.SeriesLines(colors=color, names=label),
            static=True,
        )

    contract = result["contract"]
    scenario_names = result["scenario_names"]
    for index, scenario_name in enumerate(scenario_names):
        rr.set_time(_FAILURE_ATLAS_TIMELINE, sequence=index)
        for suffix, _color, _label in styles:
            item = result["controllers"][suffix]["by_scenario"][scenario_name]
            summary = item["summary"]
            gate = item["contract"]
            for panel, value in (
                (
                    "mean_error_fov_percent",
                    100.0 * summary["mean_absolute_error_fov_fraction"],
                ),
                (
                    "p95_error_fov_percent",
                    100.0 * summary["p95_absolute_error_fov_fraction"],
                ),
                ("loss_percent", 100.0 * summary["loss_of_view_fraction"]),
                (
                    "avoidable_loss_percent",
                    100.0 * summary["avoidable_loss_fraction"],
                ),
                ("command_variation", summary["command_variation_per_s"]),
                (
                    "contract_checks",
                    gate["passed_check_count"] if gate is not None else 0.0,
                ),
            ):
                rr.log(
                    f"failure_atlas/scenario/{panel}/{suffix}",
                    rr.Scalars(value),
                )
        applicable = scenario_name in contract["tracked_scenarios"]
        for panel, value in (
            (
                "mean_error_fov_percent",
                100.0
                * contract["maximum_mean_absolute_error_fov_fraction"],
            ),
            (
                "p95_error_fov_percent",
                100.0
                * contract["maximum_p95_absolute_error_fov_fraction"],
            ),
            (
                "loss_percent",
                100.0 * contract["maximum_loss_of_view_fraction"],
            ),
            (
                "avoidable_loss_percent",
                100.0 * contract["maximum_avoidable_loss_fraction"],
            ),
            (
                "command_variation",
                contract["maximum_command_variation_per_s"],
            ),
            ("contract_checks", 10.0),
        ):
            if applicable:
                rr.log(
                    f"failure_atlas/scenario/{panel}/contract_limit",
                    rr.Scalars(value),
                )

    cause_names = sorted(
        {
            cause
            for controller in result["controllers"].values()
            for cause in controller["tracked_summary"][
                "loss_event_causes"
            ]
        }
    )
    for index, cause in enumerate(cause_names):
        rr.set_time(_FAILURE_ATLAS_TIMELINE, sequence=index)
        for suffix, _color, _label in styles:
            count = result["controllers"][suffix]["tracked_summary"][
                "loss_event_causes"
            ].get(cause, 0)
            rr.log(
                f"failure_atlas/causes/events/{suffix}",
                rr.Scalars(count),
            )


def write_performance_verification_dashboard(
    control_result: dict[str, Any],
    replication_result: dict[str, Any],
    *,
    output: Path | None = None,
    spawn: bool = False,
) -> None:
    """Write or show paired baseline/O2 gains, regressions, and stability."""
    if (output is None) == (not spawn):
        raise ValueError("choose exactly one of output or spawn")
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            "Rerun is optional; install with `pip install -e '.[visualization]'`"
        ) from error

    blueprint = _performance_verification_blueprint(rrb)
    rr.init(f"{_APP_ID}_performance_verification_v1")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)
    rr.log(
        "performance/summary",
        rr.TextDocument(
            _performance_verification_markdown(
                control_result,
                replication_result,
            ),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    for mode in ("rate", "position"):
        for metric in ("mean_error_deg", "loss_of_view_percent"):
            root = f"performance/scenario/{mode}/{metric}"
            rr.log(
                f"{root}/analytical",
                rr.SeriesLines(
                    colors=[50, 150, 255],
                    names="analytical baseline",
                ),
                static=True,
            )
            rr.log(
                f"{root}/o2_gru",
                rr.SeriesLines(
                    colors=[215, 90, 255],
                    names="O2 GRU",
                ),
                static=True,
            )
        for metric in (
            "mean_error_delta_deg",
            "p95_error_delta_deg",
            "loss_of_view_delta_percent",
            "control_cost_delta",
        ):
            root = f"performance/paired/{mode}/{metric}"
            rr.log(
                f"{root}/zero",
                rr.SeriesLines(colors=[130, 130, 140], names="no change"),
                static=True,
            )
            rr.log(
                f"{root}/o2_minus_analytical",
                rr.SeriesLines(
                    colors=[215, 90, 255],
                    names="O2 - analytical",
                ),
                static=True,
            )
        root = f"performance/stability/{mode}/mean_error_deg"
        for suffix, color, name in (
            ("analytical", [50, 150, 255], "analytical baseline"),
            ("o2_seed", [215, 90, 255], "O2 training seed"),
            ("o2_mean", [255, 200, 40], "O2 seed mean"),
        ):
            rr.log(
                f"{root}/{suffix}",
                rr.SeriesLines(colors=color, names=name),
                static=True,
            )

    scenario_items = list(control_result["scenario_aggregates"].items())
    for scenario_index, (_scenario, records) in enumerate(scenario_items):
        rr.set_time(
            _PERFORMANCE_TIMELINE,
            sequence=scenario_index,
        )
        for mode in ("rate", "position"):
            for suffix, controller in (
                ("analytical", f"analytical_{mode}"),
                ("o2_gru", f"gru_o2_{mode}"),
            ):
                metrics = records[controller]["mean_metrics"]
                root = f"performance/scenario/{mode}"
                rr.log(
                    f"{root}/mean_error_deg/{suffix}",
                    rr.Scalars(metrics["mean_absolute_error_deg"]),
                )
                rr.log(
                    f"{root}/loss_of_view_percent/{suffix}",
                    rr.Scalars(100.0 * metrics["loss_of_view_fraction"]),
                )

    for mode in ("rate", "position"):
        for variant_index, record in enumerate(
            _paired_performance_records(control_result, mode)
        ):
            rr.set_time(
                _PERFORMANCE_TIMELINE,
                sequence=variant_index,
            )
            root = f"performance/paired/{mode}"
            for metric in (
                "mean_error_delta_deg",
                "p95_error_delta_deg",
                "loss_of_view_delta_percent",
                "control_cost_delta",
            ):
                rr.log(f"{root}/{metric}/zero", rr.Scalars(0.0))
                rr.log(
                    f"{root}/{metric}/o2_minus_analytical",
                    rr.Scalars(record[metric]),
                )

    replication = replication_result["replication_summary"]
    for seed_index, seed_result in enumerate(
        replication_result["training_seed_results"]
    ):
        rr.set_time(_PERFORMANCE_TIMELINE, sequence=seed_index)
        for mode in ("rate", "position"):
            root = f"performance/stability/{mode}/mean_error_deg"
            analytical = replication[mode]["analytical_reference"]
            learned = seed_result["closed_loop_summary"][f"gru_o2_{mode}"]
            learned_mean = replication[mode]["learned_metric_distribution"]
            rr.log(
                f"{root}/analytical",
                rr.Scalars(analytical["mean_absolute_error_deg"]),
            )
            rr.log(
                f"{root}/o2_seed",
                rr.Scalars(learned["mean_absolute_error_deg"]),
            )
            rr.log(
                f"{root}/o2_mean",
                rr.Scalars(learned_mean["mean_absolute_error_deg"]["mean"]),
            )


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
        choices=(
            "causality",
            "closed-loop",
            "benchmark-suite",
            "recovery",
            "calibration",
            "replication",
            "performance",
            "adaptive-position",
            "visibility-risk",
            "predictive-position",
            "failure-atlas",
            "controller-arena",
            "challenge-arena",
        ),
        default="causality",
        help=(
            "select a causality, controller, stress, recovery, calibration, "
            "training-seed replication, performance-verification, or adaptive "
            "position V2, visibility-risk V2.1, predictive-position V3, "
            "failure-atlas, or "
            "controller-arena or predictive challenge-arena dashboard"
        ),
    )
    parser.add_argument(
        "--adaptive-position-results",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_position_v2_fresh.json"),
        help="fresh adaptive-position V2 protocol result JSON",
    )
    parser.add_argument(
        "--visibility-risk-results",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_position_v21.json"),
        help="visibility-risk V2.1 confirmation result JSON",
    )
    parser.add_argument(
        "--failure-atlas-results",
        type=Path,
        default=Path("artifacts/gimbal_position_failure_atlas.json"),
        help="absolute performance-contract and failure-atlas result JSON",
    )
    parser.add_argument(
        "--predictive-position-results",
        type=Path,
        default=Path("artifacts/gimbal_predictive_position_v3.json"),
        help="constrained predictive-position V3 confirmation result JSON",
    )
    parser.add_argument(
        "--predictive-position-v31-results",
        type=Path,
        default=Path("artifacts/gimbal_predictive_position_v31.json"),
        help="optional dual-risk V3.1 development result JSON",
    )
    parser.add_argument(
        "--arena-scenario",
        choices=(
            "nominal_combined",
            "high_latency",
            "dropout_noise",
            "slow_servo",
            "aggressive_motion",
            "travel_limit_recovery",
        ),
        help="confirmation scenario for the three-controller visual arena",
    )
    parser.add_argument(
        "--arena-world-seed",
        type=int,
        help="confirmation world seed for the controller arena",
    )
    parser.add_argument(
        "--arena-training-seed",
        type=int,
        help="GRU initialization seed for the controller arena",
    )
    parser.add_argument(
        "--arena-reactive-gain",
        type=float,
        default=0.85,
        help="reactive position gain for the challenge arena",
    )
    parser.add_argument(
        "--arena-ghost-horizons-ms",
        type=float,
        nargs="+",
        default=(100.0, 200.0, 300.0),
        help="future bbox ghost horizons in milliseconds",
    )
    parser.add_argument(
        "--performance-results",
        type=Path,
        default=Path(
            "artifacts/gimbal_mixed_gru_closed_loop_comparison.json"
        ),
        help="paired analytical/O2 closed-loop comparison result JSON",
    )
    parser.add_argument(
        "--uncertainty-calibration",
        type=Path,
        default=Path("artifacts/gimbal_o2_uncertainty_calibration.json"),
        help="validation-fit uncertainty calibration result JSON",
    )
    parser.add_argument(
        "--replication-results",
        type=Path,
        default=Path("artifacts/gimbal_o2_replication.json"),
        help="O2 multi-training-seed replication result JSON",
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
            "detector_micro_bursts",
            "target_reversal_outage",
            "negative_travel_limit_reentry",
            "body_maneuver_outage",
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
    if args.demo in {"controller-arena", "challenge-arena"}:
        from .controller_arena import (
            build_gimbal_challenge_arena,
            build_visibility_risk_controller_arena,
        )

        result = _load_visibility_risk_result(args.visibility_risk_results)
        arena = (
            build_gimbal_challenge_arena(
                result,
                scenario_name=args.arena_scenario or "high_latency",
                world_seed=args.arena_world_seed or 82000,
                training_seed=args.arena_training_seed or 17,
                reactive_gain=args.arena_reactive_gain,
                ghost_horizons_s=tuple(
                    value / 1000.0
                    for value in args.arena_ghost_horizons_ms
                ),
                device=args.device,
            )
            if args.demo == "challenge-arena"
            else build_visibility_risk_controller_arena(
                result,
                scenario_name=args.arena_scenario,
                world_seed=args.arena_world_seed,
                training_seed=args.arena_training_seed,
                device=args.device,
            )
        )
        write_controller_arena(
            arena,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "failure-atlas":
        result = _load_failure_atlas_result(args.failure_atlas_results)
        write_failure_atlas_dashboard(
            result,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "predictive-position":
        result = _load_predictive_position_result(
            args.predictive_position_results
        )
        v31_result = (
            _load_predictive_position_v31_result(
                args.predictive_position_v31_results
            )
            if args.predictive_position_v31_results.exists()
            else None
        )
        write_predictive_position_dashboard(
            result,
            v31_result,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "visibility-risk":
        result = _load_visibility_risk_result(args.visibility_risk_results)
        write_visibility_risk_dashboard(
            result,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "adaptive-position":
        result = _load_adaptive_position_result(args.adaptive_position_results)
        write_adaptive_position_dashboard(
            result,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "performance":
        control_result = _load_gru_control_result(args.performance_results)
        replication_result = _load_gru_replication_result(
            args.replication_results
        )
        write_performance_verification_dashboard(
            control_result,
            replication_result,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "replication":
        result = _load_gru_replication_result(args.replication_results)
        write_gru_replication_dashboard(
            result,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "calibration":
        result = _load_uncertainty_calibration_result(
            args.uncertainty_calibration
        )
        write_uncertainty_calibration_dashboard(
            result,
            output=args.output,
            spawn=spawn,
        )
    elif args.demo == "recovery":
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
