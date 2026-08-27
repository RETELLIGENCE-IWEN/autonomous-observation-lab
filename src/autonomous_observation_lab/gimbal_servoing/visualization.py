"""Rerun visualization for the paired gimbal-causality demonstration."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

from .demos import DemoEpisode, DemoFrame, paired_cause_demo


_APP_ID = "autonomous_observation_lab_gimbal_demo"
_TIMELINE = "sim_time"


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare identical bbox motion caused by gimbal vs target motion."
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
    parser.add_argument("--seed", type=int, default=21)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    episodes = paired_cause_demo(seed=args.seed)
    write_rerun_demo(
        episodes,
        output=args.output,
        spawn=args.spawn or args.output is None,
    )


if __name__ == "__main__":
    main()
