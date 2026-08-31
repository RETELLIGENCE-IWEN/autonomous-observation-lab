#!/usr/bin/env python3
"""Generate exact-data SVG figures for the predictive-gimbal journey paper."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "aol-matplotlib-cache"),
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Matplotlib is required; install with `pip install -e '.[reporting]'`."
    ) from error


ANALYTICAL_COLOR = "#3977b8"
LEARNED_COLOR = "#9b59b6"
IMPROVEMENT_COLOR = "#188977"
REGRESSION_COLOR = "#cf5c5c"
NEUTRAL_COLOR = "#7b8491"
GRID_COLOR = "#d9dee7"


def _load(path: Path, experiment: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment") != experiment:
        raise ValueError(f"{path} is not a {experiment} artifact")
    return value


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": "#515966",
            "axes.labelcolor": "#303640",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "autonomous-observation-lab",
            "xtick.color": "#303640",
            "ytick.color": "#303640",
        }
    )


def _save(fig: Any, path: Path) -> None:
    output_format = path.suffix.removeprefix(".")
    metadata = (
        {
            "Creator": "Autonomous Observation Lab figure generator",
            "Date": None,
        }
        if output_format == "svg"
        else {"Creator": "Autonomous Observation Lab figure generator"}
    )
    fig.savefig(
        path,
        format=output_format,
        bbox_inches="tight",
        metadata=metadata,
    )
    plt.close(fig)
    if output_format == "svg":
        svg = path.read_text(encoding="utf-8")
        path.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
            encoding="utf-8",
        )


def _label_bars(ax: Any, bars: Any, precision: int = 2) -> None:
    labels = [f"{bar.get_height():.{precision}f}" for bar in bars]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)


def _closed_loop_figure(replication: dict[str, Any], output: Path) -> None:
    summary = replication["replication_summary"]
    specifications = (
        ("mean_absolute_error_deg", "Mean error", "degrees", 1.0, 2),
        ("p95_absolute_error_deg", "Episode P95", "degrees", 1.0, 2),
        ("loss_of_view_fraction", "Loss of view", "percent", 100.0, 2),
        ("mean_control_cost", "Control cost", "cost", 1.0, 3),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    modes = ("rate", "position")
    x = [0, 1]
    width = 0.34
    for ax, (metric, title, unit, scale, precision) in zip(
        axes.flat,
        specifications,
        strict=True,
    ):
        analytical = [
            scale * summary[mode]["analytical_reference"][metric]
            for mode in modes
        ]
        learned = [
            scale
            * summary[mode]["learned_metric_distribution"][metric]["mean"]
            for mode in modes
        ]
        learned_std = [
            scale
            * summary[mode]["learned_metric_distribution"][metric][
                "sample_std"
            ]
            for mode in modes
        ]
        baseline_bars = ax.bar(
            [value - width / 2 for value in x],
            analytical,
            width,
            color=ANALYTICAL_COLOR,
            label="Analytical constant velocity",
        )
        learned_bars = ax.bar(
            [value + width / 2 for value in x],
            learned,
            width,
            yerr=learned_std,
            capsize=4,
            color=LEARNED_COLOR,
            label="O2 GRU mean ± seed SD",
        )
        _label_bars(ax, baseline_bars, precision)
        _label_bars(ax, learned_bars, precision)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks(x, ("Rate command", "Position command"))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.margins(y=0.16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Closed-loop tracking: analytical baseline vs replicated O2 GRU",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.12, hspace=0.38, wspace=0.28)
    _save(fig, output)


def _scenario_figure(control: dict[str, Any], output: Path) -> None:
    scenarios = list(control["scenario_aggregates"])
    labels = (
        "Nominal",
        "High\nlatency",
        "Dropout +\nnoise",
        "Slow\nservo",
        "Aggressive\nmotion",
        "Travel\nlimit",
    )
    if len(scenarios) != len(labels):
        labels = tuple(name.replace("_", "\n") for name in scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), sharex=True)
    metric_specs = (
        ("mean_absolute_error_deg", "Mean-error delta", "degrees", 1.0),
        ("loss_of_view_fraction", "Loss-of-view delta", "percentage points", 100.0),
    )
    for column, mode in enumerate(("rate", "position")):
        for row, (metric, title, unit, scale) in enumerate(metric_specs):
            ax = axes[row, column]
            deltas = []
            for scenario in scenarios:
                records = control["scenario_aggregates"][scenario]
                baseline = records[f"analytical_{mode}"]["mean_metrics"]
                learned = records[f"gru_o2_{mode}"]["mean_metrics"]
                deltas.append(scale * (learned[metric] - baseline[metric]))
            colors = [
                IMPROVEMENT_COLOR if value < 0.0 else REGRESSION_COLOR
                for value in deltas
            ]
            bars = ax.bar(range(len(scenarios)), deltas, color=colors)
            ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
            ax.set_title(f"{mode.capitalize()} command: {title}")
            ax.set_ylabel(f"O2 − analytical ({unit})")
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_xticks(range(len(scenarios)), labels)
            ax.tick_params(axis="x", labelsize=8)
            for bar, value in zip(bars, deltas, strict=True):
                offset = 3 if value >= 0.0 else -12
                ax.annotate(
                    f"{value:+.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0.0 else "top",
                    fontsize=7.5,
                )
    fig.suptitle(
        "Where the learned controller helps—and where geometry dominates",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Green bars favor O2; red bars indicate regression. Frozen seed-17 controller artifact.",
        ha="center",
        color="#4b535f",
    )
    fig.subplots_adjust(bottom=0.13, hspace=0.42, wspace=0.25)
    _save(fig, output)


def _calibration_figure(calibration: dict[str, Any], output: Path) -> None:
    before = calibration["test"]["uncalibrated"]
    after = calibration["test"]["calibrated"]
    names = ("Bearing", "Angular rate")
    before_values = [
        100.0 * before["bearing_two_sigma_coverage"],
        100.0 * before["rate_two_sigma_coverage"],
    ]
    after_values = [
        100.0 * after["bearing_two_sigma_coverage"],
        100.0 * after["rate_two_sigma_coverage"],
    ]
    x = [0, 1]
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    before_bars = ax.bar(
        [value - width / 2 for value in x],
        before_values,
        width,
        color=NEUTRAL_COLOR,
        label="Before calibration",
    )
    after_bars = ax.bar(
        [value + width / 2 for value in x],
        after_values,
        width,
        color=LEARNED_COLOR,
        label="After global scaling",
    )
    _label_bars(ax, before_bars, 2)
    _label_bars(ax, after_bars, 2)
    ax.axhline(
        95.45,
        color=IMPROVEMENT_COLOR,
        linestyle="--",
        linewidth=1.5,
        label="Nominal Gaussian 2σ: 95.45%",
    )
    ax.set_xticks(x, names)
    ax.set_ylabel("Held-out 2σ coverage (%)")
    ax.set_ylim(90.0, 97.0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(
        "Global uncertainty scaling improves held-out tail coverage",
        fontsize=14,
        fontweight="bold",
    )
    _save(fig, output)


def _recovery_figure(edge: dict[str, Any], output: Path) -> None:
    summary = edge["test_result"]["summary"]
    hold = summary["gru_o2_rate_hold"]
    candidate = summary["gru_o2_rate_belief"]
    specifications = (
        ("Mean error", "mean_absolute_error_deg", 1.0, "degrees"),
        ("P95 error", "p95_absolute_error_deg", 1.0, "degrees"),
        ("Loss of view", "loss_of_view_fraction", 100.0, "pp"),
        ("Control cost", "mean_control_cost", 1.0, "cost"),
        ("Unrecovered", "total_unrecovered_loss_events", 1.0, "events"),
    )
    fig, axes = plt.subplots(1, len(specifications), figsize=(13.0, 3.2))
    for ax, (label, metric, scale, unit) in zip(
        axes,
        specifications,
        strict=True,
    ):
        delta = scale * (candidate[metric] - hold[metric])
        color = IMPROVEMENT_COLOR if delta < 0.0 else REGRESSION_COLOR
        ax.bar([0], [delta], width=0.58, color=color)
        ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=1.0)
        magnitude = max(abs(delta) * 1.7, 0.08)
        ax.set_ylim(-magnitude, magnitude)
        ax.set_xticks([])
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(f"Edge − hold ({unit})", fontsize=8)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.text(
            0,
            delta,
            f"{delta:+.3f}" if abs(delta) < 1.0 else f"{delta:+.2f}",
            ha="center",
            va="bottom" if delta >= 0.0 else "top",
            fontsize=9,
            fontweight="bold",
        )
    verdict = "PASS" if edge["fresh_test_gate"]["passed"] else "REJECT"
    fig.suptitle(
        f"Fresh recovery safety gate: {verdict}",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Lower is better. Cost and mean error improve slightly, but tail error, visibility, and events regress.",
        ha="center",
        color="#4b535f",
    )
    fig.subplots_adjust(bottom=0.20, top=0.74, wspace=0.72)
    _save(fig, output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SVG figures for the predictive-gimbal paper."
    )
    parser.add_argument(
        "--control-results",
        type=Path,
        default=Path("artifacts/gimbal_mixed_gru_closed_loop_comparison.json"),
    )
    parser.add_argument(
        "--replication-results",
        type=Path,
        default=Path("artifacts/gimbal_o2_replication.json"),
    )
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("artifacts/gimbal_o2_uncertainty_calibration.json"),
    )
    parser.add_argument(
        "--edge-recovery-results",
        type=Path,
        default=Path("artifacts/gimbal_edge_recovery_protocol.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "docs/research_tracks/predictive_gimbal_servoing/figures"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _style()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    control = _load(
        args.control_results,
        "gimbal_gru_closed_loop_comparison_v1",
    )
    replication = _load(
        args.replication_results,
        "gimbal_gru_o2_replication_v1",
    )
    calibration = _load(
        args.calibration_results,
        "gimbal_uncertainty_calibration_v1",
    )
    edge = _load(
        args.edge_recovery_results,
        "gimbal_edge_recovery_development_test_v1",
    )
    _closed_loop_figure(
        replication,
        args.output_directory / "closed_loop_performance.svg",
    )
    _scenario_figure(
        control,
        args.output_directory / "scenario_deltas.svg",
    )
    _calibration_figure(
        calibration,
        args.output_directory / "uncertainty_calibration.svg",
    )
    _recovery_figure(
        edge,
        args.output_directory / "recovery_safety_gate.svg",
    )
    print(f"Wrote four SVG figures to {args.output_directory}")


if __name__ == "__main__":
    main()
