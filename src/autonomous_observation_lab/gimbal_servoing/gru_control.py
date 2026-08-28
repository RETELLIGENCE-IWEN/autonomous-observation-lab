"""Checkpoint-driven closed-loop evaluation for causal gimbal GRUs.

This module requires the optional ``learning`` dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .closed_loop import (
    ClosedLoopScenario,
    ControllerRun,
    TrackingMetrics,
    run_closed_loop_controller,
)
from .config import GimbalCommandMode, ObservationProfile
from .controllers import (
    ProportionalController,
    ProportionalPositionController,
    SearchFallbackController,
    TargetStatePositionController,
    TargetStateRateController,
)
from .dataset import FEATURE_NAMES, GimbalDatasetManifest
from .estimators import (
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
)
from .gru import (
    CausalTargetStateGRU,
    GRUInferenceConfig,
    GRUTargetStateEstimator,
    load_gru_checkpoint,
)
from .serialization import closed_loop_scenario_from_dict


LEARNED_PROFILES = (
    ObservationProfile.SERVO_AWARE,
    ObservationProfile.DISTURBANCE_AWARE,
)


@dataclass(frozen=True)
class GRUControlEvaluationConfig:
    rate_feedback_gain_s_inv: float = 2.5
    maximum_staleness_s: float = 0.50
    proportional_rate_gain: float = 1.35
    proportional_position_gain: float = 0.85
    search_rate_normalized: float = 0.25
    search_position_fraction: float = 0.90
    search_reversal_margin_rad: float = math.radians(2.0)
    device: str = "cpu"
    include_position_diagnostic: bool = True

    def __post_init__(self) -> None:
        if self.rate_feedback_gain_s_inv < 0.0:
            raise ValueError("rate feedback gain must be non-negative")
        if self.maximum_staleness_s < 0.0:
            raise ValueError("maximum staleness must be non-negative")
        if self.proportional_rate_gain < 0.0:
            raise ValueError("proportional rate gain must be non-negative")
        if self.proportional_position_gain < 0.0:
            raise ValueError("proportional position gain must be non-negative")
        for name in ("search_rate_normalized", "search_position_fraction"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.search_reversal_margin_rad < 0.0:
            raise ValueError("search reversal margin must be non-negative")


def _manifest_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    return path if path.suffix == ".json" else path.with_suffix(".json")


def _load_manifest(dataset_path: str | Path) -> GimbalDatasetManifest:
    path = _manifest_path(dataset_path)
    return GimbalDatasetManifest.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _scenario_variants(
    manifest: GimbalDatasetManifest,
) -> tuple[tuple[int, int, ClosedLoopScenario], ...]:
    raw_variants = manifest.generation.get("scenario_variants")
    if not isinstance(raw_variants, list):
        raise ValueError("control evaluation requires recorded scenario variants")
    variants: list[tuple[int, int, ClosedLoopScenario]] = []
    keys: set[tuple[int, int]] = set()
    for value in raw_variants:
        if not isinstance(value, dict):
            raise ValueError("scenario variant entries must be objects")
        seed = int(value["seed"])
        scenario_index = int(value["scenario_index"])
        key = (seed, scenario_index)
        if key in keys:
            raise ValueError("scenario variant keys must be unique")
        keys.add(key)
        scenario = closed_loop_scenario_from_dict(value["scenario"])
        if scenario.name != manifest.scenario_names[scenario_index]:
            raise ValueError("scenario variant name/index mismatch")
        variants.append((seed, scenario_index, scenario))
    expected = {
        (seed, scenario_index)
        for seed in manifest.seeds
        for scenario_index in range(len(manifest.scenario_names))
    }
    if keys != expected:
        raise ValueError("manifest does not contain a complete scenario grid")
    return tuple(variants)


def _load_profile_checkpoint(
    path: str | Path,
    profile: ObservationProfile,
    manifests: dict[str, GimbalDatasetManifest],
    device: str,
) -> CausalTargetStateGRU:
    model, metadata = load_gru_checkpoint(path, device=device)
    if metadata.get("profile") != profile.value:
        raise ValueError(f"checkpoint profile does not match {profile.value}")
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema does not match deployment")
    hashes = metadata.get("dataset_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("checkpoint is missing dataset hashes")
    for split, manifest in manifests.items():
        if hashes.get(split) != manifest.configuration_hash:
            raise ValueError(f"checkpoint {split} dataset hash mismatch")
    return model


def _analytical_estimator(
    scenario: ClosedLoopScenario,
) -> ConstantVelocityTargetEstimator:
    config = scenario.config
    minimum_horizon_s = (
        config.camera.detection_latency_s
        + config.camera.detection_latency_jitter_s
        + 2.0 * config.camera.frame_period_s
    )
    prediction_horizon_s = max(0.30, minimum_horizon_s)
    return ConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            center_noise_std_normalized=(
                config.camera.center_noise_std_normalized
            ),
            velocity_filter_coefficient=0.40,
            uncertainty_filter_coefficient=0.20,
            max_prediction_horizon_s=prediction_horizon_s,
            history_horizon_s=max(1.0, prediction_horizon_s + 0.50),
        )
    )


def _control_cost(run: ControllerRun) -> float:
    metrics = run.metrics
    config = run.episode.config
    objective = config.objective
    mean_action_change = (
        metrics.command_variation_per_s / config.timing.control_rate_hz
    )
    return float(
        objective.error_weight * metrics.rms_error_normalized**2
        + objective.loss_of_view_penalty * metrics.loss_of_view_fraction
        + objective.action_effort_weight * metrics.command_rms_normalized**2
        + objective.action_change_weight * mean_action_change
    )


def _learned_run(
    *,
    scenario: ClosedLoopScenario,
    seed: int,
    model: CausalTargetStateGRU,
    profile: ObservationProfile,
    horizon_index: int,
    command_mode: GimbalCommandMode,
    evaluation: GRUControlEvaluationConfig,
    search_fallback: bool,
) -> ControllerRun:
    config = replace(
        scenario.config,
        observation_profile=profile,
        command_mode=command_mode,
    )
    estimator = GRUTargetStateEstimator(
        model,
        config,
        GRUInferenceConfig(
            observation_profile=profile,
            horizon_index=horizon_index,
            maximum_staleness_s=evaluation.maximum_staleness_s,
            device=evaluation.device,
        ),
    )
    profile_short = "o1" if profile is ObservationProfile.SERVO_AWARE else "o2"
    mode_short = "rate" if command_mode is GimbalCommandMode.RATE else "position"
    name = f"gru_{profile_short}_{mode_short}"
    if command_mode is GimbalCommandMode.RATE:
        delegate: TargetStateRateController | TargetStatePositionController = (
            TargetStateRateController(
                estimator=estimator,
                max_rate_rad_s=config.servo.max_rate_rad_s,
                proportional_gain_s_inv=(
                    evaluation.rate_feedback_gain_s_inv
                ),
                name=name,
            )
        )
    else:
        delegate = TargetStatePositionController(
            estimator=estimator,
            servo=config.servo,
            command_preview_s=0.0,
            name=name,
        )
    controller = delegate
    if search_fallback:
        name += "_search"
        controller = SearchFallbackController(
            delegate=delegate,
            servo=config.servo,
            command_mode=command_mode,
            search_rate_normalized=evaluation.search_rate_normalized,
            search_position_fraction=evaluation.search_position_fraction,
            reversal_margin_rad=evaluation.search_reversal_margin_rad,
            name=name,
        )
    return run_closed_loop_controller(
        name=name,
        description=(
            f"{profile.value} GRU with {mode_short} adapter"
            + (" and travel-envelope search fallback." if search_fallback else ".")
        ),
        scenario=scenario,
        config=config,
        controller=controller,
        seed=seed,
    )


def _baseline_runs(
    scenario: ClosedLoopScenario,
    seed: int,
    evaluation: GRUControlEvaluationConfig,
) -> tuple[ControllerRun, ...]:
    base = scenario.config
    rate_config = replace(
        base,
        command_mode=GimbalCommandMode.RATE,
        observation_profile=ObservationProfile.SERVO_AWARE,
    )
    position_config = replace(
        base,
        command_mode=GimbalCommandMode.POSITION,
        observation_profile=ObservationProfile.SERVO_AWARE,
    )
    specifications: list[tuple[str, str, Any, Any]] = [
        (
            "proportional_rate",
            "Delayed bbox feedback driving desired rate.",
            rate_config,
            ProportionalController(gain=evaluation.proportional_rate_gain),
        ),
        (
            "analytical_rate",
            "Constant-velocity target state driving desired rate.",
            rate_config,
            TargetStateRateController(
                estimator=_analytical_estimator(scenario),
                max_rate_rad_s=rate_config.servo.max_rate_rad_s,
                proportional_gain_s_inv=evaluation.rate_feedback_gain_s_inv,
                name="analytical_rate",
            ),
        ),
    ]
    if evaluation.include_position_diagnostic:
        specifications.extend(
            (
                (
                    "proportional_position",
                    "Delayed bbox feedback updating an absolute setpoint.",
                    position_config,
                    ProportionalPositionController(
                        servo=position_config.servo,
                        selected_axis_fov_rad=(
                            position_config.camera.selected_axis_fov_rad
                        ),
                        gain=evaluation.proportional_position_gain,
                    ),
                ),
                (
                    "analytical_position",
                    "Constant-velocity target state driving an absolute setpoint.",
                    position_config,
                    TargetStatePositionController(
                        estimator=_analytical_estimator(scenario),
                        servo=position_config.servo,
                        command_preview_s=(
                            position_config.servo.command_latency_s
                            + position_config.servo.rate_time_constant_s
                        ),
                        name="analytical_position",
                    ),
                ),
            )
        )
    return tuple(
        run_closed_loop_controller(
            name=name,
            description=description,
            scenario=scenario,
            config=config,
            controller=controller,
            seed=seed,
        )
        for name, description, config, controller in specifications
    )


def _aggregate_runs(runs: list[ControllerRun]) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot aggregate an empty run list")
    metric_names = [field.name for field in fields(TrackingMetrics)]
    means = {
        name: float(np.mean([getattr(run.metrics, name) for run in runs]))
        for name in metric_names
    }
    recovered_counts = [
        run.metrics.loss_of_view_events
        - run.metrics.unrecovered_loss_events
        for run in runs
    ]
    total_recovered = sum(recovered_counts)
    weighted_recovery = sum(
        count * run.metrics.mean_recovery_time_s
        for count, run in zip(recovered_counts, runs, strict=True)
    )
    return {
        "episode_count": len(runs),
        "mean_control_cost": float(np.mean([_control_cost(run) for run in runs])),
        "episodes_with_loss_fraction": float(
            np.mean([run.metrics.loss_of_view_events > 0 for run in runs])
        ),
        "total_loss_of_view_events": int(
            sum(run.metrics.loss_of_view_events for run in runs)
        ),
        "total_unrecovered_loss_events": int(
            sum(run.metrics.unrecovered_loss_events for run in runs)
        ),
        "event_weighted_mean_recovery_time_s": (
            weighted_recovery / total_recovered if total_recovered else 0.0
        ),
        "max_recovery_time_s": max(
            run.metrics.max_recovery_time_s for run in runs
        ),
        "mean_metrics": means,
    }


def _select_horizon(
    *,
    variants: tuple[tuple[int, int, ClosedLoopScenario], ...],
    model: CausalTargetStateGRU,
    profile: ObservationProfile,
    command_mode: GimbalCommandMode,
    evaluation: GRUControlEvaluationConfig,
) -> tuple[int, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for horizon_index, horizon_s in enumerate(
        model.config.prediction_horizons_s
    ):
        runs = [
            _learned_run(
                scenario=scenario,
                seed=seed,
                model=model,
                profile=profile,
                horizon_index=horizon_index,
                command_mode=command_mode,
                evaluation=evaluation,
                search_fallback=False,
            )
            for seed, _scenario_index, scenario in variants
        ]
        aggregate = _aggregate_runs(runs)
        candidates.append(
            {
                "horizon_index": horizon_index,
                "horizon_s": horizon_s,
                "mean_control_cost": aggregate["mean_control_cost"],
                "mean_absolute_error_deg": aggregate["mean_metrics"][
                    "mean_absolute_error_deg"
                ],
                "p95_absolute_error_deg": aggregate["mean_metrics"][
                    "p95_absolute_error_deg"
                ],
                "loss_of_view_fraction": aggregate["mean_metrics"][
                    "loss_of_view_fraction"
                ],
            }
        )
    selected = min(candidates, key=lambda item: item["mean_control_cost"])
    return int(selected["horizon_index"]), candidates


def _paired_comparison(
    candidate_runs: list[ControllerRun],
    reference_runs: list[ControllerRun],
) -> dict[str, Any]:
    if len(candidate_runs) != len(reference_runs) or not candidate_runs:
        raise ValueError("paired controller runs must have equal non-zero length")
    comparisons: dict[str, Any] = {}

    def standard_error(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=1) / math.sqrt(len(values)))

    metric_names = (
        "mean_absolute_error_deg",
        "p95_absolute_error_deg",
        "loss_of_view_fraction",
        "rate_saturation_fraction",
        "command_rms_normalized",
        "command_variation_per_s",
    )
    for name in metric_names:
        differences = np.asarray(
            [
                getattr(candidate.metrics, name)
                - getattr(reference.metrics, name)
                for candidate, reference in zip(
                    candidate_runs, reference_runs, strict=True
                )
            ],
            dtype=np.float64,
        )
        comparisons[name] = {
            "mean_delta": float(np.mean(differences)),
            "standard_error": standard_error(differences),
            "candidate_win_fraction": float(np.mean(differences < 0.0)),
        }
    cost_differences = np.asarray(
        [
            _control_cost(candidate) - _control_cost(reference)
            for candidate, reference in zip(
                candidate_runs, reference_runs, strict=True
            )
        ],
        dtype=np.float64,
    )
    comparisons["control_cost"] = {
        "mean_delta": float(np.mean(cost_differences)),
        "standard_error": standard_error(cost_differences),
        "candidate_win_fraction": float(np.mean(cost_differences < 0.0)),
    }
    return comparisons


def _command_modes_from_behaviors(
    behavior_names: tuple[str, ...],
) -> tuple[str, ...]:
    modes = []
    if any(name.endswith("_rate") for name in behavior_names):
        modes.append(GimbalCommandMode.RATE.value)
    if any(name.endswith("_position") for name in behavior_names):
        modes.append(GimbalCommandMode.POSITION.value)
    return tuple(modes)


def evaluate_gru_closed_loop(
    *,
    train_data: str | Path,
    validation_data: str | Path,
    test_data: str | Path,
    o1_checkpoint: str | Path,
    o2_checkpoint: str | Path,
    evaluation: GRUControlEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Tune forecast horizon on validation and evaluate paired test control."""
    evaluation = evaluation or GRUControlEvaluationConfig()
    manifests = {
        "train": _load_manifest(train_data),
        "validation": _load_manifest(validation_data),
        "test": _load_manifest(test_data),
    }
    seed_sets = [set(manifest.seeds) for manifest in manifests.values()]
    if any(
        left & right
        for index, left in enumerate(seed_sets)
        for right in seed_sets[index + 1 :]
    ):
        raise ValueError("train, validation, and test seed blocks must be disjoint")
    validation_variants = _scenario_variants(manifests["validation"])
    test_variants = _scenario_variants(manifests["test"])
    checkpoints = {
        ObservationProfile.SERVO_AWARE: Path(o1_checkpoint),
        ObservationProfile.DISTURBANCE_AWARE: Path(o2_checkpoint),
    }
    models = {
        profile: _load_profile_checkpoint(
            path, profile, manifests, evaluation.device
        )
        for profile, path in checkpoints.items()
    }

    selected_horizons: dict[tuple[ObservationProfile, GimbalCommandMode], int] = {}
    validation_selection: dict[str, Any] = {}
    for profile in LEARNED_PROFILES:
        validation_selection[profile.value] = {}
        for command_mode in (
            GimbalCommandMode.RATE,
            GimbalCommandMode.POSITION,
        ):
            if (
                command_mode is GimbalCommandMode.POSITION
                and not evaluation.include_position_diagnostic
            ):
                continue
            selected, candidates = _select_horizon(
                variants=validation_variants,
                model=models[profile],
                profile=profile,
                command_mode=command_mode,
                evaluation=evaluation,
            )
            selected_horizons[(profile, command_mode)] = selected
            validation_selection[profile.value][command_mode.value] = {
                "selected_horizon_index": selected,
                "selected_horizon_s": (
                    models[profile].config.prediction_horizons_s[selected]
                ),
                "candidates": candidates,
            }

    training_command_modes = _command_modes_from_behaviors(
        manifests["train"].behavior_names
    )
    run_records: list[dict[str, Any]] = []
    runs_by_controller: dict[str, list[ControllerRun]] = {}
    runs_by_scenario: dict[str, dict[str, list[ControllerRun]]] = {}
    for seed, scenario_index, scenario in test_variants:
        runs = list(_baseline_runs(scenario, seed, evaluation))
        for profile in LEARNED_PROFILES:
            rate_horizon_index = selected_horizons[
                (profile, GimbalCommandMode.RATE)
            ]
            runs.append(
                _learned_run(
                    scenario=scenario,
                    seed=seed,
                    model=models[profile],
                    profile=profile,
                    horizon_index=rate_horizon_index,
                    command_mode=GimbalCommandMode.RATE,
                    evaluation=evaluation,
                    search_fallback=False,
                )
            )
            if evaluation.include_position_diagnostic:
                position_horizon_index = selected_horizons[
                    (profile, GimbalCommandMode.POSITION)
                ]
                runs.append(
                    _learned_run(
                        scenario=scenario,
                        seed=seed,
                        model=models[profile],
                        profile=profile,
                        horizon_index=position_horizon_index,
                        command_mode=GimbalCommandMode.POSITION,
                        evaluation=evaluation,
                        search_fallback=False,
                    )
                )
        for command_mode in (
            GimbalCommandMode.RATE,
            GimbalCommandMode.POSITION,
        ):
            if (
                command_mode is GimbalCommandMode.POSITION
                and not evaluation.include_position_diagnostic
            ):
                continue
            runs.append(
                _learned_run(
                    scenario=scenario,
                    seed=seed,
                    model=models[ObservationProfile.DISTURBANCE_AWARE],
                    profile=ObservationProfile.DISTURBANCE_AWARE,
                    horizon_index=selected_horizons[
                        (ObservationProfile.DISTURBANCE_AWARE, command_mode)
                    ],
                    command_mode=command_mode,
                    evaluation=evaluation,
                    search_fallback=True,
                )
            )
        for run in runs:
            name = run.episode.name
            runs_by_controller.setdefault(name, []).append(run)
            runs_by_scenario.setdefault(scenario.name, {}).setdefault(
                name, []
            ).append(run)
            mode = run.episode.config.command_mode
            run_records.append(
                {
                    "seed": seed,
                    "scenario_index": scenario_index,
                    "scenario_name": scenario.name,
                    "controller": name,
                    "command_mode": mode.value,
                    "training_command_mode_supported": (
                        mode.value in training_command_modes
                    ),
                    "tracking_metrics": asdict(run.metrics),
                    "estimator_metrics": (
                        asdict(run.estimator_metrics)
                        if run.estimator_metrics is not None
                        else None
                    ),
                    "control_cost": _control_cost(run),
                }
            )

    aggregates = {
        name: _aggregate_runs(runs)
        for name, runs in runs_by_controller.items()
    }
    scenario_aggregates = {
        scenario_name: {
            name: _aggregate_runs(runs)
            for name, runs in controller_runs.items()
        }
        for scenario_name, controller_runs in runs_by_scenario.items()
    }
    paired_references = {
        "gru_o1_rate": "analytical_rate",
        "gru_o2_rate": "analytical_rate",
        "gru_o2_rate_search": "analytical_rate",
    }
    if evaluation.include_position_diagnostic:
        paired_references.update(
            {
                "gru_o1_position": "analytical_position",
                "gru_o2_position": "analytical_position",
                "gru_o2_position_search": "analytical_position",
            }
        )
    paired_comparisons = {
        name: {
            "reference": reference,
            "metrics": _paired_comparison(
                runs_by_controller[name], runs_by_controller[reference]
            ),
        }
        for name, reference in paired_references.items()
    }
    summary = {
        name: {
            "mean_absolute_error_deg": aggregate["mean_metrics"][
                "mean_absolute_error_deg"
            ],
            "p95_absolute_error_deg": aggregate["mean_metrics"][
                "p95_absolute_error_deg"
            ],
            "loss_of_view_fraction": aggregate["mean_metrics"][
                "loss_of_view_fraction"
            ],
            "rate_saturation_fraction": aggregate["mean_metrics"][
                "rate_saturation_fraction"
            ],
            "command_rms_normalized": aggregate["mean_metrics"][
                "command_rms_normalized"
            ],
            "command_variation_per_s": aggregate["mean_metrics"][
                "command_variation_per_s"
            ],
            "mean_control_cost": aggregate["mean_control_cost"],
            "total_unrecovered_loss_events": aggregate[
                "total_unrecovered_loss_events"
            ],
            "event_weighted_mean_recovery_time_s": aggregate[
                "event_weighted_mean_recovery_time_s"
            ],
        }
        for name, aggregate in aggregates.items()
    }
    return {
        "experiment": "gimbal_gru_closed_loop_comparison_v1",
        "evaluation_config": asdict(evaluation),
        "dataset_hashes": {
            split: manifest.configuration_hash
            for split, manifest in manifests.items()
        },
        "checkpoint_paths": {
            profile.value: str(path) for profile, path in checkpoints.items()
        },
        "training_behavior_names": list(manifests["train"].behavior_names),
        "training_command_modes": list(training_command_modes),
        "position_results_are_out_of_support": (
            GimbalCommandMode.POSITION.value not in training_command_modes
        ),
        "validation_horizon_selection": validation_selection,
        "test_variant_count": len(test_variants),
        "summary": summary,
        "aggregates": aggregates,
        "scenario_aggregates": scenario_aggregates,
        "paired_comparisons": paired_comparisons,
        "runs": run_records,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained gimbal GRUs in paired closed-loop control."
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--o1-checkpoint", type=Path, required=True)
    parser.add_argument("--o2-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-staleness", type=float, default=0.50)
    parser.add_argument("--rate-feedback-gain", type=float, default=2.5)
    parser.add_argument("--search-rate", type=float, default=0.25)
    parser.add_argument("--search-position", type=float, default=0.90)
    parser.add_argument("--search-margin-deg", type=float, default=2.0)
    parser.add_argument("--skip-position-diagnostic", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_gru_closed_loop(
        train_data=args.train_data,
        validation_data=args.validation_data,
        test_data=args.test_data,
        o1_checkpoint=args.o1_checkpoint,
        o2_checkpoint=args.o2_checkpoint,
        evaluation=GRUControlEvaluationConfig(
            rate_feedback_gain_s_inv=args.rate_feedback_gain,
            maximum_staleness_s=args.maximum_staleness,
            search_rate_normalized=args.search_rate,
            search_position_fraction=args.search_position,
            search_reversal_margin_rad=math.radians(args.search_margin_deg),
            device=args.device,
            include_position_diagnostic=not args.skip_position_diagnostic,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {result['test_variant_count']} paired variants to {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
