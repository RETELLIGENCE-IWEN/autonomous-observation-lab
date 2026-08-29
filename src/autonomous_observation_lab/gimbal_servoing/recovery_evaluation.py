"""Paired evaluation of belief-guided gimbal loss-of-view recovery.

This module requires the optional ``learning`` dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .closed_loop import (
    ClosedLoopScenario,
    ControllerRun,
    TrackingMetrics,
    run_closed_loop_controller,
)
from .config import GimbalCommandMode, ObservationProfile
from .controllers import (
    SearchFallbackController,
    TargetStatePositionController,
    TargetStateRateController,
)
from .dataset import FEATURE_NAMES, _canonical
from .gru import (
    CausalTargetStateGRU,
    GRUInferenceConfig,
    GRUTargetStateEstimator,
    load_gru_checkpoint,
)
from .gru_control import _analytical_estimator, _control_cost
from .randomization import (
    GimbalDomainRandomizationConfig,
    randomize_closed_loop_scenario,
)
from .recovery import (
    BeliefRecoveryConfig,
    BeliefRecoveryController,
    RecoveryState,
    RecoveryTransition,
)
from .recovery_scenarios import (
    recovery_domain_randomization,
    recovery_scenarios,
)
from .serialization import closed_loop_scenario_from_dict
from .uncertainty_calibration import (
    UncertaintyCalibration,
    load_uncertainty_calibration,
    uncertainty_calibration_from_dict,
)


EstimatorKind = Literal["analytical", "gru_o2"]
RecoveryStrategy = Literal["hold", "blind", "belief"]


@dataclass(frozen=True)
class RecoveryEvaluationConfig:
    seeds: tuple[int, ...] = tuple(range(41000, 41008))
    rate_feedback_gain_s_inv: float = 2.5
    maximum_staleness_s: float = 0.50
    blind_search_rate_normalized: float = 0.25
    blind_search_position_fraction: float = 0.90
    search_boundary_margin_rad: float = math.radians(2.0)
    belief: BeliefRecoveryConfig = BeliefRecoveryConfig()
    randomization: GimbalDomainRandomizationConfig = (
        recovery_domain_randomization()
    )
    uncertainty_calibration: UncertaintyCalibration | None = None
    include_position: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("evaluation seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("evaluation seeds must be non-negative")
        if self.rate_feedback_gain_s_inv < 0.0:
            raise ValueError("rate feedback gain must be non-negative")
        if self.maximum_staleness_s < 0.0:
            raise ValueError("maximum staleness must be non-negative")
        for name in (
            "blind_search_rate_normalized",
            "blind_search_position_fraction",
        ):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.search_boundary_margin_rad < 0.0:
            raise ValueError("search boundary margin must be non-negative")


@dataclass(frozen=True)
class RecoveryReplayRun:
    run: ControllerRun
    state_trace: tuple[tuple[float, RecoveryState], ...]
    transitions: tuple[RecoveryTransition, ...]


@dataclass(frozen=True)
class RecoveryReplay:
    scenario: ClosedLoopScenario
    seed: int
    estimator_kind: EstimatorKind
    command_mode: GimbalCommandMode
    runs: tuple[RecoveryReplayRun, ...]
    aggregate_summary: dict[str, dict[str, Any]]
    variant_count: int
    source_results: Path


def _load_control_selection(
    path: str | Path,
    checkpoint_metadata: dict[str, Any],
    model: CausalTargetStateGRU,
) -> dict[GimbalCommandMode, int]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    result_hashes = result.get("dataset_hashes")
    checkpoint_hashes = checkpoint_metadata.get("dataset_hashes")
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError("checkpoint is missing dataset hashes")
    if not isinstance(result_hashes, dict):
        raise ValueError("control result is missing dataset hashes")
    if checkpoint_hashes != result_hashes:
        raise ValueError("control result and checkpoint dataset hashes differ")
    raw_profile = result.get("validation_horizon_selection", {}).get(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    if not isinstance(raw_profile, dict):
        raise ValueError("control result has no O2 validation selection")
    selected: dict[GimbalCommandMode, int] = {}
    for mode in (GimbalCommandMode.RATE, GimbalCommandMode.POSITION):
        raw_mode = raw_profile.get(mode.value)
        if not isinstance(raw_mode, dict):
            raise ValueError(f"control result has no {mode.value} selection")
        index = int(raw_mode["selected_horizon_index"])
        if not 0 <= index < model.horizon_count:
            raise ValueError("selected horizon index is outside checkpoint")
        reported_horizon = float(raw_mode["selected_horizon_s"])
        actual_horizon = model.config.prediction_horizons_s[index]
        if not math.isclose(reported_horizon, actual_horizon):
            raise ValueError("selected horizon value is inconsistent")
        selected[mode] = index
    return selected


def _load_o2_model(
    checkpoint: str | Path,
    control_results: str | Path,
    device: str,
) -> tuple[
    CausalTargetStateGRU,
    dict[GimbalCommandMode, int],
    dict[str, str],
]:
    model, metadata = load_gru_checkpoint(checkpoint, device=device)
    if metadata.get("profile") != ObservationProfile.DISTURBANCE_AWARE.value:
        raise ValueError("recovery evaluation requires an O2 checkpoint")
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("checkpoint feature schema does not match deployment")
    selected = _load_control_selection(
        control_results, metadata, model
    )
    return model, selected, dict(metadata["dataset_hashes"])


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_uncertainty_calibration(
    calibration: UncertaintyCalibration | None,
    *,
    model: CausalTargetStateGRU,
    dataset_hashes: dict[str, str],
    checkpoint: str | Path,
) -> None:
    if calibration is None:
        return
    if calibration.profile is not ObservationProfile.DISTURBANCE_AWARE:
        raise ValueError("recovery requires an O2 uncertainty calibration")
    if calibration.prediction_horizons_s != model.config.prediction_horizons_s:
        raise ValueError("calibration and checkpoint horizons differ")
    if calibration.checkpoint_sha256 != _sha256(checkpoint):
        raise ValueError("calibration checkpoint checksum mismatch")
    if (
        calibration.validation_dataset_hash
        != dataset_hashes.get("validation")
        or calibration.test_dataset_hash != dataset_hashes.get("test")
    ):
        raise ValueError("calibration and checkpoint dataset hashes differ")


def _target_state_controller(
    *,
    scenario: ClosedLoopScenario,
    model: CausalTargetStateGRU,
    horizon_index: int,
    estimator_kind: EstimatorKind,
    command_mode: GimbalCommandMode,
    evaluation: RecoveryEvaluationConfig,
) -> TargetStateRateController | TargetStatePositionController:
    config = replace(
        scenario.config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        command_mode=command_mode,
    )
    if estimator_kind == "analytical":
        estimator = _analytical_estimator(scenario)
    else:
        estimator = GRUTargetStateEstimator(
            model,
            config,
            GRUInferenceConfig(
                observation_profile=ObservationProfile.DISTURBANCE_AWARE,
                horizon_index=horizon_index,
                maximum_staleness_s=evaluation.maximum_staleness_s,
                uncertainty_calibration=evaluation.uncertainty_calibration,
                device=evaluation.device,
            ),
        )
    name = f"{estimator_kind}_{command_mode.name.lower()}"
    if command_mode is GimbalCommandMode.RATE:
        return TargetStateRateController(
            estimator=estimator,
            max_rate_rad_s=config.servo.max_rate_rad_s,
            proportional_gain_s_inv=evaluation.rate_feedback_gain_s_inv,
            name=name,
        )
    preview_s = (
        config.servo.command_latency_s + config.servo.rate_time_constant_s
        if estimator_kind == "analytical"
        else 0.0
    )
    return TargetStatePositionController(
        estimator=estimator,
        servo=config.servo,
        command_preview_s=preview_s,
        name=name,
    )


def _run_strategy(
    *,
    scenario: ClosedLoopScenario,
    seed: int,
    model: CausalTargetStateGRU,
    horizon_index: int,
    estimator_kind: EstimatorKind,
    command_mode: GimbalCommandMode,
    strategy: RecoveryStrategy,
    evaluation: RecoveryEvaluationConfig,
) -> tuple[ControllerRun, BeliefRecoveryController | None]:
    config = replace(
        scenario.config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        command_mode=command_mode,
    )
    delegate = _target_state_controller(
        scenario=scenario,
        model=model,
        horizon_index=horizon_index,
        estimator_kind=estimator_kind,
        command_mode=command_mode,
        evaluation=evaluation,
    )
    name = f"{estimator_kind}_{command_mode.name.lower()}_{strategy}"
    belief_controller: BeliefRecoveryController | None = None
    if strategy == "hold":
        controller = delegate
        description = "Native estimator staleness behavior with no search."
    elif strategy == "blind":
        controller = SearchFallbackController(
            delegate=delegate,
            servo=config.servo,
            command_mode=command_mode,
            search_rate_normalized=(
                evaluation.blind_search_rate_normalized
            ),
            search_position_fraction=(
                evaluation.blind_search_position_fraction
            ),
            reversal_margin_rad=evaluation.search_boundary_margin_rad,
            name=name,
        )
        description = "Travel-envelope sweep after estimator invalidation."
    else:
        belief_controller = BeliefRecoveryController(
            delegate=delegate,
            servo=config.servo,
            command_mode=command_mode,
            recovery=evaluation.belief,
            name=name,
        )
        controller = belief_controller
        description = "Belief-guided coast, bounded search, and reacquisition."
    run = run_closed_loop_controller(
        name=name,
        description=description,
        scenario=scenario,
        config=config,
        controller=controller,
        seed=seed,
    )
    return run, belief_controller


def _recovery_diagnostics(
    run: ControllerRun,
    controller: BeliefRecoveryController,
) -> dict[str, Any]:
    trace = controller.state_trace
    state_counts = {
        state.value: sum(current is state for _, current in trace)
        for state in RecoveryState
    }
    count = max(1, len(trace))
    visible = [
        frame.diagnostics.target_in_view
        for frame in run.episode.frames[: len(trace)]
    ]
    false_search_steps = sum(
        state is RecoveryState.SEARCH and target_visible
        for (_, state), target_visible in zip(trace, visible, strict=True)
    )
    visible_steps = sum(visible)
    action_by_time = {
        round(time_s, 9): action.command_normalized
        for time_s, action in controller.action_trace
    }
    trace_times = [round(time_s, 9) for time_s, _ in trace]
    reacquire_jumps: list[float] = []
    for transition in controller.transitions:
        if transition.current is not RecoveryState.REACQUIRE:
            continue
        key = round(transition.time_s, 9)
        try:
            index = trace_times.index(key)
        except ValueError:
            continue
        if index > 0:
            reacquire_jumps.append(
                abs(
                    action_by_time[key]
                    - action_by_time[trace_times[index - 1]]
                )
            )
    return {
        "state_fraction": {
            state: value / count for state, value in state_counts.items()
        },
        "transition_count": len(controller.transitions),
        "transitions": [asdict(transition) for transition in controller.transitions],
        "search_while_target_visible_fraction": (
            false_search_steps / visible_steps if visible_steps else 0.0
        ),
        "maximum_reacquire_command_jump_normalized": (
            max(reacquire_jumps) if reacquire_jumps else 0.0
        ),
    }


def _aggregate_runs(runs: list[ControllerRun]) -> dict[str, Any]:
    metric_names = [field.name for field in fields(TrackingMetrics)]
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
        "total_loss_of_view_events": int(
            sum(run.metrics.loss_of_view_events for run in runs)
        ),
        "total_unrecovered_loss_events": int(
            sum(run.metrics.unrecovered_loss_events for run in runs)
        ),
        "event_weighted_mean_recovery_time_s": (
            weighted_recovery / total_recovered if total_recovered else 0.0
        ),
        "mean_metrics": {
            name: float(np.mean([getattr(run.metrics, name) for run in runs]))
            for name in metric_names
        },
    }


def _aggregate_recovery(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(records),
        "mean_state_fraction": {
            state.value: float(
                np.mean(
                    [record["state_fraction"][state.value] for record in records]
                )
            )
            for state in RecoveryState
        },
        "mean_transition_count": float(
            np.mean([record["transition_count"] for record in records])
        ),
        "mean_search_while_target_visible_fraction": float(
            np.mean(
                [
                    record["search_while_target_visible_fraction"]
                    for record in records
                ]
            )
        ),
        "maximum_reacquire_command_jump_normalized": max(
            record["maximum_reacquire_command_jump_normalized"]
            for record in records
        ),
    }


def _paired_delta(
    candidates: list[ControllerRun], references: list[ControllerRun]
) -> dict[str, Any]:
    if len(candidates) != len(references) or not candidates:
        raise ValueError("paired runs must have equal non-zero length")
    metric_names = (
        "mean_absolute_error_deg",
        "p95_absolute_error_deg",
        "loss_of_view_fraction",
        "mean_recovery_time_s",
        "unrecovered_loss_events",
        "command_variation_per_s",
    )
    result: dict[str, Any] = {}

    def standard_error(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=1) / math.sqrt(len(values)))

    for name in metric_names:
        deltas = np.asarray(
            [
                getattr(candidate.metrics, name)
                - getattr(reference.metrics, name)
                for candidate, reference in zip(
                    candidates, references, strict=True
                )
            ]
        )
        result[name] = {
            "mean_delta": float(np.mean(deltas)),
            "standard_error": standard_error(deltas),
            "candidate_win_fraction": float(np.mean(deltas < 0.0)),
        }
    cost_deltas = np.asarray(
        [
            _control_cost(candidate) - _control_cost(reference)
            for candidate, reference in zip(candidates, references, strict=True)
        ]
    )
    result["control_cost"] = {
        "mean_delta": float(np.mean(cost_deltas)),
        "standard_error": standard_error(cost_deltas),
        "candidate_win_fraction": float(np.mean(cost_deltas < 0.0)),
    }
    return result


def evaluate_belief_recovery(
    *,
    o2_checkpoint: str | Path,
    control_results: str | Path,
    evaluation: RecoveryEvaluationConfig | None = None,
    scenarios: tuple[ClosedLoopScenario, ...] | None = None,
) -> dict[str, Any]:
    """Compare hold, blind sweep, and belief recovery on paired variants."""
    evaluation = evaluation or RecoveryEvaluationConfig()
    selected_scenarios = (
        recovery_scenarios() if scenarios is None else scenarios
    )
    if not selected_scenarios:
        raise ValueError("at least one recovery scenario is required")
    model, horizon_indices, dataset_hashes = _load_o2_model(
        o2_checkpoint, control_results, evaluation.device
    )
    _validate_uncertainty_calibration(
        evaluation.uncertainty_calibration,
        model=model,
        dataset_hashes=dataset_hashes,
        checkpoint=o2_checkpoint,
    )
    command_modes = [GimbalCommandMode.RATE]
    if evaluation.include_position:
        command_modes.append(GimbalCommandMode.POSITION)

    runs_by_controller: dict[str, list[ControllerRun]] = {}
    runs_by_scenario: dict[str, dict[str, list[ControllerRun]]] = {}
    recovery_by_controller: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    scenario_variants: list[dict[str, Any]] = []
    for seed in evaluation.seeds:
        for scenario_index, base_scenario in enumerate(selected_scenarios):
            scenario = randomize_closed_loop_scenario(
                base_scenario,
                seed=seed,
                config=evaluation.randomization,
            )
            scenario_variants.append(
                {
                    "seed": seed,
                    "scenario_index": scenario_index,
                    "scenario": _canonical(scenario),
                }
            )
            for command_mode in command_modes:
                for estimator_kind in ("analytical", "gru_o2"):
                    for strategy in ("hold", "blind", "belief"):
                        run, belief_controller = _run_strategy(
                            scenario=scenario,
                            seed=seed,
                            model=model,
                            horizon_index=horizon_indices[command_mode],
                            estimator_kind=estimator_kind,
                            command_mode=command_mode,
                            strategy=strategy,
                            evaluation=evaluation,
                        )
                        name = run.episode.name
                        runs_by_controller.setdefault(name, []).append(run)
                        runs_by_scenario.setdefault(
                            scenario.name, {}
                        ).setdefault(name, []).append(run)
                        recovery_record = None
                        if belief_controller is not None:
                            recovery_record = _recovery_diagnostics(
                                run, belief_controller
                            )
                            recovery_by_controller.setdefault(name, []).append(
                                recovery_record
                            )
                        records.append(
                            {
                                "seed": seed,
                                "scenario_index": scenario_index,
                                "scenario_name": scenario.name,
                                "controller": name,
                                "command_mode": command_mode.value,
                                "tracking_metrics": asdict(run.metrics),
                                "estimator_metrics": (
                                    asdict(run.estimator_metrics)
                                    if run.estimator_metrics is not None
                                    else None
                                ),
                                "control_cost": _control_cost(run),
                                "recovery_diagnostics": recovery_record,
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
    recovery_aggregates = {
        name: _aggregate_recovery(controller_records)
        for name, controller_records in recovery_by_controller.items()
    }
    comparisons: dict[str, Any] = {}
    for command_mode in command_modes:
        mode = command_mode.name.lower()
        for estimator_kind in ("analytical", "gru_o2"):
            candidate_name = f"{estimator_kind}_{mode}_belief"
            comparisons[candidate_name] = {}
            for strategy in ("hold", "blind"):
                reference_name = f"{estimator_kind}_{mode}_{strategy}"
                comparisons[candidate_name][strategy] = _paired_delta(
                    runs_by_controller[candidate_name],
                    runs_by_controller[reference_name],
                )
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
        "experiment": "gimbal_belief_recovery_v1",
        "evaluation_config": asdict(evaluation),
        "o2_checkpoint": str(o2_checkpoint),
        "o2_checkpoint_sha256": _sha256(o2_checkpoint),
        "control_results": str(control_results),
        "control_results_sha256": _sha256(control_results),
        "dataset_hashes": dataset_hashes,
        "selected_horizons": {
            mode.value: {
                "index": index,
                "seconds": model.config.prediction_horizons_s[index],
            }
            for mode, index in horizon_indices.items()
        },
        "scenario_names": [scenario.name for scenario in selected_scenarios],
        "scenario_variants": scenario_variants,
        "variant_count": len(evaluation.seeds) * len(selected_scenarios),
        "summary": summary,
        "aggregates": aggregates,
        "scenario_aggregates": scenario_aggregates,
        "recovery_aggregates": recovery_aggregates,
        "paired_comparisons": comparisons,
        "runs": records,
    }


def _replay_evaluation_config(
    result: dict[str, Any], *, device: str
) -> RecoveryEvaluationConfig:
    raw = result.get("evaluation_config")
    if not isinstance(raw, dict):
        raise ValueError("recovery result has no evaluation configuration")
    belief = raw.get("belief")
    if not isinstance(belief, dict):
        raise ValueError("recovery result has no belief configuration")
    raw_calibration = raw.get("uncertainty_calibration")
    calibration = (
        uncertainty_calibration_from_dict(raw_calibration)
        if isinstance(raw_calibration, dict)
        else None
    )
    return RecoveryEvaluationConfig(
        seeds=tuple(int(seed) for seed in raw["seeds"]),
        rate_feedback_gain_s_inv=float(raw["rate_feedback_gain_s_inv"]),
        maximum_staleness_s=float(raw["maximum_staleness_s"]),
        blind_search_rate_normalized=float(
            raw["blind_search_rate_normalized"]
        ),
        blind_search_position_fraction=float(
            raw["blind_search_position_fraction"]
        ),
        search_boundary_margin_rad=float(
            raw["search_boundary_margin_rad"]
        ),
        belief=BeliefRecoveryConfig(**belief),
        uncertainty_calibration=calibration,
        include_position=bool(raw["include_position"]),
        device=device,
    )


def replay_recovery_variant(
    *,
    results: str | Path,
    scenario_name: str,
    seed: int,
    estimator_kind: EstimatorKind = "gru_o2",
    command_mode: GimbalCommandMode = GimbalCommandMode.RATE,
    o2_checkpoint: str | Path | None = None,
    control_results: str | Path | None = None,
    device: str = "cpu",
) -> RecoveryReplay:
    """Replay hold/blind/belief on one exact recorded recovery variant."""
    results_path = Path(results)
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("experiment") != "gimbal_belief_recovery_v1":
        raise ValueError("unsupported recovery result schema")
    checkpoint_path = Path(o2_checkpoint or result["o2_checkpoint"])
    control_path = Path(control_results or result["control_results"])
    if _sha256(checkpoint_path) != result.get("o2_checkpoint_sha256"):
        raise ValueError("O2 checkpoint checksum does not match recovery result")
    if _sha256(control_path) != result.get("control_results_sha256"):
        raise ValueError("control-result checksum does not match recovery result")
    model, horizon_indices, dataset_hashes = _load_o2_model(
        checkpoint_path, control_path, device
    )
    if dataset_hashes != result.get("dataset_hashes"):
        raise ValueError("recovery result dataset hashes are inconsistent")
    selected = result.get("selected_horizons", {}).get(command_mode.value)
    if not isinstance(selected, dict):
        raise ValueError("recovery result has no selected command-mode horizon")
    if int(selected["index"]) != horizon_indices[command_mode]:
        raise ValueError("recorded and validated horizon selections differ")

    matching_variants = [
        entry
        for entry in result.get("scenario_variants", ())
        if int(entry["seed"]) == seed
        and entry.get("scenario", {}).get("name") == scenario_name
    ]
    if len(matching_variants) != 1:
        raise ValueError(
            "recovery result must contain exactly one matching scenario/seed"
        )
    scenario = closed_loop_scenario_from_dict(
        matching_variants[0]["scenario"]
    )
    evaluation = _replay_evaluation_config(result, device=device)
    replay_runs: list[RecoveryReplayRun] = []
    for strategy in ("hold", "blind", "belief"):
        run, belief_controller = _run_strategy(
            scenario=scenario,
            seed=seed,
            model=model,
            horizon_index=horizon_indices[command_mode],
            estimator_kind=estimator_kind,
            command_mode=command_mode,
            strategy=strategy,
            evaluation=evaluation,
        )
        replay_runs.append(
            RecoveryReplayRun(
                run=run,
                state_trace=(
                    tuple(belief_controller.state_trace)
                    if belief_controller is not None
                    else ()
                ),
                transitions=(
                    tuple(belief_controller.transitions)
                    if belief_controller is not None
                    else ()
                ),
            )
        )
    names = {replay.run.episode.name for replay in replay_runs}
    raw_summary = result.get("summary", {})
    aggregate_summary = {
        name: dict(raw_summary[name]) for name in names if name in raw_summary
    }
    return RecoveryReplay(
        scenario=scenario,
        seed=seed,
        estimator_kind=estimator_kind,
        command_mode=command_mode,
        runs=tuple(replay_runs),
        aggregate_summary=aggregate_summary,
        variant_count=int(result.get("variant_count", 0)),
        source_results=results_path,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate belief-guided gimbal loss-of-view recovery."
    )
    parser.add_argument("--o2-checkpoint", type=Path, required=True)
    parser.add_argument("--control-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=41000)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--uncertainty-calibration",
        type=Path,
        help="validation-fit uncertainty calibration JSON for O2 inference",
    )
    parser.add_argument("--skip-position", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    calibration = (
        load_uncertainty_calibration(args.uncertainty_calibration)
        if args.uncertainty_calibration is not None
        else None
    )
    evaluation = RecoveryEvaluationConfig(
        seeds=tuple(range(args.seed_start, args.seed_start + args.episodes)),
        uncertainty_calibration=calibration,
        include_position=not args.skip_position,
        device=args.device,
    )
    result = evaluate_belief_recovery(
        o2_checkpoint=args.o2_checkpoint,
        control_results=args.control_results,
        evaluation=evaluation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {result['variant_count']} paired variants to {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
