"""Development/test protocol for uncertainty-aware gimbal recovery.

This module requires the optional ``learning`` dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .closed_loop import ClosedLoopScenario, ControllerRun
from .config import GimbalCommandMode
from .randomization import (
    GimbalDomainRandomizationConfig,
    randomize_closed_loop_scenario,
)
from .recovery import BeliefRecoveryConfig
from .recovery_evaluation import (
    RecoveryEvaluationConfig,
    _aggregate_recovery,
    _aggregate_runs,
    _load_o2_model,
    _recovery_diagnostics,
    _run_strategy,
    _validate_uncertainty_calibration,
    evaluate_belief_recovery,
)
from .recovery_scenarios import (
    EXPANDED_RECOVERY_SUITE_VERSION,
    expanded_recovery_scenarios,
    recovery_domain_randomization,
    recovery_scenarios,
)
from .uncertainty_calibration import (
    UncertaintyCalibration,
    load_uncertainty_calibration,
)


RECOVERY_PROTOCOL_SCHEMA_VERSION = "gimbal_recovery_development_test_v1"
RECOVERY_ROBUSTNESS_SCHEMA_VERSION = (
    "gimbal_recovery_robustness_development_test_v1"
)


@dataclass(frozen=True)
class RecoveryDevelopmentTestConfig:
    development_seeds: tuple[int, ...] = tuple(range(42000, 42008))
    test_seeds: tuple[int, ...] = tuple(range(43000, 43008))
    maximum_coast_candidates_s: tuple[float, ...] = (0.45, 0.65, 0.85)
    maximum_coast_bearing_std_candidates_rad: tuple[float, ...] = (
        math.radians(12.0),
        math.radians(18.0),
        math.radians(24.0),
    )
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("development_seeds", "test_seeds"):
            seeds = getattr(self, name)
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError(f"{name} must be non-empty and unique")
            if any(seed < 0 for seed in seeds):
                raise ValueError(f"{name} must be non-negative")
        if set(self.development_seeds) & set(self.test_seeds):
            raise ValueError("development and test seed blocks must be disjoint")
        for name in (
            "maximum_coast_candidates_s",
            "maximum_coast_bearing_std_candidates_rad",
        ):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be non-empty and unique")
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{name} must be finite and positive")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_belief_configs(
    protocol: RecoveryDevelopmentTestConfig,
    base: BeliefRecoveryConfig,
) -> tuple[BeliefRecoveryConfig, ...]:
    candidates = []
    for maximum_coast_s in protocol.maximum_coast_candidates_s:
        if maximum_coast_s < base.dropout_grace_s:
            raise ValueError("coast candidate is shorter than dropout grace")
        for maximum_std in (
            protocol.maximum_coast_bearing_std_candidates_rad
        ):
            candidates.append(
                replace(
                    base,
                    maximum_coast_s=maximum_coast_s,
                    maximum_coast_bearing_std_rad=maximum_std,
                )
            )
    return tuple(candidates)


def _candidate_summary(
    runs: list[ControllerRun], recovery_records: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = _tracking_summary(runs)
    recovery = _aggregate_recovery(recovery_records)
    return {
        **summary,
        "search_while_target_visible_fraction": recovery[
            "mean_search_while_target_visible_fraction"
        ],
        "mean_state_fraction": recovery["mean_state_fraction"],
    }


def _tracking_summary(runs: list[ControllerRun]) -> dict[str, Any]:
    aggregate = _aggregate_runs(runs)
    return {
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


def _selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    """Prefer recoverability first, then control cost and false search."""
    return (
        float(summary["total_unrecovered_loss_events"]),
        float(summary["mean_control_cost"]),
        float(summary["search_while_target_visible_fraction"]),
        float(summary["p95_absolute_error_deg"]),
    )


def run_recovery_development_test(
    *,
    o2_checkpoint: str | Path,
    control_results: str | Path,
    uncertainty_calibration: UncertaintyCalibration,
    protocol: RecoveryDevelopmentTestConfig | None = None,
    base_belief: BeliefRecoveryConfig | None = None,
    scenarios: tuple[ClosedLoopScenario, ...] | None = None,
    randomization: GimbalDomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    """Tune on development seeds, freeze, then evaluate test exactly once."""
    protocol = protocol or RecoveryDevelopmentTestConfig()
    base_belief = base_belief or BeliefRecoveryConfig()
    scenarios = recovery_scenarios() if scenarios is None else scenarios
    randomization = (
        recovery_domain_randomization()
        if randomization is None
        else randomization
    )
    if not scenarios:
        raise ValueError("at least one recovery scenario is required")
    model, horizon_indices, dataset_hashes = _load_o2_model(
        o2_checkpoint, control_results, protocol.device
    )
    _validate_uncertainty_calibration(
        uncertainty_calibration,
        model=model,
        dataset_hashes=dataset_hashes,
        checkpoint=o2_checkpoint,
    )
    development_variants = tuple(
        (
            seed,
            randomize_closed_loop_scenario(
                scenario,
                seed=seed,
                config=randomization,
            ),
        )
        for seed in protocol.development_seeds
        for scenario in scenarios
    )
    candidate_records = []
    for candidate_index, belief in enumerate(
        _candidate_belief_configs(protocol, base_belief)
    ):
        evaluation = RecoveryEvaluationConfig(
            seeds=protocol.development_seeds,
            belief=belief,
            randomization=randomization,
            uncertainty_calibration=uncertainty_calibration,
            include_position=False,
            device=protocol.device,
        )
        runs = []
        recovery_records = []
        for seed, scenario in development_variants:
            run, controller = _run_strategy(
                scenario=scenario,
                seed=seed,
                model=model,
                horizon_index=horizon_indices[GimbalCommandMode.RATE],
                estimator_kind="gru_o2",
                command_mode=GimbalCommandMode.RATE,
                strategy="belief",
                evaluation=evaluation,
            )
            if controller is None:
                raise RuntimeError("belief candidate has no recovery controller")
            runs.append(run)
            recovery_records.append(_recovery_diagnostics(run, controller))
        summary = _candidate_summary(runs, recovery_records)
        candidate_records.append(
            {
                "candidate_index": candidate_index,
                "belief_config": asdict(belief),
                "selection_key": list(_selection_key(summary)),
                "summary": summary,
            }
        )

    selected = min(
        candidate_records,
        key=lambda record: tuple(record["selection_key"]),
    )
    selected_belief = BeliefRecoveryConfig(**selected["belief_config"])
    test_evaluation = RecoveryEvaluationConfig(
        seeds=protocol.test_seeds,
        belief=selected_belief,
        randomization=randomization,
        uncertainty_calibration=uncertainty_calibration,
        include_position=False,
        device=protocol.device,
    )
    test_result = evaluate_belief_recovery(
        o2_checkpoint=o2_checkpoint,
        control_results=control_results,
        evaluation=test_evaluation,
        scenarios=scenarios,
    )
    return {
        "experiment": RECOVERY_PROTOCOL_SCHEMA_VERSION,
        "protocol_config": asdict(protocol),
        "selection_rule": (
            "lexicographic: unrecovered events, mean control cost, "
            "search while visible, p95 error"
        ),
        "o2_checkpoint": str(o2_checkpoint),
        "o2_checkpoint_sha256": _sha256(o2_checkpoint),
        "control_results": str(control_results),
        "control_results_sha256": _sha256(control_results),
        "uncertainty_calibration_schema": (
            uncertainty_calibration.schema_version
        ),
        "uncertainty_calibration": asdict(uncertainty_calibration),
        "dataset_hashes": dataset_hashes,
        "development_world_variant_count": len(development_variants),
        "development_candidates": candidate_records,
        "selected_candidate_index": selected["candidate_index"],
        "selected_belief_config": selected["belief_config"],
        "test_access_policy": (
            "test seeds evaluated once after development selection was frozen"
        ),
        "test_result": test_result,
    }


def recovery_robustness_protocol_config(
    *, device: str = "cpu"
) -> RecoveryDevelopmentTestConfig:
    """Return the fresh seed blocks and denser threshold grid for robustness."""
    return RecoveryDevelopmentTestConfig(
        development_seeds=tuple(range(44000, 44008)),
        test_seeds=tuple(range(45000, 45008)),
        maximum_coast_candidates_s=(0.35, 0.50, 0.65, 0.80),
        maximum_coast_bearing_std_candidates_rad=tuple(
            math.radians(value) for value in (10.0, 14.0, 18.0, 24.0)
        ),
        device=device,
    )


def _scenario_tracking_summaries(
    runs_by_scenario: dict[str, list[ControllerRun]],
) -> dict[str, dict[str, Any]]:
    return {
        scenario_name: _tracking_summary(runs)
        for scenario_name, runs in runs_by_scenario.items()
    }


def _recoverability_gate(
    *,
    candidate_summary: dict[str, Any],
    candidate_scenarios: dict[str, dict[str, Any]],
    hold_summary: dict[str, Any],
    hold_scenarios: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_names = set(candidate_scenarios)
    hold_names = set(hold_scenarios)
    if candidate_names != hold_names:
        raise ValueError("candidate and hold scenario summaries differ")
    scenario_deltas = {
        name: int(
            candidate_scenarios[name]["total_unrecovered_loss_events"]
            - hold_scenarios[name]["total_unrecovered_loss_events"]
        )
        for name in sorted(candidate_names)
    }
    total_delta = int(
        candidate_summary["total_unrecovered_loss_events"]
        - hold_summary["total_unrecovered_loss_events"]
    )
    maximum_scenario_delta = max(scenario_deltas.values(), default=0)
    return {
        "rule": (
            "candidate unrecovered events must not exceed native hold in "
            "aggregate or in any individual scenario family"
        ),
        "passed": total_delta <= 0 and maximum_scenario_delta <= 0,
        "total_unrecovered_event_delta_vs_hold": total_delta,
        "maximum_scenario_unrecovered_event_delta_vs_hold": (
            maximum_scenario_delta
        ),
        "scenario_unrecovered_event_delta_vs_hold": scenario_deltas,
    }


def _robustness_selection_key(
    record: dict[str, Any],
) -> tuple[float, ...]:
    summary = record["summary"]
    gate = record["development_recoverability_gate"]
    return (
        0.0 if gate["passed"] else 1.0,
        float(max(0, gate["maximum_scenario_unrecovered_event_delta_vs_hold"])),
        float(summary["total_unrecovered_loss_events"]),
        float(summary["mean_control_cost"]),
        float(summary["search_while_target_visible_fraction"]),
        float(summary["p95_absolute_error_deg"]),
    )


def run_recovery_robustness_development_test(
    *,
    o2_checkpoint: str | Path,
    control_results: str | Path,
    uncertainty_calibration: UncertaintyCalibration,
    protocol: RecoveryDevelopmentTestConfig | None = None,
    base_belief: BeliefRecoveryConfig | None = None,
    scenarios: tuple[ClosedLoopScenario, ...] | None = None,
    randomization: GimbalDomainRandomizationConfig | None = None,
) -> dict[str, Any]:
    """Select with a per-scenario safety gate, then access fresh test once."""
    protocol = protocol or recovery_robustness_protocol_config()
    base_belief = base_belief or BeliefRecoveryConfig()
    scenarios = (
        expanded_recovery_scenarios() if scenarios is None else scenarios
    )
    randomization = (
        recovery_domain_randomization()
        if randomization is None
        else randomization
    )
    if not scenarios:
        raise ValueError("at least one recovery scenario is required")
    scenario_names = tuple(scenario.name for scenario in scenarios)
    if len(scenario_names) != len(set(scenario_names)):
        raise ValueError("recovery scenario names must be unique")

    model, horizon_indices, dataset_hashes = _load_o2_model(
        o2_checkpoint, control_results, protocol.device
    )
    _validate_uncertainty_calibration(
        uncertainty_calibration,
        model=model,
        dataset_hashes=dataset_hashes,
        checkpoint=o2_checkpoint,
    )
    development_variants = tuple(
        (
            seed,
            randomize_closed_loop_scenario(
                scenario,
                seed=seed,
                config=randomization,
            ),
        )
        for seed in protocol.development_seeds
        for scenario in scenarios
    )

    hold_evaluation = RecoveryEvaluationConfig(
        seeds=protocol.development_seeds,
        belief=base_belief,
        randomization=randomization,
        uncertainty_calibration=uncertainty_calibration,
        include_position=False,
        device=protocol.device,
    )
    hold_runs = []
    hold_runs_by_scenario: dict[str, list[ControllerRun]] = {}
    for seed, scenario in development_variants:
        run, _controller = _run_strategy(
            scenario=scenario,
            seed=seed,
            model=model,
            horizon_index=horizon_indices[GimbalCommandMode.RATE],
            estimator_kind="gru_o2",
            command_mode=GimbalCommandMode.RATE,
            strategy="hold",
            evaluation=hold_evaluation,
        )
        hold_runs.append(run)
        hold_runs_by_scenario.setdefault(scenario.name, []).append(run)
    hold_summary = _tracking_summary(hold_runs)
    hold_scenario_summaries = _scenario_tracking_summaries(
        hold_runs_by_scenario
    )

    candidate_records = []
    for candidate_index, belief in enumerate(
        _candidate_belief_configs(protocol, base_belief)
    ):
        evaluation = replace(hold_evaluation, belief=belief)
        runs = []
        recovery_records = []
        runs_by_scenario: dict[str, list[ControllerRun]] = {}
        for seed, scenario in development_variants:
            run, controller = _run_strategy(
                scenario=scenario,
                seed=seed,
                model=model,
                horizon_index=horizon_indices[GimbalCommandMode.RATE],
                estimator_kind="gru_o2",
                command_mode=GimbalCommandMode.RATE,
                strategy="belief",
                evaluation=evaluation,
            )
            if controller is None:
                raise RuntimeError("belief candidate has no recovery controller")
            runs.append(run)
            runs_by_scenario.setdefault(scenario.name, []).append(run)
            recovery_records.append(_recovery_diagnostics(run, controller))
        summary = _candidate_summary(runs, recovery_records)
        scenario_summaries = _scenario_tracking_summaries(runs_by_scenario)
        gate = _recoverability_gate(
            candidate_summary=summary,
            candidate_scenarios=scenario_summaries,
            hold_summary=hold_summary,
            hold_scenarios=hold_scenario_summaries,
        )
        record = {
            "candidate_index": candidate_index,
            "belief_config": asdict(belief),
            "summary": summary,
            "scenario_summaries": scenario_summaries,
            "development_recoverability_gate": gate,
        }
        record["selection_key"] = list(_robustness_selection_key(record))
        candidate_records.append(record)

    selected = min(
        candidate_records,
        key=lambda record: tuple(record["selection_key"]),
    )
    selected_belief = BeliefRecoveryConfig(**selected["belief_config"])
    development_gate_passed = bool(
        selected["development_recoverability_gate"]["passed"]
    )
    test_evaluation = RecoveryEvaluationConfig(
        seeds=protocol.test_seeds,
        belief=selected_belief,
        randomization=randomization,
        uncertainty_calibration=uncertainty_calibration,
        include_position=False,
        device=protocol.device,
    )
    test_result = evaluate_belief_recovery(
        o2_checkpoint=o2_checkpoint,
        control_results=control_results,
        evaluation=test_evaluation,
        scenarios=scenarios,
    )
    test_hold = test_result["summary"]["gru_o2_rate_hold"]
    test_belief = test_result["summary"]["gru_o2_rate_belief"]
    test_hold_scenarios = {
        name: aggregates["gru_o2_rate_hold"]
        for name, aggregates in test_result["scenario_aggregates"].items()
    }
    test_belief_scenarios = {
        name: aggregates["gru_o2_rate_belief"]
        for name, aggregates in test_result["scenario_aggregates"].items()
    }
    fresh_test_recoverability = _recoverability_gate(
        candidate_summary=test_belief,
        candidate_scenarios=test_belief_scenarios,
        hold_summary=test_hold,
        hold_scenarios=test_hold_scenarios,
    )
    cost_delta = float(
        test_belief["mean_control_cost"] - test_hold["mean_control_cost"]
    )
    fresh_test_gate = {
        "rule": (
            "development-selected belief recovery must retain recoverability "
            "and reduce mean control cost versus native hold on fresh test"
        ),
        "passed": (
            development_gate_passed
            and fresh_test_recoverability["passed"]
            and cost_delta < 0.0
        ),
        "development_gate_passed": development_gate_passed,
        "recoverability": fresh_test_recoverability,
        "mean_control_cost_delta_vs_hold": cost_delta,
        "mean_absolute_error_delta_deg_vs_hold": float(
            test_belief["mean_absolute_error_deg"]
            - test_hold["mean_absolute_error_deg"]
        ),
        "p95_absolute_error_delta_deg_vs_hold": float(
            test_belief["p95_absolute_error_deg"]
            - test_hold["p95_absolute_error_deg"]
        ),
        "loss_of_view_delta_vs_hold": float(
            test_belief["loss_of_view_fraction"]
            - test_hold["loss_of_view_fraction"]
        ),
    }
    deployment_recommendation = (
        "belief_recovery_eligible"
        if fresh_test_gate["passed"]
        else "native_hold"
    )
    return {
        "experiment": RECOVERY_ROBUSTNESS_SCHEMA_VERSION,
        "scenario_suite_version": EXPANDED_RECOVERY_SUITE_VERSION,
        "scenario_names": list(scenario_names),
        "scenario_descriptions": {
            scenario.name: scenario.description for scenario in scenarios
        },
        "protocol_config": asdict(protocol),
        "selection_rule": (
            "pass per-scenario/aggregate recoverability gate, then minimize "
            "maximum scenario event regression, total unrecovered events, "
            "mean control cost, false search, and p95 error"
        ),
        "o2_checkpoint": str(o2_checkpoint),
        "o2_checkpoint_sha256": _sha256(o2_checkpoint),
        "control_results": str(control_results),
        "control_results_sha256": _sha256(control_results),
        "uncertainty_calibration_schema": (
            uncertainty_calibration.schema_version
        ),
        "uncertainty_calibration": asdict(uncertainty_calibration),
        "dataset_hashes": dataset_hashes,
        "development_world_variant_count": len(development_variants),
        "development_hold_summary": hold_summary,
        "development_hold_scenario_summaries": hold_scenario_summaries,
        "development_candidates": candidate_records,
        "selected_candidate_index": selected["candidate_index"],
        "selected_belief_config": selected["belief_config"],
        "selected_candidate_development_gate_passed": (
            development_gate_passed
        ),
        "deployment_strategy_after_development": (
            "belief" if development_gate_passed else "hold"
        ),
        "test_access_policy": (
            f"test seeds {list(protocol.test_seeds)} evaluated once after "
            "the threshold and development safety decision were frozen"
        ),
        "test_result": test_result,
        "fresh_test_gate": fresh_test_gate,
        "deployment_recommendation": deployment_recommendation,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune gimbal recovery on development and test once."
    )
    parser.add_argument("--o2-checkpoint", type=Path, required=True)
    parser.add_argument("--control-results", type=Path, required=True)
    parser.add_argument("--uncertainty-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--development-seed-start", type=int, default=42000)
    parser.add_argument("--test-seed-start", type=int, default=43000)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument(
        "--coast-duration-s",
        type=float,
        action="append",
        dest="coast_durations_s",
    )
    parser.add_argument(
        "--coast-bearing-std-deg",
        type=float,
        action="append",
        dest="coast_std_deg",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _robustness_parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select gimbal recovery on the expanded suite with a native-hold "
            "safety gate, then test once."
        )
    )
    parser.add_argument("--o2-checkpoint", type=Path, required=True)
    parser.add_argument("--control-results", type=Path, required=True)
    parser.add_argument("--uncertainty-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--development-seed-start", type=int, default=44000)
    parser.add_argument("--test-seed-start", type=int, default=45000)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument(
        "--coast-duration-s",
        type=float,
        action="append",
        dest="coast_durations_s",
    )
    parser.add_argument(
        "--coast-bearing-std-deg",
        type=float,
        action="append",
        dest="coast_std_deg",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    development_seeds = tuple(
        range(
            args.development_seed_start,
            args.development_seed_start + args.episodes,
        )
    )
    test_seeds = tuple(
        range(args.test_seed_start, args.test_seed_start + args.episodes)
    )
    defaults = RecoveryDevelopmentTestConfig()
    protocol = RecoveryDevelopmentTestConfig(
        development_seeds=development_seeds,
        test_seeds=test_seeds,
        maximum_coast_candidates_s=tuple(
            args.coast_durations_s or defaults.maximum_coast_candidates_s
        ),
        maximum_coast_bearing_std_candidates_rad=tuple(
            math.radians(value)
            for value in (
                args.coast_std_deg
                or tuple(
                    math.degrees(value)
                    for value in defaults.maximum_coast_bearing_std_candidates_rad
                )
            )
        ),
        device=args.device,
    )
    calibration = load_uncertainty_calibration(
        args.uncertainty_calibration
    )
    result = run_recovery_development_test(
        o2_checkpoint=args.o2_checkpoint,
        control_results=args.control_results,
        uncertainty_calibration=calibration,
        protocol=protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.write_text(
        json.dumps(result["test_result"], indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_candidate_index": result[
                    "selected_candidate_index"
                ],
                "selected_belief_config": result["selected_belief_config"],
                "test_summary": result["test_result"]["summary"],
            },
            indent=2,
        )
    )


def robustness_main(argv: Sequence[str] | None = None) -> None:
    args = _robustness_parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    development_seeds = tuple(
        range(
            args.development_seed_start,
            args.development_seed_start + args.episodes,
        )
    )
    test_seeds = tuple(
        range(args.test_seed_start, args.test_seed_start + args.episodes)
    )
    defaults = recovery_robustness_protocol_config(device=args.device)
    protocol = RecoveryDevelopmentTestConfig(
        development_seeds=development_seeds,
        test_seeds=test_seeds,
        maximum_coast_candidates_s=tuple(
            args.coast_durations_s or defaults.maximum_coast_candidates_s
        ),
        maximum_coast_bearing_std_candidates_rad=tuple(
            math.radians(value)
            for value in (
                args.coast_std_deg
                or tuple(
                    math.degrees(value)
                    for value in defaults.maximum_coast_bearing_std_candidates_rad
                )
            )
        ),
        device=args.device,
    )
    calibration = load_uncertainty_calibration(
        args.uncertainty_calibration
    )
    result = run_recovery_robustness_development_test(
        o2_checkpoint=args.o2_checkpoint,
        control_results=args.control_results,
        uncertainty_calibration=calibration,
        protocol=protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.write_text(
        json.dumps(result["test_result"], indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_candidate_index": result[
                    "selected_candidate_index"
                ],
                "selected_belief_config": result[
                    "selected_belief_config"
                ],
                "development_gate_passed": result[
                    "selected_candidate_development_gate_passed"
                ],
                "fresh_test_gate": result["fresh_test_gate"],
                "deployment_recommendation": result[
                    "deployment_recommendation"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
