"""Development/confirmation protocol for visibility-risk position V2.1.

V2.1 is selected on freshly randomized 81000-series worlds and evaluated once
on a disjoint 82000-series confirmation block. The earlier 80000-series V2
fresh result is historical evidence and is never used for selection here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adaptive_position import (
    AdaptivePositionCandidate,
    AdaptivePositionProtocolConfig,
    _adapter_diagnostic_summary,
    _flatten,
    _fresh_variants,
    _load_checkpoint,
    _load_manifest,
    _run_grid,
    _scenario_summaries_from_variants,
    _seed_summaries,
    _summary,
)
from .closed_loop import ControllerRun
from .controllers import AdaptivePositionControllerConfig
from .gru import CausalTargetStateGRU
from .gru_control import _aggregate_runs


ADAPTIVE_POSITION_V21_SCHEMA_VERSION = (
    "gimbal_adaptive_position_v21_protocol_v1"
)
DEFAULT_DEVELOPMENT_SEEDS = tuple(range(81000, 81008))
DEFAULT_CONFIRMATION_SEEDS = tuple(range(82000, 82008))
HISTORICAL_V2_FRESH_SEEDS = (80000, 80001, 80002, 80003)


def adaptive_position_v2_config() -> AdaptivePositionControllerConfig:
    """Return the validation-selected V2 adapter as the frozen reference."""

    return AdaptivePositionControllerConfig(
        actuator_arrival_time_scale=0.85,
        position_response_fraction=0.0,
        full_trust_std_ratio=1.0,
        zero_trust_std_ratio=4.0,
        minimum_prediction_weight=0.50,
        setpoint_rate_limit_scale=6.0,
        setpoint_acceleration_limit_scale=12.0,
        setpoint_jerk_rise_time_s=0.015,
    )


def default_visibility_risk_candidates() -> tuple[
    AdaptivePositionCandidate, ...
]:
    base = adaptive_position_v2_config()
    shared = {
        "visibility_risk_onset_fraction": 0.55,
        "visibility_risk_full_fraction": 0.85,
        "visibility_uncertainty_sigma": 1.0,
    }
    return (
        AdaptivePositionCandidate(
            "authority_only",
            replace(
                base,
                **shared,
                risk_acceleration_limit_multiplier=2.0,
                risk_jerk_limit_multiplier=4.0,
            ),
        ),
        AdaptivePositionCandidate(
            "preview_090",
            replace(base, **shared, risk_horizon_boost_s=0.090),
        ),
        AdaptivePositionCandidate(
            "preview_125",
            replace(base, **shared, risk_horizon_boost_s=0.125),
        ),
        AdaptivePositionCandidate(
            "preview_175",
            replace(base, **shared, risk_horizon_boost_s=0.175),
        ),
        AdaptivePositionCandidate(
            "preview_authority_050",
            replace(
                base,
                **shared,
                risk_horizon_boost_s=0.050,
                risk_acceleration_limit_multiplier=2.0,
                risk_jerk_limit_multiplier=4.0,
            ),
        ),
    )


@dataclass(frozen=True)
class VisibilityRiskProtocolConfig:
    maximum_staleness_s: float = 0.50
    development_max_mean_error_regression_deg: float = 0.25
    development_max_p95_regression_deg: float = 0.50
    development_max_loss_of_view_regression: float = 0.005
    development_max_variation_regression_fraction: float = 0.10
    development_minimum_unrecovered_event_reduction: int = 1
    confirmation_max_mean_error_regression_deg: float = 0.25
    confirmation_max_p95_regression_deg: float = 0.50
    confirmation_max_loss_of_view_regression: float = 0.005
    confirmation_max_variation_regression_fraction: float = 0.10
    confirmation_minimum_variation_reduction_vs_fixed_fraction: float = 0.05
    confirmation_max_scenario_p95_regression_deg: float = 2.0
    confirmation_max_scenario_loss_of_view_regression: float = 0.02
    device: str = "cpu"
    candidates: tuple[AdaptivePositionCandidate, ...] = (
        default_visibility_risk_candidates()
    )

    def __post_init__(self) -> None:
        for name in (
            "maximum_staleness_s",
            "development_max_mean_error_regression_deg",
            "development_max_p95_regression_deg",
            "development_max_loss_of_view_regression",
            "development_max_variation_regression_fraction",
            "confirmation_max_mean_error_regression_deg",
            "confirmation_max_p95_regression_deg",
            "confirmation_max_loss_of_view_regression",
            "confirmation_max_variation_regression_fraction",
            "confirmation_minimum_variation_reduction_vs_fixed_fraction",
            "confirmation_max_scenario_p95_regression_deg",
            "confirmation_max_scenario_loss_of_view_regression",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.development_minimum_unrecovered_event_reduction < 1:
            raise ValueError(
                "development requires at least one fewer unrecovered event"
            )
        if not self.candidates:
            raise ValueError("at least one V2.1 candidate is required")
        names = [candidate.name for candidate in self.candidates]
        if len(names) != len(set(names)):
            raise ValueError("V2.1 candidate names must be unique")


def _delta(candidate: dict[str, Any], reference: dict[str, Any], key: str) -> float:
    return float(candidate[key]) - float(reference[key])


def _variation_reduction(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> float:
    reference_value = float(reference["command_variation_per_s"])
    if reference_value <= 0.0:
        return 0.0
    return (
        reference_value - float(candidate["command_variation_per_s"])
    ) / reference_value


def _comparison(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    return {
        "deltas": {
            key: _delta(candidate, reference, key)
            for key in (
                "mean_absolute_error_deg",
                "p95_absolute_error_deg",
                "loss_of_view_fraction",
                "command_variation_per_s",
                "mean_control_cost",
            )
        },
        "unrecovered_event_delta": (
            candidate["total_unrecovered_loss_events"]
            - reference["total_unrecovered_loss_events"]
        ),
        "command_variation_reduction_fraction": _variation_reduction(
            candidate, reference
        ),
    }


def _development_gate(
    *,
    candidate: dict[str, Any],
    v2: dict[str, Any],
    fixed: dict[str, Any],
    scenario_summaries: dict[str, Any],
    protocol: VisibilityRiskProtocolConfig,
) -> dict[str, Any]:
    variation_regression = -_variation_reduction(candidate, v2)
    event_reduction = (
        v2["total_unrecovered_loss_events"]
        - candidate["total_unrecovered_loss_events"]
    )
    checks = {
        "mean_error": _delta(candidate, v2, "mean_absolute_error_deg")
        <= protocol.development_max_mean_error_regression_deg,
        "p95_error": _delta(candidate, v2, "p95_absolute_error_deg")
        <= protocol.development_max_p95_regression_deg,
        "loss_of_view": _delta(candidate, v2, "loss_of_view_fraction")
        <= protocol.development_max_loss_of_view_regression,
        "variation_vs_v2": variation_regression
        <= protocol.development_max_variation_regression_fraction,
        "variation_vs_fixed": _variation_reduction(candidate, fixed)
        >= protocol.confirmation_minimum_variation_reduction_vs_fixed_fraction,
        "unrecovered_event_reduction": event_reduction
        >= protocol.development_minimum_unrecovered_event_reduction,
        "unrecovered_events_vs_fixed": candidate[
            "total_unrecovered_loss_events"
        ]
        <= fixed["total_unrecovered_loss_events"],
        "no_scenario_event_regression": all(
            value["v2"]["total_unrecovered_loss_events"]
            <= value["fixed"]["total_unrecovered_loss_events"]
            for value in scenario_summaries.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "vs_v2": _comparison(candidate, v2),
        "vs_fixed": _comparison(candidate, fixed),
    }


def _confirmation_gate(
    *,
    candidate: dict[str, Any],
    v2: dict[str, Any],
    fixed: dict[str, Any],
    seed_summaries: dict[str, Any],
    scenario_summaries: dict[str, Any],
    fixed_scenario_summaries: dict[str, Any],
    protocol: VisibilityRiskProtocolConfig,
) -> dict[str, Any]:
    aggregate_checks = {
        "mean_error": _delta(candidate, v2, "mean_absolute_error_deg")
        <= protocol.confirmation_max_mean_error_regression_deg,
        "p95_error": _delta(candidate, v2, "p95_absolute_error_deg")
        <= protocol.confirmation_max_p95_regression_deg,
        "loss_of_view": _delta(candidate, v2, "loss_of_view_fraction")
        <= protocol.confirmation_max_loss_of_view_regression,
        "variation_vs_v2": -_variation_reduction(candidate, v2)
        <= protocol.confirmation_max_variation_regression_fraction,
        "variation_vs_fixed": _variation_reduction(candidate, fixed)
        >= protocol.confirmation_minimum_variation_reduction_vs_fixed_fraction,
        "unrecovered_events_vs_v2": candidate[
            "total_unrecovered_loss_events"
        ]
        <= v2["total_unrecovered_loss_events"],
        "unrecovered_events_vs_fixed": candidate[
            "total_unrecovered_loss_events"
        ]
        <= fixed["total_unrecovered_loss_events"],
    }
    seed_checks = {
        seed: (
            values["deltas"]["mean_absolute_error_deg"]
            <= protocol.confirmation_max_mean_error_regression_deg
            and values["deltas"]["p95_absolute_error_deg"]
            <= protocol.confirmation_max_p95_regression_deg
            and values["deltas"]["loss_of_view_fraction"]
            <= protocol.confirmation_max_loss_of_view_regression
            and values["v2"]["total_unrecovered_loss_events"]
            <= values["fixed"]["total_unrecovered_loss_events"]
        )
        for seed, values in seed_summaries.items()
        if seed != "distributions"
    }
    scenario_checks = {}
    for name, values in scenario_summaries.items():
        fixed_values = fixed_scenario_summaries[name]
        scenario_checks[name] = (
            values["deltas"]["p95_absolute_error_deg"]
            <= protocol.confirmation_max_scenario_p95_regression_deg
            and values["deltas"]["loss_of_view_fraction"]
            <= protocol.confirmation_max_scenario_loss_of_view_regression
            and values["v2"]["total_unrecovered_loss_events"]
            <= values["fixed"]["total_unrecovered_loss_events"]
            and values["v2"]["total_unrecovered_loss_events"]
            <= fixed_values["fixed"]["total_unrecovered_loss_events"]
        )
    return {
        "passed": (
            all(aggregate_checks.values())
            and all(seed_checks.values())
            and all(scenario_checks.values())
        ),
        "aggregate_checks": aggregate_checks,
        "per_training_seed_checks": seed_checks,
        "per_scenario_checks": scenario_checks,
        "vs_v2": _comparison(candidate, v2),
        "vs_fixed": _comparison(candidate, fixed),
    }


def _risk_trace(
    v2: ControllerRun,
    candidate: ControllerRun,
    *,
    world_seed: int,
    training_seed: int,
    scenario_name: str,
) -> dict[str, Any]:
    length = min(
        len(v2.adapter_diagnostics),
        len(candidate.adapter_diagnostics),
        len(v2.episode.frames),
        len(candidate.episode.frames),
    )
    records = []
    for index in range(length):
        v2_item = v2.adapter_diagnostics[index]
        candidate_item = candidate.adapter_diagnostics[index]
        v2_frame = v2.episode.frames[index]
        candidate_frame = candidate.episode.frames[index]
        records.append(
            {
                "time_s": candidate_frame.diagnostics.time_s,
                "target_body_bearing_deg": math.degrees(
                    candidate_frame.diagnostics.target_bearing_rad
                    - candidate_frame.diagnostics.body_bearing_rad
                ),
                "v2_gimbal_angle_deg": math.degrees(
                    v2_frame.diagnostics.gimbal_angle_rad
                ),
                "v21_gimbal_angle_deg": math.degrees(
                    candidate_frame.diagnostics.gimbal_angle_rad
                ),
                "v2_command_deg": math.degrees(
                    float(v2_item.get("shaped_target_angle_rad", 0.0))
                ),
                "v21_command_deg": math.degrees(
                    float(candidate_item.get("shaped_target_angle_rad", 0.0))
                ),
                "v21_raw_target_deg": math.degrees(
                    float(candidate_item.get("raw_target_angle_rad", 0.0))
                ),
                "v2_in_view": v2_frame.diagnostics.target_in_view,
                "v21_in_view": candidate_frame.diagnostics.target_in_view,
                "visibility_risk": float(
                    candidate_item.get("visibility_risk", 0.0)
                ),
                "predicted_fov_fraction": float(
                    candidate_item.get("predicted_fov_fraction", 0.0)
                ),
                "horizon_boost_s": float(
                    candidate_item.get("horizon_boost_s", 0.0)
                ),
                "effective_horizon_s": float(
                    candidate_item.get("effective_horizon_s", 0.0)
                ),
            }
        )
    return {
        "world_seed": world_seed,
        "training_seed": training_seed,
        "scenario_name": scenario_name,
        "records": records,
    }


def _select_trace(
    *,
    variants: Sequence[tuple[int, int, Any]],
    v2_runs: dict[int, list[ControllerRun]],
    candidate_runs: dict[int, list[ControllerRun]],
) -> dict[str, Any]:
    choices = [
        (
            v2.metrics.unrecovered_loss_events
            - candidate.metrics.unrecovered_loss_events,
            training_seed,
            index,
        )
        for training_seed in candidate_runs
        for index, (v2, candidate) in enumerate(
            zip(v2_runs[training_seed], candidate_runs[training_seed], strict=True)
        )
    ]
    improvement, training_seed, index = max(choices)
    if improvement <= 0:
        index = next(
            (
                position
                for position, variant in enumerate(variants)
                if variant[2].name == "aggressive_motion"
            ),
            0,
        )
        training_seed = next(iter(candidate_runs))
    variant = variants[index]
    return _risk_trace(
        v2_runs[training_seed][index],
        candidate_runs[training_seed][index],
        world_seed=variant[0],
        training_seed=training_seed,
        scenario_name=variant[2].name,
    )


def _scenario_summaries(
    *,
    variants: Sequence[tuple[int, int, Any]],
    candidate_runs: dict[int, list[ControllerRun]],
    reference_runs: dict[int, list[ControllerRun]],
) -> dict[str, Any]:
    return _scenario_summaries_from_variants(
        variants=variants,
        candidate_runs=candidate_runs,
        reference_runs=reference_runs,
    )


def evaluate_visibility_risk_v21(
    *,
    validation_data: str | Path,
    test_data: str | Path,
    checkpoints: dict[int, str | Path],
    protocol: VisibilityRiskProtocolConfig | None = None,
    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS,
    confirmation_seeds: tuple[int, ...] = DEFAULT_CONFIRMATION_SEEDS,
) -> dict[str, Any]:
    """Select on development worlds, freeze, then open confirmation once."""

    protocol = protocol or VisibilityRiskProtocolConfig()
    if len(checkpoints) < 2:
        raise ValueError("V2.1 evaluation requires at least two training seeds")
    if not development_seeds or not confirmation_seeds:
        raise ValueError("development and confirmation seeds must be non-empty")
    if len(set(development_seeds)) != len(development_seeds) or len(
        set(confirmation_seeds)
    ) != len(confirmation_seeds):
        raise ValueError("protocol seeds must be unique within each block")
    if set(development_seeds) & set(confirmation_seeds):
        raise ValueError("development and confirmation seeds overlap")
    if set(HISTORICAL_V2_FRESH_SEEDS) & (
        set(development_seeds) | set(confirmation_seeds)
    ):
        raise ValueError("V2.1 cannot reuse the historical V2 fresh seeds")

    manifests = {
        "validation": _load_manifest(validation_data),
        "test": _load_manifest(test_data),
    }
    occupied = set(manifests["validation"].seeds) | set(manifests["test"].seeds)
    if occupied & (set(development_seeds) | set(confirmation_seeds)):
        raise ValueError("V2.1 protocol seeds overlap recorded datasets")
    checkpoint_paths = {
        int(seed): Path(path) for seed, path in checkpoints.items()
    }
    models: dict[int, CausalTargetStateGRU] = {}
    horizon_indices: dict[int, int] = {}
    for training_seed, path in checkpoint_paths.items():
        model, _metadata, horizon_index = _load_checkpoint(
            path,
            training_seed,
            manifests,
            protocol.device,
        )
        models[training_seed] = model
        horizon_indices[training_seed] = horizon_index

    base_candidate = AdaptivePositionCandidate(
        "adaptive_v2",
        adaptive_position_v2_config(),
    )
    grid = (base_candidate,) + protocol.candidates
    runtime_protocol = AdaptivePositionProtocolConfig(
        maximum_staleness_s=protocol.maximum_staleness_s,
        device=protocol.device,
        candidates=grid,
    )
    development_variants = _fresh_variants(development_seeds)
    fixed_development, adaptive_development = _run_grid(
        variants=development_variants,
        models=models,
        horizon_indices=horizon_indices,
        protocol=runtime_protocol,
        candidates=grid,
    )
    fixed_development_summary = _summary(
        _aggregate_runs(_flatten(fixed_development))
    )
    v2_development = adaptive_development[base_candidate.name]
    v2_development_summary = _summary(
        _aggregate_runs(_flatten(v2_development))
    )
    development_candidates = []
    for candidate in protocol.candidates:
        runs = adaptive_development[candidate.name]
        summary = _summary(_aggregate_runs(_flatten(runs)))
        scenario_summaries = _scenario_summaries(
            variants=development_variants,
            candidate_runs=runs,
            reference_runs=v2_development,
        )
        development_candidates.append(
            {
                "name": candidate.name,
                "controller_config": asdict(candidate.controller),
                "summary": summary,
                "adapter_diagnostics": _adapter_diagnostic_summary(
                    _flatten(runs)
                ),
                "by_scenario_vs_v2": scenario_summaries,
                "gate": _development_gate(
                    candidate=summary,
                    v2=v2_development_summary,
                    fixed=fixed_development_summary,
                    scenario_summaries=scenario_summaries,
                    protocol=protocol,
                ),
            }
        )
    eligible = [item for item in development_candidates if item["gate"]["passed"]]
    development_result = {
        "world_seeds": list(development_seeds),
        "variant_count_per_training_seed": len(development_variants),
        "fixed_summary": fixed_development_summary,
        "v2_summary": v2_development_summary,
        "candidates": development_candidates,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": None,
        "selected_candidate_passed_gate": False,
    }
    common = {
        "experiment": ADAPTIVE_POSITION_V21_SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "training_seeds": list(models),
        "checkpoints": {
            str(seed): str(path) for seed, path in checkpoint_paths.items()
        },
        "fixed_horizons": {
            str(seed): {
                "horizon_index": horizon_indices[seed],
                "horizon_s": models[seed].config.prediction_horizons_s[
                    horizon_indices[seed]
                ],
            }
            for seed in models
        },
        "selection_policy": (
            "The 81000-series development block ranks only candidates that "
            "remove at least one unrecovered event, remain within declared "
            "tracking/activity bounds, preserve at least five percent command-"
            "variation reduction versus fixed horizon, and add no scenario-"
            "level terminal loss. The disjoint 82000-series block remains "
            "closed unless a candidate passes."
        ),
    }
    if not eligible:
        return {
            **common,
            "development": development_result,
            "confirmation": {
                "opened": False,
                "reason": "no V2.1 candidate passed the development gate",
                "recommendation": "retain_adaptive_position_v2",
            },
        }

    selected_record = min(
        eligible,
        key=lambda item: (
            item["summary"]["total_unrecovered_loss_events"],
            item["summary"]["command_variation_per_s"],
            item["summary"]["mean_control_cost"],
        ),
    )
    development_result["selected_candidate"] = selected_record["name"]
    development_result["selected_candidate_passed_gate"] = True
    selected_candidate = next(
        candidate
        for candidate in protocol.candidates
        if candidate.name == selected_record["name"]
    )

    confirmation_variants = _fresh_variants(confirmation_seeds)
    confirmation_grid = (base_candidate, selected_candidate)
    fixed_confirmation, adaptive_confirmation = _run_grid(
        variants=confirmation_variants,
        models=models,
        horizon_indices=horizon_indices,
        protocol=runtime_protocol,
        candidates=confirmation_grid,
    )
    v2_confirmation = adaptive_confirmation[base_candidate.name]
    candidate_confirmation = adaptive_confirmation[selected_candidate.name]
    fixed_summary = _summary(_aggregate_runs(_flatten(fixed_confirmation)))
    v2_summary = _summary(_aggregate_runs(_flatten(v2_confirmation)))
    candidate_summary = _summary(
        _aggregate_runs(_flatten(candidate_confirmation))
    )
    seed_summaries = _seed_summaries(
        candidate_confirmation,
        v2_confirmation,
    )
    scenario_summaries = _scenario_summaries(
        variants=confirmation_variants,
        candidate_runs=candidate_confirmation,
        reference_runs=v2_confirmation,
    )
    fixed_scenario_summaries = _scenario_summaries(
        variants=confirmation_variants,
        candidate_runs=candidate_confirmation,
        reference_runs=fixed_confirmation,
    )
    gate = _confirmation_gate(
        candidate=candidate_summary,
        v2=v2_summary,
        fixed=fixed_summary,
        seed_summaries=seed_summaries,
        scenario_summaries=scenario_summaries,
        fixed_scenario_summaries=fixed_scenario_summaries,
        protocol=protocol,
    )
    return {
        **common,
        "development": development_result,
        "confirmation": {
            "opened": True,
            "world_seeds": list(confirmation_seeds),
            "variant_count_per_training_seed": len(confirmation_variants),
            "fixed_summary": fixed_summary,
            "v2_summary": v2_summary,
            "v21_summary": candidate_summary,
            "adapter_diagnostics": _adapter_diagnostic_summary(
                _flatten(candidate_confirmation)
            ),
            "by_training_seed_vs_v2": seed_summaries,
            "by_scenario_vs_v2": scenario_summaries,
            "by_scenario_vs_fixed": fixed_scenario_summaries,
            "acceptance_gate": gate,
            "recommendation": (
                "adaptive_position_v21"
                if gate["passed"]
                else "retain_adaptive_position_v2"
            ),
        },
        "representative_trace": _select_trace(
            variants=confirmation_variants,
            v2_runs=v2_confirmation,
            candidate_runs=candidate_confirmation,
        ),
    }


def _checkpoint_argument(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("checkpoint must be TRAINING_SEED=PATH")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint seed must be an integer") from error
    return seed, Path(path_text)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and confirm visibility-risk adaptive position V2.1."
    )
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=_checkpoint_argument,
        action="append",
        required=True,
        help="repeat TRAINING_SEED=PATH for each O2 initialization",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--development-seed",
        type=int,
        action="append",
        help="repeat to override the default 81000-series development block",
    )
    parser.add_argument(
        "--confirmation-seed",
        type=int,
        action="append",
        help="repeat to override the default 82000-series confirmation block",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise SystemExit("checkpoint training seeds must be unique")
    result = evaluate_visibility_risk_v21(
        validation_data=args.validation_data,
        test_data=args.test_data,
        checkpoints=checkpoints,
        protocol=VisibilityRiskProtocolConfig(device=args.device),
        development_seeds=(
            tuple(args.development_seed)
            if args.development_seed
            else DEFAULT_DEVELOPMENT_SEEDS
        ),
        confirmation_seeds=(
            tuple(args.confirmation_seed)
            if args.confirmation_seed
            else DEFAULT_CONFIRMATION_SEEDS
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["confirmation"], indent=2))


if __name__ == "__main__":
    main()
