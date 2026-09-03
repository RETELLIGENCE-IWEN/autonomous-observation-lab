"""Development-locked conventional baseline for predictive gimbal control.

The protocol tunes only conventional estimator/filter choices and a practical
feedback gain on the established 81000-series development worlds. It then
replays the frozen choices on the already-open 82000-series confirmation
worlds. The learned and classical predictors share the exact V2.1 position
adapter so their comparison isolates target-state estimation quality.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .adaptive_position import (
    AdaptivePositionProtocolConfig,
    _adaptive_run,
    _fresh_variants,
    _summary,
)
from .closed_loop import (
    ClosedLoopScenario,
    ControllerRun,
    run_closed_loop_controller,
)
from .config import GimbalCommandMode, ObservationProfile
from .controllers import (
    AdaptivePositionControllerConfig,
    AdaptiveTargetStatePositionController,
    ProportionalPositionController,
)
from .estimators import (
    ConstantVelocityEstimatorConfig,
    MultiHorizonConstantVelocityTargetEstimator,
)
from .gru import CausalTargetStateGRU, load_gru_checkpoint
from .gru_control import _aggregate_runs


CONVENTIONAL_CHAMPION_SCHEMA_VERSION = (
    "gimbal_conventional_champion_v1_protocol_v1"
)
DEFAULT_DEVELOPMENT_SEEDS = tuple(range(81000, 81008))
DEFAULT_CONFIRMATION_SEEDS = tuple(range(82000, 82008))


@dataclass(frozen=True)
class FeedbackScheduleCandidate:
    """Dimensioned gain schedule over configured sensing/plant delay."""

    name: str
    delay_gain_product_s: float
    maximum_gain: float = 0.40
    minimum_gain: float = 0.05
    position_response_fraction: float = 0.15

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("candidate name must be identifier-like")
        for name in (
            "delay_gain_product_s",
            "maximum_gain",
            "minimum_gain",
            "position_response_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_gain > self.maximum_gain:
            raise ValueError("minimum gain must not exceed maximum gain")

    def gain_for(self, scenario: ClosedLoopScenario) -> float:
        config = scenario.config
        loop_delay_s = (
            config.camera.detection_latency_s
            + config.servo.command_latency_s
            + config.servo.rate_time_constant_s
            + self.position_response_fraction
            / config.servo.position_gain_s_inv
        )
        return max(
            self.minimum_gain,
            min(self.maximum_gain, self.delay_gain_product_s / loop_delay_s),
        )


def default_feedback_schedules() -> tuple[FeedbackScheduleCandidate, ...]:
    return tuple(
        FeedbackScheduleCandidate(
            f"delay_scheduled_{int(round(1000.0 * product_s)):03d}",
            delay_gain_product_s=product_s,
        )
        for product_s in (0.05, 0.07, 0.09, 0.11, 0.13)
    )


@dataclass(frozen=True)
class ClassicalEstimatorCandidate:
    """Hardware-independent filter choices for a deployable estimator."""

    name: str
    velocity_filter_coefficient: float
    uncertainty_filter_coefficient: float = 0.20
    process_acceleration_std_rad_s2: float = math.radians(80.0)
    body_rate_compensation: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError("candidate name must be identifier-like")
        for name in (
            "velocity_filter_coefficient",
            "uncertainty_filter_coefficient",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if (
            not math.isfinite(self.process_acceleration_std_rad_s2)
            or self.process_acceleration_std_rad_s2 < 0.0
        ):
            raise ValueError(
                "process_acceleration_std_rad_s2 must be finite and non-negative"
            )
        if not isinstance(self.body_rate_compensation, bool):
            raise ValueError("body_rate_compensation must be boolean")


def default_classical_candidates() -> tuple[ClassicalEstimatorCandidate, ...]:
    return (
        ClassicalEstimatorCandidate(
            "relative_cv_a040",
            velocity_filter_coefficient=0.40,
            body_rate_compensation=False,
        ),
        ClassicalEstimatorCandidate("imu_cv_a025", 0.25),
        ClassicalEstimatorCandidate("imu_cv_a040", 0.40),
        ClassicalEstimatorCandidate("imu_cv_a055", 0.55),
        ClassicalEstimatorCandidate("imu_cv_a070", 0.70),
        ClassicalEstimatorCandidate("imu_cv_a085", 0.85),
    )


@dataclass(frozen=True)
class ConventionalChampionProtocolConfig:
    maximum_staleness_s: float = 0.50
    feedback_schedules: tuple[FeedbackScheduleCandidate, ...] = (
        default_feedback_schedules()
    )
    feedback_max_aggregate_cost_regression: float = 0.05
    feedback_max_high_latency_cost_regression: float = 0.01
    classical_candidates: tuple[ClassicalEstimatorCandidate, ...] = (
        default_classical_candidates()
    )
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_staleness_s)
            or self.maximum_staleness_s <= 0.0
        ):
            raise ValueError("maximum staleness must be finite and positive")
        if not self.feedback_schedules:
            raise ValueError("at least one feedback schedule is required")
        schedule_names = [item.name for item in self.feedback_schedules]
        if len(set(schedule_names)) != len(schedule_names):
            raise ValueError("feedback schedule names must be unique")
        for name in (
            "feedback_max_aggregate_cost_regression",
            "feedback_max_high_latency_cost_regression",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.classical_candidates:
            raise ValueError("at least one classical candidate is required")
        names = [candidate.name for candidate in self.classical_candidates]
        if len(set(names)) != len(names):
            raise ValueError("classical candidate names must be unique")


def locked_feedback_schedule() -> FeedbackScheduleCandidate:
    """Development-selected schedule used by practical feedback."""

    return FeedbackScheduleCandidate(
        "delay_scheduled_090",
        delay_gain_product_s=0.09,
    )


def locked_classical_estimator_candidate() -> ClassicalEstimatorCandidate:
    """Development-selected estimator used by Conventional Champion v1."""

    return ClassicalEstimatorCandidate("imu_cv_a070", 0.70)


def _position_config(scenario: ClosedLoopScenario) -> Any:
    return replace(
        scenario.config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        command_mode=GimbalCommandMode.POSITION,
    )


def _classical_estimator(
    *,
    scenario: ClosedLoopScenario,
    candidate: ClassicalEstimatorCandidate,
    prediction_horizons_s: tuple[float, ...],
    maximum_staleness_s: float,
) -> MultiHorizonConstantVelocityTargetEstimator:
    config = _position_config(scenario)
    maximum_projection_s = maximum_staleness_s + prediction_horizons_s[-1]
    return MultiHorizonConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            center_noise_std_normalized=(
                config.camera.center_noise_std_normalized
            ),
            velocity_filter_coefficient=(
                candidate.velocity_filter_coefficient
            ),
            uncertainty_filter_coefficient=(
                candidate.uncertainty_filter_coefficient
            ),
            max_prediction_horizon_s=maximum_projection_s,
            history_horizon_s=max(1.0, maximum_projection_s + 0.50),
            process_acceleration_std_rad_s2=(
                candidate.process_acceleration_std_rad_s2
            ),
            body_rate_compensation=candidate.body_rate_compensation,
        ),
        prediction_horizons_s=prediction_horizons_s,
    )


def practical_feedback_run(
    *,
    scenario: ClosedLoopScenario,
    seed: int,
    gain: float | None = None,
    schedule: FeedbackScheduleCandidate | None = None,
    name: str = "practical_feedback_position",
) -> ControllerRun:
    """Run a tuned, intentionally simple deployable feedback controller."""

    config = _position_config(scenario)
    if gain is not None and schedule is not None:
        raise ValueError("choose either a fixed gain or a gain schedule")
    selected_schedule = schedule or locked_feedback_schedule()
    selected_gain = (
        selected_schedule.gain_for(scenario) if gain is None else gain
    )
    return run_closed_loop_controller(
        name=name,
        description=(
            "Development-selected latency-scheduled bbox feedback with an "
            "absolute position command."
        ),
        scenario=scenario,
        config=config,
        controller=ProportionalPositionController(
            servo=config.servo,
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            gain=selected_gain,
            name=name,
        ),
        seed=seed,
    )


def conventional_champion_run(
    *,
    scenario: ClosedLoopScenario,
    seed: int,
    adapter: AdaptivePositionControllerConfig,
    prediction_horizons_s: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3),
    estimator_candidate: ClassicalEstimatorCandidate | None = None,
    maximum_staleness_s: float = 0.50,
    name: str = "conventional_champion_v1",
) -> ControllerRun:
    """Run the delay-aware, IMU-compensated conventional champion."""

    config = _position_config(scenario)
    candidate = estimator_candidate or locked_classical_estimator_candidate()
    estimator = _classical_estimator(
        scenario=scenario,
        candidate=candidate,
        prediction_horizons_s=prediction_horizons_s,
        maximum_staleness_s=maximum_staleness_s,
    )
    controller = AdaptiveTargetStatePositionController(
        estimator=estimator,
        servo=config.servo,
        config=adapter,
        selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
        name=name,
    )
    return run_closed_loop_controller(
        name=name,
        description=(
            "Causal IMU-compensated constant-velocity estimator with the "
            "shared V2.1 hardware-aware position adapter."
        ),
        scenario=scenario,
        config=config,
        controller=controller,
        seed=seed,
    )


def _summarize(runs: list[ControllerRun]) -> dict[str, Any]:
    aggregate = _aggregate_runs(runs)
    summary = _summary(aggregate)
    episode_count = int(aggregate["episode_count"])
    return {
        **summary,
        "episode_count": episode_count,
        "mean_unrecovered_loss_events_per_episode": (
            summary["total_unrecovered_loss_events"] / episode_count
        ),
    }


def _by_scenario(
    variants: Sequence[tuple[int, int, ClosedLoopScenario]],
    runs: Sequence[ControllerRun],
) -> dict[str, Any]:
    buckets: dict[str, list[ControllerRun]] = {}
    for variant, run in zip(variants, runs, strict=True):
        buckets.setdefault(variant[2].name, []).append(run)
    return {name: _summarize(values) for name, values in buckets.items()}


def _comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "mean_absolute_error_deg",
        "p95_absolute_error_deg",
        "loss_of_view_fraction",
        "command_variation_per_s",
        "mean_control_cost",
    )
    return {
        "deltas": {
            key: float(candidate[key]) - float(reference[key]) for key in keys
        },
        "mean_unrecovered_event_delta_per_episode": (
            float(candidate["mean_unrecovered_loss_events_per_episode"])
            - float(reference["mean_unrecovered_loss_events_per_episode"])
        ),
    }


def _validate_seed_blocks(
    development_seeds: tuple[int, ...],
    confirmation_seeds: tuple[int, ...],
) -> None:
    if not development_seeds or not confirmation_seeds:
        raise ValueError("development and confirmation seeds must be non-empty")
    if len(set(development_seeds)) != len(development_seeds) or len(
        set(confirmation_seeds)
    ) != len(confirmation_seeds):
        raise ValueError("protocol seeds must be unique within each block")
    if set(development_seeds) & set(confirmation_seeds):
        raise ValueError("development and confirmation seeds overlap")


def _load_models(
    result: dict[str, Any],
    device: str,
) -> dict[int, CausalTargetStateGRU]:
    raw = result.get("checkpoints")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("visibility-risk result has no checkpoints")
    return {
        int(seed): load_gru_checkpoint(Path(path), device=device)[0]
        for seed, path in raw.items()
    }


def _selected_adapter_config(
    result: dict[str, Any],
) -> AdaptivePositionControllerConfig:
    development = result.get("development")
    if not isinstance(development, dict):
        raise ValueError("visibility-risk result has no development block")
    selected = development.get("selected_candidate")
    candidates = development.get("candidates")
    if not isinstance(selected, str) or not isinstance(candidates, list):
        raise ValueError("visibility-risk result has no selected candidate")
    record = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and item.get("name") == selected
        ),
        None,
    )
    if record is None or not isinstance(record.get("controller_config"), dict):
        raise ValueError("selected V2.1 adapter configuration is missing")
    return AdaptivePositionControllerConfig(**record["controller_config"])


def evaluate_conventional_champion_v1(
    *,
    visibility_risk_result: dict[str, Any],
    protocol: ConventionalChampionProtocolConfig | None = None,
    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS,
    confirmation_seeds: tuple[int, ...] = DEFAULT_CONFIRMATION_SEEDS,
) -> dict[str, Any]:
    """Select conventional baselines on development and replay confirmation."""

    protocol = protocol or ConventionalChampionProtocolConfig()
    _validate_seed_blocks(development_seeds, confirmation_seeds)
    confirmation = visibility_risk_result.get("confirmation")
    if not isinstance(confirmation, dict) or not confirmation.get("opened"):
        raise ValueError("visibility-risk confirmation block was not opened")
    recorded_development = set(
        visibility_risk_result.get("development", {}).get("world_seeds", ())
    )
    recorded_confirmation = set(confirmation.get("world_seeds", ()))
    if not set(development_seeds) <= recorded_development:
        raise ValueError("development seeds are outside the established block")
    if not set(confirmation_seeds) <= recorded_confirmation:
        raise ValueError("confirmation seeds are outside the established block")

    adapter = _selected_adapter_config(visibility_risk_result)
    models = _load_models(visibility_risk_result, protocol.device)
    horizons_by_seed = {
        seed: model.config.prediction_horizons_s
        for seed, model in models.items()
    }
    unique_horizons = set(horizons_by_seed.values())
    if len(unique_horizons) != 1:
        raise ValueError("learned checkpoints use different forecast horizons")
    prediction_horizons_s = next(iter(unique_horizons))

    development_variants = _fresh_variants(development_seeds)
    feedback_records = []
    for schedule in protocol.feedback_schedules:
        runs = [
            practical_feedback_run(
                scenario=scenario,
                seed=world_seed,
                schedule=schedule,
            )
            for world_seed, _scenario_index, scenario in development_variants
        ]
        feedback_records.append(
            {
                "name": schedule.name,
                "schedule_config": asdict(schedule),
                "summary": _summarize(runs),
                "by_scenario": _by_scenario(development_variants, runs),
            }
        )
    minimum_aggregate_cost = min(
        item["summary"]["mean_control_cost"] for item in feedback_records
    )
    minimum_high_latency_cost = min(
        item["by_scenario"]["high_latency"]["mean_control_cost"]
        for item in feedback_records
    )
    eligible_feedback = [
        item
        for item in feedback_records
        if item["summary"]["mean_control_cost"]
        <= minimum_aggregate_cost
        + protocol.feedback_max_aggregate_cost_regression
        and item["by_scenario"]["high_latency"]["mean_control_cost"]
        <= minimum_high_latency_cost
        + protocol.feedback_max_high_latency_cost_regression
    ]
    selected_feedback = min(
        eligible_feedback,
        key=lambda item: item["summary"]["command_variation_per_s"],
    )
    selected_feedback_schedule = next(
        schedule
        for schedule in protocol.feedback_schedules
        if schedule.name == selected_feedback["name"]
    )

    classical_records = []
    for candidate in protocol.classical_candidates:
        runs = [
            conventional_champion_run(
                scenario=scenario,
                seed=world_seed,
                adapter=adapter,
                prediction_horizons_s=prediction_horizons_s,
                estimator_candidate=candidate,
                maximum_staleness_s=protocol.maximum_staleness_s,
            )
            for world_seed, _scenario_index, scenario in development_variants
        ]
        classical_records.append(
            {
                "name": candidate.name,
                "estimator_config": asdict(candidate),
                "summary": _summarize(runs),
                "by_scenario": _by_scenario(development_variants, runs),
            }
        )
    selected_classical = min(
        classical_records,
        key=lambda item: (
            item["summary"]["mean_control_cost"],
            item["summary"]["command_variation_per_s"],
        ),
    )
    selected_candidate = next(
        candidate
        for candidate in protocol.classical_candidates
        if candidate.name == selected_classical["name"]
    )

    confirmation_variants = _fresh_variants(confirmation_seeds)
    feedback_runs = [
        practical_feedback_run(
            scenario=scenario,
            seed=world_seed,
            schedule=selected_feedback_schedule,
        )
        for world_seed, _scenario_index, scenario in confirmation_variants
    ]
    champion_runs = [
        conventional_champion_run(
            scenario=scenario,
            seed=world_seed,
            adapter=adapter,
            prediction_horizons_s=prediction_horizons_s,
            estimator_candidate=selected_candidate,
            maximum_staleness_s=protocol.maximum_staleness_s,
        )
        for world_seed, _scenario_index, scenario in confirmation_variants
    ]
    learned_runtime = AdaptivePositionProtocolConfig(
        maximum_staleness_s=protocol.maximum_staleness_s,
        device=protocol.device,
    )
    learned_by_seed: dict[int, list[ControllerRun]] = {}
    for training_seed, model in models.items():
        learned_by_seed[training_seed] = [
            _adaptive_run(
                scenario=scenario,
                seed=world_seed,
                model=model,
                adapter=adapter,
                evaluation=learned_runtime,
                name="dream_to_center",
            )
            for world_seed, _scenario_index, scenario in confirmation_variants
        ]
    learned_runs = [
        run for runs in learned_by_seed.values() for run in runs
    ]
    feedback_summary = _summarize(feedback_runs)
    champion_summary = _summarize(champion_runs)
    learned_summary = _summarize(learned_runs)
    learned_scenario_variants = tuple(
        variant
        for _training_seed in learned_by_seed
        for variant in confirmation_variants
    )
    return {
        "experiment": CONVENTIONAL_CHAMPION_SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "shared_adapter_config": asdict(adapter),
        "prediction_horizons_s": list(prediction_horizons_s),
        "development": {
            "world_seeds": list(development_seeds),
            "selection_policy": (
                "Choose the smoothest delay schedule within the declared "
                "aggregate and high-latency control-cost plateaus."
            ),
            "feedback_candidates": feedback_records,
            "selected_feedback_schedule": selected_feedback["name"],
            "classical_candidates": classical_records,
            "selected_classical_candidate": selected_classical["name"],
        },
        "confirmation": {
            "opened": True,
            "evidence_status": (
                "historical_replay_of_the_already_open_v21_confirmation_block"
            ),
            "world_seeds": list(confirmation_seeds),
            "training_seeds": list(models),
            "practical_feedback": {
                "summary": feedback_summary,
                "by_scenario": _by_scenario(
                    confirmation_variants,
                    feedback_runs,
                ),
            },
            "conventional_champion_v1": {
                "summary": champion_summary,
                "by_scenario": _by_scenario(
                    confirmation_variants,
                    champion_runs,
                ),
                "vs_practical_feedback": _comparison(
                    champion_summary,
                    feedback_summary,
                ),
            },
            "dream_to_center": {
                "summary": learned_summary,
                "by_scenario": _by_scenario(
                    learned_scenario_variants,
                    learned_runs,
                ),
                "by_training_seed": {
                    str(seed): _summarize(runs)
                    for seed, runs in learned_by_seed.items()
                },
                "vs_conventional_champion_v1": _comparison(
                    learned_summary,
                    champion_summary,
                ),
            },
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and replay Conventional Champion v1."
    )
    parser.add_argument(
        "--visibility-risk-results",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_position_v21.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_conventional_champion_v1.json"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    visibility_result = json.loads(
        args.visibility_risk_results.read_text(encoding="utf-8")
    )
    result = evaluate_conventional_champion_v1(
        visibility_risk_result=visibility_result,
        protocol=ConventionalChampionProtocolConfig(device=args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["confirmation"], indent=2))


if __name__ == "__main__":
    main()
