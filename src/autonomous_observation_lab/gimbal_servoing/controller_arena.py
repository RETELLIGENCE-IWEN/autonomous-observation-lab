"""Deterministic fixed/V2/V2.1 replay for the visual controller arena."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .adaptive_position import (
    AdaptivePositionProtocolConfig,
    _adaptive_run,
    _fixed_run,
)
from .adaptive_position_v21 import (
    ADAPTIVE_POSITION_V21_SCHEMA_VERSION,
    adaptive_position_v2_config,
)
from .closed_loop import (
    ClosedLoopComparison,
    ControllerRun,
    closed_loop_scenarios,
)
from .controllers import AdaptivePositionControllerConfig
from .gru import load_gru_checkpoint
from .randomization import randomize_closed_loop_scenario


@dataclass(frozen=True)
class ControllerArena:
    comparison: ClosedLoopComparison
    world_seed: int
    training_seed: int
    scenario_name: str
    selected_v21_candidate: str


def _rename_run(
    run: ControllerRun,
    *,
    name: str,
    description: str,
) -> ControllerRun:
    return replace(
        run,
        episode=replace(
            run.episode,
            name=name,
            description=description,
        ),
    )


def _selected_candidate_config(
    result: dict[str, Any],
) -> tuple[str, AdaptivePositionControllerConfig]:
    development = result["development"]
    selected = development.get("selected_candidate")
    if not isinstance(selected, str) or not selected:
        raise ValueError("visibility-risk result has no selected V2.1 candidate")
    candidates = development.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("visibility-risk result has no development candidates")
    record = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and item.get("name") == selected
        ),
        None,
    )
    if record is None or not isinstance(record.get("controller_config"), dict):
        raise ValueError("selected V2.1 controller configuration is missing")
    return selected, AdaptivePositionControllerConfig(
        **record["controller_config"]
    )


def build_visibility_risk_controller_arena(
    result: dict[str, Any],
    *,
    scenario_name: str | None = None,
    world_seed: int | None = None,
    training_seed: int | None = None,
    device: str = "cpu",
) -> ControllerArena:
    """Reconstruct three paired controllers from a frozen V2.1 result."""

    if result.get("experiment") != ADAPTIVE_POSITION_V21_SCHEMA_VERSION:
        raise ValueError("unsupported visibility-risk result")
    confirmation = result.get("confirmation")
    if not isinstance(confirmation, dict) or not confirmation.get("opened"):
        raise ValueError("visibility-risk confirmation block was not opened")
    trace = result.get("representative_trace")
    if not isinstance(trace, dict):
        raise ValueError("visibility-risk result has no representative trace")

    selected_scenario = scenario_name or str(trace["scenario_name"])
    selected_world_seed = (
        int(world_seed) if world_seed is not None else int(trace["world_seed"])
    )
    selected_training_seed = (
        int(training_seed)
        if training_seed is not None
        else int(trace["training_seed"])
    )
    if selected_world_seed not in confirmation.get("world_seeds", []):
        raise ValueError("arena world seed is outside the confirmation block")
    if selected_training_seed not in result.get("training_seeds", []):
        raise ValueError("arena training seed is unavailable")

    scenarios = {scenario.name: scenario for scenario in closed_loop_scenarios()}
    if selected_scenario not in scenarios:
        raise ValueError("arena scenario is unavailable")
    scenario = randomize_closed_loop_scenario(
        scenarios[selected_scenario],
        seed=selected_world_seed,
    )

    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("visibility-risk result has no checkpoints")
    checkpoint = checkpoints.get(str(selected_training_seed))
    if not isinstance(checkpoint, str):
        raise ValueError("selected arena checkpoint is unavailable")
    model, metadata = load_gru_checkpoint(Path(checkpoint), device=device)
    if metadata.get("training_config", {}).get("seed") not in (
        None,
        selected_training_seed,
    ):
        raise ValueError("arena checkpoint training seed mismatch")

    fixed_horizons = result.get("fixed_horizons")
    if not isinstance(fixed_horizons, dict):
        raise ValueError("visibility-risk result has no fixed horizons")
    fixed_horizon = fixed_horizons.get(str(selected_training_seed))
    if not isinstance(fixed_horizon, dict):
        raise ValueError("selected fixed horizon is unavailable")
    horizon_index = int(fixed_horizon["horizon_index"])
    if not 0 <= horizon_index < model.horizon_count:
        raise ValueError("selected fixed horizon is invalid")

    selected_candidate, v21_config = _selected_candidate_config(result)
    runtime = AdaptivePositionProtocolConfig(
        maximum_staleness_s=float(result["protocol"]["maximum_staleness_s"]),
        device=device,
    )
    fixed = _rename_run(
        _fixed_run(
            scenario=scenario,
            seed=selected_world_seed,
            model=model,
            horizon_index=horizon_index,
            evaluation=runtime,
        ),
        name="arena_fixed_horizon",
        description=(
            f"Fixed learned horizon: "
            f"{1000.0 * model.config.prediction_horizons_s[horizon_index]:.0f} ms"
        ),
    )
    v2 = _rename_run(
        _adaptive_run(
            scenario=scenario,
            seed=selected_world_seed,
            model=model,
            adapter=adaptive_position_v2_config(),
            evaluation=runtime,
            name="arena_adaptive_v2",
        ),
        name="arena_adaptive_v2",
        description="Adaptive timing, uncertainty trust, and smooth shaping",
    )
    v21 = _rename_run(
        _adaptive_run(
            scenario=scenario,
            seed=selected_world_seed,
            model=model,
            adapter=v21_config,
            evaluation=runtime,
            name="arena_visibility_risk_v21",
        ),
        name="arena_visibility_risk_v21",
        description=(
            "Visibility-risk forecast preview with the V2 smooth shaper"
        ),
    )
    comparison = ClosedLoopComparison(
        scenario_name=selected_scenario,
        description=(
            f"Exact paired world seed {selected_world_seed}, GRU training seed "
            f"{selected_training_seed}. All controllers share target/body "
            "motion, detections, hardware, and initial state."
        ),
        runs=(fixed, v2, v21),
    )
    return ControllerArena(
        comparison=comparison,
        world_seed=selected_world_seed,
        training_seed=selected_training_seed,
        scenario_name=selected_scenario,
        selected_v21_candidate=selected_candidate,
    )
