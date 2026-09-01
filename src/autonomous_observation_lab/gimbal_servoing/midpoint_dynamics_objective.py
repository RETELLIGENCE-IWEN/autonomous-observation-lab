"""Fresh development protocol for acceleration-aware integrated GRU heads."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .control_action_objective import (
    CONTROL_ACTION_OBJECTIVE_SCHEMA_VERSION,
    ControlActionCandidate,
    ControlActionObjectiveConfig,
    evaluate_control_action_objective,
)


MIDPOINT_DYNAMICS_OBJECTIVE_SCHEMA_VERSION = (
    "gimbal_midpoint_dynamics_action_objective_v51_development_v1"
)


def midpoint_dynamics_candidates() -> tuple[ControlActionCandidate, ...]:
    """Ablate latent midpoint-rate integration and dual command losses."""

    return (
        ControlActionCandidate(
            "independent_consistent_reference",
            0.0,
            0.0,
            25.0,
            "independent",
        ),
        ControlActionCandidate(
            "midpoint_state",
            0.0,
            0.0,
            0.0,
            "integrated_midpoint",
        ),
        ControlActionCandidate(
            "midpoint_dual_gentle",
            0.10,
            0.10,
            0.0,
            "integrated_midpoint",
        ),
        ControlActionCandidate(
            "midpoint_position_priority",
            0.05,
            0.15,
            0.0,
            "integrated_midpoint",
        ),
        ControlActionCandidate(
            "midpoint_dual_moderate",
            0.25,
            0.25,
            0.0,
            "integrated_midpoint",
        ),
    )


def evaluate_midpoint_dynamics_objective(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_directory: str | Path,
    config: ControlActionObjectiveConfig | None = None,
):
    configured = replace(
        config or ControlActionObjectiveConfig(),
        candidates=midpoint_dynamics_candidates(),
    )
    result = evaluate_control_action_objective(
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_directory=checkpoint_directory,
        config=configured,
    )
    if result["experiment"] != CONTROL_ACTION_OBJECTIVE_SCHEMA_VERSION:
        raise AssertionError("unexpected action-objective result schema")
    result["experiment"] = MIDPOINT_DYNAMICS_OBJECTIVE_SCHEMA_VERSION
    if result["selected_candidate"] is not None:
        result["recommendation"] = "replicate_midpoint_action_objective"
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate midpoint-integrated GRU dynamics and action losses."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_train.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_dynamics_validation.npz"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_dynamics_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_dynamics_development.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_midpoint_dynamics_objective(
        train_path=args.train_data,
        validation_path=args.validation_data,
        checkpoint_directory=args.checkpoint_directory,
        config=ControlActionObjectiveConfig(
            epochs=args.epochs,
            device=args.device,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"selected={result['selected_candidate']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
