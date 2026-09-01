"""V7 structural factorial for hard dynamics and the actual V2.1 adapter."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .adaptive_curriculum_objective import (
    ADAPTIVE_CURRICULUM_OBJECTIVE_SCHEMA_VERSION,
    AdaptiveCurriculumCandidate,
    AdaptiveCurriculumObjectiveConfig,
    evaluate_adaptive_curriculum_objective,
)


MIDPOINT_ADAPTER_OBJECTIVE_SCHEMA_VERSION = (
    "gimbal_midpoint_adapter_objective_v7_development_v1"
)


def midpoint_adapter_candidates() -> tuple[AdaptiveCurriculumCandidate, ...]:
    """Isolate hard dynamics, adapter loss, and a mild curriculum."""

    return (
        AdaptiveCurriculumCandidate(
            "v4_reference",
            0.0,
            False,
            25.0,
            "independent",
        ),
        AdaptiveCurriculumCandidate(
            "midpoint_state_reference",
            0.0,
            False,
            0.0,
            "integrated_midpoint",
        ),
        AdaptiveCurriculumCandidate(
            "midpoint_adapter_gentle",
            0.15,
            False,
            0.0,
            "integrated_midpoint",
        ),
        AdaptiveCurriculumCandidate(
            "midpoint_adapter_gentle_mild_curriculum",
            0.15,
            True,
            0.0,
            "integrated_midpoint",
            1.0,
        ),
        AdaptiveCurriculumCandidate(
            "midpoint_adapter_moderate_mild_curriculum",
            0.25,
            True,
            0.0,
            "integrated_midpoint",
            1.0,
        ),
    )


def evaluate_midpoint_adapter_objective(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_directory: str | Path,
    config: AdaptiveCurriculumObjectiveConfig | None = None,
):
    configured = replace(
        config or AdaptiveCurriculumObjectiveConfig(),
        candidates=midpoint_adapter_candidates(),
    )
    result = evaluate_adaptive_curriculum_objective(
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_directory=checkpoint_directory,
        config=configured,
    )
    if result["experiment"] != ADAPTIVE_CURRICULUM_OBJECTIVE_SCHEMA_VERSION:
        raise AssertionError("unexpected adaptive-curriculum result schema")
    result["experiment"] = MIDPOINT_ADAPTER_OBJECTIVE_SCHEMA_VERSION
    if result["selected_candidate"] is not None:
        result["recommendation"] = "replicate_v7_candidate_before_fresh_test"
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hard midpoint dynamics with the V2.1 adapter loss."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_train.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_validation.npz"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_development.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_midpoint_adapter_objective(
        train_path=args.train_data,
        validation_path=args.validation_data,
        checkpoint_directory=args.checkpoint_directory,
        config=AdaptiveCurriculumObjectiveConfig(
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
