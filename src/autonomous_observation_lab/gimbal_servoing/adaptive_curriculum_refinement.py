"""Fresh V6.1 refinement of adapter-loss and dynamic-consistency balance."""

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


ADAPTIVE_CURRICULUM_REFINEMENT_SCHEMA_VERSION = (
    "gimbal_adaptive_curriculum_objective_v61_refinement_v1"
)


def adaptive_curriculum_refinement_candidates() -> tuple[
    AdaptiveCurriculumCandidate, ...
]:
    """Predeclare the coefficient balance before opening the fresh block."""

    return (
        AdaptiveCurriculumCandidate("v4_reference", 0.0, False, 25.0),
        AdaptiveCurriculumCandidate("combined_25", 0.25, True, 25.0),
        AdaptiveCurriculumCandidate("combined_50", 0.25, True, 50.0),
        AdaptiveCurriculumCandidate("combined_100", 0.25, True, 100.0),
        AdaptiveCurriculumCandidate("combined_gentle_50", 0.15, True, 50.0),
    )


def evaluate_adaptive_curriculum_refinement(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    checkpoint_directory: str | Path,
    config: AdaptiveCurriculumObjectiveConfig | None = None,
):
    configured = replace(
        config or AdaptiveCurriculumObjectiveConfig(),
        candidates=adaptive_curriculum_refinement_candidates(),
    )
    result = evaluate_adaptive_curriculum_objective(
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_directory=checkpoint_directory,
        config=configured,
    )
    if result["experiment"] != ADAPTIVE_CURRICULUM_OBJECTIVE_SCHEMA_VERSION:
        raise AssertionError("unexpected adaptive-curriculum result schema")
    result["experiment"] = ADAPTIVE_CURRICULUM_REFINEMENT_SCHEMA_VERSION
    if result["selected_candidate"] is not None:
        result["recommendation"] = "replicate_v61_candidate_before_fresh_test"
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine V6 adapter and dynamic-consistency loss balance."
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/gimbal_control_aware_train.npz"),
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_curriculum_refinement_validation.npz"),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_curriculum_refinement_checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_curriculum_refinement.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_adaptive_curriculum_refinement(
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
