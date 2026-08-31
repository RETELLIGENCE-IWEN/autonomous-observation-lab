"""Fresh corrective protocol for dual-risk constrained position V3.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .controllers import PredictivePositionOptimizerConfig
from .performance_atlas import PerformanceContract, load_performance_contract
from .predictive_position_v3 import (
    DEFAULT_CONFIRMATION_SEEDS as V3_CONFIRMATION_SEEDS,
    DEFAULT_DEVELOPMENT_SEEDS as V3_DEVELOPMENT_SEEDS,
    HISTORICAL_WORLD_SEEDS as PRE_V3_WORLD_SEEDS,
    PREDICTIVE_POSITION_V3_SCHEMA_VERSION,
    PredictivePositionV3Candidate,
    PredictivePositionV3ProtocolConfig,
    evaluate_predictive_position_v3,
)


PREDICTIVE_POSITION_V31_SCHEMA_VERSION = (
    "gimbal_dual_risk_predictive_position_v31_protocol_v1"
)
DEFAULT_DEVELOPMENT_SEEDS = tuple(range(85000, 85008))
DEFAULT_CONFIRMATION_SEEDS = tuple(range(86000, 86008))
HISTORICAL_WORLD_SEEDS = (
    PRE_V3_WORLD_SEEDS + V3_DEVELOPMENT_SEEDS + V3_CONFIRMATION_SEEDS
)


def default_v31_candidates() -> tuple[PredictivePositionV3Candidate, ...]:
    """Predeclared dual-risk gates; all other V3 settings remain frozen."""

    shared: dict[str, Any] = {
        "maximum_optimization_horizon_s": 0.10,
        "terminal_tracking_weight": 1.0,
        "rate_matching_weight": 0.10,
        "visibility_weight": 4.0,
        "command_effect_response_fraction": 0.50,
        "minimum_optimizer_position_gain_s_inv": 4.0,
        "command_change_weight": 0.06,
        "command_rate_change_weight": 0.010,
    }
    return (
        PredictivePositionV3Candidate(
            "dual_risk_early",
            PredictivePositionOptimizerConfig(
                **shared,
                activation_gate_mode="minimum",
                activation_rate_onset_fraction=0.30,
                activation_rate_full_fraction=0.55,
                activation_visibility_onset_fraction=0.45,
                activation_visibility_full_fraction=0.70,
            ),
        ),
        PredictivePositionV3Candidate(
            "dual_risk_balanced",
            PredictivePositionOptimizerConfig(
                **shared,
                activation_gate_mode="minimum",
                activation_rate_onset_fraction=0.40,
                activation_rate_full_fraction=0.65,
                activation_visibility_onset_fraction=0.55,
                activation_visibility_full_fraction=0.78,
            ),
        ),
        PredictivePositionV3Candidate(
            "dual_risk_late",
            PredictivePositionOptimizerConfig(
                **shared,
                activation_gate_mode="minimum",
                activation_rate_onset_fraction=0.50,
                activation_rate_full_fraction=0.72,
                activation_visibility_onset_fraction=0.65,
                activation_visibility_full_fraction=0.85,
            ),
        ),
        PredictivePositionV3Candidate(
            "dual_risk_product",
            PredictivePositionOptimizerConfig(
                **shared,
                activation_gate_mode="product",
                activation_rate_onset_fraction=0.25,
                activation_rate_full_fraction=0.55,
                activation_visibility_onset_fraction=0.40,
                activation_visibility_full_fraction=0.70,
            ),
        ),
    )


def default_v31_protocol(*, device: str = "cpu") -> PredictivePositionV3ProtocolConfig:
    """Return the frozen V3.1 selection and confirmation requirements."""

    return PredictivePositionV3ProtocolConfig(
        development_max_mean_error_regression_fov_fraction=0.005,
        development_max_p95_regression_fov_fraction=0.005,
        device=device,
        candidates=default_v31_candidates(),
    )


def _rename_v31_result(result: dict[str, Any]) -> None:
    result["experiment"] = PREDICTIVE_POSITION_V31_SCHEMA_VERSION
    confirmation = result["confirmation"]
    if "predictive_position_v3" in confirmation:
        confirmation["predictive_position_v31"] = confirmation.pop(
            "predictive_position_v3"
        )
    if confirmation.get("opened"):
        confirmation["recommendation"] = (
            "constrained_predictive_position_v31"
            if confirmation["acceptance_gate"]["passed"]
            else "retain_visibility_risk_v21"
        )
    trace = result.get("representative_trace", {})
    for record in trace.get("records", []):
        for old_name in tuple(record):
            if old_name.startswith("v3_"):
                record["v31_" + old_name[3:]] = record.pop(old_name)


def evaluate_predictive_position_v31(
    *,
    v3_results: str | Path,
    visibility_risk_results: str | Path,
    protocol: PredictivePositionV3ProtocolConfig | None = None,
    contract: PerformanceContract | None = None,
    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS,
    confirmation_seeds: tuple[int, ...] = DEFAULT_CONFIRMATION_SEEDS,
) -> dict[str, Any]:
    """Develop on 85k worlds and open 86k only for one frozen winner."""

    predecessor_path = Path(v3_results)
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("experiment") != PREDICTIVE_POSITION_V3_SCHEMA_VERSION:
        raise ValueError("V3.1 requires the completed V3 protocol result")
    predecessor_confirmation = predecessor.get("confirmation", {})
    if not predecessor_confirmation.get("opened"):
        raise ValueError("V3.1 requires an opened V3 confirmation")
    if predecessor_confirmation.get("recommendation") != "retain_visibility_risk_v21":
        raise ValueError("V3.1 is only corrective after a rejected V3 candidate")

    requested_seeds = set(development_seeds) | set(confirmation_seeds)
    if requested_seeds & set(HISTORICAL_WORLD_SEEDS):
        raise ValueError("V3.1 cannot reuse historical evaluation seeds")

    visibility_path = Path(visibility_risk_results)
    visibility_hash = hashlib.sha256(visibility_path.read_bytes()).hexdigest()
    if visibility_hash != predecessor.get("source_sha256"):
        raise ValueError("V3.1 V2.1 source does not match the V3 predecessor")

    result = evaluate_predictive_position_v3(
        visibility_risk_results=visibility_path,
        protocol=protocol or default_v31_protocol(),
        contract=contract or PerformanceContract(),
        development_seeds=development_seeds,
        confirmation_seeds=confirmation_seeds,
    )
    _rename_v31_result(result)
    result["predecessor_v3_result"] = str(predecessor_path)
    result["predecessor_v3_sha256"] = hashlib.sha256(
        predecessor_path.read_bytes()
    ).hexdigest()
    result["correction"] = (
        "Require joint predicted rate-capacity and visibility risk before "
        "departing from the frozen V2.1 fallback."
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and confirm dual-risk predictive position V3.1."
    )
    parser.add_argument(
        "--v3-results",
        type=Path,
        default=Path("artifacts/gimbal_predictive_position_v3.json"),
    )
    parser.add_argument(
        "--visibility-risk-results",
        type=Path,
        default=Path("artifacts/gimbal_adaptive_position_v21.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_predictive_position_v31.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/gimbal_performance_contract.json"),
    )
    parser.add_argument("--development-seed", type=int, action="append")
    parser.add_argument("--confirmation-seed", type=int, action="append")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = evaluate_predictive_position_v31(
        v3_results=args.v3_results,
        visibility_risk_results=args.visibility_risk_results,
        protocol=default_v31_protocol(device=args.device),
        contract=load_performance_contract(args.contract),
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
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    confirmation = result["confirmation"]
    if confirmation["opened"]:
        gate = confirmation["acceptance_gate"]
        print(
            "V3.1 confirmation: "
            f"{'PASS' if gate['passed'] else 'FAIL'}; "
            f"recommendation={confirmation['recommendation']}"
        )
    else:
        print("V3.1 confirmation remained closed")


if __name__ == "__main__":
    main()
