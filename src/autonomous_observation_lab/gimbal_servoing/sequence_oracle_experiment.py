"""V11 development screen for a privileged constrained sequence oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .adaptive_curriculum_objective import (
    selected_adaptive_position_v21_config,
)
from .adaptive_position_supervision import (
    compute_adaptive_position_supervision,
)
from .config import ObservationProfile
from .control_criticality import (
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from .dataset import FEATURE_NAMES, load_gimbal_dataset
from .gru import angular_residual_rad
from .multi_command_experiment import _context_window
from .sequence_oracle import (
    PrivilegedSequenceOracleConfig,
    optimize_privileged_command_sequence,
)


SEQUENCE_ORACLE_EXPERIMENT_SCHEMA_VERSION = (
    "gimbal_privileged_sequence_oracle_v11_development_v1"
)


@dataclass(frozen=True)
class SequenceOracleExperimentConfig:
    behavior_name: str = "privileged_oracle_position"
    focus_start_indices: tuple[int, ...] = (0, 48, 120)
    focus_steps: int = 16
    batch_size: int = 8
    maximum_episode_count: int = 48
    minimum_episode_count: int = 24
    minimum_tracking_improvement_fraction: float = 0.005
    device: str = "cpu"
    oracle: PrivilegedSequenceOracleConfig = (
        PrivilegedSequenceOracleConfig()
    )
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        if not self.behavior_name:
            raise ValueError("sequence-oracle behavior name must not be empty")
        if not self.focus_start_indices or any(
            value < 0 for value in self.focus_start_indices
        ):
            raise ValueError(
                "sequence-oracle focus starts must be non-negative"
            )
        if len(set(self.focus_start_indices)) != len(
            self.focus_start_indices
        ):
            raise ValueError("sequence-oracle focus starts must be unique")
        for name in (
            "focus_steps",
            "batch_size",
            "maximum_episode_count",
            "minimum_episode_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"sequence-oracle {name} must be positive")
        if self.maximum_episode_count < self.minimum_episode_count:
            raise ValueError(
                "sequence-oracle maximum episodes must cover the minimum"
            )
        if not math.isfinite(
            self.minimum_tracking_improvement_fraction
        ) or self.minimum_tracking_improvement_fraction < 0.0:
            raise ValueError(
                "sequence-oracle tracking threshold must be non-negative"
            )


@dataclass
class _Accumulator:
    tracking_squared_sum: float = 0.0
    visibility_squared_sum: float = 0.0
    smoothness_squared_sum: float = 0.0
    saturation_sum: float = 0.0
    residual_squared_sum: float = 0.0
    count: int = 0

    def report(self) -> dict[str, float | int]:
        count = max(1, self.count)
        return {
            "sample_count": self.count,
            "tracking_rmse_normalized": math.sqrt(
                self.tracking_squared_sum / count
            ),
            "visibility_rmse_normalized": math.sqrt(
                self.visibility_squared_sum / count
            ),
            "smoothness_rmse_normalized": math.sqrt(
                self.smoothness_squared_sum / count
            ),
            "saturation_rmse_normalized": math.sqrt(
                self.saturation_sum / count
            ),
            "command_residual_rmse_normalized": math.sqrt(
                self.residual_squared_sum / count
            ),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else math.inf
    return candidate / reference - 1.0


def _accumulate(
    accumulator: _Accumulator,
    *,
    tracking_error: torch.Tensor,
    visibility: torch.Tensor,
    command_difference: torch.Tensor,
    saturation: torch.Tensor,
    command_residual: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    count = int(mask.sum().item())
    if count == 0:
        return
    accumulator.tracking_squared_sum += float(
        tracking_error[mask].square().sum()
    )
    accumulator.visibility_squared_sum += float(
        visibility[mask].square().sum()
    )
    accumulator.smoothness_squared_sum += float(
        command_difference[mask].square().sum()
    )
    accumulator.saturation_sum += float(saturation[mask].sum())
    accumulator.residual_squared_sum += float(
        command_residual[mask].square().sum()
    )
    accumulator.count += count


def _gate(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    config: SequenceOracleExperimentConfig,
) -> dict[str, Any]:
    changes = {}
    for scope in ("global", "critical"):
        for metric, reference_value in reference[scope].items():
            if metric in {"sample_count", "command_residual_rmse_normalized"}:
                continue
            changes[f"{scope}_{metric}"] = _relative_change(
                float(candidate[scope][metric]),
                float(reference_value),
            )
    oracle = config.oracle
    checks = {
        "global_tracking": changes["global_tracking_rmse_normalized"]
        <= -config.minimum_tracking_improvement_fraction,
        "critical_tracking": changes["critical_tracking_rmse_normalized"]
        <= -config.minimum_tracking_improvement_fraction,
        "global_visibility": changes["global_visibility_rmse_normalized"]
        <= oracle.maximum_visibility_regression_fraction,
        "critical_visibility": changes[
            "critical_visibility_rmse_normalized"
        ]
        <= oracle.maximum_visibility_regression_fraction,
        "global_smoothness": changes["global_smoothness_rmse_normalized"]
        <= oracle.maximum_smoothness_regression_fraction,
        "critical_smoothness": changes[
            "critical_smoothness_rmse_normalized"
        ]
        <= oracle.maximum_smoothness_regression_fraction,
        "global_saturation": changes["global_saturation_rmse_normalized"]
        <= oracle.maximum_saturation_regression_fraction,
        "critical_saturation": changes[
            "critical_saturation_rmse_normalized"
        ]
        <= oracle.maximum_saturation_regression_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_changes": changes,
    }


def evaluate_sequence_oracle_experiment(
    *,
    validation_path: str | Path,
    config: SequenceOracleExperimentConfig | None = None,
) -> dict[str, Any]:
    """Screen a privileged sequence ceiling without opening fresh test data."""

    config = config or SequenceOracleExperimentConfig()
    validation_path = Path(validation_path)
    dataset = load_gimbal_dataset(validation_path)
    if config.behavior_name not in dataset.manifest.behavior_names:
        raise ValueError("sequence-oracle behavior is absent from the dataset")
    behavior_index = dataset.manifest.behavior_names.index(
        config.behavior_name
    )
    episode_indices = np.flatnonzero(
        dataset.behavior_index == behavior_index
    )[: config.maximum_episode_count]
    if len(episode_indices) < config.minimum_episode_count:
        raise ValueError("sequence-oracle development set is too small")
    lengths = np.sum(dataset.sequence_mask[episode_indices], axis=1)
    maximum_end = max(config.focus_start_indices) + config.focus_steps
    if np.any(lengths <= maximum_end):
        raise ValueError("sequence-oracle focus exceeds an episode")

    adapter = selected_adaptive_position_v21_config()
    supervision = compute_adaptive_position_supervision(
        dataset,
        adapter=adapter,
        profile=ObservationProfile.DISTURBANCE_AWARE,
    )
    criticality = compute_control_criticality(
        dataset,
        config=config.criticality,
    )
    profile_index = dataset.manifest.observation_profiles.index(
        ObservationProfile.DISTURBANCE_AWARE.value
    )
    previous_action_index = FEATURE_NAMES.index(
        "previous_action_normalized"
    )
    angle_index = FEATURE_NAMES.index("gimbal_angle_rad")
    device = torch.device(config.device)
    reference_global = _Accumulator()
    reference_critical = _Accumulator()
    candidate_global = _Accumulator()
    candidate_critical = _Accumulator()
    selected_blends: list[float] = []
    improved_episode_count = 0
    evaluated_episode_count = 0
    replay_maximum_angle_error_rad = 0.0
    run_records = []

    for start in config.focus_start_indices:
        end = start + config.focus_steps
        replay_end = end
        for offset in range(0, len(episode_indices), config.batch_size):
            selected = episode_indices[offset : offset + config.batch_size]
            context = _context_window(
                supervision,
                selected,
                0,
                replay_end,
                device,
            )
            command = torch.from_numpy(
                dataset.oracle_actions[selected, :replay_end, 1]
            ).float().to(device)
            target = torch.from_numpy(
                dataset.targets[selected, 1 : replay_end + 1, 0, 0]
            ).float().to(device)
            mask = torch.from_numpy(
                dataset.sequence_mask[selected, :replay_end]
            ).bool().to(device)
            initial_time = torch.from_numpy(
                dataset.time_s[selected, 0]
            ).float().to(device)
            previous_action = torch.from_numpy(
                dataset.features[
                    selected,
                    profile_index,
                    0,
                    previous_action_index,
                ]
            ).float().to(device)
            oracle_config = replace(
                config.oracle,
                focus_start_index=start,
                focus_steps=config.focus_steps,
            )
            result = optimize_privileged_command_sequence(
                command,
                target,
                context,
                mask,
                initial_time,
                previous_action,
                config=oracle_config,
            )
            focus_slice = slice(start, end)
            focus_mask = mask[:, focus_slice]
            half_fov = 0.5 * context.selected_axis_fov_rad[:, focus_slice]
            base_tracking = angular_residual_rad(
                target[:, focus_slice],
                result.base_angle_rad[:, focus_slice],
            ) / half_fov
            selected_tracking = angular_residual_rad(
                target[:, focus_slice],
                result.selected_angle_rad[:, focus_slice],
            ) / half_fov
            base_visibility = torch.relu(
                torch.abs(base_tracking)
                - oracle_config.visibility_margin_fraction
            )
            selected_visibility = torch.relu(
                torch.abs(selected_tracking)
                - oracle_config.visibility_margin_fraction
            )
            preceding_base = (
                previous_action
                if start == 0
                else result.base_command_normalized[:, start - 1]
            )
            preceding_selected = (
                previous_action
                if start == 0
                else result.selected_command_normalized[:, start - 1]
            )
            base_difference = torch.diff(
                torch.cat(
                    (
                        preceding_base[:, None],
                        result.base_command_normalized[:, focus_slice],
                    ),
                    dim=1,
                ),
                dim=1,
            )
            selected_difference = torch.diff(
                torch.cat(
                    (
                        preceding_selected[:, None],
                        result.selected_command_normalized[:, focus_slice],
                    ),
                    dim=1,
                ),
                dim=1,
            )
            residual = (
                result.selected_command_normalized[:, focus_slice]
                - result.base_command_normalized[:, focus_slice]
            )
            zero_residual = torch.zeros_like(residual)
            critical_mask = torch.from_numpy(
                criticality.critical_mask[
                    selected,
                    start + 1 : end + 1,
                    0,
                ]
            ).bool().to(device)
            global_mask = focus_mask
            selected_critical = focus_mask & critical_mask
            base_saturation = result.base_saturation_fraction[:, focus_slice]
            selected_saturation = (
                result.selected_saturation_fraction[:, focus_slice]
            )
            for accumulator, tracking, visibility, difference, saturation, command_residual, selected_mask in (
                (
                    reference_global,
                    base_tracking,
                    base_visibility,
                    base_difference,
                    base_saturation,
                    zero_residual,
                    global_mask,
                ),
                (
                    reference_critical,
                    base_tracking,
                    base_visibility,
                    base_difference,
                    base_saturation,
                    zero_residual,
                    selected_critical,
                ),
                (
                    candidate_global,
                    selected_tracking,
                    selected_visibility,
                    selected_difference,
                    selected_saturation,
                    residual,
                    global_mask,
                ),
                (
                    candidate_critical,
                    selected_tracking,
                    selected_visibility,
                    selected_difference,
                    selected_saturation,
                    residual,
                    selected_critical,
                ),
            ):
                _accumulate(
                    accumulator,
                    tracking_error=tracking,
                    visibility=visibility,
                    command_difference=difference,
                    saturation=saturation,
                    command_residual=command_residual,
                    mask=selected_mask,
                )
            base_episode = result.base_metrics["tracking_mse"]
            selected_episode = result.selected_metrics["tracking_mse"]
            improved_episode_count += int(
                torch.sum(selected_episode < base_episode - 1e-12).item()
            )
            evaluated_episode_count += len(selected)
            selected_blends.extend(
                result.selected_blend_fraction.cpu().tolist()
            )
            if replay_end < dataset.sequence_mask.shape[1]:
                logged_next_angle = torch.from_numpy(
                    dataset.features[
                        selected,
                        profile_index,
                        1 : replay_end + 1,
                        angle_index,
                    ]
                ).float().to(device)
                replay_maximum_angle_error_rad = max(
                    replay_maximum_angle_error_rad,
                    float(
                        torch.max(
                            torch.abs(
                                result.base_angle_rad - logged_next_angle
                            )
                        )
                    ),
                )
            run_records.append(
                {
                    "focus_start_index": start,
                    "episode_indices": selected.tolist(),
                    "mean_selected_blend_fraction": float(
                        torch.mean(result.selected_blend_fraction)
                    ),
                    "improved_episode_count": int(
                        torch.sum(selected_episode < base_episode - 1e-12)
                    ),
                    "optimization_initial": result.optimization_history[0],
                    "optimization_final": result.optimization_history[-1],
                }
            )

    reference = {
        "global": reference_global.report(),
        "critical": reference_critical.report(),
    }
    candidate = {
        "global": candidate_global.report(),
        "critical": candidate_critical.report(),
    }
    gate = _gate(candidate, reference, config)
    return {
        "experiment": SEQUENCE_ORACLE_EXPERIMENT_SCHEMA_VERSION,
        "config": asdict(config),
        "dataset": {
            "path": str(validation_path),
            "sha256": _sha256(validation_path),
            "selected_behavior": config.behavior_name,
            "selected_episode_count": len(episode_indices),
            "criticality": control_criticality_report(
                dataset,
                criticality,
                config=config.criticality,
            ),
            "fresh_test": {"opened": False},
        },
        "oracle_scope": {
            "privileged_inputs": [
                "future_body_relative_target_bearing",
                "serialized_servo_and_camera_configuration",
                "exact_episode_start_state",
            ],
            "baseline": "logged_privileged_oracle_position_commands",
            "commands_before_focus_frozen": True,
            "replayed_from_episode_start": True,
            "exact_blend_selection": True,
        },
        "reference": reference,
        "candidate": candidate,
        "gate": gate,
        "diagnostics": {
            "evaluated_episode_windows": evaluated_episode_count,
            "improved_episode_fraction": (
                improved_episode_count / max(1, evaluated_episode_count)
            ),
            "mean_selected_blend_fraction": float(
                np.mean(selected_blends)
            ),
            "zero_blend_fraction": float(
                np.mean(np.asarray(selected_blends) == 0.0)
            ),
            "baseline_replay_maximum_angle_error_rad": (
                replay_maximum_angle_error_rad
            ),
        },
        "runs": run_records,
        "recommendation": (
            "distill_privileged_sequence_oracle"
            if gate["passed"]
            else "revise_sequence_oracle_before_distillation"
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen the V11 privileged constrained sequence oracle."
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_validation.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_sequence_oracle_v11.json"),
    )
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--focus-steps", type=int, default=16)
    parser.add_argument("--maximum-episodes", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    defaults = SequenceOracleExperimentConfig()
    result = evaluate_sequence_oracle_experiment(
        validation_path=args.validation_data,
        config=replace(
            defaults,
            focus_steps=args.focus_steps,
            maximum_episode_count=args.maximum_episodes,
            device=args.device,
            oracle=replace(
                defaults.oracle,
                focus_steps=args.focus_steps,
                optimization_iterations=args.iterations,
            ),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"passed={result['gate']['passed']}; "
        f"recommendation={result['recommendation']}"
    )


if __name__ == "__main__":
    main()
