"""V14 constrained sequence-oracle screen around the deployable controller."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .adaptive_curriculum_objective import selected_adaptive_position_v21_config
from .adaptive_position_supervision import compute_adaptive_position_supervision
from .config import ObservationProfile
from .control_criticality import (
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from .dataset import FEATURE_NAMES, load_gimbal_dataset
from .gru import angular_residual_rad, load_gru_checkpoint
from .multi_command_experiment import _window_batch
from .multi_command_policy import (
    CausalRecurrentPositionResidualPolicy,
    RecurrentPositionResidualPolicyConfig,
    recurrent_policy_input_dim,
    rollout_counterfactual_window,
)
from .sequence_oracle import (
    PrivilegedSequenceOracleConfig,
    optimize_privileged_command_sequence,
)
from .sequence_oracle_experiment import (
    _Accumulator,
    _accumulate,
    _gate,
    _sha256,
)


DEPLOYABLE_SEQUENCE_ORACLE_SCHEMA_VERSION = (
    "gimbal_deployable_sequence_oracle_v14_development_v1"
)


@dataclass(frozen=True)
class DeployableSequenceOracleExperimentConfig:
    behavior_name: str = "privileged_oracle_position"
    focus_start_index: int = 0
    focus_steps: int = 16
    batch_size: int = 8
    maximum_episode_count: int = 48
    minimum_episode_count: int = 24
    minimum_tracking_improvement_fraction: float = 0.005
    device: str = "cpu"
    zero_policy_hidden_dim: int = 8
    zero_policy_embedding_dim: int = 8
    oracle: PrivilegedSequenceOracleConfig = PrivilegedSequenceOracleConfig(
        focus_start_index=0,
        focus_steps=16,
        optimization_iterations=24,
    )
    criticality: ControlCriticalityConfig = ControlCriticalityConfig()

    def __post_init__(self) -> None:
        if not self.behavior_name:
            raise ValueError("deployable-oracle behavior must not be empty")
        if self.focus_start_index < 0:
            raise ValueError("deployable-oracle focus start must be non-negative")
        for name in (
            "focus_steps",
            "batch_size",
            "maximum_episode_count",
            "minimum_episode_count",
            "zero_policy_hidden_dim",
            "zero_policy_embedding_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"deployable-oracle {name} must be positive")
        if self.maximum_episode_count < self.minimum_episode_count:
            raise ValueError(
                "deployable-oracle maximum episodes must cover the minimum"
            )
        if not math.isfinite(
            self.minimum_tracking_improvement_fraction
        ) or self.minimum_tracking_improvement_fraction < 0.0:
            raise ValueError(
                "deployable-oracle tracking threshold must be non-negative"
            )
        if self.focus_start_index != self.oracle.focus_start_index or (
            self.focus_steps != self.oracle.focus_steps
        ):
            raise ValueError(
                "deployable-oracle focus must match its optimizer configuration"
            )


def evaluate_deployable_sequence_oracle_experiment(
    *,
    validation_path: str | Path,
    base_checkpoint: str | Path,
    config: DeployableSequenceOracleExperimentConfig | None = None,
) -> dict[str, Any]:
    """Test the privileged ceiling before training another residual policy."""

    config = config or DeployableSequenceOracleExperimentConfig()
    validation_path = Path(validation_path)
    base_checkpoint = Path(base_checkpoint)
    dataset = load_gimbal_dataset(validation_path)
    if config.behavior_name not in dataset.manifest.behavior_names:
        raise ValueError("deployable-oracle behavior is absent from the dataset")
    behavior = dataset.manifest.behavior_names.index(config.behavior_name)
    episode_indices = np.flatnonzero(dataset.behavior_index == behavior)[
        : config.maximum_episode_count
    ]
    if len(episode_indices) < config.minimum_episode_count:
        raise ValueError("deployable-oracle development set is too small")
    replay_steps = config.focus_start_index + config.focus_steps
    lengths = np.sum(dataset.sequence_mask[episode_indices], axis=1)
    if np.any(lengths <= replay_steps):
        raise ValueError("deployable-oracle focus exceeds an episode")

    base_model, base_metadata = load_gru_checkpoint(
        base_checkpoint,
        device=config.device,
    )
    if base_model.config.mean_parameterization != "integrated_midpoint":
        raise ValueError("V14 requires the hard-midpoint state predictor")
    if base_model.config.prediction_horizons_s != (
        dataset.manifest.prediction_horizons_s
    ):
        raise ValueError("V14 checkpoint horizons differ from the dataset")
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()

    device = torch.device(config.device)
    zero_policy = CausalRecurrentPositionResidualPolicy(
        RecurrentPositionResidualPolicyConfig(
            input_dim=recurrent_policy_input_dim(base_model.horizon_count),
            hidden_dim=config.zero_policy_hidden_dim,
            embedding_dim=config.zero_policy_embedding_dim,
        )
    ).to(device)
    zero_policy.eval()
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
    previous_action_index = FEATURE_NAMES.index("previous_action_normalized")
    focus = slice(
        config.focus_start_index,
        config.focus_start_index + config.focus_steps,
    )
    reference_global = _Accumulator()
    reference_critical = _Accumulator()
    candidate_global = _Accumulator()
    candidate_critical = _Accumulator()
    selected_blends = []
    improved_episode_count = 0
    evaluated_episode_count = 0
    plant_replay_maximum_angle_error_rad = 0.0
    reference_to_logged_action_squared_sum = 0.0
    reference_to_logged_action_count = 0
    run_records = []

    for offset in range(0, len(episode_indices), config.batch_size):
        selected = episode_indices[offset : offset + config.batch_size]
        batch = _window_batch(
            dataset,
            supervision,
            selected,
            0,
            replay_steps,
            device,
        )
        with torch.no_grad():
            reference_rollout = rollout_counterfactual_window(
                base_model,
                zero_policy,
                batch,
                prediction_horizons_s=dataset.manifest.prediction_horizons_s,
                adapter=adapter,
                visibility_margin_fraction=(
                    config.oracle.visibility_margin_fraction
                ),
            )
        target_after = batch.target_bearing_rad[:, 1:]
        previous_action = batch.logged_features[
            :, 0, previous_action_index
        ]
        oracle = optimize_privileged_command_sequence(
            reference_rollout.command_normalized,
            target_after,
            batch.context,
            batch.sequence_mask,
            batch.time_s[:, 0],
            previous_action,
            config=config.oracle,
        )
        plant_replay_maximum_angle_error_rad = max(
            plant_replay_maximum_angle_error_rad,
            float(
                torch.max(
                    torch.abs(
                        oracle.base_angle_rad
                        - reference_rollout.gimbal_angle_rad
                    )
                )
            ),
        )
        focus_mask = batch.sequence_mask[:, focus]
        half_fov = 0.5 * batch.context.selected_axis_fov_rad[:, focus]
        base_tracking = angular_residual_rad(
            target_after[:, focus],
            oracle.base_angle_rad[:, focus],
        ) / half_fov
        selected_tracking = angular_residual_rad(
            target_after[:, focus],
            oracle.selected_angle_rad[:, focus],
        ) / half_fov
        base_visibility = torch.relu(
            torch.abs(base_tracking)
            - config.oracle.visibility_margin_fraction
        )
        selected_visibility = torch.relu(
            torch.abs(selected_tracking)
            - config.oracle.visibility_margin_fraction
        )
        preceding_base = (
            previous_action
            if config.focus_start_index == 0
            else oracle.base_command_normalized[
                :, config.focus_start_index - 1
            ]
        )
        preceding_selected = (
            previous_action
            if config.focus_start_index == 0
            else oracle.selected_command_normalized[
                :, config.focus_start_index - 1
            ]
        )
        base_difference = torch.diff(
            torch.cat(
                (
                    preceding_base[:, None],
                    oracle.base_command_normalized[:, focus],
                ),
                dim=1,
            ),
            dim=1,
        )
        selected_difference = torch.diff(
            torch.cat(
                (
                    preceding_selected[:, None],
                    oracle.selected_command_normalized[:, focus],
                ),
                dim=1,
            ),
            dim=1,
        )
        residual = (
            oracle.selected_command_normalized[:, focus]
            - oracle.base_command_normalized[:, focus]
        )
        zero_residual = torch.zeros_like(residual)
        critical_mask = torch.from_numpy(
            criticality.critical_mask[
                selected,
                config.focus_start_index
                + 1 : config.focus_start_index
                + config.focus_steps
                + 1,
                0,
            ]
        ).bool().to(device)
        global_mask = focus_mask
        selected_critical = focus_mask & critical_mask
        for accumulator, tracking, visibility, difference, saturation, command_residual, mask in (
            (
                reference_global,
                base_tracking,
                base_visibility,
                base_difference,
                oracle.base_saturation_fraction[:, focus],
                zero_residual,
                global_mask,
            ),
            (
                reference_critical,
                base_tracking,
                base_visibility,
                base_difference,
                oracle.base_saturation_fraction[:, focus],
                zero_residual,
                selected_critical,
            ),
            (
                candidate_global,
                selected_tracking,
                selected_visibility,
                selected_difference,
                oracle.selected_saturation_fraction[:, focus],
                residual,
                global_mask,
            ),
            (
                candidate_critical,
                selected_tracking,
                selected_visibility,
                selected_difference,
                oracle.selected_saturation_fraction[:, focus],
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
                mask=mask,
            )
        logged_action = torch.from_numpy(
            dataset.oracle_actions[selected, :replay_steps, 1]
        ).float().to(device)
        reference_to_logged_action_squared_sum += float(
            (
                reference_rollout.command_normalized[:, focus]
                - logged_action[:, focus]
            )
            .square()
            .mul(focus_mask)
            .sum()
        )
        reference_to_logged_action_count += int(focus_mask.sum())
        improved = (
            oracle.selected_metrics["tracking_mse"]
            < oracle.base_metrics["tracking_mse"] - 1e-12
        )
        improved_episode_count += int(improved.sum())
        evaluated_episode_count += len(selected)
        selected_blends.extend(oracle.selected_blend_fraction.cpu().tolist())
        run_records.append(
            {
                "episode_indices": selected.tolist(),
                "mean_selected_blend_fraction": float(
                    oracle.selected_blend_fraction.mean()
                ),
                "improved_episode_count": int(improved.sum()),
                "optimization_initial": oracle.optimization_history[0],
                "optimization_final": oracle.optimization_history[-1],
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
        "experiment": DEPLOYABLE_SEQUENCE_ORACLE_SCHEMA_VERSION,
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
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "sha256": _sha256(base_checkpoint),
            "metadata": base_metadata,
        },
        "reference_scope": {
            "state_predictor": "frozen_hard_midpoint_gru",
            "position_adapter": "v2_1_hardware_aware",
            "policy_inputs": "deployable_o2_only",
            "counterfactual_observation_feedback": True,
            "recurrent_state_from_episode_start": True,
            "persistent_command_latency_queue": True,
            "trainable_reference_parameters": 0,
        },
        "oracle_scope": {
            "future_target_bearing_is_privileged": True,
            "commands_before_focus_frozen": True,
            "replayed_from_episode_start": True,
            "exact_blend_selection": True,
            "reference_fallback_always_available": True,
        },
        "reference": reference,
        "candidate": candidate,
        "gate": gate,
        "diagnostics": {
            "evaluated_episode_windows": evaluated_episode_count,
            "improved_episode_fraction": (
                improved_episode_count / max(1, evaluated_episode_count)
            ),
            "mean_selected_blend_fraction": float(np.mean(selected_blends)),
            "zero_blend_fraction": float(
                np.mean(np.asarray(selected_blends) == 0.0)
            ),
            "plant_replay_maximum_angle_error_rad": (
                plant_replay_maximum_angle_error_rad
            ),
            "reference_to_logged_privileged_action_rmse_normalized": math.sqrt(
                reference_to_logged_action_squared_sum
                / max(1, reference_to_logged_action_count)
            ),
        },
        "runs": run_records,
        "recommendation": (
            "authorize_state_consistent_gated_distillation"
            if gate["passed"]
            else "do_not_train_deployable_reference_residual"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run V14 deployable-reference sequence-oracle screen."
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("artifacts/gimbal_midpoint_adapter_validation.npz"),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path(
            "artifacts/gimbal_midpoint_adapter_replication_checkpoints/"
            "gimbal_v7_midpoint_state_reference_seed_29.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gimbal_deployable_sequence_oracle_v14.json"),
    )
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--focus-steps", type=int, default=16)
    parser.add_argument("--maximum-episodes", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    defaults = DeployableSequenceOracleExperimentConfig()
    result = evaluate_deployable_sequence_oracle_experiment(
        validation_path=args.validation_data,
        base_checkpoint=args.base_checkpoint,
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
