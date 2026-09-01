import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.adaptive_curriculum_objective import (
    AdaptiveCurriculumObjectiveConfig,
    default_adaptive_curriculum_candidates,
    evaluate_adaptive_curriculum_objective,
)
from autonomous_observation_lab.gimbal_servoing.adaptive_curriculum_refinement import (
    adaptive_curriculum_refinement_candidates,
)
from autonomous_observation_lab.gimbal_servoing.config import ObservationProfile
from autonomous_observation_lab.gimbal_servoing.control_aware_predictor import (
    ControlAwarePredictorProtocolConfig,
    default_control_aware_candidates,
    evaluate_control_aware_predictor_development,
)
from autonomous_observation_lab.gimbal_servoing.control_aware_fresh_test import (
    ControlAwareFreshTestConfig,
    evaluate_control_aware_fresh_test,
)
from autonomous_observation_lab.gimbal_servoing.control_action_objective import (
    ControlActionObjectiveConfig,
    default_control_action_candidates,
    evaluate_control_action_objective,
)
from autonomous_observation_lab.gimbal_servoing.control_action_refinement import (
    refined_control_action_candidates,
)
from autonomous_observation_lab.gimbal_servoing.integrated_dynamics_objective import (
    integrated_dynamics_candidates,
)
from autonomous_observation_lab.gimbal_servoing.midpoint_dynamics_objective import (
    midpoint_dynamics_candidates,
)
from autonomous_observation_lab.gimbal_servoing.midpoint_adapter_objective import (
    evaluate_midpoint_adapter_objective,
    midpoint_adapter_candidates,
)
from autonomous_observation_lab.gimbal_servoing.midpoint_adapter_replication import (
    MidpointAdapterReplicationConfig,
    evaluate_midpoint_adapter_replication,
)
from autonomous_observation_lab.gimbal_servoing.midpoint_adapter_ensemble import (
    MidpointAdapterEnsembleConfig,
    evaluate_midpoint_adapter_ensemble,
)
from autonomous_observation_lab.gimbal_servoing.control_aware_replication import (
    ControlAwareReplicationConfig,
    evaluate_control_aware_replication,
)
from autonomous_observation_lab.gimbal_servoing.control_aware_closed_loop import (
    ControlAwareClosedLoopConfig,
    _paired_gate,
)
from autonomous_observation_lab.gimbal_servoing.control_criticality import (
    ControlCriticalityConfig,
)
from autonomous_observation_lab.gimbal_servoing.dataset import (
    FEATURE_NAMES,
    GimbalDatasetGenerationConfig,
    generate_gimbal_dataset,
    save_gimbal_dataset,
)
from autonomous_observation_lab.gimbal_servoing.gru import (
    CausalTargetStateGRU,
    GRUTargetStateModelConfig,
    save_gru_checkpoint,
)


def _dataset(split, seed, scenario):
    return generate_gimbal_dataset(
        GimbalDatasetGenerationConfig(
            split=split,
            seeds=(seed,),
            scenario_names=(scenario.name,),
            behavior_names=("privileged_oracle_position",),
            observation_profiles=(ObservationProfile.DISTURBANCE_AWARE,),
            prediction_horizons_s=(0.0, 0.1),
            include_oracle_ceilings=False,
        ),
        scenarios=(scenario,),
    )


def test_control_aware_candidates_isolate_required_ablation_factors():
    candidates = default_control_aware_candidates((0.0, 0.1, 0.2, 0.3))

    assert [candidate.name for candidate in candidates] == [
        "baseline_expanded",
        "critical_only",
        "consistency_only",
        "critical_consistency",
        "control_focused",
    ]
    assert not candidates[0].use_criticality_weights
    assert candidates[1].use_criticality_weights
    assert candidates[2].loss.dynamic_consistency_weight > 0.0
    assert candidates[-1].loss.horizon_weights[1] > (
        candidates[-1].loss.horizon_weights[0]
    )

    action_candidates = default_control_action_candidates()
    assert action_candidates[0].rate_action_weight == 0.0
    assert action_candidates[0].position_action_weight == 0.0
    assert any(
        candidate.rate_action_weight > 0.0
        and candidate.position_action_weight > 0.0
        for candidate in action_candidates
    )
    refined = refined_control_action_candidates()
    assert all(
        candidate.rate_action_weight <= 0.25
        and candidate.position_action_weight <= 0.35
        for candidate in refined[1:]
    )
    assert any(
        candidate.dynamic_consistency_weight > 25.0
        for candidate in refined
    )
    integrated = integrated_dynamics_candidates()
    assert integrated[0].mean_parameterization == "independent"
    assert all(
        candidate.mean_parameterization == "integrated_rate"
        for candidate in integrated[1:]
    )
    adaptive = default_adaptive_curriculum_candidates()
    assert adaptive[0].adaptive_position_action_weight == 0.0
    assert not adaptive[0].use_critical_episode_curriculum
    assert any(
        candidate.adaptive_position_action_weight > 0.0
        and not candidate.use_critical_episode_curriculum
        for candidate in adaptive
    )
    refinement = adaptive_curriculum_refinement_candidates()
    assert [candidate.dynamic_consistency_weight for candidate in refinement] == [
        25.0,
        25.0,
        50.0,
        100.0,
        50.0,
    ]
    assert refinement[-1].adaptive_position_action_weight < (
        refinement[2].adaptive_position_action_weight
    )
    structural = midpoint_adapter_candidates()
    assert structural[0].mean_parameterization == "independent"
    assert all(
        candidate.mean_parameterization == "integrated_midpoint"
        for candidate in structural[1:]
    )
    assert all(
        candidate.dynamic_consistency_weight == 0.0
        for candidate in structural[1:]
    )
    assert all(
        candidate.curriculum_concentration_strength == 1.0
        for candidate in structural[-2:]
    )
    assert any(
        candidate.adaptive_position_action_weight == 0.0
        and candidate.use_critical_episode_curriculum
        for candidate in adaptive
    )
    assert any(
        candidate.adaptive_position_action_weight > 0.0
        and candidate.use_critical_episode_curriculum
        for candidate in adaptive
    )
    midpoint = midpoint_dynamics_candidates()
    assert all(
        candidate.mean_parameterization == "integrated_midpoint"
        for candidate in midpoint[1:]
    )
    assert any(
        candidate.rate_action_weight > 0.0
        and candidate.position_action_weight > 0.0
        for candidate in integrated[1:]
    )


def test_control_aware_development_keeps_test_closed(tmp_path):
    base = nominal_scenario()
    scenario = replace(
        base,
        name="control_aware_smoke",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.2),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                miss_probability=0.0,
            ),
        ),
    )
    train_path, _ = save_gimbal_dataset(
        tmp_path / "train",
        _dataset("train", 101, scenario),
    )
    validation_path, _ = save_gimbal_dataset(
        tmp_path / "validation",
        _dataset("validation", 201, scenario),
    )
    legacy = CausalTargetStateGRU(
        GRUTargetStateModelConfig(
            input_dim=len(FEATURE_NAMES),
            prediction_horizons_s=(0.0, 0.1),
            hidden_dim=8,
            embedding_dim=8,
        )
    )
    legacy_path = save_gru_checkpoint(
        tmp_path / "legacy.pt",
        legacy,
        metadata={"purpose": "control-aware smoke"},
    )

    result = evaluate_control_aware_predictor_development(
        train_path=train_path,
        validation_path=validation_path,
        legacy_checkpoint=legacy_path,
        checkpoint_directory=tmp_path / "checkpoints",
        protocol=ControlAwarePredictorProtocolConfig(
            epochs=1,
            batch_size=1,
            hidden_dim=8,
            embedding_dim=8,
            minimum_training_episodes=1,
            minimum_validation_episodes=1,
            criticality=ControlCriticalityConfig(
                critical_weight_threshold=1.0
            ),
        ),
    )

    assert result["datasets"]["test"] == {"opened": False}
    assert len(result["candidates"]) == 5
    assert result["datasets"]["train"]["episodes"] == 1
    assert all(
        candidate["best_epoch"] == 1 for candidate in result["candidates"]
    )


def test_adaptive_curriculum_objective_keeps_fresh_test_closed(tmp_path):
    base = nominal_scenario()
    scenario = replace(
        base,
        name="adaptive_curriculum_smoke",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.2),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                miss_probability=0.0,
            ),
        ),
    )
    train_path, _ = save_gimbal_dataset(
        tmp_path / "adaptive_train",
        _dataset("train", 111, scenario),
    )
    validation_path, _ = save_gimbal_dataset(
        tmp_path / "adaptive_validation",
        _dataset("validation", 211, scenario),
    )

    result = evaluate_adaptive_curriculum_objective(
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_directory=tmp_path / "adaptive_checkpoints",
        config=AdaptiveCurriculumObjectiveConfig(
            epochs=1,
            batch_size=1,
            hidden_dim=8,
            embedding_dim=8,
            minimum_training_episodes=1,
            minimum_validation_episodes=1,
            criticality=ControlCriticalityConfig(
                critical_weight_threshold=1.0
            ),
        ),
    )

    assert result["datasets"]["fresh_test"] == {"opened": False}
    assert len(result["candidates"]) == 5
    assert all(candidate["best_epoch"] == 1 for candidate in result["candidates"])
    assert all(
        candidate["common_adapter_validation"][
            "adaptive_position_action_rmse_normalized"
        ]
        is not None
        for candidate in result["candidates"]
    )
    assert all(
        Path(candidate["checkpoint"]).exists()
        for candidate in result["candidates"]
    )


def test_midpoint_adapter_objective_enforces_hard_dynamics(tmp_path):
    base = nominal_scenario()
    scenario = replace(
        base,
        name="midpoint_adapter_smoke",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.2),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                miss_probability=0.0,
            ),
        ),
    )
    train_path, _ = save_gimbal_dataset(
        tmp_path / "midpoint_train",
        _dataset("train", 311, scenario),
    )
    validation_path, _ = save_gimbal_dataset(
        tmp_path / "midpoint_validation",
        _dataset("validation", 411, scenario),
    )

    result = evaluate_midpoint_adapter_objective(
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_directory=tmp_path / "midpoint_checkpoints",
        config=AdaptiveCurriculumObjectiveConfig(
            epochs=1,
            batch_size=1,
            hidden_dim=8,
            embedding_dim=8,
            minimum_training_episodes=1,
            minimum_validation_episodes=1,
            criticality=ControlCriticalityConfig(
                critical_weight_threshold=1.0
            ),
        ),
    )

    assert result["datasets"]["fresh_test"] == {"opened": False}
    assert len(result["candidates"]) == 5
    midpoint_records = result["candidates"][1:]
    assert all(
        record["model_config"]["mean_parameterization"]
        == "integrated_midpoint"
        for record in midpoint_records
    )
    assert all(
        record["standard_validation"]["dynamic_consistency_rmse_deg"]
        < 1e-4
        for record in midpoint_records
    )
    assert midpoint_records[-1]["curriculum"]["configuration"][
        "concentration_strength"
    ] == 1.0


def test_midpoint_adapter_replication_is_seed_matched_and_test_closed(tmp_path):
    base = nominal_scenario()
    scenario = replace(
        base,
        name="midpoint_replication_smoke",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.2),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                miss_probability=0.0,
            ),
        ),
    )
    train_path, _ = save_gimbal_dataset(
        tmp_path / "midpoint_replication_train",
        _dataset("train", 511, scenario),
    )
    validation_path, _ = save_gimbal_dataset(
        tmp_path / "midpoint_replication_validation",
        _dataset("validation", 611, scenario),
    )

    result = evaluate_midpoint_adapter_replication(
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_directory=tmp_path / "midpoint_replication_checkpoints",
        config=MidpointAdapterReplicationConfig(
            training_seeds=(7,),
            epochs=1,
            batch_size=1,
            hidden_dim=8,
            embedding_dim=8,
            minimum_training_episodes=1,
            minimum_validation_episodes=1,
            minimum_improving_seed_count=1,
            criticality=ControlCriticalityConfig(
                critical_weight_threshold=1.0
            ),
        ),
    )

    assert result["datasets"]["test"] == {"opened": False}
    assert len(result["training_seed_results"]) == 1
    record = result["training_seed_results"][0]
    assert record["training_seed"] == 7
    assert record["v4_reference"]["best_epoch"] == 1
    assert record["midpoint_state_reference"]["best_epoch"] == 1
    assert record["midpoint_state_reference"]["standard_validation"][
        "dynamic_consistency_rmse_deg"
    ] < 1e-4
    assert Path(record["v4_reference"]["checkpoint"]).exists()
    assert Path(record["midpoint_state_reference"]["checkpoint"]).exists()

    replication_path = tmp_path / "midpoint_replication.json"
    replication_path.write_text(json.dumps(result), encoding="utf-8")
    ensemble = evaluate_midpoint_adapter_ensemble(
        validation_path=validation_path,
        replication_path=replication_path,
        config=MidpointAdapterEnsembleConfig(
            batch_size=1,
            minimum_member_count=1,
            criticality=ControlCriticalityConfig(
                critical_weight_threshold=1.0
            ),
        ),
    )
    assert ensemble["datasets"]["test"] == {"opened": False}
    assert ensemble["ensembles"]["v4_reference"]["member_count"] == 1
    assert ensemble["ensembles"]["midpoint_state_reference"][
        "member_count"
    ] == 1


def test_control_aware_replication_is_seed_matched_and_keeps_test_closed(
    tmp_path,
):
    base = nominal_scenario()
    scenario = replace(
        base,
        name="control_aware_replication_smoke",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.2),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                miss_probability=0.0,
            ),
        ),
    )
    train_path, _ = save_gimbal_dataset(
        tmp_path / "replication_train",
        _dataset("train", 301, scenario),
    )
    validation_path, _ = save_gimbal_dataset(
        tmp_path / "replication_validation",
        _dataset("validation", 401, scenario),
    )

    result = evaluate_control_aware_replication(
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_directory=tmp_path / "replication_checkpoints",
        config=ControlAwareReplicationConfig(
            training_seeds=(7,),
            epochs=1,
            batch_size=1,
            hidden_dim=8,
            embedding_dim=8,
            minimum_training_episodes=1,
            minimum_validation_episodes=1,
            criticality=ControlCriticalityConfig(
                critical_weight_threshold=1.0
            ),
        ),
    )

    assert result["datasets"]["test"] == {"opened": False}
    assert result["parameter_count"] > 0
    assert len(result["training_seed_results"]) == 1
    seed_result = result["training_seed_results"][0]
    assert seed_result["training_seed"] == 7
    assert seed_result["baseline_expanded"]["best_epoch"] == 1
    assert seed_result["consistent_v4"]["best_epoch"] == 1
    assert "checkpoint" in seed_result["baseline_expanded"]
    assert "checkpoint" in seed_result["consistent_v4"]


def test_control_aware_fresh_test_uses_disjoint_data_and_frozen_pairs(tmp_path):
    base = nominal_scenario()
    scenario = replace(
        base,
        name="control_aware_fresh_test_smoke",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.2),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                miss_probability=0.0,
            ),
        ),
    )
    train_path, _ = save_gimbal_dataset(
        tmp_path / "fresh_train",
        _dataset("train", 501, scenario),
    )
    validation_path, _ = save_gimbal_dataset(
        tmp_path / "fresh_validation",
        _dataset("validation", 601, scenario),
    )
    test_path, _ = save_gimbal_dataset(
        tmp_path / "fresh_test",
        _dataset("test", 701, scenario),
    )
    checkpoint_directory = tmp_path / "fresh_checkpoints"
    checkpoint_directory.mkdir()
    for candidate in ("baseline_expanded", "consistent_v4"):
        model = CausalTargetStateGRU(
            GRUTargetStateModelConfig(
                input_dim=len(FEATURE_NAMES),
                prediction_horizons_s=(0.0, 0.1),
                hidden_dim=8,
                embedding_dim=8,
            )
        )
        save_gru_checkpoint(
            checkpoint_directory / f"gimbal_{candidate}_seed_7.pt",
            model,
            metadata={
                "experiment": (
                    "gimbal_control_aware_consistency_replication_v4_v1"
                ),
                "candidate": candidate,
                "training_seed": 7,
                "test_opened": False,
            },
        )

    result = evaluate_control_aware_fresh_test(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        checkpoint_directory=checkpoint_directory,
        config=ControlAwareFreshTestConfig(
            training_seeds=(7,),
            batch_size=1,
            minimum_test_episodes=1,
            minimum_improving_seed_count=1,
            criticality=ControlCriticalityConfig(
                critical_weight_threshold=1.0
            ),
        ),
    )

    assert result["datasets"]["test"]["opened"]
    assert result["datasets"]["test"]["seeds"] == [701]
    assert len(result["training_seed_results"]) == 1
    assert set(result["training_seed_results"][0]) >= {
        "baseline_expanded",
        "consistent_v4",
        "relative_changes",
    }


def test_control_action_objective_keeps_fresh_test_closed(tmp_path):
    base = nominal_scenario()
    scenario = replace(
        base,
        name="control_action_objective_smoke",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.2),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                miss_probability=0.0,
            ),
        ),
    )
    train_path, _ = save_gimbal_dataset(
        tmp_path / "action_train",
        _dataset("train", 801, scenario),
    )
    validation_path, _ = save_gimbal_dataset(
        tmp_path / "action_validation",
        _dataset("validation", 901, scenario),
    )

    result = evaluate_control_action_objective(
        train_path=train_path,
        validation_path=validation_path,
        checkpoint_directory=tmp_path / "action_checkpoints",
        config=ControlActionObjectiveConfig(
            epochs=1,
            batch_size=1,
            hidden_dim=8,
            embedding_dim=8,
            minimum_training_episodes=1,
            minimum_validation_episodes=1,
            criticality=ControlCriticalityConfig(
                critical_weight_threshold=1.0
            ),
        ),
    )

    assert result["datasets"]["fresh_test"] == {"reopened": False}
    assert len(result["candidates"]) == 5
    assert all(candidate["best_epoch"] == 1 for candidate in result["candidates"])
    assert all(
        candidate["common_control_validation"][
            "rate_action_rmse_normalized"
        ]
        is not None
        for candidate in result["candidates"]
    )


def test_control_aware_closed_loop_gate_requires_seed_consistency():
    reference = {
        "mean_absolute_error_fov_fraction": 0.20,
        "p95_absolute_error_fov_fraction": 0.50,
        "forecast_error_fov_fraction": 0.10,
        "command_variation_per_s": 1.0,
        "actuator_acceleration_rms_normalized": 0.5,
        "loss_of_view_fraction": 0.01,
        "avoidable_loss_fraction": 0.005,
    }
    candidate = {
        **reference,
        "mean_absolute_error_fov_fraction": 0.19,
        "command_variation_per_s": 0.98,
    }
    per_seed_reference = {
        seed: reference for seed in (17, 29, 43)
    }
    per_seed_candidate = {
        17: candidate,
        29: candidate,
        43: {**candidate, "mean_absolute_error_fov_fraction": 0.21},
    }

    gate = _paired_gate(
        candidate=candidate,
        reference=reference,
        per_seed_candidate=per_seed_candidate,
        per_seed_reference=per_seed_reference,
        config=ControlAwareClosedLoopConfig(
            world_seeds=(1,),
            scenario_names=("nominal_combined",),
        ),
    )

    assert gate["passed"]
    assert gate["mean_error_improving_training_seed_count"] == 2
