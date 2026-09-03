import json
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip("torch")

from autonomous_observation_lab.gimbal_servoing import (
    GimbalDatasetGenerationConfig,
    GimbalDomainRandomizationConfig,
    ObservationProfile,
    generate_gimbal_dataset,
    save_gimbal_dataset,
)
from autonomous_observation_lab.gimbal_servoing.adaptive_position import (
    ADAPTIVE_POSITION_SCHEMA_VERSION,
    AdaptivePositionCandidate,
    AdaptivePositionProtocolConfig,
    evaluate_adaptive_position_v2,
)
from autonomous_observation_lab.gimbal_servoing.adaptive_position_v21 import (
    ADAPTIVE_POSITION_V21_SCHEMA_VERSION,
    VisibilityRiskProtocolConfig,
    adaptive_position_v2_config,
    evaluate_visibility_risk_v21,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.controller_arena import (
    build_gimbal_challenge_arena,
    build_visibility_risk_controller_arena,
)
from autonomous_observation_lab.gimbal_servoing.controllers import (
    AdaptivePositionControllerConfig,
)
from autonomous_observation_lab.gimbal_servoing.dataset import FEATURE_NAMES
from autonomous_observation_lab.gimbal_servoing.gru import (
    CausalTargetStateGRU,
    GRUTargetStateModelConfig,
    save_gru_checkpoint,
)


def test_adaptive_position_protocol_selects_before_disjoint_test(tmp_path):
    scenario = replace(nominal_scenario(), name="adaptive_position_smoke")
    randomization = GimbalDomainRandomizationConfig()
    randomization = replace(
        randomization,
        hardware=replace(
            randomization.hardware,
            episode_duration_s=0.2,
        ),
    )
    paths = {}
    manifests = {}
    for split, seed in (("validation", 701), ("test", 801)):
        request = GimbalDatasetGenerationConfig(
            split=split,
            seeds=(seed,),
            scenario_names=(scenario.name,),
            behavior_names=("proportional_position",),
            observation_profiles=(ObservationProfile.DISTURBANCE_AWARE,),
            prediction_horizons_s=(0.0, 0.1),
            domain_randomization=randomization,
            include_oracle_ceilings=False,
        )
        dataset = generate_gimbal_dataset(request, scenarios=(scenario,))
        paths[split], _ = save_gimbal_dataset(tmp_path / split, dataset)
        manifests[split] = dataset.manifest

    checkpoints = {}
    for training_seed in (7, 11):
        torch.manual_seed(training_seed)
        model = CausalTargetStateGRU(
            GRUTargetStateModelConfig(
                input_dim=len(FEATURE_NAMES),
                prediction_horizons_s=(0.0, 0.1),
                hidden_dim=8,
                embedding_dim=8,
            )
        )
        checkpoints[training_seed] = save_gru_checkpoint(
            tmp_path / f"o2_seed_{training_seed}.pt",
            model,
            metadata={
                "profile": ObservationProfile.DISTURBANCE_AWARE.value,
                "feature_names": list(FEATURE_NAMES),
                "dataset_hashes": {
                    split: manifest.configuration_hash
                    for split, manifest in manifests.items()
                },
                "training_config": {"seed": training_seed},
                "selected_horizons": {
                    "position": {"horizon_index": 0, "horizon_s": 0.0}
                },
            },
        )

    candidate = AdaptivePositionCandidate(
        "smoke",
        AdaptivePositionControllerConfig(position_response_fraction=0.0),
    )
    result = evaluate_adaptive_position_v2(
        validation_data=paths["validation"],
        test_data=paths["test"],
        checkpoints=checkpoints,
        protocol=AdaptivePositionProtocolConfig(
            validation_max_mean_error_regression_deg=180.0,
            validation_max_p95_regression_deg=180.0,
            validation_max_loss_of_view_regression=1.0,
            validation_max_control_cost_regression=10.0,
            test_max_mean_error_regression_deg=180.0,
            test_max_p95_regression_deg=180.0,
            test_max_loss_of_view_regression=1.0,
            test_minimum_command_variation_reduction_fraction=0.0,
            test_max_scenario_p95_regression_deg=180.0,
            test_max_scenario_loss_of_view_regression=1.0,
            candidates=(candidate,),
        ),
    )

    assert result["experiment"] == ADAPTIVE_POSITION_SCHEMA_VERSION
    assert result["training_seeds"] == [7, 11]
    assert result["validation"]["variant_count_per_training_seed"] == 1
    assert result["validation"]["selected_candidate"] == "smoke"
    assert result["test"]["variant_count_per_training_seed"] == 1
    assert set(result["test"]["by_training_seed"]) == {
        "7",
        "11",
        "distributions",
    }
    assert result["representative_trace"]["world_seed"] == 801
    assert result["representative_trace"]["records"]
    json.dumps(result)

    neutral = AdaptivePositionCandidate(
        "neutral",
        adaptive_position_v2_config(),
    )
    v21_result = evaluate_visibility_risk_v21(
        validation_data=paths["validation"],
        test_data=paths["test"],
        checkpoints=checkpoints,
        protocol=VisibilityRiskProtocolConfig(candidates=(neutral,)),
        development_seeds=(901,),
        confirmation_seeds=(902,),
    )

    assert v21_result["experiment"] == ADAPTIVE_POSITION_V21_SCHEMA_VERSION
    assert v21_result["development"]["selected_candidate"] is None
    assert not v21_result["confirmation"]["opened"]
    json.dumps(v21_result)

    arena_result = {
        "experiment": ADAPTIVE_POSITION_V21_SCHEMA_VERSION,
        "protocol": {"maximum_staleness_s": 0.5},
        "training_seeds": [7, 11],
        "checkpoints": {
            str(seed): str(path) for seed, path in checkpoints.items()
        },
        "fixed_horizons": {
            str(seed): {"horizon_index": 0, "horizon_s": 0.0}
            for seed in checkpoints
        },
        "development": {
            "selected_candidate": "neutral",
            "candidates": [
                {
                    "name": "neutral",
                    "controller_config": asdict(adaptive_position_v2_config()),
                }
            ],
        },
        "confirmation": {"opened": True, "world_seeds": [901]},
        "representative_trace": {
            "scenario_name": "nominal_combined",
            "world_seed": 901,
            "training_seed": 7,
        },
    }
    arena = build_visibility_risk_controller_arena(arena_result)

    assert arena.scenario_name == "nominal_combined"
    assert [run.episode.name for run in arena.comparison.runs] == [
        "arena_fixed_horizon",
        "arena_adaptive_v2",
        "arena_visibility_risk_v21",
    ]
    assert len({len(run.episode.frames) for run in arena.comparison.runs}) == 1

    challenge = build_gimbal_challenge_arena(
        arena_result,
        scenario_name="nominal_combined",
        world_seed=901,
        training_seed=7,
    )
    assert challenge.kind == "challenge"
    assert [run.episode.name for run in challenge.comparison.runs] == [
        "challenge_reactive_position",
        "challenge_classical_predictive",
        "challenge_dream_to_center",
    ]
    assert len(
        {len(run.episode.frames) for run in challenge.comparison.runs}
    ) == 1
    assert not any(challenge.comparison.runs[0].forecasts)
    assert any(challenge.comparison.runs[1].forecasts)
    assert any(
        len(forecasts) > 1
        for forecasts in challenge.comparison.runs[2].forecasts
    )
    assert all(
        run.episode.config.servo == challenge.comparison.runs[0].episode.config.servo
        for run in challenge.comparison.runs
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_gimbal_challenge_arena(
            arena_result,
            scenario_name="nominal_combined",
            world_seed=901,
            training_seed=7,
            ghost_horizons_s=(0.2, 0.1),
        )
