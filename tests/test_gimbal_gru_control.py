from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from autonomous_observation_lab.gimbal_servoing import (
    GimbalDatasetGenerationConfig,
    GimbalDomainRandomizationConfig,
    ObservationProfile,
    generate_gimbal_dataset,
    save_gimbal_dataset,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.dataset import FEATURE_NAMES
from autonomous_observation_lab.gimbal_servoing.gru import (
    CausalTargetStateGRU,
    GRUTargetStateModelConfig,
    save_gru_checkpoint,
)
from autonomous_observation_lab.gimbal_servoing.gru_control import (
    evaluate_gru_closed_loop,
)


def test_checkpoint_driven_control_evaluation_uses_paired_splits(tmp_path):
    base = nominal_scenario()
    scenario = replace(base, name="control_test")
    randomization = GimbalDomainRandomizationConfig()
    randomization = replace(
        randomization,
        hardware=replace(
            randomization.hardware,
            episode_duration_s=0.25,
        ),
    )

    paths = {}
    manifests = {}
    for split, seed in (("train", 801), ("validation", 901), ("test", 1001)):
        request = GimbalDatasetGenerationConfig(
            split=split,
            seeds=(seed,),
            scenario_names=(scenario.name,),
            behavior_names=("proportional_rate", "proportional_position"),
            observation_profiles=(
                ObservationProfile.SERVO_AWARE,
                ObservationProfile.DISTURBANCE_AWARE,
            ),
            prediction_horizons_s=(0.0,),
            domain_randomization=randomization,
            include_oracle_ceilings=False,
        )
        dataset = generate_gimbal_dataset(request, scenarios=(scenario,))
        paths[split], _ = save_gimbal_dataset(tmp_path / split, dataset)
        manifests[split] = dataset.manifest

    checkpoint_paths = {}
    for short, profile in (
        ("o1", ObservationProfile.SERVO_AWARE),
        ("o2", ObservationProfile.DISTURBANCE_AWARE),
    ):
        model = CausalTargetStateGRU(
            GRUTargetStateModelConfig(
                input_dim=len(FEATURE_NAMES),
                prediction_horizons_s=(0.0,),
                hidden_dim=8,
                embedding_dim=8,
            )
        )
        checkpoint_paths[profile] = save_gru_checkpoint(
            tmp_path / f"{short}.pt",
            model,
            metadata={
                "profile": profile.value,
                "feature_names": list(FEATURE_NAMES),
                "dataset_hashes": {
                    split: manifest.configuration_hash
                    for split, manifest in manifests.items()
                },
            },
        )

    result = evaluate_gru_closed_loop(
        train_data=paths["train"],
        validation_data=paths["validation"],
        test_data=paths["test"],
        o1_checkpoint=checkpoint_paths[ObservationProfile.SERVO_AWARE],
        o2_checkpoint=checkpoint_paths[
            ObservationProfile.DISTURBANCE_AWARE
        ],
    )

    assert result["test_variant_count"] == 1
    assert not result["position_results_are_out_of_support"]
    assert set(result["summary"]) == {
        "proportional_rate",
        "analytical_rate",
        "proportional_position",
        "analytical_position",
        "gru_o1_rate",
        "gru_o1_position",
        "gru_o2_rate",
        "gru_o2_position",
        "gru_o2_rate_search",
        "gru_o2_position_search",
    }
    assert result["validation_horizon_selection"][
        ObservationProfile.DISTURBANCE_AWARE.value
    ]["desired_position"]["selected_horizon_s"] == 0.0
    assert result["paired_comparisons"]["gru_o2_rate"]["reference"] == (
        "analytical_rate"
    )
