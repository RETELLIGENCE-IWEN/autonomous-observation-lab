from dataclasses import replace

import numpy as np

from autonomous_observation_lab.gimbal_servoing import (
    GimbalDatasetGenerationConfig,
    GimbalDomainRandomizationConfig,
    HardwareRandomizationConfig,
    ObservationProfile,
    generate_gimbal_dataset,
    randomize_closed_loop_scenario,
    closed_loop_scenario_from_dict,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    nominal_scenario,
)


def short_randomization() -> GimbalDomainRandomizationConfig:
    defaults = GimbalDomainRandomizationConfig()
    return replace(
        defaults,
        hardware=replace(defaults.hardware, episode_duration_s=0.4),
    )


def test_domain_randomization_is_replayable_and_seed_dependent():
    base = nominal_scenario()
    randomization = short_randomization()
    first = randomize_closed_loop_scenario(
        base, seed=101, config=randomization
    )
    replay = randomize_closed_loop_scenario(
        base, seed=101, config=randomization
    )
    other = randomize_closed_loop_scenario(
        base, seed=102, config=randomization
    )

    assert first == replay
    assert first != other
    assert first.config.timing.episode_duration_s == 0.4
    assert first.config.servo.min_angle_rad < 0.0
    assert first.config.servo.max_angle_rad > 0.0
    assert (
        first.target_motion.state_at(0.2)
        != other.target_motion.state_at(0.2)
    )
    assert first.body_motion.state_at(0.2) != other.body_motion.state_at(0.2)
    assert first.config.servo != other.config.servo
    assert first.config.camera != other.config.camera


def test_randomized_dataset_pairs_behaviors_and_profiles_on_one_world():
    scenario = nominal_scenario()
    request = GimbalDatasetGenerationConfig(
        split="train",
        seeds=(301, 302),
        scenario_names=(scenario.name,),
        behavior_names=(
            "privileged_oracle_rate",
            "privileged_oracle_position",
        ),
        observation_profiles=(
            ObservationProfile.VISION_ONLY,
            ObservationProfile.SERVO_AWARE,
            ObservationProfile.DISTURBANCE_AWARE,
        ),
        prediction_horizons_s=(0.0, 0.1),
        domain_randomization=short_randomization(),
        include_oracle_ceilings=False,
    )
    first = generate_gimbal_dataset(request, scenarios=(scenario,))
    replay = generate_gimbal_dataset(request, scenarios=(scenario,))

    assert first.manifest.configuration_hash == replay.manifest.configuration_hash
    for name in first.manifest.array_shapes:
        np.testing.assert_array_equal(getattr(first, name), getattr(replay, name))

    variants = first.manifest.generation["scenario_variants"]
    assert len(variants) == 2
    assert {variant["seed"] for variant in variants} == {301, 302}
    reconstructed = closed_loop_scenario_from_dict(variants[0]["scenario"])
    expected = randomize_closed_loop_scenario(
        scenario, seed=301, config=request.domain_randomization
    )
    assert reconstructed == expected
    np.testing.assert_array_equal(first.targets[0], first.targets[1])
    np.testing.assert_array_equal(first.target_mask[0], first.target_mask[1])
    np.testing.assert_array_equal(first.time_s[0], first.time_s[1])
    assert not np.array_equal(first.targets[0], first.targets[2])

    image_error_index = first.manifest.feature_names.index(
        "image_error_normalized"
    )
    np.testing.assert_array_equal(
        first.features[0, 0, :, image_error_index],
        first.features[0, 2, :, image_error_index],
    )


def test_randomization_configuration_remains_fully_replaceable():
    default = HardwareRandomizationConfig()
    fixed_duration = replace(default, episode_duration_s=2.5)
    randomization = GimbalDomainRandomizationConfig(hardware=fixed_duration)
    scenario = randomize_closed_loop_scenario(
        nominal_scenario(), seed=77, config=randomization
    )

    assert scenario.config.timing.episode_duration_s == 2.5
    assert scenario.config.timing.integration_rate_hz == 1000.0
