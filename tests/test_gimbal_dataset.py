import math
from dataclasses import replace

import numpy as np
import pytest

from autonomous_observation_lab.gimbal_servoing import (
    FEATURE_NAMES,
    ConstantRateAngularMotion,
    GimbalCommandMode,
    GimbalDatasetGenerationConfig,
    GimbalServoEnv,
    ObservationProfile,
    OracleControlConfig,
    PrivilegedTargetStateOracle,
    StaticAngularMotion,
    encode_deployable_observation,
    evaluate_privileged_oracle_ceilings,
    generate_gimbal_dataset,
    load_gimbal_dataset,
    save_gimbal_dataset,
    validate_disjoint_seed_blocks,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    ClosedLoopScenario,
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.control_criticality import (
    ControlCriticalityConfig,
    compute_control_criticality,
    control_criticality_report,
)
from autonomous_observation_lab.gimbal_servoing.control_supervision import (
    compute_control_action_supervision,
)
from autonomous_observation_lab.gimbal_servoing.critical_curriculum import (
    CriticalEpisodeCurriculumConfig,
    compute_critical_episode_curriculum,
    critical_episode_curriculum_report,
)


def short_scenario(
    *,
    camera_noise: float = 0.0,
    name: str = "dataset_case",
) -> ClosedLoopScenario:
    base = nominal_scenario()
    return replace(
        base,
        name=name,
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.30),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                center_noise_std_normalized=camera_noise,
                miss_probability=0.0,
            ),
        ),
    )


def dataset_request(**overrides) -> GimbalDatasetGenerationConfig:
    values = dict(
        split="train",
        seeds=(101,),
        scenario_names=("dataset_case",),
        behavior_names=("predictive_rate", "privileged_oracle_position"),
        observation_profiles=(
            ObservationProfile.VISION_ONLY,
            ObservationProfile.SERVO_AWARE,
            ObservationProfile.DISTURBANCE_AWARE,
        ),
        prediction_horizons_s=(0.0, 0.1, 0.2),
        include_oracle_ceilings=False,
    )
    values.update(overrides)
    return GimbalDatasetGenerationConfig(**values)


def test_privileged_oracle_reports_body_relative_truth_and_both_commands():
    scenario = short_scenario()
    target = ConstantRateAngularMotion(
        initial_angle_rad=math.radians(10.0),
        rate_rad_s=math.radians(8.0),
    )
    body = ConstantRateAngularMotion(
        initial_angle_rad=math.radians(2.0),
        rate_rad_s=math.radians(3.0),
    )
    env = GimbalServoEnv(
        scenario.config,
        target_motion=target,
        body_motion=body,
    )
    _, diagnostics = env.reset(seed=1)
    oracle = PrivilegedTargetStateOracle(
        target_motion=target,
        body_motion=body,
        servo=scenario.config.servo,
        control=OracleControlConfig(
            rate_feedback_gain_s_inv=2.0,
            position_preview_s=0.1,
        ),
    )

    current = oracle.state_from_diagnostics(diagnostics)
    future = oracle.state_at(0.2)
    actions = oracle.action_targets(diagnostics)

    assert current.body_relative_bearing_rad == pytest.approx(
        math.radians(8.0)
    )
    assert current.body_relative_rate_rad_s == pytest.approx(
        math.radians(5.0)
    )
    assert future.body_relative_bearing_rad == pytest.approx(
        math.radians(9.0)
    )
    expected_rate = math.radians(5.0 + 2.0 * 8.0)
    assert actions.desired_rate_normalized == pytest.approx(
        expected_rate / scenario.config.servo.max_rate_rad_s
    )
    expected_position = math.radians(8.5)
    assert actions.desired_position_normalized == pytest.approx(
        scenario.config.servo.normalized_from_position(expected_position)
    )


def test_privileged_ceiling_exercises_rate_and_position_servo_plants():
    scenario = short_scenario()
    ceilings = evaluate_privileged_oracle_ceilings((scenario,), seed=7)

    assert len(ceilings) == 2
    assert {record.command_mode for record in ceilings} == {
        "desired_rate",
        "desired_position",
    }
    assert all(record.scenario_name == scenario.name for record in ceilings)
    assert all(record.metrics["rms_error_normalized"] >= 0.0 for record in ceilings)


def test_observation_encoder_masks_profile_capabilities_without_truth_inputs():
    scenario = short_scenario()
    config = replace(
        scenario.config,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
    )
    observation, _ = GimbalServoEnv(
        config,
        body_motion=ConstantRateAngularMotion(rate_rad_s=math.radians(5.0)),
        target_motion=StaticAngularMotion(math.radians(3.0)),
    ).reset(seed=2)
    vectors = {
        profile: encode_deployable_observation(
            observation, profile=profile, config=config
        )
        for profile in ObservationProfile
    }
    index = {name: position for position, name in enumerate(FEATURE_NAMES)}

    assert all("target" not in name for name in FEATURE_NAMES)
    assert "body_bearing_rad" not in FEATURE_NAMES
    assert vectors[ObservationProfile.VISION_ONLY][
        index["gimbal_position_valid"]
    ] == 0.0
    assert vectors[ObservationProfile.VISION_ONLY][
        index["body_rate_valid"]
    ] == 0.0
    assert vectors[ObservationProfile.SERVO_AWARE][
        index["gimbal_position_valid"]
    ] == 1.0
    assert vectors[ObservationProfile.SERVO_AWARE][
        index["body_rate_valid"]
    ] == 0.0
    assert vectors[ObservationProfile.DISTURBANCE_AWARE][
        index["body_rate_valid"]
    ] == 1.0
    assert vectors[ObservationProfile.DISTURBANCE_AWARE][
        index["image_error_rad"]
    ] == pytest.approx(
        vectors[ObservationProfile.DISTURBANCE_AWARE][
            index["image_error_normalized"]
        ]
        * 0.5
        * config.camera.selected_axis_fov_rad
    )
    assert vectors[ObservationProfile.SERVO_AWARE][
        index["gimbal_angle_rad"]
    ] == pytest.approx(observation.gimbal_angle_rad.value)
    assert vectors[ObservationProfile.DISTURBANCE_AWARE][
        index["body_rate_rad_s"]
    ] == pytest.approx(observation.body_rate_rad_s.value)
    assert vectors[ObservationProfile.VISION_ONLY][
        index["previous_rate_command_rad_s"]
    ] == pytest.approx(
        observation.previous_action_normalized
        * config.servo.max_rate_rad_s
    )
    assert vectors[ObservationProfile.VISION_ONLY][
        index["previous_position_command_rad"]
    ] == 0.0

    position_observation = replace(
        observation,
        command_mode=GimbalCommandMode.POSITION,
        previous_action_normalized=-0.4,
    )
    position_vector = encode_deployable_observation(
        position_observation,
        profile=ObservationProfile.VISION_ONLY,
        config=replace(config, command_mode=GimbalCommandMode.POSITION),
    )
    assert position_vector[index["previous_rate_command_rad_s"]] == 0.0
    assert position_vector[
        index["previous_position_command_rad"]
    ] == pytest.approx(config.servo.position_from_normalized(-0.4))


def test_dataset_replay_is_deterministic_and_profiles_share_labels():
    scenario = short_scenario()
    request = dataset_request()
    first = generate_gimbal_dataset(request, scenarios=(scenario,))
    second = generate_gimbal_dataset(request, scenarios=(scenario,))

    assert first.episode_count == 2
    assert first.profile_count == 3
    assert first.manifest.configuration_hash == second.manifest.configuration_hash
    for name in first.manifest.array_shapes:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    index = {name: position for position, name in enumerate(FEATURE_NAMES)}
    np.testing.assert_array_equal(
        first.features[:, 0, :, index["image_error_normalized"]],
        first.features[:, 2, :, index["image_error_normalized"]],
    )
    assert np.all(
        first.features[:, 0, :, index["gimbal_position_valid"]] == 0.0
    )
    assert np.all(
        first.features[:, 2, first.sequence_mask[0], :][
            ..., index["gimbal_position_valid"]
        ]
        == 1.0
    )
    assert first.targets.ndim == 4
    assert first.targets.shape[-1] == 2
    assert first.target_mask.shape == first.targets.shape[:-1]
    assert not np.all(first.target_mask[:, -1, 1:])


def test_detector_corruption_changes_features_but_not_privileged_targets():
    clean = short_scenario(camera_noise=0.0)
    noisy = short_scenario(camera_noise=0.08)
    request = dataset_request(
        behavior_names=("privileged_oracle_rate",),
        observation_profiles=(ObservationProfile.VISION_ONLY,),
    )
    clean_data = generate_gimbal_dataset(request, scenarios=(clean,))
    noisy_data = generate_gimbal_dataset(request, scenarios=(noisy,))

    assert not np.array_equal(clean_data.features, noisy_data.features)
    np.testing.assert_array_equal(clean_data.targets, noisy_data.targets)
    np.testing.assert_array_equal(clean_data.target_mask, noisy_data.target_mask)
    np.testing.assert_array_equal(clean_data.actions, noisy_data.actions)
    np.testing.assert_array_equal(
        clean_data.oracle_actions, noisy_data.oracle_actions
    )


def test_control_criticality_is_hardware_relative_and_training_only():
    scenario = short_scenario()
    dataset = generate_gimbal_dataset(
        dataset_request(
            behavior_names=("privileged_oracle_position",),
            observation_profiles=(ObservationProfile.DISTURBANCE_AWARE,),
        ),
        scenarios=(scenario,),
    )
    config = ControlCriticalityConfig(critical_weight_threshold=1.0)

    criticality = compute_control_criticality(dataset, config=config)
    report = control_criticality_report(
        dataset,
        criticality,
        config=config,
    )
    valid = dataset.target_mask & dataset.sequence_mask[:, :, None]

    assert criticality.weights.shape == dataset.target_mask.shape
    assert np.mean(criticality.weights[valid]) == pytest.approx(1.0)
    assert np.all(criticality.raw_weights[valid] >= 1.0)
    assert np.all(criticality.critical_mask[valid])
    assert report["episode_count"] == dataset.episode_count
    assert report["overall"]["critical_label_fraction"] == 1.0
    assert set(report["by_scenario"]) == {scenario.name}
    assert all("target" not in name for name in dataset.manifest.feature_names)

    uniform = compute_control_criticality(
        dataset,
        config=ControlCriticalityConfig(weighting_strength=0.0),
    )
    assert np.all(uniform.weights[valid] == pytest.approx(1.0))


def test_control_action_supervision_uses_serialized_hardware_and_stays_privileged():
    scenario = short_scenario()
    dataset = generate_gimbal_dataset(
        dataset_request(
            behavior_names=("privileged_oracle_position",),
            observation_profiles=(ObservationProfile.DISTURBANCE_AWARE,),
        ),
        scenarios=(scenario,),
    )

    supervision = compute_control_action_supervision(dataset)

    assert supervision.oracle_actions.shape == (*dataset.sequence_mask.shape, 2)
    np.testing.assert_array_equal(
        supervision.oracle_actions,
        dataset.oracle_actions,
    )
    assert np.all(
        supervision.servo_max_rate_rad_s[dataset.sequence_mask]
        == pytest.approx(scenario.config.servo.max_rate_rad_s)
    )
    assert np.all(supervision.mask == dataset.sequence_mask)
    assert all("oracle" not in name for name in dataset.manifest.feature_names)


def test_critical_episode_curriculum_concentrates_without_label_reweighting():
    scenario = short_scenario()
    dataset = generate_gimbal_dataset(
        dataset_request(),
        scenarios=(scenario,),
    )
    criticality = compute_control_criticality(dataset)
    config = CriticalEpisodeCurriculumConfig(
        concentration_strength=2.0,
        maximum_episode_weight=3.0,
    )

    curriculum = compute_critical_episode_curriculum(
        dataset,
        criticality,
        config=config,
    )
    report = critical_episode_curriculum_report(
        dataset,
        curriculum,
        config=config,
    )

    assert curriculum.episode_weights.shape == (dataset.episode_count,)
    assert np.mean(curriculum.episode_weights) == pytest.approx(1.0)
    assert curriculum.expected_sampled_critical_label_fraction >= (
        curriculum.observed_critical_label_fraction
    )
    assert report["episode_count"] == dataset.episode_count
    assert report["mean_episode_weight"] == pytest.approx(1.0)


def test_dataset_round_trip_and_split_seed_validation(tmp_path):
    scenario = short_scenario()
    train = generate_gimbal_dataset(dataset_request(), scenarios=(scenario,))
    dataset_path, manifest_path = save_gimbal_dataset(
        tmp_path / "gimbal_train", train
    )
    restored = load_gimbal_dataset(dataset_path)

    assert dataset_path.exists()
    assert manifest_path.exists()
    assert restored.manifest == train.manifest
    for name in train.manifest.array_shapes:
        np.testing.assert_array_equal(getattr(restored, name), getattr(train, name))

    validation = generate_gimbal_dataset(
        dataset_request(split="validation", seeds=(202,)),
        scenarios=(scenario,),
    )
    validate_disjoint_seed_blocks((train.manifest, validation.manifest))
    overlapping = replace(validation.manifest, seeds=train.manifest.seeds)
    with pytest.raises(ValueError, match="reuse seeds"):
        validate_disjoint_seed_blocks((train.manifest, overlapping))
