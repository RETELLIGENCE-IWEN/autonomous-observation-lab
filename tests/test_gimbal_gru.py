import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autonomous_observation_lab.gimbal_servoing import (
    GimbalServoEnv,
    GimbalDomainRandomizationConfig,
    ObservationProfile,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    ClosedLoopScenario,
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.dataset import (
    FEATURE_NAMES,
    GimbalDatasetGenerationConfig,
    generate_gimbal_dataset,
    save_gimbal_dataset,
)
from autonomous_observation_lab.gimbal_servoing.gru import (
    CausalTargetStateGRU,
    GRUInferenceConfig,
    GRULossConfig,
    GRUTargetStateEstimator,
    GRUTargetStateModelConfig,
    angular_residual_rad,
    load_gru_checkpoint,
    save_gru_checkpoint,
    target_state_nll,
)
from autonomous_observation_lab.gimbal_servoing.gru_training import (
    GRUTrainingConfig,
    constant_velocity_predictions,
    evaluate_constant_velocity_baseline,
    evaluate_gru,
    train_gru,
)
from autonomous_observation_lab.gimbal_servoing.gru_profiles import (
    run_gru_profile_comparison,
)


def learning_scenario() -> ClosedLoopScenario:
    base = nominal_scenario()
    return replace(
        base,
        name="learning_case",
        config=replace(
            base.config,
            timing=replace(base.config.timing, episode_duration_s=0.8),
            camera=replace(
                base.config.camera,
                detection_latency_s=0.0,
                detection_latency_jitter_s=0.0,
                center_noise_std_normalized=0.0,
                confidence_noise_std=0.0,
                miss_probability=0.0,
            ),
        ),
    )


def learning_dataset(split: str, seeds: tuple[int, ...]):
    scenario = learning_scenario()
    request = GimbalDatasetGenerationConfig(
        split=split,
        seeds=seeds,
        scenario_names=(scenario.name,),
        behavior_names=("privileged_oracle_rate",),
        observation_profiles=(ObservationProfile.SERVO_AWARE,),
        prediction_horizons_s=(0.0, 0.1),
        include_oracle_ceilings=False,
    )
    return generate_gimbal_dataset(request, scenarios=(scenario,))


def small_model() -> CausalTargetStateGRU:
    return CausalTargetStateGRU(
        GRUTargetStateModelConfig(
            input_dim=len(FEATURE_NAMES),
            prediction_horizons_s=(0.0, 0.1),
            hidden_dim=16,
            embedding_dim=12,
        )
    )


def test_gru_is_causal_and_streaming_matches_batched_forward():
    torch.manual_seed(3)
    model = small_model().eval()
    features = torch.randn(2, 9, len(FEATURE_NAMES))
    reference = model(features)
    changed = features.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 20.0
    changed_output = model(changed)

    torch.testing.assert_close(reference.mean[:, :5], changed_output.mean[:, :5])
    torch.testing.assert_close(reference.std[:, :5], changed_output.std[:, :5])

    hidden = None
    step_means = []
    step_stds = []
    for time_index in range(features.shape[1]):
        step = model.forward_step(features[:, time_index], hidden)
        hidden = step.hidden
        step_means.append(step.mean)
        step_stds.append(step.std)
    torch.testing.assert_close(reference.mean, torch.stack(step_means, dim=1))
    torch.testing.assert_close(reference.std, torch.stack(step_stds, dim=1))
    assert torch.all(reference.std > 0.0)


def test_probabilistic_loss_wraps_bearing_and_ignores_masked_values():
    model = small_model()
    features = torch.randn(2, 6, len(FEATURE_NAMES))
    output = model(features)
    targets = torch.randn_like(output.mean)
    target_mask = torch.ones(targets.shape[:-1], dtype=torch.bool)
    sequence_mask = torch.ones(targets.shape[:2], dtype=torch.bool)
    target_mask[:, -1, 1] = False
    sequence_mask[:, -2:] = False

    reference = target_state_nll(
        output,
        targets,
        target_mask,
        sequence_mask,
        GRULossConfig(),
    )
    corrupted = targets.clone()
    combined_mask = target_mask & sequence_mask.unsqueeze(-1)
    corrupted[~combined_mask] = 10_000.0
    replay = target_state_nll(
        output,
        corrupted,
        target_mask,
        sequence_mask,
    )
    torch.testing.assert_close(reference.total, replay.total)
    reference.total.backward()
    assert all(
        parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )

    wrapped = angular_residual_rad(
        torch.tensor([math.pi - 0.1]),
        torch.tensor([-math.pi + 0.1]),
    )
    assert wrapped.item() == pytest.approx(-0.2, abs=1e-6)


def test_streaming_estimator_produces_controller_compatible_state():
    scenario = learning_scenario()
    config = replace(
        scenario.config,
        observation_profile=ObservationProfile.SERVO_AWARE,
    )
    observation, _ = GimbalServoEnv(
        config,
        target_motion=scenario.target_motion,
        body_motion=scenario.body_motion,
    ).reset(seed=4)
    estimator = GRUTargetStateEstimator(
        small_model(),
        config,
        GRUInferenceConfig(
            observation_profile=ObservationProfile.SERVO_AWARE,
            horizon_index=0,
            maximum_staleness_s=0.2,
        ),
    )
    estimate = estimator.update(observation)

    assert estimate.valid
    assert estimate.measurement_time_s.valid
    assert estimate.bearing_std_rad.value > 0.0
    assert estimate.rate_std_rad_s.value > 0.0
    assert estimate.time_s == pytest.approx(observation.time_s)


@pytest.fixture(scope="module")
def trained_gru():
    train = learning_dataset("train", (11, 12, 13))
    validation = learning_dataset("validation", (21,))
    model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=train.manifest.prediction_horizons_s,
        hidden_dim=16,
        embedding_dim=16,
    )
    training_config = GRUTrainingConfig(
        epochs=4,
        batch_size=2,
        learning_rate=3e-3,
        seed=9,
    )
    first = train_gru(
        train,
        validation,
        ObservationProfile.SERVO_AWARE,
        model_config=model_config,
        training_config=training_config,
    )
    second = train_gru(
        train,
        validation,
        ObservationProfile.SERVO_AWARE,
        model_config=model_config,
        training_config=training_config,
    )
    return train, validation, first, second


def test_training_is_deterministic_and_improves_validation_loss(trained_gru):
    _, _, first, second = trained_gru
    assert first.history == second.history
    assert first.best_validation.loss is not None
    assert first.initial_validation.loss is not None
    assert first.best_validation.loss < first.initial_validation.loss
    assert first.best_epoch >= 1


def test_checkpoint_round_trip_and_analytical_comparison(trained_gru, tmp_path):
    _, validation, training, _ = trained_gru
    before = evaluate_gru(
        training.model,
        validation,
        ObservationProfile.SERVO_AWARE,
    )
    checkpoint = save_gru_checkpoint(
        tmp_path / "gimbal_gru.pt",
        training.model,
        metadata={"purpose": "test", "best_epoch": training.best_epoch},
    )
    restored, metadata = load_gru_checkpoint(checkpoint)
    after = evaluate_gru(
        restored,
        validation,
        ObservationProfile.SERVO_AWARE,
    )
    analytical = evaluate_constant_velocity_baseline(
        validation,
        ObservationProfile.SERVO_AWARE,
    )
    _, _, analytical_mask, _ = constant_velocity_predictions(
        validation, ObservationProfile.SERVO_AWARE
    )
    learned_matched = evaluate_gru(
        restored,
        validation,
        ObservationProfile.SERVO_AWARE,
        evaluation_mask=analytical_mask,
    )

    assert metadata == {"purpose": "test", "best_epoch": training.best_epoch}
    assert after == before
    assert analytical.availability_fraction > 0.5
    assert learned_matched.availability_fraction == pytest.approx(
        analytical.availability_fraction
    )
    assert analytical.bearing_rmse_deg is not None
    assert analytical.rate_rmse_deg_s is not None
    assert np.isfinite(analytical.bearing_rmse_deg)
    assert np.isfinite(analytical.rate_rmse_deg_s)


def test_analytical_replay_uses_randomized_episode_hardware():
    scenario = learning_scenario()
    randomization = GimbalDomainRandomizationConfig()
    randomization = replace(
        randomization,
        hardware=replace(
            randomization.hardware,
            episode_duration_s=0.8,
        ),
    )
    request = GimbalDatasetGenerationConfig(
        split="test",
        seeds=(404,),
        scenario_names=(scenario.name,),
        behavior_names=("privileged_oracle_rate",),
        observation_profiles=(ObservationProfile.SERVO_AWARE,),
        prediction_horizons_s=(0.0, 0.1),
        domain_randomization=randomization,
        include_oracle_ceilings=False,
    )
    dataset = generate_gimbal_dataset(request, scenarios=(scenario,))
    metrics = evaluate_constant_velocity_baseline(
        dataset, ObservationProfile.SERVO_AWARE
    )

    assert metrics.availability_fraction > 0.5
    assert metrics.bearing_rmse_deg is not None
    assert np.isfinite(metrics.bearing_rmse_deg)


def test_profile_comparison_uses_matched_architecture_and_data(tmp_path):
    scenario = learning_scenario()
    randomization = GimbalDomainRandomizationConfig()
    randomization = replace(
        randomization,
        hardware=replace(randomization.hardware, episode_duration_s=0.6),
    )

    def make_split(split: str, seeds: tuple[int, ...], filename: str):
        request = GimbalDatasetGenerationConfig(
            split=split,
            seeds=seeds,
            scenario_names=(scenario.name,),
            behavior_names=("privileged_oracle_rate",),
            observation_profiles=tuple(ObservationProfile),
            prediction_horizons_s=(0.0, 0.1),
            domain_randomization=randomization,
            include_oracle_ceilings=False,
        )
        dataset = generate_gimbal_dataset(request, scenarios=(scenario,))
        path, _ = save_gimbal_dataset(tmp_path / filename, dataset)
        return path

    train_path = make_split("train", (501, 502), "train")
    validation_path = make_split("validation", (601,), "validation")
    test_path = make_split("test", (701,), "test")
    result = run_gru_profile_comparison(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        checkpoint_directory=tmp_path / "checkpoints",
        hidden_dim=12,
        embedding_dim=12,
        training_config=GRUTrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=3e-3,
            seed=8,
        ),
    )

    assert result["profiles"] == [profile.value for profile in ObservationProfile]
    assert result["parameter_count_per_model"] > 0
    profile_results = result["profile_results"]
    assert set(profile_results) == {
        profile.value for profile in ObservationProfile
    }
    assert (
        profile_results[ObservationProfile.VISION_ONLY.value][
            "constant_velocity_test"
        ]["availability_fraction"]
        == 0.0
    )
    assert (
        profile_results[ObservationProfile.SERVO_AWARE.value][
            "constant_velocity_test"
        ]["availability_fraction"]
        > 0.5
    )
    assert all(
        (tmp_path / "checkpoints" / f"gimbal_gru_o{index}.pt").exists()
        for index in range(3)
    )
