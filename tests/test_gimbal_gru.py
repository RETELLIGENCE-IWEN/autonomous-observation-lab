import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autonomous_observation_lab.gimbal_servoing import (
    GimbalAction,
    GimbalCommandMode,
    GimbalServoEnv,
    GimbalDomainRandomizationConfig,
    ObservationProfile,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    ClosedLoopScenario,
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.adaptive_position_supervision import (
    compute_adaptive_position_supervision,
)
from autonomous_observation_lab.gimbal_servoing.adaptive_position_v21 import (
    default_visibility_risk_candidates,
)
from autonomous_observation_lab.gimbal_servoing.dataset import (
    FEATURE_NAMES,
    GimbalDatasetGenerationConfig,
    generate_gimbal_dataset,
    save_gimbal_dataset,
)
from autonomous_observation_lab.gimbal_servoing.gru import (
    CausalTargetStateGRU,
    CausalTargetStateGRUEnsemble,
    GRUAdaptivePositionLossContext,
    GRUControlLossContext,
    GRUInferenceConfig,
    GRULossConfig,
    GRUPositionPlantRolloutConfig,
    GRUTargetStateEstimator,
    GRUTargetStateModelConfig,
    GRUTargetStateOutput,
    adaptive_position_surrogate_actions,
    angular_residual_rad,
    differentiable_position_servo_rollout,
    gru_parameter_count,
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


def test_gru_ensemble_matches_identical_members_and_streaming_forward():
    torch.manual_seed(13)
    first = small_model().eval()
    second = small_model().eval()
    second.load_state_dict(first.state_dict())
    ensemble = CausalTargetStateGRUEnsemble((first, second)).eval()
    features = torch.randn(2, 7, len(FEATURE_NAMES))
    reference = first(features)
    combined = ensemble(features)

    torch.testing.assert_close(combined.mean, reference.mean)
    torch.testing.assert_close(combined.std, reference.std)
    assert combined.hidden.shape[0] == 2

    hidden = None
    step_means = []
    step_stds = []
    for time_index in range(features.shape[1]):
        output = ensemble.forward_step(features[:, time_index], hidden)
        hidden = output.hidden
        step_means.append(output.mean)
        step_stds.append(output.std)
    torch.testing.assert_close(combined.mean, torch.stack(step_means, dim=1))
    torch.testing.assert_close(combined.std, torch.stack(step_stds, dim=1))


def test_integrated_rate_head_is_dynamically_constrained_by_construction():
    model = CausalTargetStateGRU(
        GRUTargetStateModelConfig(
            input_dim=len(FEATURE_NAMES),
            prediction_horizons_s=(0.0, 0.1, 0.25),
            hidden_dim=16,
            embedding_dim=12,
            mean_parameterization="integrated_rate",
        )
    )
    independent = CausalTargetStateGRU(
        replace(model.config, mean_parameterization="independent")
    )
    output = model(torch.randn(2, 5, len(FEATURE_NAMES)))
    intervals = torch.tensor([0.1, 0.15])
    bearing_step = angular_residual_rad(
        output.mean[..., 1:, 0],
        output.mean[..., :-1, 0],
    )
    integrated_rate = 0.5 * (
        output.mean[..., 1:, 1] + output.mean[..., :-1, 1]
    ) * intervals

    torch.testing.assert_close(bearing_step, integrated_rate, atol=1e-6, rtol=1e-6)
    assert gru_parameter_count(model) < gru_parameter_count(independent)

    midpoint_model = CausalTargetStateGRU(
        replace(model.config, mean_parameterization="integrated_midpoint")
    )
    midpoint_output = midpoint_model(
        torch.randn(2, 5, len(FEATURE_NAMES))
    )
    assert midpoint_output.interval_rate_rad_s is not None
    midpoint_step = angular_residual_rad(
        midpoint_output.mean[..., 1:, 0],
        midpoint_output.mean[..., :-1, 0],
    )
    simpson_rate = (
        midpoint_output.mean[..., :-1, 1]
        + 4.0 * midpoint_output.interval_rate_rad_s
        + midpoint_output.mean[..., 1:, 1]
    ) * intervals / 6.0
    torch.testing.assert_close(midpoint_step, simpson_rate, atol=1e-6, rtol=1e-6)


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
    assert any(
        parameter.grad is not None
        and torch.any(torch.abs(parameter.grad) > 0.0)
        for parameter in model.parameters()
    )

    wrapped = angular_residual_rad(
        torch.tensor([math.pi - 0.1]),
        torch.tensor([-math.pi + 0.1]),
    )
    assert wrapped.item() == pytest.approx(-0.2, abs=1e-6)


def test_adaptive_position_teacher_matches_differentiable_truth_replay():
    dataset = learning_dataset("train", (161, 162))
    adapter = default_visibility_risk_candidates()[2].controller
    supervision = compute_adaptive_position_supervision(
        dataset,
        adapter=adapter,
        profile=ObservationProfile.SERVO_AWARE,
    )

    def tensor(name: str, *, boolean: bool = False):
        value = torch.from_numpy(getattr(supervision, name))
        return value.bool() if boolean else value.float()

    context = GRUAdaptivePositionLossContext(
        teacher_action_normalized=tensor("teacher_action_normalized"),
        mask=tensor("mask", boolean=True),
        gimbal_angle_rad=tensor("gimbal_angle_rad"),
        gimbal_rate_rad_s=tensor("gimbal_rate_rad_s"),
        control_dt_s=tensor("control_dt_s"),
        selected_axis_fov_rad=tensor("selected_axis_fov_rad"),
        servo_min_angle_rad=tensor("servo_min_angle_rad"),
        servo_max_angle_rad=tensor("servo_max_angle_rad"),
        servo_max_rate_rad_s=tensor("servo_max_rate_rad_s"),
        servo_max_acceleration_rad_s2=tensor(
            "servo_max_acceleration_rad_s2"
        ),
        servo_position_gain_s_inv=tensor("servo_position_gain_s_inv"),
        servo_position_tolerance_rad=tensor("servo_position_tolerance_rad"),
        servo_position_quantization_rad=tensor(
            "servo_position_quantization_rad"
        ),
        servo_command_polarity=tensor("servo_command_polarity"),
        servo_command_latency_s=tensor("servo_command_latency_s"),
        servo_rate_time_constant_s=tensor("servo_rate_time_constant_s"),
        control_period_s=tensor("control_period_s"),
        integration_period_s=tensor("integration_period_s"),
        camera_frame_period_s=tensor("camera_frame_period_s"),
    )
    truth = torch.from_numpy(dataset.targets).float()
    truth_output = GRUTargetStateOutput(
        mean=truth,
        std=torch.full_like(truth, 1e-8),
        hidden=torch.empty(0),
    )
    sequence_mask = torch.from_numpy(dataset.sequence_mask)
    actions = adaptive_position_surrogate_actions(
        truth_output,
        context,
        dataset.manifest.prediction_horizons_s,
        adapter,
        sequence_mask,
    )
    selected = context.mask & sequence_mask
    torch.testing.assert_close(
        actions[selected],
        context.teacher_action_normalized[selected],
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.all(torch.abs(actions) <= 1.0)

    model = small_model()
    profile_index = dataset.manifest.observation_profiles.index(
        ObservationProfile.SERVO_AWARE.value
    )
    output = model(
        torch.from_numpy(dataset.features[:, profile_index]).float()
    )
    plant_loss_config = GRULossConfig(
        adaptive_position_action_weight=0.25,
        adaptive_position_config=adapter,
        position_plant_tracking_weight=0.20,
        position_plant_response_weight=0.10,
        position_plant_regret_weight=0.10,
        position_plant_visibility_weight=0.05,
        position_plant_smoothness_weight=0.01,
        position_plant_saturation_weight=0.01,
        position_plant_config=GRUPositionPlantRolloutConfig(
            horizon_index=1,
            integration_period_override_s=0.01,
        ),
    )
    loss = target_state_nll(
        output,
        truth,
        torch.from_numpy(dataset.target_mask),
        sequence_mask,
        plant_loss_config,
        prediction_horizons_s=dataset.manifest.prediction_horizons_s,
        adaptive_position_context=context,
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.adaptive_position_action_rmse_normalized)
    assert torch.isfinite(loss.position_plant_tracking_rmse_normalized)
    assert torch.isfinite(loss.position_plant_response_rmse_normalized)
    assert torch.isfinite(loss.position_plant_regret_rmse_normalized)
    assert torch.isfinite(loss.position_plant_visibility_rmse_normalized)
    assert torch.isfinite(loss.position_plant_smoothness_rmse_normalized)
    assert torch.isfinite(loss.position_plant_saturation_rmse_normalized)
    assert all(
        parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.any(torch.abs(parameter.grad) > 0.0)
        for parameter in model.parameters()
    )
    evaluation = evaluate_gru(
        model,
        dataset,
        ObservationProfile.SERVO_AWARE,
        batch_size=2,
        loss_config=plant_loss_config,
    )
    assert evaluation.loss is not None
    assert evaluation.position_plant_tracking_rmse_normalized is not None
    assert evaluation.position_plant_response_rmse_normalized is not None
    assert evaluation.position_plant_regret_rmse_normalized is not None
    assert evaluation.position_plant_smoothness_rmse_normalized is not None


def test_differentiable_position_plant_matches_multi_step_simulator():
    base = nominal_scenario()
    servo = replace(
        base.config.servo,
        command_latency_s=0.0073,
        rate_time_constant_s=0.025,
        position_tolerance_rad=0.0007,
        position_quantization_rad=0.0013,
    )
    config = replace(
        base.config,
        servo=servo,
        command_mode=GimbalCommandMode.POSITION,
        timing=replace(
            base.config.timing,
            control_rate_hz=20.0,
            integration_rate_hz=1000.0,
            episode_duration_s=0.1,
        ),
        camera=replace(base.config.camera, frame_rate_hz=20.0),
        scenario=replace(
            base.config.scenario,
            initial_gimbal_angle_rad=0.12,
            initial_gimbal_rate_rad_s=-0.25,
        ),
    )
    env = GimbalServoEnv(config)
    env.reset(seed=37)
    command_value = 0.15
    for _ in range(2):
        result = env.step(
            GimbalAction.position(command_value)
        )

    def value(number: float) -> torch.Tensor:
        return torch.tensor([[number]], dtype=torch.float64)

    context = GRUAdaptivePositionLossContext(
        teacher_action_normalized=value(0.0),
        mask=torch.tensor([[True]]),
        gimbal_angle_rad=value(config.scenario.initial_gimbal_angle_rad),
        gimbal_rate_rad_s=value(config.scenario.initial_gimbal_rate_rad_s),
        control_dt_s=value(config.timing.control_period_s),
        selected_axis_fov_rad=value(config.camera.selected_axis_fov_rad),
        servo_min_angle_rad=value(servo.min_angle_rad),
        servo_max_angle_rad=value(servo.max_angle_rad),
        servo_max_rate_rad_s=value(servo.max_rate_rad_s),
        servo_max_acceleration_rad_s2=value(servo.max_acceleration_rad_s2),
        servo_position_gain_s_inv=value(servo.position_gain_s_inv),
        servo_position_tolerance_rad=value(servo.position_tolerance_rad),
        servo_position_quantization_rad=value(servo.position_quantization_rad),
        servo_command_polarity=value(float(servo.command_polarity)),
        servo_command_latency_s=value(servo.command_latency_s),
        servo_rate_time_constant_s=value(servo.rate_time_constant_s),
        control_period_s=value(config.timing.control_period_s),
        integration_period_s=value(config.timing.integration_period_s),
        camera_frame_period_s=value(config.camera.frame_period_s),
    )
    command = value(command_value).requires_grad_()
    rollout = differentiable_position_servo_rollout(
        command,
        context,
        duration_s=0.1,
    )

    assert rollout.angle_rad.item() == pytest.approx(
        result.diagnostics.gimbal_angle_rad,
        abs=2e-12,
    )
    assert rollout.rate_rad_s.item() == pytest.approx(
        result.diagnostics.gimbal_rate_rad_s,
        abs=2e-12,
    )
    rollout.angle_rad.sum().backward()
    assert command.grad is not None
    assert torch.isfinite(command.grad).all()
    assert torch.abs(command.grad).item() > 0.0


def test_control_aware_loss_penalizes_inconsistent_prediction_heads():
    consistent_mean = torch.tensor(
        [[[[0.0, 1.0], [0.1, 1.0], [0.2, 1.0]]]]
    )
    inconsistent_mean = consistent_mean.clone()
    inconsistent_mean[..., 1, 0] = 0.3
    std = torch.ones_like(consistent_mean)
    targets = torch.zeros_like(consistent_mean)
    target_mask = torch.ones(targets.shape[:-1], dtype=torch.bool)
    sequence_mask = torch.ones(targets.shape[:2], dtype=torch.bool)
    config = GRULossConfig(
        bearing_weight=0.0,
        rate_weight=0.0,
        mean_error_weight=0.0,
        dynamic_consistency_weight=1.0,
    )

    consistent = target_state_nll(
        GRUTargetStateOutput(consistent_mean, std, torch.empty(0)),
        targets,
        target_mask,
        sequence_mask,
        config,
        prediction_horizons_s=(0.0, 0.1, 0.2),
    )
    inconsistent = target_state_nll(
        GRUTargetStateOutput(inconsistent_mean, std, torch.empty(0)),
        targets,
        target_mask,
        sequence_mask,
        config,
        prediction_horizons_s=(0.0, 0.1, 0.2),
    )

    assert consistent.dynamic_consistency_rmse_rad == pytest.approx(0.0)
    assert inconsistent.total > consistent.total


def test_control_aware_loss_applies_horizon_and_critical_label_weights():
    mean = torch.zeros(1, 1, 2, 2)
    std = torch.ones_like(mean)
    targets = torch.zeros_like(mean)
    targets[..., 1, 0] = 1.0
    target_mask = torch.ones(targets.shape[:-1], dtype=torch.bool)
    sequence_mask = torch.ones(targets.shape[:2], dtype=torch.bool)
    output = GRUTargetStateOutput(mean, std, torch.empty(0))
    config = GRULossConfig(
        bearing_weight=0.0,
        rate_weight=0.0,
        mean_error_weight=1.0,
        horizon_weights=(1.0, 2.0),
    )

    uniform = target_state_nll(
        output,
        targets,
        target_mask,
        sequence_mask,
        config,
    )
    critical = target_state_nll(
        output,
        targets,
        target_mask,
        sequence_mask,
        config,
        label_weights=torch.tensor([[[1.0, 4.0]]]),
    )

    assert critical.total > uniform.total


def test_control_aware_loss_matches_hardware_normalized_oracle_actions():
    mean = torch.tensor([[[[0.2, 0.1], [0.3, 0.1]]]])
    std = torch.ones_like(mean)
    targets = mean.clone()
    target_mask = torch.ones(targets.shape[:-1], dtype=torch.bool)
    sequence_mask = torch.ones(targets.shape[:2], dtype=torch.bool)
    context = GRUControlLossContext(
        oracle_actions=torch.tensor([[[0.4, 0.4]]]),
        gimbal_angle_rad=torch.tensor([[0.05]]),
        servo_max_rate_rad_s=torch.tensor([[1.0]]),
        servo_min_angle_rad=torch.tensor([[-0.5]]),
        servo_max_angle_rad=torch.tensor([[0.5]]),
        rate_feedback_gain_s_inv=torch.tensor([[2.0]]),
        position_preview_s=torch.tensor([[0.0]]),
        mask=torch.tensor([[True]]),
    )
    config = GRULossConfig(
        bearing_weight=0.0,
        rate_weight=0.0,
        mean_error_weight=0.0,
        rate_action_weight=1.0,
        position_action_weight=1.0,
    )

    matched = target_state_nll(
        GRUTargetStateOutput(mean, std, torch.empty(0)),
        targets,
        target_mask,
        sequence_mask,
        config,
        control_context=context,
    )
    shifted_mean = mean.clone()
    shifted_mean[..., 0, 0] = 0.3
    shifted = target_state_nll(
        GRUTargetStateOutput(shifted_mean, std, torch.empty(0)),
        targets,
        target_mask,
        sequence_mask,
        config,
        control_context=context,
    )

    assert matched.rate_action_rmse_normalized == pytest.approx(0.0)
    assert matched.position_action_rmse_normalized == pytest.approx(0.0)
    assert shifted.total > matched.total


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
    model = small_model()
    estimator = GRUTargetStateEstimator(
        model,
        config,
        GRUInferenceConfig(
            observation_profile=ObservationProfile.SERVO_AWARE,
            horizon_index=0,
            maximum_staleness_s=0.2,
        ),
    )
    scaled_estimator = GRUTargetStateEstimator(
        model,
        config,
        GRUInferenceConfig(
            observation_profile=ObservationProfile.SERVO_AWARE,
            horizon_index=0,
            maximum_staleness_s=0.2,
            bearing_std_scale=1.25,
            rate_std_scale=0.75,
        ),
    )

    class FreshScale:
        def scales_for_observation(
            self, horizon_index, current_observation, detection_gap_s
        ):
            assert horizon_index == 0
            assert current_observation is observation
            assert detection_gap_s == pytest.approx(0.0)
            return 2.0, 3.0

    with pytest.raises(ValueError, match="either fixed scales"):
        GRUInferenceConfig(
            observation_profile=ObservationProfile.SERVO_AWARE,
            bearing_std_scale=1.1,
            uncertainty_calibration=FreshScale(),
        )

    contextual_estimator = GRUTargetStateEstimator(
        model,
        config,
        GRUInferenceConfig(
            observation_profile=ObservationProfile.SERVO_AWARE,
            horizon_index=0,
            maximum_staleness_s=0.2,
            uncertainty_calibration=FreshScale(),
        ),
    )
    estimate = estimator.update(observation)
    scaled = scaled_estimator.update(observation)
    contextual = contextual_estimator.update(observation)

    assert estimate.valid
    assert estimate.measurement_time_s.valid
    assert estimate.bearing_std_rad.value > 0.0
    assert estimate.rate_std_rad_s.value > 0.0
    assert estimate.time_s == pytest.approx(observation.time_s)
    assert scaled.body_relative_bearing_rad.value == pytest.approx(
        estimate.body_relative_bearing_rad.value
    )
    assert scaled.body_relative_rate_rad_s.value == pytest.approx(
        estimate.body_relative_rate_rad_s.value
    )
    assert scaled.bearing_std_rad.value == pytest.approx(
        1.25 * estimate.bearing_std_rad.value
    )
    assert scaled.rate_std_rad_s.value == pytest.approx(
        0.75 * estimate.rate_std_rad_s.value
    )
    assert contextual.bearing_std_rad.value == pytest.approx(
        2.0 * estimate.bearing_std_rad.value
    )
    assert contextual.rate_std_rad_s.value == pytest.approx(
        3.0 * estimate.rate_std_rad_s.value
    )


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


def test_episode_weighted_training_is_deterministic():
    train = learning_dataset("train", (31, 32, 33))
    validation = learning_dataset("validation", (41,))
    model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=train.manifest.prediction_horizons_s,
        hidden_dim=8,
        embedding_dim=8,
    )
    training_config = GRUTrainingConfig(
        epochs=1,
        batch_size=2,
        seed=13,
    )
    episode_weights = np.asarray((0.5, 1.0, 2.0), dtype=np.float32)

    first = train_gru(
        train,
        validation,
        ObservationProfile.SERVO_AWARE,
        model_config=model_config,
        training_config=training_config,
        training_episode_weights=episode_weights,
    )
    second = train_gru(
        train,
        validation,
        ObservationProfile.SERVO_AWARE,
        model_config=model_config,
        training_config=training_config,
        training_episode_weights=episode_weights,
    )

    assert first.history == second.history
    with pytest.raises(ValueError, match="episode weights shape"):
        train_gru(
            train,
            validation,
            ObservationProfile.SERVO_AWARE,
            model_config=model_config,
            training_config=training_config,
            training_episode_weights=np.ones(2, dtype=np.float32),
        )


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
