import math
from dataclasses import fields, replace

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
    CausalTargetStateGRUWithPositionResidual,
    GRUAdaptivePositionLossContext,
    GRUControlLossContext,
    GRUInferenceConfig,
    GRULossConfig,
    GRUPositionPlantRolloutConfig,
    GRUPositionResidualConfig,
    GRUTargetStateEstimator,
    GRUTargetStateModelConfig,
    GRUTargetStateOutput,
    adaptive_position_surrogate_actions,
    angular_residual_rad,
    differentiable_position_servo_rollout,
    differentiable_position_servo_sequence,
    gru_parameter_count,
    load_gru_checkpoint,
    load_gru_position_residual_checkpoint,
    save_gru_checkpoint,
    save_gru_position_residual_checkpoint,
    target_state_nll,
)
from autonomous_observation_lab.gimbal_servoing.gru_training import (
    GRUReferenceAnchorConfig,
    GRUTrainingConfig,
    constant_velocity_predictions,
    evaluate_constant_velocity_baseline,
    evaluate_gru,
    train_gru,
)
from autonomous_observation_lab.gimbal_servoing.gru_profiles import (
    run_gru_profile_comparison,
)
from autonomous_observation_lab.gimbal_servoing.multi_command_policy import (
    CausalRecurrentPositionResidualPolicy,
    CounterfactualWindowBatch,
    RecurrentPositionResidualPolicyConfig,
    counterfactual_capture_source_indices,
    recurrent_policy_input_dim,
    rollout_counterfactual_window,
)
from autonomous_observation_lab.gimbal_servoing.on_policy_distillation import (
    rollout_counterfactual_position_policy,
)
from autonomous_observation_lab.gimbal_servoing.sequence_oracle import (
    PrivilegedSequenceOracleConfig,
    optimize_privileged_command_sequence,
)
from autonomous_observation_lab.gimbal_servoing.sequence_distillation import (
    CausalHardwareConditionedPositionPolicy,
    SequenceDistillationPolicyConfig,
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


def test_position_residual_head_preserves_state_and_round_trips(tmp_path):
    torch.manual_seed(5)
    base = small_model().eval()
    model = CausalTargetStateGRUWithPositionResidual(
        base,
        GRUPositionResidualConfig(
            hidden_dim=7,
            maximum_half_fov_fraction=0.2,
        ),
    ).eval()
    features = torch.randn(2, 6, len(FEATURE_NAMES))
    reference = base(features)
    initial = model(features)

    torch.testing.assert_close(initial.mean, reference.mean)
    torch.testing.assert_close(initial.std, reference.std)
    torch.testing.assert_close(
        initial.position_target_residual_fov_fraction,
        torch.zeros(2, 6),
    )
    final = model.residual_head[-1]
    assert isinstance(final, torch.nn.Linear)
    final.bias.data.fill_(10.0)
    shifted = model(features)
    assert torch.all(
        shifted.position_target_residual_fov_fraction <= 0.2
    )
    assert torch.all(
        shifted.position_target_residual_fov_fraction > 0.19
    )
    assert not any(
        parameter.requires_grad for parameter in model.base_model.parameters()
    )

    checkpoint = save_gru_position_residual_checkpoint(
        tmp_path / "position_residual.pt",
        model,
        {"purpose": "test"},
    )
    restored, metadata = load_gru_position_residual_checkpoint(checkpoint)
    replay = restored(features)
    torch.testing.assert_close(
        replay.position_target_residual_fov_fraction,
        shifted.position_target_residual_fov_fraction,
    )
    torch.testing.assert_close(replay.mean, shifted.mean)
    assert metadata == {"purpose": "test"}


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


def test_differentiable_position_sequence_preserves_command_queue():
    base = nominal_scenario()
    servo = replace(
        base.config.servo,
        command_latency_s=0.0127,
        rate_time_constant_s=0.023,
        position_tolerance_rad=0.0006,
        position_quantization_rad=0.0011,
    )
    commands = (0.15, -0.20, 0.35, 0.05)
    config = replace(
        base.config,
        servo=servo,
        command_mode=GimbalCommandMode.POSITION,
        timing=replace(
            base.config.timing,
            control_rate_hz=20.0,
            integration_rate_hz=1000.0,
            episode_duration_s=0.20,
        ),
        camera=replace(base.config.camera, frame_rate_hz=27.0),
        scenario=replace(
            base.config.scenario,
            initial_gimbal_angle_rad=0.12,
            initial_gimbal_rate_rad_s=-0.25,
        ),
    )
    env = GimbalServoEnv(config)
    env.reset(seed=41)
    expected_angles = []
    expected_rates = []
    expected_applied = []
    for command_value in commands:
        result = env.step(GimbalAction.position(command_value))
        expected_angles.append(result.diagnostics.gimbal_angle_rad)
        expected_rates.append(result.diagnostics.gimbal_rate_rad_s)
        expected_applied.append(
            result.diagnostics.applied_position_command_rad
        )

    shape = (1, len(commands))

    def values(number: float) -> torch.Tensor:
        return torch.full(shape, number, dtype=torch.float64)

    context = GRUAdaptivePositionLossContext(
        teacher_action_normalized=values(0.0),
        mask=torch.ones(shape, dtype=torch.bool),
        gimbal_angle_rad=values(config.scenario.initial_gimbal_angle_rad),
        gimbal_rate_rad_s=values(config.scenario.initial_gimbal_rate_rad_s),
        control_dt_s=values(config.timing.control_period_s),
        selected_axis_fov_rad=values(config.camera.selected_axis_fov_rad),
        servo_min_angle_rad=values(servo.min_angle_rad),
        servo_max_angle_rad=values(servo.max_angle_rad),
        servo_max_rate_rad_s=values(servo.max_rate_rad_s),
        servo_max_acceleration_rad_s2=values(
            servo.max_acceleration_rad_s2
        ),
        servo_position_gain_s_inv=values(servo.position_gain_s_inv),
        servo_position_tolerance_rad=values(
            servo.position_tolerance_rad
        ),
        servo_position_quantization_rad=values(
            servo.position_quantization_rad
        ),
        servo_command_polarity=values(float(servo.command_polarity)),
        servo_command_latency_s=values(servo.command_latency_s),
        servo_rate_time_constant_s=values(servo.rate_time_constant_s),
        control_period_s=values(config.timing.control_period_s),
        integration_period_s=values(config.timing.integration_period_s),
        camera_frame_period_s=values(config.camera.frame_period_s),
    )
    command = torch.tensor([commands], dtype=torch.float64, requires_grad=True)
    rollout = differentiable_position_servo_sequence(
        command,
        context,
        torch.ones(shape, dtype=torch.bool),
    )

    torch.testing.assert_close(
        rollout.angle_rad,
        torch.tensor([expected_angles], dtype=torch.float64),
        atol=2e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rollout.rate_rad_s,
        torch.tensor([expected_rates], dtype=torch.float64),
        atol=2e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rollout.applied_position_rad,
        torch.tensor([expected_applied], dtype=torch.float64),
        atol=2e-12,
        rtol=0.0,
    )
    rollout.angle_rad[:, -1].sum().backward()
    assert command.grad is not None
    assert torch.isfinite(command.grad).all()
    assert torch.count_nonzero(command.grad).item() >= 2


def test_privileged_sequence_oracle_improves_a_feasible_focus_segment():
    shape = (1, 10)

    def values(number: float) -> torch.Tensor:
        return torch.full(shape, number, dtype=torch.float64)

    context = GRUAdaptivePositionLossContext(
        teacher_action_normalized=values(0.0),
        mask=torch.ones(shape, dtype=torch.bool),
        gimbal_angle_rad=values(0.0),
        gimbal_rate_rad_s=values(0.0),
        control_dt_s=values(0.04),
        selected_axis_fov_rad=values(math.radians(60.0)),
        servo_min_angle_rad=values(-1.0),
        servo_max_angle_rad=values(1.0),
        servo_max_rate_rad_s=values(10.0),
        servo_max_acceleration_rad_s2=values(1000.0),
        servo_position_gain_s_inv=values(5.0),
        servo_position_tolerance_rad=values(0.0),
        servo_position_quantization_rad=values(0.0),
        servo_command_polarity=values(1.0),
        servo_command_latency_s=values(0.01),
        servo_rate_time_constant_s=values(0.03),
        control_period_s=values(0.04),
        integration_period_s=values(0.001),
        camera_frame_period_s=values(1.0 / 30.0),
    )
    base = values(0.05)
    result = optimize_privileged_command_sequence(
        base,
        values(0.30),
        context,
        torch.ones(shape, dtype=torch.bool),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        config=PrivilegedSequenceOracleConfig(
            focus_start_index=2,
            focus_steps=8,
            optimization_iterations=20,
            learning_rate=0.15,
            maximum_command_residual=0.50,
            maximum_smoothness_regression_fraction=100.0,
            maximum_smoothness_regression_absolute_mse=0.10,
            blend_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        ),
    )

    assert result.selected_metrics["tracking_mse"].item() < (
        result.base_metrics["tracking_mse"].item()
    )
    torch.testing.assert_close(
        result.selected_command_normalized[:, :2],
        base[:, :2],
    )
    assert torch.all(torch.abs(result.selected_command_normalized) <= 1.0)
    assert result.selected_blend_fraction.item() > 0.0


def test_counterfactual_window_is_causal_and_updates_servo_features():
    dataset = learning_dataset("train", (171, 172))
    adapter = default_visibility_risk_candidates()[2].controller
    supervision = compute_adaptive_position_supervision(
        dataset,
        adapter=adapter,
        profile=ObservationProfile.SERVO_AWARE,
    )
    profile_index = dataset.manifest.observation_profiles.index(
        ObservationProfile.SERVO_AWARE.value
    )
    start = 1
    step_count = 3
    end = start + step_count
    logged = torch.from_numpy(
        dataset.features[:, profile_index, start:end]
    ).float()
    warmup = torch.from_numpy(
        dataset.features[:, profile_index, :start]
    ).float()
    target_bearing = torch.from_numpy(
        dataset.targets[:, start : end + 1, 0, 0]
    ).float()
    time_s = torch.from_numpy(dataset.time_s[:, start : end + 1]).float()
    capture_source = counterfactual_capture_source_indices(time_s, logged)
    context = GRUAdaptivePositionLossContext(
        **{
            field.name: (
                torch.from_numpy(
                    getattr(supervision, field.name)[:, start:end]
                ).bool()
                if field.name == "mask"
                else torch.from_numpy(
                    getattr(supervision, field.name)[:, start:end]
                ).float()
            )
            for field in fields(GRUAdaptivePositionLossContext)
        }
    )
    window = CounterfactualWindowBatch(
        logged_features=logged,
        warmup_features=warmup,
        target_bearing_rad=target_bearing,
        time_s=time_s,
        capture_source_index=capture_source,
        context=context,
        sequence_mask=torch.ones(
            dataset.episode_count,
            step_count,
            dtype=torch.bool,
        ),
    )
    model = small_model().eval()
    policy = CausalRecurrentPositionResidualPolicy(
        RecurrentPositionResidualPolicyConfig(
            input_dim=recurrent_policy_input_dim(model.horizon_count),
            hidden_dim=8,
            embedding_dim=8,
        )
    )
    rollout = rollout_counterfactual_window(
        model,
        policy,
        window,
        prediction_horizons_s=dataset.manifest.prediction_horizons_s,
        adapter=adapter,
    )

    torch.testing.assert_close(
        rollout.policy_residual_normalized,
        torch.zeros_like(rollout.policy_residual_normalized),
    )
    angle_index = FEATURE_NAMES.index("gimbal_angle_rad")
    torch.testing.assert_close(
        rollout.synthetic_features[:, 1:, angle_index],
        rollout.gimbal_angle_rad[:, :-1],
    )
    assert torch.all(torch.isfinite(rollout.tracking_error_normalized))

    changed_logged = logged.clone()
    changed_logged[:, -1] = torch.randn_like(changed_logged[:, -1]) * 50.0
    changed = rollout_counterfactual_window(
        model,
        policy,
        replace(window, logged_features=changed_logged),
        prediction_horizons_s=dataset.manifest.prediction_horizons_s,
        adapter=adapter,
    )
    torch.testing.assert_close(
        rollout.command_normalized[:, :-1],
        changed.command_normalized[:, :-1],
    )
    rollout.policy_residual_normalized.sum().backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0.0)
        for parameter in policy.parameters()
    )

    final = policy.head[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias.fill_(0.25)
    direct = rollout_counterfactual_window(
        model,
        policy,
        window,
        prediction_horizons_s=dataset.manifest.prediction_horizons_s,
        adapter=adapter,
        residual_application="command_normalized",
    )
    position_index = FEATURE_NAMES.index("previous_position_command_rad")
    minimum_angle = context.servo_min_angle_rad[:, 0]
    maximum_angle = context.servo_max_angle_rad[:, 0]
    issued = direct.command_normalized[:, 0] * torch.where(
        direct.command_normalized[:, 0] >= 0.0,
        maximum_angle,
        -minimum_angle,
    )
    torch.testing.assert_close(
        direct.synthetic_features[:, 1, position_index],
        issued,
    )
    assert torch.any(
        direct.command_normalized[:, 0] != rollout.command_normalized[:, 0]
    )

    actor = CausalHardwareConditionedPositionPolicy(
        SequenceDistillationPolicyConfig(hidden_dim=8, embedding_dim=8)
    )
    actor_commands = actor(logged, context)
    actor_changed = actor(changed_logged, context)
    torch.testing.assert_close(
        actor_commands[:, :-1],
        actor_changed[:, :-1],
    )
    actor_commands.sum().backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0.0)
        for parameter in actor.parameters()
    )

    actor_rollout = rollout_counterfactual_position_policy(
        actor,
        logged,
        target_bearing,
        time_s,
        capture_source,
        context,
        window.sequence_mask,
    )
    torch.testing.assert_close(
        actor_rollout.synthetic_features[:, 1:, angle_index],
        actor_rollout.gimbal_angle_rad[:, :-1],
    )
    assert actor_rollout.command_normalized.shape == (
        dataset.episode_count,
        step_count,
    )
    assert torch.all(torch.isfinite(actor_rollout.tracking_error_normalized))


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


def test_reference_anchor_limits_function_drift():
    train = learning_dataset("train", (51, 52, 53))
    validation = learning_dataset("validation", (61,))
    model_config = GRUTargetStateModelConfig(
        input_dim=len(FEATURE_NAMES),
        prediction_horizons_s=train.manifest.prediction_horizons_s,
        hidden_dim=8,
        embedding_dim=8,
    )
    torch.manual_seed(23)
    reference_model = CausalTargetStateGRU(model_config)
    reference_state = {
        name: value.detach().clone()
        for name, value in reference_model.state_dict().items()
    }
    training_config = GRUTrainingConfig(
        epochs=2,
        batch_size=2,
        learning_rate=3e-3,
        seed=19,
    )
    unanchored = train_gru(
        train,
        validation,
        ObservationProfile.SERVO_AWARE,
        model_config=model_config,
        training_config=training_config,
        initial_state_dict=reference_state,
        retain_epoch_states=True,
    )
    anchored = train_gru(
        train,
        validation,
        ObservationProfile.SERVO_AWARE,
        model_config=model_config,
        training_config=training_config,
        initial_state_dict=reference_state,
        reference_anchor_config=GRUReferenceAnchorConfig(
            bearing_weight=100.0,
            rate_weight=100.0,
            project_conflicting_gradients=True,
        ),
        training_reference_anchor_weights=np.ones_like(
            train.target_mask,
            dtype=np.float32,
        ),
        retain_epoch_states=True,
    )

    def displacement(state):
        return sum(
            torch.sum((state[name] - reference_state[name]).square()).item()
            for name in reference_state
        )

    assert displacement(anchored.epoch_state_dicts[-1]) < displacement(
        unanchored.epoch_state_dicts[-1]
    )
    assert anchored.history[-1].training_reference_anchor_loss > 0.0
    assert 0.0 <= (
        anchored.history[-1].reference_anchor_conflict_fraction
    ) <= 1.0
    with pytest.raises(ValueError, match="projection epsilon"):
        GRUReferenceAnchorConfig(projection_epsilon=0.0)
    with pytest.raises(ValueError, match="initial state dictionary"):
        train_gru(
            train,
            validation,
            ObservationProfile.SERVO_AWARE,
            model_config=model_config,
            training_config=replace(training_config, epochs=1),
            reference_anchor_config=GRUReferenceAnchorConfig(
                bearing_weight=1.0
            ),
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
