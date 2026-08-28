import math
from dataclasses import replace

import numpy as np
import pytest

from autonomous_observation_lab.gimbal_servoing import (
    BodyRateFeedforwardController,
    CameraConfig,
    GimbalAction,
    GimbalCommandMode,
    GimbalServoEnv,
    GimbalServoingConfig,
    ObservationProfile,
    ProportionalController,
    ProportionalPositionController,
    RatePulseAngularMotion,
    ScenarioConfig,
    ServoConfig,
    SinusoidalAngularMotion,
    StaticAngularMotion,
    TimingConfig,
    WrongSignController,
    ZeroController,
)


def ideal_config(**overrides) -> GimbalServoingConfig:
    config = GimbalServoingConfig(
        servo=ServoConfig(
            min_angle_rad=math.radians(-90.0),
            max_angle_rad=math.radians(90.0),
            max_rate_rad_s=math.radians(120.0),
            max_acceleration_rad_s2=math.radians(2000.0),
            rate_time_constant_s=0.010,
            command_latency_s=0.0,
        ),
        camera=CameraConfig(
            selected_axis_fov_rad=math.radians(60.0),
            orthogonal_fov_rad=math.radians(45.0),
            frame_rate_hz=50.0,
            detection_latency_s=0.0,
        ),
        timing=TimingConfig(
            control_rate_hz=50.0,
            integration_rate_hz=1000.0,
            episode_duration_s=2.0,
        ),
    )
    return replace(config, **overrides)


def rollout(env, controller):
    observation, diagnostics = env.reset(seed=123)
    controller.reset()
    trace = [diagnostics]
    while True:
        result = env.step(controller.act(observation))
        trace.append(result.diagnostics)
        observation = result.observation
        if result.truncated:
            return trace


def test_body_forward_is_zero_and_positive_action_reduces_positive_error():
    target = StaticAngularMotion(math.radians(10.0))
    env = GimbalServoEnv(ideal_config(), target_motion=target)
    observation, diagnostics = env.reset(seed=1)

    assert diagnostics.gimbal_angle_rad == 0.0
    assert diagnostics.optical_axis_bearing_rad == 0.0
    assert observation.image_error_normalized.value > 0.0

    initial_error = diagnostics.true_image_error_normalized
    for _ in range(10):
        diagnostics = env.step(GimbalAction(0.25)).diagnostics
    assert diagnostics.gimbal_angle_rad > 0.0
    assert diagnostics.true_image_error_normalized < initial_error


def test_action_requires_exactly_one_bounded_command():
    assert GimbalAction.rate(0.25).mode is GimbalCommandMode.RATE
    assert GimbalAction.position(-0.25).mode is GimbalCommandMode.POSITION
    with pytest.raises(ValueError, match="exactly one"):
        GimbalAction()
    with pytest.raises(ValueError, match="exactly one"):
        GimbalAction(0.1, 0.2)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        GimbalAction.position(1.1)


def test_command_profile_rejects_the_other_action_type():
    rate_env = GimbalServoEnv(ideal_config())
    rate_env.reset(seed=1)
    with pytest.raises(ValueError, match="expects desired_rate"):
        rate_env.step(GimbalAction.position(0.0))

    position_config = replace(
        ideal_config(), command_mode=GimbalCommandMode.POSITION
    )
    position_env = GimbalServoEnv(position_config)
    position_env.reset(seed=1)
    with pytest.raises(ValueError, match="expects desired_position"):
        position_env.step(GimbalAction.rate(0.0))


def test_position_zero_commands_body_forward():
    base = ideal_config()
    config = replace(
        base,
        command_mode=GimbalCommandMode.POSITION,
        scenario=replace(
            base.scenario, initial_gimbal_angle_rad=math.radians(20.0)
        ),
    )
    env = GimbalServoEnv(config)
    _, diagnostics = env.reset(seed=2)
    assert diagnostics.gimbal_angle_rad == pytest.approx(math.radians(20.0))

    while True:
        result = env.step(GimbalAction.position(0.0))
        if result.truncated:
            break
    assert result.diagnostics.requested_position_rad == pytest.approx(0.0)
    assert abs(result.diagnostics.gimbal_angle_rad) < math.radians(0.2)


@pytest.mark.parametrize(
    ("command", "expected_angle_deg"), [(0.5, 40.0), (-0.5, -20.0)]
)
def test_position_normalization_preserves_zero_with_asymmetric_travel(
    command, expected_angle_deg
):
    base = ideal_config()
    servo = replace(
        base.servo,
        min_angle_rad=math.radians(-40.0),
        max_angle_rad=math.radians(80.0),
    )
    config = replace(
        base,
        servo=servo,
        command_mode=GimbalCommandMode.POSITION,
    )
    env = GimbalServoEnv(config)
    env.reset(seed=2)
    while True:
        result = env.step(GimbalAction.position(command))
        if result.truncated:
            break

    expected = math.radians(expected_angle_deg)
    assert result.diagnostics.applied_position_command_rad == pytest.approx(expected)
    assert result.diagnostics.gimbal_angle_rad == pytest.approx(
        expected, abs=math.radians(0.2)
    )


def test_proportional_position_controller_centers_nominal_target():
    base = ideal_config()
    config = replace(
        base,
        command_mode=GimbalCommandMode.POSITION,
        observation_profile=ObservationProfile.SERVO_AWARE,
    )
    controller = ProportionalPositionController(
        servo=config.servo,
        selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
    )
    trace = rollout(
        GimbalServoEnv(
            config, target_motion=StaticAngularMotion(math.radians(10.0))
        ),
        controller,
    )
    assert abs(trace[-1].true_image_error_normalized) < 0.01


def test_seeded_noisy_delayed_replay_is_identical():
    base = ideal_config()
    camera = replace(
        base.camera,
        detection_latency_s=0.035,
        detection_latency_jitter_s=0.012,
        center_noise_std_normalized=0.015,
        size_noise_std_fraction=0.01,
        confidence_noise_std=0.03,
        miss_probability=0.20,
    )
    config = replace(base, camera=camera)
    traces = []
    for _ in range(2):
        env = GimbalServoEnv(
            config,
            body_motion=SinusoidalAngularMotion(
                amplitude_rad=math.radians(5.0), frequency_hz=0.8
            ),
        )
        observation, _ = env.reset(seed=991)
        trace = []
        for index in range(30):
            result = env.step(GimbalAction(0.15 if index < 10 else -0.05))
            observation = result.observation
            trace.append(
                (
                    observation.frame_updated,
                    observation.measurement_age_s,
                    observation.image_error_normalized,
                    observation.confidence,
                    result.diagnostics.gimbal_angle_rad,
                )
            )
        traces.append(trace)
    assert traces[0] == traces[1]


def test_random_imperfection_streams_are_independent():
    base = ideal_config()
    common_camera = replace(
        base.camera,
        detection_latency_jitter_s=0.010,
        miss_probability=0.35,
    )
    clean = replace(base, camera=common_camera)
    noisy = replace(
        base,
        camera=replace(common_camera, center_noise_std_normalized=0.05),
    )

    schedules = []
    for config in (clean, noisy):
        env = GimbalServoEnv(config)
        observation, _ = env.reset(seed=77)
        schedule = []
        for _ in range(40):
            observation = env.step(GimbalAction(0.0)).observation
            schedule.append(
                (
                    observation.frame_updated,
                    observation.measurement_age_s,
                    observation.detection_valid,
                )
            )
        schedules.append(schedule)
    assert schedules[0] == schedules[1]


def test_forced_dropout_intervals_are_exact_and_seed_independent():
    base = ideal_config()
    config = replace(
        base,
        camera=replace(
            base.camera,
            forced_dropout_intervals_s=((0.10, 0.20),),
        ),
        timing=replace(base.timing, episode_duration_s=0.30),
    )
    schedules = []
    for seed in (1, 999):
        env = GimbalServoEnv(config)
        observation, _ = env.reset(seed=seed)
        schedule = [(observation.time_s, observation.detection_valid)]
        while True:
            result = env.step(GimbalAction.rate(0.0))
            observation = result.observation
            if observation.frame_updated:
                schedule.append(
                    (observation.time_s, observation.detection_valid)
                )
            if result.truncated:
                break
        schedules.append(schedule)

    assert schedules[0] == schedules[1]
    assert all(
        valid == (not 0.10 <= round(time_s, 8) < 0.20)
        for time_s, valid in schedules[0]
    )


def test_forced_dropout_intervals_must_be_ordered_and_disjoint():
    with pytest.raises(ValueError, match="sorted and disjoint"):
        CameraConfig(
            forced_dropout_intervals_s=((0.4, 0.8), (0.2, 0.3))
        )
    with pytest.raises(ValueError, match="finite and increasing"):
        CameraConfig(forced_dropout_intervals_s=((0.3, 0.3),))


@pytest.mark.parametrize(
    ("profile", "servo_valid", "body_valid"),
    [
        (ObservationProfile.VISION_ONLY, False, False),
        (ObservationProfile.SERVO_AWARE, True, False),
        (ObservationProfile.DISTURBANCE_AWARE, True, True),
    ],
)
def test_observation_profiles_use_explicit_validity_masks(
    profile, servo_valid, body_valid
):
    config = replace(ideal_config(), observation_profile=profile)
    observation, _ = GimbalServoEnv(config).reset(seed=4)
    assert observation.gimbal_angle_rad.valid is servo_valid
    assert observation.gimbal_rate_rad_s.valid is servo_valid
    assert observation.body_rate_rad_s.valid is body_valid


def test_command_latency_is_a_causal_queue():
    base = ideal_config()
    config = replace(
        base,
        servo=replace(base.servo, command_latency_s=0.10),
        timing=replace(base.timing, control_rate_hz=100.0),
    )
    env = GimbalServoEnv(config)
    env.reset(seed=9)
    for _ in range(9):
        diagnostics = env.step(GimbalAction(0.5)).diagnostics
    assert diagnostics.time_s == pytest.approx(0.09)
    assert diagnostics.gimbal_angle_rad == pytest.approx(0.0)

    for _ in range(3):
        diagnostics = env.step(GimbalAction(0.5)).diagnostics
    assert diagnostics.time_s == pytest.approx(0.12)
    assert diagnostics.gimbal_angle_rad > 0.0


def test_nominal_plant_separates_zero_proportional_and_wrong_sign_control():
    target = StaticAngularMotion(math.radians(10.0))
    config = ideal_config()
    controllers = (
        ZeroController(),
        ProportionalController(gain=1.2),
        WrongSignController(gain=1.2),
    )
    final_errors = {}
    for controller in controllers:
        trace = rollout(GimbalServoEnv(config, target_motion=target), controller)
        final_errors[controller.name] = abs(trace[-1].true_image_error_normalized)

    assert final_errors["proportional"] < 0.05
    assert final_errors["proportional"] < final_errors["zero"]
    assert final_errors["wrong_sign"] > final_errors["zero"]


def test_body_rate_feedforward_improves_delayed_periodic_tracking():
    base = ideal_config()
    config = replace(
        base,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
        camera=replace(base.camera, detection_latency_s=0.080),
        servo=replace(base.servo, command_latency_s=0.020),
        timing=replace(base.timing, episode_duration_s=4.0),
    )
    body_motion = SinusoidalAngularMotion(
        amplitude_rad=math.radians(8.0), frequency_hz=0.5
    )
    controllers = (
        ProportionalController(gain=1.5),
        BodyRateFeedforwardController(
            max_rate_rad_s=config.servo.max_rate_rad_s,
            proportional_gain=1.5,
        ),
    )
    errors = {}
    for controller in controllers:
        trace = rollout(
            GimbalServoEnv(config, body_motion=body_motion), controller
        )
        tail = np.array([sample.true_image_error_normalized for sample in trace[25:]])
        errors[controller.name] = float(np.sqrt(np.mean(tail**2)))

    assert errors["body_rate_feedforward"] < 0.7 * errors["proportional"]


def test_unannounced_pulse_is_not_visible_to_o0_before_image_motion():
    base = ideal_config()
    config_o0 = replace(
        base,
        observation_profile=ObservationProfile.VISION_ONLY,
        timing=replace(base.timing, control_rate_hz=20.0, episode_duration_s=1.2),
    )
    pulse = RatePulseAngularMotion(
        onset_s=1.0,
        duration_s=0.05,
        rate_rad_s=math.radians(100.0),
    )
    static_env = GimbalServoEnv(config_o0)
    pulse_env = GimbalServoEnv(config_o0, body_motion=pulse)
    static_observation, _ = static_env.reset(seed=5)
    pulse_observation, _ = pulse_env.reset(seed=5)
    for _ in range(20):
        static_observation = static_env.step(GimbalAction(0.0)).observation
        pulse_observation = pulse_env.step(GimbalAction(0.0)).observation

    assert static_observation.time_s == pytest.approx(1.0)
    assert static_observation.image_error_normalized == pulse_observation.image_error_normalized
    assert not pulse_observation.body_rate_rad_s.valid

    config_o2 = replace(
        config_o0, observation_profile=ObservationProfile.DISTURBANCE_AWARE
    )
    o2_env = GimbalServoEnv(config_o2, body_motion=pulse)
    observation, _ = o2_env.reset(seed=5)
    for _ in range(20):
        observation = o2_env.step(GimbalAction(0.0)).observation
    assert observation.body_rate_rad_s.valid
    assert observation.body_rate_rad_s.value == pytest.approx(pulse.rate_rad_s)
