import math
from dataclasses import dataclass

from .config import (
    CameraConfig,
    GimbalCommandMode,
    GimbalServoingConfig,
    ObservationProfile,
    ScenarioConfig,
    ServoConfig,
    TimingConfig,
)
from .disturbances import SampledAngularMotion, StaticAngularMotion
from .env import GimbalServoEnv
from .types import GimbalAction, GimbalDiagnostics, GimbalObservation


@dataclass(frozen=True)
class DemoFrame:
    step: int
    action: GimbalAction
    observation: GimbalObservation
    diagnostics: GimbalDiagnostics


@dataclass(frozen=True)
class DemoEpisode:
    name: str
    description: str
    config: GimbalServoingConfig
    frames: tuple[DemoFrame, ...]


def demo_config() -> GimbalServoingConfig:
    """A visible, intentionally non-ideal position-servo configuration."""
    return GimbalServoingConfig(
        servo=ServoConfig(
            min_angle_rad=math.radians(-45.0),
            max_angle_rad=math.radians(45.0),
            max_rate_rad_s=math.radians(55.0),
            max_acceleration_rad_s2=math.radians(220.0),
            rate_time_constant_s=0.080,
            command_latency_s=0.080,
            position_gain_s_inv=5.0,
            position_tolerance_rad=math.radians(0.10),
        ),
        camera=CameraConfig(
            selected_axis_fov_rad=math.radians(60.0),
            orthogonal_fov_rad=math.radians(45.0),
            frame_rate_hz=30.0,
            detection_latency_s=0.100,
            confidence_mean=0.95,
        ),
        timing=TimingConfig(
            control_rate_hz=30.0,
            integration_rate_hz=1000.0,
            episode_duration_s=5.5,
        ),
        scenario=ScenarioConfig(
            target_angular_width_rad=math.radians(4.0),
            target_angular_height_rad=math.radians(4.0),
        ),
        observation_profile=ObservationProfile.SERVO_AWARE,
        command_mode=GimbalCommandMode.POSITION,
    )


def _gimbal_position_schedule(time_s: float) -> GimbalAction:
    if time_s < 0.75:
        command = 0.0
    elif time_s < 2.25:
        command = 0.40
    elif time_s < 3.75:
        command = -0.30
    else:
        command = 0.0
    return GimbalAction.position(command)


def _rollout(
    *,
    name: str,
    description: str,
    config: GimbalServoingConfig,
    env: GimbalServoEnv,
    action_at,
    seed: int,
) -> DemoEpisode:
    observation, diagnostics = env.reset(seed)
    initial_action = action_at(0.0)
    frames = [
        DemoFrame(
            step=0,
            action=initial_action,
            observation=observation,
            diagnostics=diagnostics,
        )
    ]
    step = 0
    while True:
        action = action_at(observation.time_s)
        result = env.step(action)
        step += 1
        frames.append(
            DemoFrame(
                step=step,
                action=action,
                observation=result.observation,
                diagnostics=result.diagnostics,
            )
        )
        observation = result.observation
        if result.truncated:
            break
    return DemoEpisode(
        name=name,
        description=description,
        config=config,
        frames=tuple(frames),
    )


def stationary_target_moving_gimbal(seed: int = 21) -> DemoEpisode:
    config = demo_config()
    env = GimbalServoEnv(
        config,
        body_motion=StaticAngularMotion(),
        target_motion=StaticAngularMotion(),
    )
    return _rollout(
        name="gimbal_moves",
        description=(
            "Stationary target; delayed position steps move the gimbal and bbox."
        ),
        config=config,
        env=env,
        action_at=_gimbal_position_schedule,
        seed=seed,
    )


def moving_target_stationary_gimbal(
    reference: DemoEpisode,
    seed: int = 21,
) -> DemoEpisode:
    """Reproduce the reference image error using target motion alone."""
    times = tuple(frame.diagnostics.time_s for frame in reference.frames)
    half_fov = 0.5 * reference.config.camera.selected_axis_fov_rad
    target_angles = tuple(
        frame.diagnostics.true_image_error_normalized * half_fov
        for frame in reference.frames
    )
    target_motion = SampledAngularMotion(times, target_angles)
    config = reference.config
    env = GimbalServoEnv(
        config,
        body_motion=StaticAngularMotion(),
        target_motion=target_motion,
    )
    return _rollout(
        name="target_moves",
        description=(
            "Stationary body-forward gimbal; target motion reproduces the same bbox trace."
        ),
        config=config,
        env=env,
        action_at=lambda _time_s: GimbalAction.position(0.0),
        seed=seed,
    )


def paired_cause_demo(seed: int = 21) -> tuple[DemoEpisode, DemoEpisode]:
    gimbal_episode = stationary_target_moving_gimbal(seed)
    target_episode = moving_target_stationary_gimbal(gimbal_episode, seed)
    return gimbal_episode, target_episode
