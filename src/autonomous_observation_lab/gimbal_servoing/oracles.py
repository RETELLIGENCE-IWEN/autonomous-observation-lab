"""Privileged simulator-truth interfaces for training and ceiling evaluation.

Nothing in this module is a deployable policy input.  The oracle consumes
``GimbalDiagnostics`` or the simulator's exogenous motion definitions directly;
keeping that dependency explicit prevents truth values from silently entering
``GimbalObservation`` feature pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import GimbalCommandMode, GimbalServoingConfig, ServoConfig
from .demos import DemoEpisode, DemoFrame
from .disturbances import AngularMotion
from .env import GimbalServoEnv, wrap_angle_rad
from .types import GimbalAction, GimbalDiagnostics


@dataclass(frozen=True)
class PrivilegedTargetState:
    """Ideal body-relative target state at one simulator time."""

    time_s: float
    body_relative_bearing_rad: float
    body_relative_rate_rad_s: float


@dataclass(frozen=True)
class OracleControlConfig:
    """Configurable conversion from ideal target state to control targets."""

    rate_feedback_gain_s_inv: float = 4.0
    position_preview_s: float = 0.0

    def __post_init__(self) -> None:
        if self.rate_feedback_gain_s_inv < 0.0:
            raise ValueError("rate feedback gain must be non-negative")
        if self.position_preview_s < 0.0:
            raise ValueError("position preview must be non-negative")


@dataclass(frozen=True)
class OracleActionTargets:
    """Ideal normalized targets for both supported actuator interfaces."""

    desired_rate_normalized: float
    desired_position_normalized: float


@dataclass(frozen=True)
class PrivilegedTargetStateOracle:
    """Read simulator truth without exposing it through actor observations."""

    target_motion: AngularMotion
    body_motion: AngularMotion
    servo: ServoConfig
    control: OracleControlConfig = OracleControlConfig()

    def state_at(self, time_s: float) -> PrivilegedTargetState:
        """Query ideal exogenous motion, including labels at future horizons."""
        target_bearing, target_rate = self.target_motion.state_at(time_s)
        body_bearing, body_rate = self.body_motion.state_at(time_s)
        return PrivilegedTargetState(
            time_s=time_s,
            body_relative_bearing_rad=wrap_angle_rad(
                target_bearing - body_bearing
            ),
            body_relative_rate_rad_s=target_rate - body_rate,
        )

    def state_from_diagnostics(
        self, diagnostics: GimbalDiagnostics
    ) -> PrivilegedTargetState:
        """Read the current target state from the truth-only diagnostics path."""
        return PrivilegedTargetState(
            time_s=diagnostics.time_s,
            body_relative_bearing_rad=wrap_angle_rad(
                diagnostics.target_bearing_rad
                - diagnostics.body_bearing_rad
            ),
            body_relative_rate_rad_s=(
                diagnostics.target_rate_rad_s
                - diagnostics.body_rate_rad_s
            ),
        )

    def action_targets(
        self, diagnostics: GimbalDiagnostics
    ) -> OracleActionTargets:
        state = self.state_from_diagnostics(diagnostics)
        bearing_error = wrap_angle_rad(
            state.body_relative_bearing_rad - diagnostics.gimbal_angle_rad
        )
        desired_rate = (
            state.body_relative_rate_rad_s
            + self.control.rate_feedback_gain_s_inv * bearing_error
        )
        rate_normalized = float(
            np.clip(desired_rate / self.servo.max_rate_rad_s, -1.0, 1.0)
        )

        desired_position = wrap_angle_rad(
            state.body_relative_bearing_rad
            + self.control.position_preview_s
            * state.body_relative_rate_rad_s
        )
        desired_position = float(
            np.clip(
                desired_position,
                self.servo.min_angle_rad,
                self.servo.max_angle_rad,
            )
        )
        position_normalized = self.servo.normalized_from_position(
            desired_position
        )
        return OracleActionTargets(
            desired_rate_normalized=rate_normalized,
            desired_position_normalized=position_normalized,
        )


@dataclass(frozen=True)
class PrivilegedOracleController:
    """Truth-only controller used to measure an upper-bound control rollout.

    Its ``act`` signature intentionally accepts diagnostics rather than a
    ``GimbalObservation``.  It therefore cannot be passed accidentally where a
    deployable controller is expected.
    """

    oracle: PrivilegedTargetStateOracle
    command_mode: GimbalCommandMode

    @property
    def name(self) -> str:
        return f"privileged_oracle_{self.command_mode.name.lower()}"

    def act(self, diagnostics: GimbalDiagnostics) -> GimbalAction:
        targets = self.oracle.action_targets(diagnostics)
        if self.command_mode is GimbalCommandMode.RATE:
            return GimbalAction.rate(targets.desired_rate_normalized)
        return GimbalAction.position(targets.desired_position_normalized)


def rollout_privileged_oracle(
    *,
    config: GimbalServoingConfig,
    target_motion: AngularMotion,
    body_motion: AngularMotion,
    command_mode: GimbalCommandMode,
    seed: int,
    oracle_control: OracleControlConfig | None = None,
    name: str | None = None,
    description: str | None = None,
) -> DemoEpisode:
    """Run a truth-only controller through the real configured servo plant."""
    rollout_config = replace(config, command_mode=command_mode)
    oracle = PrivilegedTargetStateOracle(
        target_motion=target_motion,
        body_motion=body_motion,
        servo=rollout_config.servo,
        control=oracle_control or OracleControlConfig(),
    )
    controller = PrivilegedOracleController(oracle, command_mode)
    env = GimbalServoEnv(
        rollout_config,
        target_motion=target_motion,
        body_motion=body_motion,
    )
    observation, diagnostics = env.reset(seed)
    action = controller.act(diagnostics)
    frames = [DemoFrame(0, action, observation, diagnostics)]
    step = 0
    while True:
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
        if result.truncated:
            break
        action = controller.act(result.diagnostics)
    return DemoEpisode(
        name=name or controller.name,
        description=description
        or "Privileged target-state feedback through the configured servo plant.",
        config=rollout_config,
        frames=tuple(frames),
    )
