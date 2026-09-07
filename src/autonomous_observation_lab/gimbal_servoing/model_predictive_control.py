"""Deployable identified finite-horizon control for the C4 baseline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import GimbalCommandMode, ServoConfig
from .estimators import (
    MultiHorizonTargetStateEstimator,
    TargetStateEstimate,
)
from .types import GimbalAction, GimbalObservation


@dataclass(frozen=True)
class IdentifiedMPCControllerConfig:
    """Dimensionless objective and identified-plant settings for C4.

    The servo values themselves are supplied by ``ServoConfig``. The three
    model scales let a deployment load identified values relative to that
    hardware configuration and support controlled plant-mismatch studies.
    """

    optimization_iterations: int = 14
    gradient_step_scale: float = 0.85
    tracking_weight: float = 1.0
    terminal_tracking_multiplier: float = 3.0
    rate_matching_weight: float = 0.10
    visibility_weight: float = 3.0
    visibility_margin_fraction: float = 0.85
    uncertainty_sigma: float = 0.5
    command_change_weight: float = 0.05
    command_effort_weight: float = 0.002
    rate_command_slew_scale: float = 1.0
    position_command_slew_scale: float = 2.0
    model_rate_time_constant_scale: float = 1.0
    model_command_latency_scale: float = 1.0
    model_position_gain_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.optimization_iterations <= 0:
            raise ValueError("MPC optimization iterations must be positive")
        for name in (
            "gradient_step_scale",
            "tracking_weight",
            "terminal_tracking_multiplier",
            "rate_matching_weight",
            "visibility_weight",
            "uncertainty_sigma",
            "command_change_weight",
            "command_effort_weight",
            "rate_command_slew_scale",
            "position_command_slew_scale",
            "model_rate_time_constant_scale",
            "model_command_latency_scale",
            "model_position_gain_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.gradient_step_scale <= 0.0:
            raise ValueError("MPC gradient step scale must be positive")
        if self.terminal_tracking_multiplier < 1.0:
            raise ValueError(
                "MPC terminal tracking multiplier must be at least one"
            )
        if not 0.0 < self.visibility_margin_fraction <= 1.0:
            raise ValueError("MPC visibility margin must be in (0, 1]")
        for name in (
            "rate_command_slew_scale",
            "position_command_slew_scale",
            "model_rate_time_constant_scale",
            "model_position_gain_scale",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class IdentifiedMPCDiagnostics:
    valid: bool
    horizon_s: float
    horizon_steps: int
    selected_command_normalized: float
    objective: float
    predicted_terminal_error_fov_fraction: float
    predicted_peak_error_fov_fraction: float
    predicted_terminal_rate_error_normalized: float
    predicted_visibility_excess: float
    optimization_iterations: int
    constraint_active: bool
    warm_started: bool
    modeled_command_latency_s: float
    modeled_rate_time_constant_s: float
    modeled_position_gain_s_inv: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass
class IdentifiedMPCGimbalController:
    """Receding-horizon DMC with a causal estimator and identified plant.

    The quadratic dynamic matrix is rebuilt from the configured plant at each
    update. Projected gradient iterations enforce command bounds and
    hardware-relative command slew constraints. Only the first command is
    issued; the remaining plan is a warm start for the next update.
    """

    estimator: MultiHorizonTargetStateEstimator
    servo: ServoConfig
    selected_axis_fov_rad: float
    command_mode: GimbalCommandMode
    nominal_control_period_s: float
    config: IdentifiedMPCControllerConfig = IdentifiedMPCControllerConfig()
    name: str = "identified_mpc"
    last_estimate: TargetStateEstimate = field(init=False, repr=False)
    last_diagnostics: IdentifiedMPCDiagnostics = field(
        init=False,
        repr=False,
    )
    _initial_applied_command: float | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_command: float | None = field(init=False, default=None, repr=False)
    _command_history: list[tuple[float, float]] = field(
        init=False,
        default_factory=list,
        repr=False,
    )
    _warm_plan: np.ndarray | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        horizons = self.estimator.prediction_horizons_s
        if len(horizons) < 2 or horizons[0] != 0.0:
            raise ValueError("MPC estimator horizons must begin at zero")
        if any(
            right <= left for left, right in zip(horizons, horizons[1:])
        ):
            raise ValueError("MPC estimator horizons must be increasing")
        steps = np.diff(np.asarray(horizons))
        if not np.allclose(
            steps,
            self.nominal_control_period_s,
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError("MPC horizons must match the control period")
        if not math.isfinite(self.selected_axis_fov_rad) or (
            self.selected_axis_fov_rad <= 0.0
        ):
            raise ValueError("MPC selected-axis FOV must be positive")
        if not math.isfinite(self.nominal_control_period_s) or (
            self.nominal_control_period_s <= 0.0
        ):
            raise ValueError("MPC control period must be positive")
        self.reset()

    @property
    def _horizon_steps(self) -> int:
        return len(self.estimator.prediction_horizons_s) - 1

    @property
    def _modeled_latency_s(self) -> float:
        return (
            self.config.model_command_latency_scale
            * self.servo.command_latency_s
        )

    @property
    def _modeled_time_constant_s(self) -> float:
        return (
            self.config.model_rate_time_constant_scale
            * self.servo.rate_time_constant_s
        )

    @property
    def _modeled_position_gain_s_inv(self) -> float:
        return (
            self.config.model_position_gain_scale
            * self.servo.position_gain_s_inv
        )

    def _missing_diagnostics(self) -> IdentifiedMPCDiagnostics:
        return IdentifiedMPCDiagnostics(
            valid=False,
            horizon_s=self.estimator.prediction_horizons_s[-1],
            horizon_steps=self._horizon_steps,
            selected_command_normalized=0.0,
            objective=0.0,
            predicted_terminal_error_fov_fraction=0.0,
            predicted_peak_error_fov_fraction=0.0,
            predicted_terminal_rate_error_normalized=0.0,
            predicted_visibility_excess=0.0,
            optimization_iterations=0,
            constraint_active=False,
            warm_started=False,
            modeled_command_latency_s=self._modeled_latency_s,
            modeled_rate_time_constant_s=self._modeled_time_constant_s,
            modeled_position_gain_s_inv=(
                self._modeled_position_gain_s_inv
            ),
        )

    def reset(self) -> None:
        self.estimator.reset()
        self.last_estimate = TargetStateEstimate.missing(0.0)
        self._initial_applied_command = None
        self._last_command = None
        self._command_history.clear()
        self._warm_plan = None
        self.last_diagnostics = self._missing_diagnostics()

    def _command_bounds(self) -> tuple[float, float, float]:
        if self.command_mode is GimbalCommandMode.RATE:
            return (
                -self.servo.max_rate_rad_s,
                self.servo.max_rate_rad_s,
                self.servo.max_rate_rad_s,
            )
        scale = max(
            abs(self.servo.min_angle_rad),
            abs(self.servo.max_angle_rad),
        )
        return self.servo.min_angle_rad, self.servo.max_angle_rad, scale

    def _maximum_command_step(self, dt_s: float) -> float:
        if self.command_mode is GimbalCommandMode.RATE:
            return (
                self.config.rate_command_slew_scale
                * self.servo.max_acceleration_rad_s2
                * dt_s
            )
        return (
            self.config.position_command_slew_scale
            * self.servo.max_rate_rad_s
            * dt_s
        )

    def _project_sequence(
        self,
        command: np.ndarray,
        previous_command: float,
        dt_s: float,
    ) -> np.ndarray:
        lower, upper, _scale = self._command_bounds()
        maximum_step = self._maximum_command_step(dt_s)
        projected = np.clip(command, lower, upper).copy()
        preceding = previous_command
        for index in range(len(projected)):
            projected[index] = np.clip(
                projected[index],
                preceding - maximum_step,
                preceding + maximum_step,
            )
            projected[index] = np.clip(projected[index], lower, upper)
            preceding = float(projected[index])
        return projected

    def _plant_matrices(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        time_constant_s = self._modeled_time_constant_s
        decay = (
            math.exp(-dt_s / time_constant_s)
            if time_constant_s > 0.0
            else 0.0
        )
        response = 1.0 - decay
        polarity = float(self.servo.command_polarity)
        if self.command_mode is GimbalCommandMode.RATE:
            gain = polarity
            state = np.array(
                [[1.0, dt_s * decay], [0.0, decay]],
                dtype=np.float64,
            )
            control = np.array(
                [dt_s * response * gain, response * gain],
                dtype=np.float64,
            )
            return state, control
        position_gain = self._modeled_position_gain_s_inv
        feedback = response * position_gain
        state = np.array(
            [
                [1.0 - dt_s * feedback, dt_s * decay],
                [-feedback, decay],
            ],
            dtype=np.float64,
        )
        control = np.array(
            [dt_s * feedback * polarity, feedback * polarity],
            dtype=np.float64,
        )
        return state, control

    def _applied_command_at(
        self,
        time_s: float,
        future_issues: tuple[tuple[float, float], ...],
    ) -> float:
        assert self._initial_applied_command is not None
        cutoff_s = time_s - self._modeled_latency_s
        command = self._initial_applied_command
        for issue_time_s, issued_command in (
            *self._command_history,
            *future_issues,
        ):
            if issue_time_s <= cutoff_s + 1e-12:
                command = issued_command
            else:
                break
        return command

    def _linear_prediction(
        self,
        command: np.ndarray,
        observation: GimbalObservation,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert observation.gimbal_angle_rad.valid
        assert observation.gimbal_rate_rad_s.valid
        state_matrix, control_vector = self._plant_matrices(dt_s)
        state = np.array(
            [
                observation.gimbal_angle_rad.value,
                observation.gimbal_rate_rad_s.value,
            ],
            dtype=np.float64,
        )
        future_issues = tuple(
            (observation.time_s + index * dt_s, float(value))
            for index, value in enumerate(command)
        )
        angle = np.empty(len(command), dtype=np.float64)
        rate = np.empty(len(command), dtype=np.float64)
        for step in range(len(command)):
            applied = self._applied_command_at(
                observation.time_s + (step + 1) * dt_s,
                future_issues,
            )
            state = state_matrix @ state + control_vector * applied
            angle[step] = state[0]
            rate[step] = state[1]
        return angle, rate

    def _dynamic_matrices(
        self,
        observation: GimbalObservation,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = self._horizon_steps
        zero = np.zeros(count, dtype=np.float64)
        free_angle, free_rate = self._linear_prediction(
            zero,
            observation,
            dt_s,
        )
        angle_response = np.empty((count, count), dtype=np.float64)
        rate_response = np.empty((count, count), dtype=np.float64)
        for index in range(count):
            impulse = zero.copy()
            impulse[index] = 1.0
            angle, rate = self._linear_prediction(
                impulse,
                observation,
                dt_s,
            )
            angle_response[:, index] = angle - free_angle
            rate_response[:, index] = rate - free_rate
        return free_angle, free_rate, angle_response, rate_response

    @staticmethod
    def _difference_matrix(count: int) -> np.ndarray:
        difference = np.eye(count, dtype=np.float64)
        if count > 1:
            difference[1:, :-1] -= np.eye(count - 1, dtype=np.float64)
        return difference

    def _initial_plan(
        self,
        previous_command: float,
        dt_s: float,
    ) -> tuple[np.ndarray, bool]:
        count = self._horizon_steps
        warm_started = self._warm_plan is not None and len(
            self._warm_plan
        ) == count
        if warm_started:
            assert self._warm_plan is not None
            command = np.concatenate(
                (self._warm_plan[1:], self._warm_plan[-1:])
            )
        else:
            command = np.full(count, previous_command, dtype=np.float64)
        return (
            self._project_sequence(command, previous_command, dt_s),
            warm_started,
        )

    def _objective_and_gradient(
        self,
        command: np.ndarray,
        *,
        target_angle: np.ndarray,
        target_rate: np.ndarray,
        target_std: np.ndarray,
        free_angle: np.ndarray,
        free_rate: np.ndarray,
        angle_response: np.ndarray,
        rate_response: np.ndarray,
        previous_command: float,
        command_scale: float,
        terminal_weights: np.ndarray,
    ) -> tuple[float, np.ndarray, dict[str, float]]:
        count = len(command)
        half_fov = 0.5 * self.selected_axis_fov_rad
        predicted_angle = free_angle + angle_response @ command
        predicted_rate = free_rate + rate_response @ command
        error = (predicted_angle - target_angle) / half_fov
        rate_error = (
            predicted_rate - target_rate
        ) / self.servo.max_rate_rad_s
        normalized_angle_response = angle_response / half_fov
        normalized_rate_response = (
            rate_response / self.servo.max_rate_rad_s
        )
        weighted_error = terminal_weights * error
        weighted_rate_error = terminal_weights * rate_error
        objective = self.config.tracking_weight * float(
            np.dot(error, weighted_error) / count
        )
        gradient = (
            2.0
            * self.config.tracking_weight
            / count
            * normalized_angle_response.T
            @ weighted_error
        )
        objective += self.config.rate_matching_weight * float(
            np.dot(rate_error, weighted_rate_error) / count
        )
        gradient += (
            2.0
            * self.config.rate_matching_weight
            / count
            * normalized_rate_response.T
            @ weighted_rate_error
        )

        difference_matrix = self._difference_matrix(count)
        previous = np.zeros(count, dtype=np.float64)
        previous[0] = previous_command
        command_difference = (
            difference_matrix @ command - previous
        ) / command_scale
        normalized_difference = difference_matrix / command_scale
        objective += self.config.command_change_weight * float(
            np.dot(command_difference, command_difference) / count
        )
        gradient += (
            2.0
            * self.config.command_change_weight
            / count
            * normalized_difference.T
            @ command_difference
        )
        normalized_command = command / command_scale
        objective += self.config.command_effort_weight * float(
            np.dot(normalized_command, normalized_command) / count
        )
        gradient += (
            2.0
            * self.config.command_effort_weight
            / count
            * command
            / command_scale**2
        )

        robust_fraction = (
            np.abs(error)
            + self.config.uncertainty_sigma * target_std / half_fov
        )
        visibility_excess = np.maximum(
            robust_fraction - self.config.visibility_margin_fraction,
            0.0,
        )
        objective += self.config.visibility_weight * float(
            np.dot(visibility_excess, visibility_excess) / count
        )
        gradient += (
            2.0
            * self.config.visibility_weight
            / count
            * normalized_angle_response.T
            @ (np.sign(error) * visibility_excess)
        )
        return objective, gradient, {
            "terminal_error": abs(float(error[-1])),
            "peak_error": float(np.max(np.abs(error))),
            "terminal_rate_error": abs(float(rate_error[-1])),
            "visibility_excess": float(np.max(visibility_excess)),
        }

    def _step_size(
        self,
        angle_response: np.ndarray,
        rate_response: np.ndarray,
        command_scale: float,
        terminal_weights: np.ndarray,
    ) -> float:
        count = angle_response.shape[0]
        angle = angle_response / (0.5 * self.selected_axis_fov_rad)
        rate = rate_response / self.servo.max_rate_rad_s
        difference = self._difference_matrix(count) / command_scale
        curvature = (
            self.config.tracking_weight
            * angle.T
            @ (terminal_weights[:, None] * angle)
            + self.config.rate_matching_weight
            * rate.T
            @ (terminal_weights[:, None] * rate)
            + self.config.command_change_weight
            * difference.T
            @ difference
            + self.config.command_effort_weight
            * np.eye(count)
            / command_scale**2
            + self.config.visibility_weight * angle.T @ angle
        ) / count
        largest = max(
            1e-12,
            float(np.max(np.linalg.eigvalsh(curvature))),
        )
        return self.config.gradient_step_scale / (2.0 * largest)

    def _normalized_command(self, command: float) -> float:
        if self.command_mode is GimbalCommandMode.RATE:
            return float(
                np.clip(command / self.servo.max_rate_rad_s, -1.0, 1.0)
            )
        clipped = float(
            np.clip(
                command,
                self.servo.min_angle_rad,
                self.servo.max_angle_rad,
            )
        )
        return self.servo.normalized_from_position(clipped)

    def _initialize_commands(self, observation: GimbalObservation) -> None:
        if self._initial_applied_command is not None:
            return
        if self.command_mode is GimbalCommandMode.RATE:
            initial = 0.0
        else:
            if not observation.gimbal_angle_rad.valid:
                raise ValueError("position MPC requires gimbal angle")
            initial = float(
                np.clip(
                    observation.gimbal_angle_rad.value
                    * self.servo.command_polarity,
                    self.servo.min_angle_rad,
                    self.servo.max_angle_rad,
                )
            )
        self._initial_applied_command = initial
        self._last_command = initial

    def _prune_history(self, time_s: float, dt_s: float) -> None:
        cutoff = time_s - self._modeled_latency_s - 2.0 * dt_s
        while (
            len(self._command_history) > 1
            and self._command_history[1][0] <= cutoff
        ):
            self._initial_applied_command = self._command_history.pop(0)[1]

    def act(self, observation: GimbalObservation) -> GimbalAction:
        if observation.command_mode is not self.command_mode:
            raise ValueError("observation command mode does not match MPC")
        if not observation.gimbal_angle_rad.valid:
            raise ValueError("MPC requires servo angle feedback")
        if not observation.gimbal_rate_rad_s.valid:
            raise ValueError("MPC requires servo rate feedback")
        self._initialize_commands(observation)
        assert self._last_command is not None
        estimates = self.estimator.update_all(observation)
        self.last_estimate = estimates[0]
        dt_s = (
            observation.control_dt_s
            if observation.control_dt_s > 0.0
            else self.nominal_control_period_s
        )
        self._prune_history(observation.time_s, dt_s)
        if not estimates or not all(estimate.valid for estimate in estimates):
            command = self._normalized_command(self._last_command)
            self.last_diagnostics = self._missing_diagnostics()
            if self.command_mode is GimbalCommandMode.RATE:
                return GimbalAction.rate(command)
            return GimbalAction.position(command)

        target_angle = np.unwrap(
            np.asarray(
                [
                    observation.gimbal_angle_rad.value,
                    *(
                        estimate.body_relative_bearing_rad.value
                        for estimate in estimates[1:]
                    ),
                ],
                dtype=np.float64,
            )
        )[1:]
        target_rate = np.asarray(
            [
                estimate.body_relative_rate_rad_s.value
                for estimate in estimates[1:]
            ],
            dtype=np.float64,
        )
        target_std = np.asarray(
            [estimate.bearing_std_rad.value for estimate in estimates[1:]],
            dtype=np.float64,
        )
        (
            free_angle,
            free_rate,
            angle_response,
            rate_response,
        ) = self._dynamic_matrices(observation, dt_s)
        lower, upper, command_scale = self._command_bounds()
        del lower, upper
        terminal_weights = np.ones(self._horizon_steps, dtype=np.float64)
        terminal_weights[-1] = self.config.terminal_tracking_multiplier
        planned, warm_started = self._initial_plan(self._last_command, dt_s)
        step_size = self._step_size(
            angle_response,
            rate_response,
            command_scale,
            terminal_weights,
        )
        constraint_active = False
        for _iteration in range(self.config.optimization_iterations):
            _objective, gradient, _prediction = self._objective_and_gradient(
                planned,
                target_angle=target_angle,
                target_rate=target_rate,
                target_std=target_std,
                free_angle=free_angle,
                free_rate=free_rate,
                angle_response=angle_response,
                rate_response=rate_response,
                previous_command=self._last_command,
                command_scale=command_scale,
                terminal_weights=terminal_weights,
            )
            proposal = planned - step_size * gradient
            projected = self._project_sequence(
                proposal,
                self._last_command,
                dt_s,
            )
            constraint_active = constraint_active or not np.allclose(
                proposal,
                projected,
                rtol=0.0,
                atol=1e-10,
            )
            planned = projected
        objective, _gradient, prediction = self._objective_and_gradient(
            planned,
            target_angle=target_angle,
            target_rate=target_rate,
            target_std=target_std,
            free_angle=free_angle,
            free_rate=free_rate,
            angle_response=angle_response,
            rate_response=rate_response,
            previous_command=self._last_command,
            command_scale=command_scale,
            terminal_weights=terminal_weights,
        )
        self._warm_plan = planned.copy()
        self._last_command = float(planned[0])
        self._command_history.append(
            (observation.time_s, self._last_command)
        )
        normalized = self._normalized_command(self._last_command)
        self.last_diagnostics = IdentifiedMPCDiagnostics(
            valid=True,
            horizon_s=self.estimator.prediction_horizons_s[-1],
            horizon_steps=self._horizon_steps,
            selected_command_normalized=normalized,
            objective=objective,
            predicted_terminal_error_fov_fraction=prediction[
                "terminal_error"
            ],
            predicted_peak_error_fov_fraction=prediction["peak_error"],
            predicted_terminal_rate_error_normalized=prediction[
                "terminal_rate_error"
            ],
            predicted_visibility_excess=prediction["visibility_excess"],
            optimization_iterations=self.config.optimization_iterations,
            constraint_active=constraint_active,
            warm_started=warm_started,
            modeled_command_latency_s=self._modeled_latency_s,
            modeled_rate_time_constant_s=self._modeled_time_constant_s,
            modeled_position_gain_s_inv=(
                self._modeled_position_gain_s_inv
            ),
        )
        if self.command_mode is GimbalCommandMode.RATE:
            return GimbalAction.rate(normalized)
        return GimbalAction.position(normalized)
