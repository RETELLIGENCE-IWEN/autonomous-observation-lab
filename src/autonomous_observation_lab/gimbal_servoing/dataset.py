"""Deterministic target-state dataset generation for predictive servoing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .closed_loop import ClosedLoopScenario, closed_loop_scenarios, tracking_metrics
from .config import GimbalCommandMode, GimbalServoingConfig, ObservationProfile
from .controllers import (
    ProportionalController,
    ProportionalPositionController,
    TargetStatePositionController,
    TargetStateRateController,
)
from .env import GimbalServoEnv
from .estimators import (
    ConstantVelocityEstimatorConfig,
    ConstantVelocityTargetEstimator,
)
from .oracles import (
    OracleControlConfig,
    PrivilegedOracleController,
    PrivilegedTargetStateOracle,
    rollout_privileged_oracle,
)
from .randomization import (
    GimbalDomainRandomizationConfig,
    randomize_closed_loop_scenario,
)
from .types import GimbalAction, GimbalObservation, MaskedScalar


SCHEMA_VERSION = "gimbal_target_state_v2"
SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {"gimbal_target_state_v1", SCHEMA_VERSION}
)
VALID_SPLITS = frozenset(
    {"train", "validation", "development", "test", "shift_test"}
)
BEHAVIOR_NAMES = (
    "proportional_rate",
    "proportional_position",
    "predictive_rate",
    "predictive_position",
    "privileged_oracle_rate",
    "privileged_oracle_position",
)
LEGACY_FEATURE_NAMES = (
    "control_dt_s",
    "frame_updated",
    "measurement_age_s",
    "measurement_age_valid",
    "image_error_normalized",
    "image_error_valid",
    "bbox_width_fraction",
    "bbox_width_valid",
    "bbox_height_fraction",
    "bbox_height_valid",
    "confidence",
    "confidence_valid",
    "gimbal_position_normalized",
    "gimbal_position_valid",
    "gimbal_rate_normalized",
    "gimbal_rate_valid",
    "body_rate_normalized",
    "body_rate_valid",
    "previous_action_normalized",
    "command_mode_rate",
    "command_mode_position",
)
FEATURE_NAMES = (
    "control_dt_s",
    "frame_updated",
    "measurement_age_s",
    "measurement_age_valid",
    "image_error_normalized",
    "image_error_rad",
    "image_error_valid",
    "bbox_width_fraction",
    "bbox_width_valid",
    "bbox_height_fraction",
    "bbox_height_valid",
    "confidence",
    "confidence_valid",
    "gimbal_position_normalized",
    "gimbal_angle_rad",
    "gimbal_position_valid",
    "gimbal_rate_normalized",
    "gimbal_rate_rad_s",
    "gimbal_rate_valid",
    "body_rate_normalized",
    "body_rate_rad_s",
    "body_rate_valid",
    "previous_action_normalized",
    "previous_rate_command_rad_s",
    "previous_position_command_rad",
    "command_mode_rate",
    "command_mode_position",
)
ACTION_NAMES = (
    "desired_rate_normalized",
    "desired_position_normalized",
    "rate_command_valid",
    "position_command_valid",
)
TARGET_NAMES = (
    "body_relative_bearing_rad",
    "body_relative_rate_rad_s",
)
ORACLE_ACTION_NAMES = (
    "desired_rate_normalized",
    "desired_position_normalized",
)


class _DeployableController(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: GimbalObservation) -> GimbalAction: ...


@dataclass(frozen=True)
class GimbalDatasetGenerationConfig:
    """Full deterministic request used to construct a dataset manifest."""

    split: str = "train"
    seeds: tuple[int, ...] = (1000,)
    scenario_names: tuple[str, ...] = (
        "nominal_combined",
        "high_latency",
        "dropout_noise",
        "slow_servo",
        "aggressive_motion",
        "travel_limit_recovery",
    )
    behavior_names: tuple[str, ...] = (
        "proportional_rate",
        "predictive_rate",
        "privileged_oracle_rate",
    )
    observation_profiles: tuple[ObservationProfile, ...] = (
        ObservationProfile.VISION_ONLY,
        ObservationProfile.SERVO_AWARE,
        ObservationProfile.DISTURBANCE_AWARE,
    )
    prediction_horizons_s: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    oracle_control: OracleControlConfig = OracleControlConfig()
    domain_randomization: GimbalDomainRandomizationConfig | None = None
    include_oracle_ceilings: bool = True

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in self.seeds
        ):
            raise ValueError("seeds must contain integers")
        for name, values in (
            ("scenario_names", self.scenario_names),
            ("behavior_names", self.behavior_names),
            ("observation_profiles", self.observation_profiles),
            ("prediction_horizons_s", self.prediction_horizons_s),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be non-empty and unique")
        unknown = set(self.behavior_names) - set(BEHAVIOR_NAMES)
        if unknown:
            raise ValueError(f"unknown behavior names: {sorted(unknown)}")
        if any(
            not math.isfinite(horizon) or horizon < 0.0
            for horizon in self.prediction_horizons_s
        ):
            raise ValueError("prediction horizons must be non-negative")
        if tuple(sorted(self.prediction_horizons_s)) != self.prediction_horizons_s:
            raise ValueError("prediction horizons must be in ascending order")
        if any(
            not isinstance(profile, ObservationProfile)
            for profile in self.observation_profiles
        ):
            raise ValueError("observation profiles must use ObservationProfile")
        if self.domain_randomization is not None and not isinstance(
            self.domain_randomization, GimbalDomainRandomizationConfig
        ):
            raise ValueError(
                "domain_randomization must use GimbalDomainRandomizationConfig"
            )


@dataclass(frozen=True)
class OracleCeilingRecord:
    scenario_name: str
    command_mode: str
    seed: int
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class GimbalDatasetManifest:
    schema_version: str
    split: str
    configuration_hash: str
    seeds: tuple[int, ...]
    scenario_names: tuple[str, ...]
    behavior_names: tuple[str, ...]
    observation_profiles: tuple[str, ...]
    prediction_horizons_s: tuple[float, ...]
    feature_names: tuple[str, ...]
    action_names: tuple[str, ...]
    target_names: tuple[str, ...]
    oracle_action_names: tuple[str, ...]
    generation: dict[str, Any]
    array_shapes: dict[str, tuple[int, ...]]
    array_dtypes: dict[str, str]
    oracle_ceilings: tuple[OracleCeilingRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return _canonical(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GimbalDatasetManifest":
        ceilings = tuple(
            OracleCeilingRecord(
                scenario_name=item["scenario_name"],
                command_mode=item["command_mode"],
                seed=int(item["seed"]),
                metrics={
                    key: metric for key, metric in item["metrics"].items()
                },
            )
            for item in value["oracle_ceilings"]
        )
        return cls(
            schema_version=value["schema_version"],
            split=value["split"],
            configuration_hash=value["configuration_hash"],
            seeds=tuple(int(seed) for seed in value["seeds"]),
            scenario_names=tuple(value["scenario_names"]),
            behavior_names=tuple(value["behavior_names"]),
            observation_profiles=tuple(value["observation_profiles"]),
            prediction_horizons_s=tuple(
                float(horizon) for horizon in value["prediction_horizons_s"]
            ),
            feature_names=tuple(value["feature_names"]),
            action_names=tuple(value["action_names"]),
            target_names=tuple(value["target_names"]),
            oracle_action_names=tuple(value["oracle_action_names"]),
            generation=value["generation"],
            array_shapes={
                key: tuple(int(size) for size in shape)
                for key, shape in value["array_shapes"].items()
            },
            array_dtypes={
                key: str(dtype) for key, dtype in value["array_dtypes"].items()
            },
            oracle_ceilings=ceilings,
        )


@dataclass(frozen=True)
class GimbalTargetStateDataset:
    """Padded episodes with separate deployable and privileged tensors."""

    manifest: GimbalDatasetManifest
    features: np.ndarray
    sequence_mask: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    oracle_actions: np.ndarray
    time_s: np.ndarray
    episode_seed: np.ndarray
    scenario_index: np.ndarray
    behavior_index: np.ndarray

    def __post_init__(self) -> None:
        _validate_dataset(self)

    @property
    def episode_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def profile_count(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True)
class _EpisodeArrays:
    features: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    oracle_actions: np.ndarray
    time_s: np.ndarray
    seed: int
    scenario_index: int
    behavior_index: int


def _masked(value: MaskedScalar, available: bool = True) -> tuple[float, float]:
    valid = available and value.valid
    return (float(value.value) if valid else 0.0, float(valid))


def encode_deployable_observation(
    observation: GimbalObservation,
    *,
    profile: ObservationProfile,
    config: GimbalServoingConfig,
) -> np.ndarray:
    """Encode actor-visible values only, with explicit masks for every option."""
    age = _masked(observation.measurement_age_s)
    image_error = _masked(observation.image_error_normalized)
    width = _masked(observation.bbox_width_fraction)
    height = _masked(observation.bbox_height_fraction)
    confidence = _masked(observation.confidence)

    servo_available = profile in {
        ObservationProfile.SERVO_AWARE,
        ObservationProfile.DISTURBANCE_AWARE,
    }
    image_error_rad = (
        image_error[0] * 0.5 * config.camera.selected_axis_fov_rad
        if image_error[1]
        else 0.0
    )
    angle_rad, angle_valid = _masked(
        observation.gimbal_angle_rad, servo_available
    )
    angle = angle_rad
    if angle_valid:
        angle = config.servo.normalized_from_position(angle_rad)
    rate_rad_s, rate_valid = _masked(
        observation.gimbal_rate_rad_s, servo_available
    )
    rate = rate_rad_s
    if rate_valid:
        rate /= config.servo.max_rate_rad_s
    body_rate_rad_s, body_rate_valid = _masked(
        observation.body_rate_rad_s,
        profile is ObservationProfile.DISTURBANCE_AWARE,
    )
    body_rate = body_rate_rad_s
    if body_rate_valid:
        body_rate /= config.servo.max_rate_rad_s

    rate_mode = observation.command_mode is GimbalCommandMode.RATE
    previous_rate_command_rad_s = (
        observation.previous_action_normalized * config.servo.max_rate_rad_s
        if rate_mode
        else 0.0
    )
    previous_position_command_rad = (
        0.0
        if rate_mode
        else config.servo.position_from_normalized(
            observation.previous_action_normalized
        )
    )
    vector = np.asarray(
        (
            observation.control_dt_s,
            float(observation.frame_updated),
            *age,
            image_error[0],
            image_error_rad,
            image_error[1],
            *width,
            *height,
            *confidence,
            angle,
            angle_rad,
            angle_valid,
            rate,
            rate_rad_s,
            rate_valid,
            body_rate,
            body_rate_rad_s,
            body_rate_valid,
            observation.previous_action_normalized,
            previous_rate_command_rad_s,
            previous_position_command_rad,
            float(rate_mode),
            float(not rate_mode),
        ),
        dtype=np.float32,
    )
    if vector.shape != (len(FEATURE_NAMES),):
        raise AssertionError("feature schema and encoder disagree")
    return vector


def encode_action(action: GimbalAction) -> np.ndarray:
    if action.mode is GimbalCommandMode.RATE:
        values = (action.command_normalized, 0.0, 1.0, 0.0)
    else:
        values = (0.0, action.command_normalized, 0.0, 1.0)
    return np.asarray(values, dtype=np.float32)


def _target_estimator(
    config: GimbalServoingConfig,
) -> ConstantVelocityTargetEstimator:
    minimum_horizon_s = (
        config.camera.detection_latency_s
        + config.camera.detection_latency_jitter_s
        + 2.0 * config.camera.frame_period_s
    )
    prediction_horizon_s = max(0.30, minimum_horizon_s)
    return ConstantVelocityTargetEstimator(
        ConstantVelocityEstimatorConfig(
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            center_noise_std_normalized=(
                config.camera.center_noise_std_normalized
            ),
            velocity_filter_coefficient=0.40,
            uncertainty_filter_coefficient=0.20,
            max_prediction_horizon_s=prediction_horizon_s,
            history_horizon_s=max(1.0, prediction_horizon_s + 0.50),
        )
    )


def _behavior_mode(name: str) -> GimbalCommandMode:
    return (
        GimbalCommandMode.POSITION
        if name.endswith("_position")
        else GimbalCommandMode.RATE
    )


def _deployable_behavior(
    name: str, config: GimbalServoingConfig
) -> _DeployableController | None:
    if name.startswith("privileged_oracle_"):
        return None
    if name == "proportional_rate":
        return ProportionalController(gain=1.35)
    if name == "proportional_position":
        return ProportionalPositionController(
            servo=config.servo,
            selected_axis_fov_rad=config.camera.selected_axis_fov_rad,
            gain=0.85,
        )
    if name == "predictive_rate":
        return TargetStateRateController(
            estimator=_target_estimator(config),
            max_rate_rad_s=config.servo.max_rate_rad_s,
            proportional_gain_s_inv=2.5,
            name=name,
        )
    if name == "predictive_position":
        return TargetStatePositionController(
            estimator=_target_estimator(config),
            servo=config.servo,
            command_preview_s=(
                config.servo.command_latency_s
                + config.servo.rate_time_constant_s
            ),
            name=name,
        )
    raise ValueError(f"unknown behavior: {name}")


def _collect_episode(
    *,
    scenario: ClosedLoopScenario,
    scenario_index: int,
    behavior_name: str,
    behavior_index: int,
    profiles: tuple[ObservationProfile, ...],
    horizons_s: tuple[float, ...],
    oracle_control: OracleControlConfig,
    seed: int,
) -> _EpisodeArrays:
    command_mode = _behavior_mode(behavior_name)
    config = replace(
        scenario.config,
        command_mode=command_mode,
        observation_profile=ObservationProfile.DISTURBANCE_AWARE,
    )
    env = GimbalServoEnv(
        config,
        target_motion=scenario.target_motion,
        body_motion=scenario.body_motion,
    )
    oracle = PrivilegedTargetStateOracle(
        target_motion=scenario.target_motion,
        body_motion=scenario.body_motion,
        servo=config.servo,
        control=oracle_control,
    )
    privileged_controller = PrivilegedOracleController(oracle, command_mode)
    deployable_controller = _deployable_behavior(behavior_name, config)
    if deployable_controller is not None:
        deployable_controller.reset()

    observation, diagnostics = env.reset(seed)
    feature_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    oracle_action_rows: list[np.ndarray] = []
    times: list[float] = []
    end_time_s = config.timing.episode_duration_s

    while True:
        feature_rows.append(
            np.stack(
                [
                    encode_deployable_observation(
                        observation, profile=profile, config=config
                    )
                    for profile in profiles
                ]
            )
        )
        oracle_actions = oracle.action_targets(diagnostics)
        oracle_action_rows.append(
            np.asarray(
                (
                    oracle_actions.desired_rate_normalized,
                    oracle_actions.desired_position_normalized,
                ),
                dtype=np.float32,
            )
        )

        states = []
        valid_horizons = []
        for horizon_s in horizons_s:
            label_time_s = observation.time_s + horizon_s
            valid = label_time_s <= end_time_s + 1e-12
            valid_horizons.append(valid)
            if valid:
                state = (
                    oracle.state_from_diagnostics(diagnostics)
                    if horizon_s == 0.0
                    else oracle.state_at(label_time_s)
                )
                states.append(
                    (
                        state.body_relative_bearing_rad,
                        state.body_relative_rate_rad_s,
                    )
                )
            else:
                states.append((0.0, 0.0))
        target_rows.append(np.asarray(states, dtype=np.float32))
        target_masks.append(np.asarray(valid_horizons, dtype=np.bool_))
        times.append(observation.time_s)

        action = (
            privileged_controller.act(diagnostics)
            if deployable_controller is None
            else deployable_controller.act(observation)
        )
        action_rows.append(encode_action(action))
        result = env.step(action)
        if result.truncated:
            break
        observation = result.observation
        diagnostics = result.diagnostics

    return _EpisodeArrays(
        features=np.stack(feature_rows),
        actions=np.stack(action_rows),
        targets=np.stack(target_rows),
        target_mask=np.stack(target_masks),
        oracle_actions=np.stack(oracle_action_rows),
        time_s=np.asarray(times, dtype=np.float64),
        seed=seed,
        scenario_index=scenario_index,
        behavior_index=behavior_index,
    )


def evaluate_privileged_oracle_ceilings(
    scenarios: Sequence[ClosedLoopScenario],
    *,
    seed: int,
    oracle_control: OracleControlConfig | None = None,
) -> tuple[OracleCeilingRecord, ...]:
    """Evaluate ideal state feedback through each scenario's non-ideal plant."""
    records = []
    for scenario in scenarios:
        for mode in GimbalCommandMode:
            episode = rollout_privileged_oracle(
                config=scenario.config,
                target_motion=scenario.target_motion,
                body_motion=scenario.body_motion,
                command_mode=mode,
                seed=seed,
                oracle_control=oracle_control,
                name=f"{scenario.name}_oracle_{mode.name.lower()}",
            )
            metrics = tracking_metrics(episode)
            records.append(
                OracleCeilingRecord(
                    scenario_name=scenario.name,
                    command_mode=mode.value,
                    seed=seed,
                    metrics={
                        field.name: getattr(metrics, field.name)
                        for field in fields(metrics)
                    },
                )
            )
    return tuple(records)


def generate_gimbal_dataset(
    request: GimbalDatasetGenerationConfig,
    *,
    scenarios: Sequence[ClosedLoopScenario] | None = None,
) -> GimbalTargetStateDataset:
    """Generate a deterministic Cartesian product of seeds/scenarios/behaviors."""
    available = {
        scenario.name: scenario
        for scenario in (scenarios or closed_loop_scenarios())
    }
    missing = set(request.scenario_names) - set(available)
    if missing:
        raise ValueError(f"unknown scenario names: {sorted(missing)}")
    selected = tuple(available[name] for name in request.scenario_names)

    episodes = []
    scenario_variants = []
    first_seed_variants = []
    for seed in request.seeds:
        for scenario_index, scenario in enumerate(selected):
            episode_scenario = (
                randomize_closed_loop_scenario(
                    scenario,
                    seed=seed,
                    config=request.domain_randomization,
                )
                if request.domain_randomization is not None
                else scenario
            )
            if request.domain_randomization is not None:
                scenario_variants.append(
                    {
                        "seed": seed,
                        "scenario_index": scenario_index,
                        "scenario": _canonical(episode_scenario),
                    }
                )
                if seed == request.seeds[0]:
                    first_seed_variants.append(episode_scenario)
            for behavior_index, behavior_name in enumerate(
                request.behavior_names
            ):
                episodes.append(
                    _collect_episode(
                        scenario=episode_scenario,
                        scenario_index=scenario_index,
                        behavior_name=behavior_name,
                        behavior_index=behavior_index,
                        profiles=request.observation_profiles,
                        horizons_s=request.prediction_horizons_s,
                        oracle_control=request.oracle_control,
                        seed=seed,
                    )
                )

    episode_count = len(episodes)
    profile_count = len(request.observation_profiles)
    max_steps = max(episode.features.shape[0] for episode in episodes)
    feature_count = len(FEATURE_NAMES)
    horizon_count = len(request.prediction_horizons_s)
    features_array = np.zeros(
        (episode_count, profile_count, max_steps, feature_count),
        dtype=np.float32,
    )
    sequence_mask = np.zeros((episode_count, max_steps), dtype=np.bool_)
    actions = np.zeros(
        (episode_count, max_steps, len(ACTION_NAMES)), dtype=np.float32
    )
    targets = np.zeros(
        (episode_count, max_steps, horizon_count, len(TARGET_NAMES)),
        dtype=np.float32,
    )
    target_mask = np.zeros(
        (episode_count, max_steps, horizon_count), dtype=np.bool_
    )
    oracle_actions = np.zeros(
        (episode_count, max_steps, len(ORACLE_ACTION_NAMES)),
        dtype=np.float32,
    )
    time_s = np.zeros((episode_count, max_steps), dtype=np.float64)
    episode_seed = np.zeros(episode_count, dtype=np.int64)
    scenario_index = np.zeros(episode_count, dtype=np.int32)
    behavior_index = np.zeros(episode_count, dtype=np.int32)

    for index, episode in enumerate(episodes):
        length = episode.features.shape[0]
        features_array[index, :, :length] = np.transpose(
            episode.features, (1, 0, 2)
        )
        sequence_mask[index, :length] = True
        actions[index, :length] = episode.actions
        targets[index, :length] = episode.targets
        target_mask[index, :length] = episode.target_mask
        oracle_actions[index, :length] = episode.oracle_actions
        time_s[index, :length] = episode.time_s
        episode_seed[index] = episode.seed
        scenario_index[index] = episode.scenario_index
        behavior_index[index] = episode.behavior_index

    arrays = {
        "features": features_array,
        "sequence_mask": sequence_mask,
        "actions": actions,
        "targets": targets,
        "target_mask": target_mask,
        "oracle_actions": oracle_actions,
        "time_s": time_s,
        "episode_seed": episode_seed,
        "scenario_index": scenario_index,
        "behavior_index": behavior_index,
    }
    generation = {
        "request": _canonical(request),
        "collector_observation_profile": (
            ObservationProfile.DISTURBANCE_AWARE.value
        ),
        "scenarios": [_canonical(scenario) for scenario in selected],
    }
    if scenario_variants:
        generation["scenario_variants"] = scenario_variants
    configuration_hash = hashlib.sha256(
        json.dumps(
            generation, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    ceilings = (
        evaluate_privileged_oracle_ceilings(
            tuple(first_seed_variants) if first_seed_variants else selected,
            seed=request.seeds[0],
            oracle_control=request.oracle_control,
        )
        if request.include_oracle_ceilings
        else ()
    )
    manifest = GimbalDatasetManifest(
        schema_version=SCHEMA_VERSION,
        split=request.split,
        configuration_hash=configuration_hash,
        seeds=request.seeds,
        scenario_names=request.scenario_names,
        behavior_names=request.behavior_names,
        observation_profiles=tuple(
            profile.value for profile in request.observation_profiles
        ),
        prediction_horizons_s=request.prediction_horizons_s,
        feature_names=FEATURE_NAMES,
        action_names=ACTION_NAMES,
        target_names=TARGET_NAMES,
        oracle_action_names=ORACLE_ACTION_NAMES,
        generation=generation,
        array_shapes={key: value.shape for key, value in arrays.items()},
        array_dtypes={key: str(value.dtype) for key, value in arrays.items()},
        oracle_ceilings=ceilings,
    )
    return GimbalTargetStateDataset(manifest=manifest, **arrays)


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            **{
                field.name: _canonical(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _dataset_arrays(dataset: GimbalTargetStateDataset) -> dict[str, np.ndarray]:
    return {
        name: getattr(dataset, name)
        for name in (
            "features",
            "sequence_mask",
            "actions",
            "targets",
            "target_mask",
            "oracle_actions",
            "time_s",
            "episode_seed",
            "scenario_index",
            "behavior_index",
        )
    }


def _validate_dataset(dataset: GimbalTargetStateDataset) -> None:
    arrays = _dataset_arrays(dataset)
    expected_shapes = dataset.manifest.array_shapes
    expected_dtypes = dataset.manifest.array_dtypes
    if set(arrays) != set(expected_shapes) or set(arrays) != set(expected_dtypes):
        raise ValueError("manifest array names do not match the dataset")
    for name, array in arrays.items():
        if tuple(array.shape) != tuple(expected_shapes[name]):
            raise ValueError(f"array shape mismatch for {name}")
        if str(array.dtype) != expected_dtypes[name]:
            raise ValueError(f"array dtype mismatch for {name}")

    episode_count, profile_count, max_steps, feature_count = (
        dataset.features.shape
    )
    expected_feature_names = (
        LEGACY_FEATURE_NAMES
        if dataset.manifest.schema_version == "gimbal_target_state_v1"
        else FEATURE_NAMES
    )
    if dataset.manifest.feature_names != expected_feature_names:
        raise ValueError("unsupported feature schema")
    if dataset.manifest.action_names != ACTION_NAMES:
        raise ValueError("unsupported action schema")
    if dataset.manifest.target_names != TARGET_NAMES:
        raise ValueError("unsupported target schema")
    if dataset.manifest.oracle_action_names != ORACLE_ACTION_NAMES:
        raise ValueError("unsupported oracle action schema")
    if feature_count != len(dataset.manifest.feature_names):
        raise ValueError("feature dimension does not match manifest")
    if profile_count != len(dataset.manifest.observation_profiles):
        raise ValueError("profile dimension does not match manifest")
    if dataset.sequence_mask.shape != (episode_count, max_steps):
        raise ValueError("sequence mask has an invalid shape")
    if dataset.actions.shape != (episode_count, max_steps, len(ACTION_NAMES)):
        raise ValueError("action dimensions are invalid")
    if dataset.targets.shape[:2] != (episode_count, max_steps):
        raise ValueError("target leading dimensions are invalid")
    if dataset.targets.shape[2] != len(
        dataset.manifest.prediction_horizons_s
    ):
        raise ValueError("target horizon dimension does not match manifest")
    if dataset.targets.shape[-1] != len(TARGET_NAMES):
        raise ValueError("target state dimension does not match manifest")
    if dataset.target_mask.shape != dataset.targets.shape[:-1]:
        raise ValueError("target mask dimensions are invalid")
    if dataset.oracle_actions.shape != (
        episode_count,
        max_steps,
        len(ORACLE_ACTION_NAMES),
    ):
        raise ValueError("oracle action dimensions are invalid")
    if dataset.time_s.shape != (episode_count, max_steps):
        raise ValueError("time dimensions are invalid")
    for name in ("episode_seed", "scenario_index", "behavior_index"):
        if getattr(dataset, name).shape != (episode_count,):
            raise ValueError(f"{name} dimensions are invalid")
    if np.any(np.diff(dataset.sequence_mask.astype(np.int8), axis=1) > 0):
        raise ValueError("sequence masks must be contiguous")
    if np.any(~np.any(dataset.sequence_mask, axis=1)):
        raise ValueError("every episode must contain at least one step")
    if np.any(dataset.target_mask & ~dataset.sequence_mask[:, :, None]):
        raise ValueError("target mask marks a padded step as valid")
    if not set(dataset.episode_seed.tolist()) <= set(dataset.manifest.seeds):
        raise ValueError("episode seed is absent from the manifest")
    if np.any(dataset.scenario_index < 0) or np.any(
        dataset.scenario_index >= len(dataset.manifest.scenario_names)
    ):
        raise ValueError("scenario index is outside the manifest")
    if np.any(dataset.behavior_index < 0) or np.any(
        dataset.behavior_index >= len(dataset.manifest.behavior_names)
    ):
        raise ValueError("behavior index is outside the manifest")
    expected_configuration_hash = hashlib.sha256(
        json.dumps(
            dataset.manifest.generation,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if dataset.manifest.configuration_hash != expected_configuration_hash:
        raise ValueError("configuration hash does not match the manifest")
    if dataset.manifest.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported dataset schema version")


def _dataset_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.suffix == "":
        return resolved.with_suffix(".npz")
    if resolved.suffix != ".npz":
        raise ValueError("dataset path must have an .npz suffix")
    return resolved


def save_gimbal_dataset(
    path: str | Path, dataset: GimbalTargetStateDataset
) -> tuple[Path, Path]:
    dataset_path = _dataset_path(path)
    manifest_path = dataset_path.with_suffix(".json")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **_dataset_arrays(dataset))
    manifest_path.write_text(
        json.dumps(dataset.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_path, manifest_path


def load_gimbal_dataset(path: str | Path) -> GimbalTargetStateDataset:
    dataset_path = _dataset_path(path)
    manifest_path = dataset_path.with_suffix(".json")
    manifest = GimbalDatasetManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return GimbalTargetStateDataset(manifest=manifest, **arrays)


def validate_disjoint_seed_blocks(
    manifests: Sequence[GimbalDatasetManifest],
) -> None:
    """Reject train/validation/test manifests that reuse simulator seeds."""
    for left_index, left in enumerate(manifests):
        for right in manifests[left_index + 1 :]:
            overlap = set(left.seeds) & set(right.seeds)
            if overlap:
                raise ValueError(
                    f"splits {left.split!r} and {right.split!r} reuse seeds: "
                    f"{sorted(overlap)}"
                )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic privileged gimbal target-state data."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="train")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument(
        "--behavior", action="append", choices=BEHAVIOR_NAMES, dest="behaviors"
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.value for profile in ObservationProfile],
        dest="profiles",
    )
    parser.add_argument("--no-oracle-ceilings", action="store_true")
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="randomize motion, camera, servo, and timing from each split seed",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    defaults = GimbalDatasetGenerationConfig()
    request = GimbalDatasetGenerationConfig(
        split=args.split,
        seeds=tuple(range(args.seed_start, args.seed_start + args.episodes)),
        scenario_names=tuple(args.scenarios or defaults.scenario_names),
        behavior_names=tuple(args.behaviors or defaults.behavior_names),
        observation_profiles=tuple(
            ObservationProfile(profile)
            for profile in (args.profiles or defaults.observation_profiles)
        ),
        prediction_horizons_s=defaults.prediction_horizons_s,
        oracle_control=defaults.oracle_control,
        domain_randomization=(
            GimbalDomainRandomizationConfig()
            if args.domain_randomization
            else None
        ),
        include_oracle_ceilings=not args.no_oracle_ceilings,
    )
    dataset = generate_gimbal_dataset(request)
    dataset_path, manifest_path = save_gimbal_dataset(args.output, dataset)
    print(
        f"wrote {dataset.episode_count} episodes to {dataset_path}\n"
        f"manifest: {manifest_path}\n"
        f"configuration hash: {dataset.manifest.configuration_hash}"
    )


if __name__ == "__main__":
    main()
