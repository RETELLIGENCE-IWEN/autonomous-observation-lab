"""Strict reconstruction of exact gimbal scenarios stored in manifests."""

from __future__ import annotations

from typing import Any

from .closed_loop import ClosedLoopScenario
from .config import (
    CameraConfig,
    GimbalCommandMode,
    GimbalServoingConfig,
    ObjectiveConfig,
    ObservationProfile,
    ScenarioConfig,
    ServoConfig,
    TimingConfig,
)
from .disturbances import (
    AngularMotion,
    ConstantRateAngularMotion,
    RatePulseAngularMotion,
    SampledAngularMotion,
    SinusoidalAngularMotion,
    StaticAngularMotion,
    SumAngularMotion,
)


def _typed_payload(value: Any, expected_type: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != expected_type:
        raise ValueError(f"expected canonical {expected_type} payload")
    return {key: item for key, item in value.items() if key != "type"}


def angular_motion_from_dict(value: Any) -> AngularMotion:
    """Reconstruct one supported angular-motion tree from canonical JSON."""
    if not isinstance(value, dict):
        raise ValueError("angular motion payload must be an object")
    motion_type = value.get("type")
    payload = {key: item for key, item in value.items() if key != "type"}
    if motion_type == "StaticAngularMotion":
        return StaticAngularMotion(**payload)
    if motion_type == "ConstantRateAngularMotion":
        return ConstantRateAngularMotion(**payload)
    if motion_type == "SinusoidalAngularMotion":
        return SinusoidalAngularMotion(**payload)
    if motion_type == "RatePulseAngularMotion":
        return RatePulseAngularMotion(**payload)
    if motion_type == "SampledAngularMotion":
        return SampledAngularMotion(
            times_s=tuple(payload["times_s"]),
            angles_rad=tuple(payload["angles_rad"]),
        )
    if motion_type == "SumAngularMotion":
        components = payload.get("components")
        if not isinstance(components, list):
            raise ValueError("sum motion components must be a list")
        return SumAngularMotion(
            tuple(angular_motion_from_dict(item) for item in components)
        )
    raise ValueError(f"unsupported angular motion type: {motion_type!r}")


def gimbal_config_from_dict(value: Any) -> GimbalServoingConfig:
    """Reconstruct a fully configurable plant and observation contract."""
    payload = _typed_payload(value, "GimbalServoingConfig")
    camera_payload = _typed_payload(payload["camera"], "CameraConfig")
    if "forced_dropout_intervals_s" in camera_payload:
        camera_payload["forced_dropout_intervals_s"] = tuple(
            tuple(interval)
            for interval in camera_payload["forced_dropout_intervals_s"]
        )
    return GimbalServoingConfig(
        servo=ServoConfig(
            **_typed_payload(payload["servo"], "ServoConfig")
        ),
        camera=CameraConfig(**camera_payload),
        timing=TimingConfig(
            **_typed_payload(payload["timing"], "TimingConfig")
        ),
        scenario=ScenarioConfig(
            **_typed_payload(payload["scenario"], "ScenarioConfig")
        ),
        objective=ObjectiveConfig(
            **_typed_payload(payload["objective"], "ObjectiveConfig")
        ),
        observation_profile=ObservationProfile(payload["observation_profile"]),
        command_mode=GimbalCommandMode(payload["command_mode"]),
    )


def closed_loop_scenario_from_dict(value: Any) -> ClosedLoopScenario:
    """Reconstruct an exact closed-loop scenario recorded by the dataset."""
    payload = _typed_payload(value, "ClosedLoopScenario")
    return ClosedLoopScenario(
        name=str(payload["name"]),
        description=str(payload["description"]),
        config=gimbal_config_from_dict(payload["config"]),
        target_motion=angular_motion_from_dict(payload["target_motion"]),
        body_motion=angular_motion_from_dict(payload["body_motion"]),
    )
