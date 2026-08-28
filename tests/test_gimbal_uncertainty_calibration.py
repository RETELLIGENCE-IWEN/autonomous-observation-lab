import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autonomous_observation_lab.gimbal_servoing import (
    GimbalCommandMode,
    GimbalDatasetGenerationConfig,
    ObservationProfile,
    generate_gimbal_dataset,
    save_gimbal_dataset,
)
from autonomous_observation_lab.gimbal_servoing.closed_loop import (
    nominal_scenario,
)
from autonomous_observation_lab.gimbal_servoing.dataset import FEATURE_NAMES
from autonomous_observation_lab.gimbal_servoing.gru import (
    CausalTargetStateGRU,
    GRUTargetStateModelConfig,
    save_gru_checkpoint,
)
from autonomous_observation_lab.gimbal_servoing.uncertainty_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CONTEXTUAL_CALIBRATION_SCHEMA_VERSION,
    CONTEXT_NAMES,
    CalibrationExperimentConfig,
    ContextualCalibrationExperimentConfig,
    ContextualGaussianUncertaintyCalibration,
    GaussianUncertaintyCalibration,
    _fit_scales,
    apply_uncertainty_calibration,
    calibrate_contextual_gru_uncertainty,
    calibrate_gru_uncertainty,
    load_uncertainty_calibration,
)
from autonomous_observation_lab.gimbal_servoing.types import (
    GimbalObservation,
    MaskedScalar,
)


def _calibration(
    *, bearing_scale: tuple[float, ...], rate_scale: tuple[float, ...]
) -> GaussianUncertaintyCalibration:
    return GaussianUncertaintyCalibration(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        profile=ObservationProfile.DISTURBANCE_AWARE,
        prediction_horizons_s=tuple(
            0.1 * index for index in range(len(bearing_scale))
        ),
        bearing_std_scale=bearing_scale,
        rate_std_scale=rate_scale,
        validation_dataset_hash="validation",
        test_dataset_hash="test",
        checkpoint_sha256="checkpoint",
        minimum_scale=0.25,
        maximum_scale=4.0,
    )


def _observation(
    *, frame_updated: bool, detection_valid: bool, measurement_age_s: float
) -> GimbalObservation:
    detected = MaskedScalar(0.0, detection_valid)
    return GimbalObservation(
        time_s=1.0,
        control_dt_s=0.01,
        frame_updated=frame_updated,
        measurement_age_s=MaskedScalar(measurement_age_s, detection_valid),
        image_error_normalized=detected,
        bbox_width_fraction=detected,
        bbox_height_fraction=detected,
        confidence=detected,
        gimbal_angle_rad=MaskedScalar(0.0, True),
        gimbal_rate_rad_s=MaskedScalar(0.0, True),
        body_rate_rad_s=MaskedScalar(0.0, True),
        command_mode=GimbalCommandMode.RATE,
        previous_action_normalized=0.0,
    )


def _contextual_calibration() -> ContextualGaussianUncertaintyCalibration:
    return ContextualGaussianUncertaintyCalibration(
        schema_version=CONTEXTUAL_CALIBRATION_SCHEMA_VERSION,
        profile=ObservationProfile.DISTURBANCE_AWARE,
        prediction_horizons_s=(0.0,),
        context_names=CONTEXT_NAMES,
        bearing_std_scale_by_context=tuple(
            (float(index + 1),) for index in range(len(CONTEXT_NAMES))
        ),
        rate_std_scale_by_context=tuple(
            (0.5 * float(index + 1),) for index in range(len(CONTEXT_NAMES))
        ),
        measurement_age_threshold_s=0.16,
        detection_gap_thresholds_s=(0.15, 0.65),
        prior_strength=512.0,
        validation_dataset_hash="validation",
        test_dataset_hash="test",
        checkpoint_sha256="checkpoint",
        minimum_scale=0.25,
        maximum_scale=8.0,
    )


def test_closed_form_gaussian_scaling_recovers_known_residual_scale():
    mean = np.zeros((1, 4, 2, 2), dtype=np.float64)
    std = np.ones_like(mean)
    targets = np.zeros_like(mean)
    targets[..., 0] = 2.0
    targets[..., 1] = 3.0
    mask = np.ones(mean.shape[:-1], dtype=np.bool_)

    bearing, rate, counts = _fit_scales(
        mean,
        std,
        targets,
        mask,
        minimum_scale=0.25,
        maximum_scale=4.0,
    )
    calibration = _calibration(
        bearing_scale=bearing,
        rate_scale=rate,
    )
    calibrated = apply_uncertainty_calibration(std, calibration)

    assert bearing == pytest.approx((2.0, 2.0))
    assert rate == pytest.approx((3.0, 3.0))
    assert counts == (4, 4)
    np.testing.assert_allclose(calibrated[..., 0], 2.0)
    np.testing.assert_allclose(calibrated[..., 1], 3.0)


def test_contextual_calibration_maps_only_deployable_measurement_state():
    calibration = _contextual_calibration()
    cases = (
        (_observation(frame_updated=True, detection_valid=True, measurement_age_s=0.1), 0.0),
        (_observation(frame_updated=True, detection_valid=True, measurement_age_s=0.2), 0.0),
        (_observation(frame_updated=False, detection_valid=True, measurement_age_s=0.1), 0.1),
        (_observation(frame_updated=False, detection_valid=True, measurement_age_s=0.2), 0.2),
        (_observation(frame_updated=True, detection_valid=False, measurement_age_s=0.0), 0.1),
        (_observation(frame_updated=True, detection_valid=False, measurement_age_s=0.0), 0.3),
        (_observation(frame_updated=True, detection_valid=False, measurement_age_s=0.0), 0.8),
        (_observation(frame_updated=True, detection_valid=False, measurement_age_s=0.0), None),
    )

    for expected, (observation, gap_s) in enumerate(cases):
        assert calibration.context_index(observation, gap_s) == expected
        assert calibration.scales_for_observation(0, observation, gap_s) == (
            float(expected + 1),
            0.5 * float(expected + 1),
        )


def test_calibration_fits_validation_and_evaluates_disjoint_test(tmp_path):
    scenario = replace(
        nominal_scenario(),
        name="calibration_smoke",
        config=replace(
            nominal_scenario().config,
            timing=replace(
                nominal_scenario().config.timing,
                episode_duration_s=0.3,
            ),
        ),
    )
    paths = {}
    manifests = {}
    for split, seed in (("validation", 5101), ("test", 6101)):
        request = GimbalDatasetGenerationConfig(
            split=split,
            seeds=(seed,),
            scenario_names=(scenario.name,),
            behavior_names=("proportional_rate",),
            observation_profiles=(
                ObservationProfile.DISTURBANCE_AWARE,
            ),
            prediction_horizons_s=(0.0,),
            include_oracle_ceilings=False,
        )
        dataset = generate_gimbal_dataset(request, scenarios=(scenario,))
        paths[split], _ = save_gimbal_dataset(tmp_path / split, dataset)
        manifests[split] = dataset.manifest

    model = CausalTargetStateGRU(
        GRUTargetStateModelConfig(
            input_dim=len(FEATURE_NAMES),
            prediction_horizons_s=(0.0,),
            hidden_dim=8,
            embedding_dim=8,
        )
    )
    checkpoint = save_gru_checkpoint(
        tmp_path / "o2.pt",
        model,
        metadata={
            "profile": ObservationProfile.DISTURBANCE_AWARE.value,
            "feature_names": list(FEATURE_NAMES),
            "dataset_hashes": {
                "validation": manifests["validation"].configuration_hash,
                "test": manifests["test"].configuration_hash,
            },
        },
    )
    result = calibrate_gru_uncertainty(
        validation_data=paths["validation"],
        test_data=paths["test"],
        checkpoint=checkpoint,
        config=CalibrationExperimentConfig(batch_size=1),
    )
    output = tmp_path / "calibration.json"
    output.write_text(json.dumps(result), encoding="utf-8")
    restored = load_uncertainty_calibration(output)
    contextual_result = calibrate_contextual_gru_uncertainty(
        validation_data=paths["validation"],
        test_data=paths["test"],
        checkpoint=checkpoint,
        config=ContextualCalibrationExperimentConfig(
            batch_size=1,
            prior_strength=16.0,
        ),
    )
    contextual_output = tmp_path / "contextual_calibration.json"
    contextual_output.write_text(
        json.dumps(contextual_result), encoding="utf-8"
    )
    contextual_restored = load_uncertainty_calibration(contextual_output)

    assert result["fit_split"] == "validation"
    assert result["evaluation_split"] == "test"
    assert result["test"]["uncalibrated"]["bearing_rmse_deg"] == pytest.approx(
        result["test"]["calibrated"]["bearing_rmse_deg"]
    )
    assert result["test"]["uncalibrated"]["rate_rmse_deg_s"] == pytest.approx(
        result["test"]["calibrated"]["rate_rmse_deg_s"]
    )
    assert all(
        math.isfinite(scale)
        for scale in restored.bearing_std_scale + restored.rate_std_scale
    )
    assert asdict(restored) == result["calibration"]
    assert isinstance(
        contextual_restored, ContextualGaussianUncertaintyCalibration
    )
    assert contextual_result["test"]["uncalibrated"][
        "bearing_rmse_deg"
    ] == pytest.approx(
        contextual_result["test"]["contextual"]["bearing_rmse_deg"]
    )
