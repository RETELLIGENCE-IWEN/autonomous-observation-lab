import math

import numpy as np
import pytest

from autonomous_observation_lab.gimbal_servoing.demos import paired_cause_demo
from autonomous_observation_lab.gimbal_servoing.disturbances import (
    SampledAngularMotion,
)


def test_sampled_motion_interpolates_angle_and_rate() -> None:
    motion = SampledAngularMotion(
        times_s=(0.0, 1.0, 3.0),
        angles_rad=(0.0, 2.0, 4.0),
    )

    assert motion.state_at(0.5) == pytest.approx((1.0, 2.0))
    assert motion.state_at(2.0) == pytest.approx((3.0, 1.0))
    assert motion.state_at(-1.0)[0] == 0.0
    assert motion.state_at(4.0)[0] == 4.0


def test_paired_demo_has_same_bbox_motion_from_different_causes() -> None:
    gimbal_moves, target_moves = paired_cause_demo()

    assert len(gimbal_moves.frames) == len(target_moves.frames)
    gimbal_times = [frame.diagnostics.time_s for frame in gimbal_moves.frames]
    target_times = [frame.diagnostics.time_s for frame in target_moves.frames]
    assert target_times == pytest.approx(gimbal_times)

    gimbal_target = np.array(
        [frame.diagnostics.target_bearing_rad for frame in gimbal_moves.frames]
    )
    gimbal_angle = np.array(
        [frame.diagnostics.gimbal_angle_rad for frame in gimbal_moves.frames]
    )
    target_target = np.array(
        [frame.diagnostics.target_bearing_rad for frame in target_moves.frames]
    )
    target_gimbal = np.array(
        [frame.diagnostics.gimbal_angle_rad for frame in target_moves.frames]
    )

    assert np.max(np.abs(gimbal_target)) == pytest.approx(0.0)
    assert np.max(np.abs(gimbal_angle)) > math.radians(10.0)
    assert np.max(np.abs(target_target)) > math.radians(10.0)
    assert np.max(np.abs(target_gimbal)) == pytest.approx(0.0)

    gimbal_error = [
        frame.diagnostics.true_image_error_normalized
        for frame in gimbal_moves.frames
    ]
    target_error = [
        frame.diagnostics.true_image_error_normalized
        for frame in target_moves.frames
    ]
    assert target_error == pytest.approx(gimbal_error, abs=1e-11)

    paired_observations = zip(
        (frame.observation for frame in gimbal_moves.frames),
        (frame.observation for frame in target_moves.frames),
        strict=True,
    )
    for gimbal_observation, target_observation in paired_observations:
        assert (
            target_observation.image_error_normalized.valid
            is gimbal_observation.image_error_normalized.valid
        )
        if gimbal_observation.image_error_normalized.valid:
            assert target_observation.image_error_normalized.value == pytest.approx(
                gimbal_observation.image_error_normalized.value,
                abs=1e-11,
            )
