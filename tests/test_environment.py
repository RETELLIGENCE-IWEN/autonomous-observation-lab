import numpy as np

from autonomous_observation_lab.benchmark import (
    Action,
    ActionKind,
    BenchmarkConfig,
    StagedEvidenceEnv,
)


def serialize_observation(observation):
    return (
        observation.step,
        observation.remaining_steps,
        tuple(
            (
                d.handle,
                tuple(d.bbox.tolist()),
                d.confidence,
                tuple(d.appearance.tolist()),
                d.motion_cue,
                d.quality,
            )
            for d in observation.detections
        ),
    )


def test_seeded_replay_is_identical():
    config = BenchmarkConfig(miss_probability=0.1)
    actions = [
        Action(ActionKind.LOOK, 0),
        Action(ActionKind.WIDE_SCAN),
        Action(ActionKind.DWELL, 1),
        Action(ActionKind.HOLD),
    ]
    traces = []
    for _ in range(2):
        env = StagedEvidenceEnv(config)
        observation, _ = env.reset(seed=1234)
        trace = [serialize_observation(observation)]
        for action in actions:
            result = env.step(action)
            trace.append(
                (
                    serialize_observation(result.observation),
                    result.reward,
                    result.info["stage"],
                )
            )
        traces.append(trace)
    assert traces[0] == traces[1]


def test_unique_target_predicate_matches_target_id():
    env = StagedEvidenceEnv(BenchmarkConfig())
    for seed in range(200):
        env.reset(seed)
        predicate_ids = [obj.object_id for obj in env.objects if obj.is_target]
        expected = [] if env.target_id is None else [env.target_id]
        assert predicate_ids == expected


def test_target_handle_is_not_fixed():
    env = StagedEvidenceEnv(BenchmarkConfig(target_present_probability=1.0))
    target_ids = []
    for seed in range(100):
        env.reset(seed)
        target_ids.append(env.target_id)
    counts = np.bincount(target_ids, minlength=env.config.num_objects)
    assert np.all(counts > 5)


def test_target_is_hidden_during_occlusion_interval():
    config = BenchmarkConfig(
        target_present_probability=1.0,
        miss_probability=0.0,
        occlusion_start=2,
        occlusion_end=4,
    )
    env = StagedEvidenceEnv(config)
    env.reset(seed=7)
    assert env.target_id is not None

    env.step(Action(ActionKind.WIDE_SCAN))
    result = env.step(Action(ActionKind.WIDE_SCAN))
    handles = {d.handle for d in result.observation.detections}
    assert env.target_id not in handles

    env.step(Action(ActionKind.WIDE_SCAN))
    result = env.step(Action(ActionKind.WIDE_SCAN))
    handles = {d.handle for d in result.observation.detections}
    assert env.target_id in handles


def test_handle_corruption_is_seeded_and_can_create_collisions():
    config = BenchmarkConfig(
        target_present_probability=1.0,
        miss_probability=0.0,
        handle_corruption_probability=1.0,
    )
    traces = []
    collision_seen = False
    for _ in range(2):
        env = StagedEvidenceEnv(config)
        observation, _ = env.reset(seed=991)
        trace = []
        for _step in range(6):
            handles = [d.handle for d in observation.detections]
            collision_seen = collision_seen or len(handles) != len(set(handles))
            trace.append(tuple(handles))
            observation = env.step(Action(ActionKind.WIDE_SCAN)).observation
        traces.append(trace)
    assert traces[0] == traces[1]
    assert collision_seen
