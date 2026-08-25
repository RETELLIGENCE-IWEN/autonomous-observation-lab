from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from autonomous_observation_lab.benchmark.config import BenchmarkConfig
from autonomous_observation_lab.benchmark.env import StagedEvidenceEnv
from autonomous_observation_lab.benchmark.policies import make_policy
from autonomous_observation_lab.benchmark.types import Action, ActionKind, Observation


ACTION_KINDS = (
    ActionKind.WIDE_SCAN,
    ActionKind.LOOK,
    ActionKind.DWELL,
    ActionKind.HOLD,
    ActionKind.COMMIT,
    ActionKind.DECLARE_ABSENT,
    ActionKind.ABSTAIN,
)


@dataclass(frozen=True)
class TensorSpec:
    num_objects: int
    signature_bits: int
    detection_dim: int
    action_dim: int
    target_dim: int
    sequence_length: int


def tensor_spec(config: BenchmarkConfig) -> TensorSpec:
    # bbox(4), confidence(1), appearance(D), appearance-valid(D),
    # motion cue/valid(2), quality(1), observed(1)
    detection_dim = 4 + 1 + 2 * config.signature_bits + 2 + 1 + 1
    action_dim = len(ACTION_KINDS) + config.num_objects
    # signature(D), motion(1), position(2), visibility(1), quality(1), target(1)
    target_dim = config.signature_bits + 6
    return TensorSpec(
        num_objects=config.num_objects,
        signature_bits=config.signature_bits,
        detection_dim=detection_dim,
        action_dim=action_dim,
        target_dim=target_dim,
        sequence_length=config.horizon,
    )


def encode_action(action: Action | None, config: BenchmarkConfig) -> np.ndarray:
    vector = np.zeros(len(ACTION_KINDS) + config.num_objects, dtype=np.float32)
    if action is None:
        return vector
    vector[ACTION_KINDS.index(action.kind)] = 1.0
    if action.object_id is not None:
        vector[len(ACTION_KINDS) + action.object_id] = 1.0
    return vector


def encode_observation(
    observation: Observation, config: BenchmarkConfig
) -> tuple[np.ndarray, np.ndarray]:
    spec = tensor_spec(config)
    features = np.zeros((config.num_objects, spec.detection_dim), dtype=np.float32)
    mask = np.zeros(config.num_objects, dtype=np.float32)
    for detection in observation.detections:
        handle = detection.handle
        if not 0 <= handle < config.num_objects:
            continue
        cursor = 0
        features[handle, cursor : cursor + 4] = detection.bbox
        cursor += 4
        features[handle, cursor] = detection.confidence
        cursor += 1
        features[handle, cursor : cursor + config.signature_bits] = detection.appearance
        cursor += config.signature_bits
        features[handle, cursor : cursor + config.signature_bits] = (
            detection.appearance_valid
        )
        cursor += config.signature_bits
        features[handle, cursor] = detection.motion_cue
        features[handle, cursor + 1] = float(detection.motion_valid)
        cursor += 2
        features[handle, cursor] = detection.quality
        features[handle, cursor + 1] = 1.0
        mask[handle] = 1.0
    return features, mask


def hidden_targets(
    env: StagedEvidenceEnv, observation: Observation
) -> tuple[np.ndarray, np.ndarray]:
    config = env.config
    spec = tensor_spec(config)
    targets = np.zeros((config.num_objects, spec.target_dim), dtype=np.float32)
    quality_by_object = {
        env._handle_to_object.get(d.handle, -1): d.quality  # noqa: SLF001
        for d in observation.detections
    }
    for obj in env.objects:
        cursor = 0
        targets[obj.object_id, cursor : cursor + config.signature_bits] = obj.signature
        cursor += config.signature_bits
        targets[obj.object_id, cursor] = obj.motion_class
        cursor += 1
        targets[obj.object_id, cursor : cursor + 2] = (
            obj.position + observation.step * obj.velocity
        )
        cursor += 2
        visible = not (
            obj.object_id == env.target_id
            and config.occlusion_start <= observation.step < config.occlusion_end
        )
        targets[obj.object_id, cursor] = float(visible)
        targets[obj.object_id, cursor + 1] = quality_by_object.get(obj.object_id, 0.0)
        targets[obj.object_id, cursor + 2] = float(obj.is_target)
    return targets, np.ones(config.num_objects, dtype=np.float32)


def _exploration_action(
    proposed: Action,
    observation: Observation,
    rng: np.random.Generator,
    config: BenchmarkConfig,
) -> Action:
    if proposed.kind is ActionKind.COMMIT and proposed.object_id is not None:
        return Action(ActionKind.LOOK, proposed.object_id)
    if proposed.kind in {ActionKind.DECLARE_ABSENT, ActionKind.ABSTAIN}:
        return Action(ActionKind.WIDE_SCAN)
    # Inject dwell and wide actions so evidence-quality dynamics are covered.
    roll = rng.random()
    visible = sorted({d.handle for d in observation.detections})
    if roll < 0.10:
        return Action(ActionKind.WIDE_SCAN)
    if roll < 0.20 and visible:
        return Action(ActionKind.DWELL, int(rng.choice(visible)))
    return proposed


def generate_trajectory(
    seed: int,
    config: BenchmarkConfig,
    policy_name: str,
) -> dict[str, np.ndarray]:
    spec = tensor_spec(config)
    env = StagedEvidenceEnv(config)
    policy = make_policy(policy_name, config)
    rng = np.random.default_rng(seed ^ 0x5EED)
    observation, _ = env.reset(seed)
    policy.reset()
    policy.observe(observation)
    previous_action: Action | None = None

    detections = np.zeros(
        (spec.sequence_length, spec.num_objects, spec.detection_dim), dtype=np.float32
    )
    detection_mask = np.zeros(
        (spec.sequence_length, spec.num_objects), dtype=np.float32
    )
    actions = np.zeros((spec.sequence_length, spec.action_dim), dtype=np.float32)
    targets = np.zeros(
        (spec.sequence_length, spec.num_objects, spec.target_dim), dtype=np.float32
    )
    target_mask = np.ones(
        (spec.sequence_length, spec.num_objects), dtype=np.float32
    )

    for time in range(spec.sequence_length):
        detections[time], detection_mask[time] = encode_observation(
            observation, config
        )
        actions[time] = encode_action(previous_action, config)
        targets[time], target_mask[time] = hidden_targets(env, observation)
        if time == spec.sequence_length - 1:
            break
        proposed = policy.act(observation)
        action = _exploration_action(proposed, observation, rng, config)
        result = env.step(action)
        observation = result.observation
        policy.observe(observation)
        previous_action = action

    return {
        "detections": detections,
        "detection_mask": detection_mask,
        "actions": actions,
        "targets": targets,
        "target_mask": target_mask,
    }


def generate_dataset(
    seeds: range,
    config: BenchmarkConfig | None = None,
    policy_names: tuple[str, ...] = (
        "random",
        "fixed_scan",
        "entropy_greedy",
        "decision_voi",
    ),
) -> dict[str, np.ndarray]:
    config = config or BenchmarkConfig()
    trajectories = [
        generate_trajectory(
            seed,
            config,
            policy_names[index % len(policy_names)],
        )
        for index, seed in enumerate(seeds)
    ]
    return {
        key: np.stack([trajectory[key] for trajectory in trajectories])
        for key in trajectories[0]
    }


class FeatureTrajectoryDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("all arrays must have equal leading length")
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(value[index]).float()
            for key, value in self.arrays.items()
        }

