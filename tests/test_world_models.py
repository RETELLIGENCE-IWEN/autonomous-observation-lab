import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autonomous_observation_lab.benchmark.config import BenchmarkConfig
from autonomous_observation_lab.world_models.data import (
    FeatureTrajectoryDataset,
    generate_dataset,
    tensor_spec,
)
from autonomous_observation_lab.world_models.models import make_model
from autonomous_observation_lab.world_models.objectives import world_model_loss


def test_dataset_generation_is_deterministic():
    config = BenchmarkConfig()
    first = generate_dataset(range(101, 105), config)
    second = generate_dataset(range(101, 105), config)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize(
    "model_name",
    [
        "deterministic_recurrent",
        "monolithic_rssm",
        "object_centric_rssm",
    ],
)
def test_model_shapes_losses_and_open_loop(model_name):
    config = BenchmarkConfig()
    spec = tensor_spec(config)
    arrays = generate_dataset(range(201, 204), config)
    dataset = FeatureTrajectoryDataset(arrays)
    batch = {
        key: torch.stack([dataset[index][key] for index in range(len(dataset))])
        for key in arrays
    }
    model = make_model(model_name, spec)
    model.eval()
    filtering = model(batch)
    open_loop = model(batch, open_loop_start=5)
    expected_shape = (
        3,
        config.horizon,
        config.num_objects,
        spec.target_dim,
    )
    assert filtering.prediction.shape == expected_shape
    assert open_loop.prediction.shape == expected_shape
    loss = world_model_loss(filtering, batch, spec)
    assert torch.isfinite(loss.total)
    assert float(loss.total.detach()) > 0.0


def test_corruption_changes_inputs_not_privileged_targets():
    stable = BenchmarkConfig(handle_corruption_probability=0.0)
    corrupt = BenchmarkConfig(handle_corruption_probability=1.0)
    stable_data = generate_dataset(range(301, 303), stable)
    corrupt_data = generate_dataset(range(301, 303), corrupt)
    assert not np.array_equal(stable_data["detections"], corrupt_data["detections"])
    d = stable.signature_bits
    privileged_indices = list(range(d + 4)) + [d + 5]
    np.testing.assert_array_equal(
        stable_data["targets"][..., privileged_indices],
        corrupt_data["targets"][..., privileged_indices],
    )
