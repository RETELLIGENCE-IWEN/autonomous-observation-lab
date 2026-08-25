import argparse
import json
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from autonomous_observation_lab.benchmark.config import BenchmarkConfig

from .data import FeatureTrajectoryDataset, generate_dataset, tensor_spec
from .models import make_model, parameter_count
from .objectives import world_model_loss


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    target_accuracy: float
    target_balanced_accuracy: float
    target_auroc: float
    target_brier: float
    position_rmse: float
    visibility_accuracy: float
    occlusion_target_accuracy: float
    occlusion_target_balanced_accuracy: float
    reacquisition_target_accuracy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    positive = target > 0.5
    negative = ~positive
    tpr = float(prediction[positive].mean()) if positive.any() else 0.5
    tnr = float((1.0 - prediction[negative]).mean()) if negative.any() else 0.5
    return 0.5 * (tpr + tnr)


def _binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.reshape(-1) > 0.5
    score = score.reshape(-1)
    positive_count = int(target.sum())
    negative_count = len(target) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.5
    order = np.argsort(score, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average tied ranks.
    unique, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    del unique
    for group in np.flatnonzero(counts > 1):
        tied = inverse == group
        ranks[tied] = ranks[tied].mean()
    rank_sum = ranks[target].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    config: BenchmarkConfig,
    device: torch.device,
    open_loop_start: int | None = None,
) -> EvaluationMetrics:
    model.eval()
    spec = tensor_spec(config)
    losses = []
    target_probabilities, target_truths = [], []
    positions, position_truths = [], []
    visibility_predictions, visibility_truths = [], []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(batch, open_loop_start=open_loop_start)
        loss = world_model_loss(output, batch, spec)
        losses.append(float(loss.total))
        d = spec.signature_bits
        target_probabilities.append(torch.sigmoid(output.prediction[..., d + 5]).cpu())
        target_truths.append(batch["targets"][..., d + 5].cpu())
        positions.append(output.prediction[..., d + 1 : d + 3].cpu())
        position_truths.append(batch["targets"][..., d + 1 : d + 3].cpu())
        visibility_predictions.append(
            torch.sigmoid(output.prediction[..., d + 3]).cpu()
        )
        visibility_truths.append(batch["targets"][..., d + 3].cpu())

    target_probability = torch.cat(target_probabilities).numpy()
    target_truth = torch.cat(target_truths).numpy()
    target_prediction = (target_probability >= 0.5).astype(np.float32)
    position = torch.cat(positions).numpy()
    position_truth = torch.cat(position_truths).numpy()
    visibility_probability = torch.cat(visibility_predictions).numpy()
    visibility_truth = torch.cat(visibility_truths).numpy()

    occlusion = slice(config.occlusion_start, config.occlusion_end)
    reacquisition = slice(
        config.occlusion_end, min(config.horizon, config.occlusion_end + 2)
    )
    return EvaluationMetrics(
        loss=float(np.mean(losses)),
        target_accuracy=float(np.mean(target_prediction == target_truth)),
        target_balanced_accuracy=_balanced_accuracy(target_truth, target_prediction),
        target_auroc=_binary_auroc(target_truth, target_probability),
        target_brier=float(np.mean((target_probability - target_truth) ** 2)),
        position_rmse=float(np.sqrt(np.mean((position - position_truth) ** 2))),
        visibility_accuracy=float(
            np.mean((visibility_probability >= 0.5) == visibility_truth)
        ),
        occlusion_target_accuracy=float(
            np.mean(target_prediction[:, occlusion] == target_truth[:, occlusion])
        ),
        occlusion_target_balanced_accuracy=_balanced_accuracy(
            target_truth[:, occlusion], target_prediction[:, occlusion]
        ),
        reacquisition_target_accuracy=float(
            np.mean(
                target_prediction[:, reacquisition]
                == target_truth[:, reacquisition]
            )
        ),
    )


def train_model(
    model_name: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: BenchmarkConfig,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    spec = tensor_spec(config)
    model = make_model(model_name, spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(batch)
            loss = world_model_loss(output, batch, spec)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            train_losses.append(float(loss.total.detach()))
        validation = evaluate_model(model, validation_loader, config, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation.loss,
            }
        )
    return model, history


def latest_frame_baseline(
    arrays: dict[str, np.ndarray], config: BenchmarkConfig
) -> dict[str, float]:
    """Non-recurrent target estimate from current raw evidence only."""
    d = config.signature_bits
    appearance_bit = arrays["detections"][..., 5]
    motion_index = 5 + 2 * d
    motion_bit = arrays["detections"][..., motion_index]
    observed = arrays["detection_mask"]
    score = appearance_bit * motion_bit * observed
    prediction = (score >= 0.5).astype(np.float32)
    truth = arrays["targets"][..., d + 5]
    occlusion = slice(config.occlusion_start, config.occlusion_end)
    return {
        "target_accuracy": float(np.mean(prediction == truth)),
        "target_balanced_accuracy": _balanced_accuracy(truth, prediction),
        "target_auroc": _binary_auroc(truth, score),
        "occlusion_target_accuracy": float(
            np.mean(prediction[:, occlusion] == truth[:, occlusion])
        ),
        "occlusion_target_balanced_accuracy": _balanced_accuracy(
            truth[:, occlusion], prediction[:, occlusion]
        ),
    }


def run_experiment(
    train_episodes: int = 800,
    validation_episodes: int = 200,
    test_episodes: int = 300,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    seed: int = 7,
    device_name: str = "cpu",
) -> dict[str, object]:
    set_seed(seed)
    config = BenchmarkConfig()
    device = torch.device(device_name)
    train_arrays = generate_dataset(range(40_000, 40_000 + train_episodes), config)
    validation_arrays = generate_dataset(
        range(50_000, 50_000 + validation_episodes), config
    )
    test_arrays = generate_dataset(range(60_000, 60_000 + test_episodes), config)
    corrupted_config = BenchmarkConfig(handle_corruption_probability=0.15)
    corrupted_arrays = generate_dataset(
        range(70_000, 70_000 + test_episodes), corrupted_config
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        FeatureTrajectoryDataset(train_arrays),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        FeatureTrajectoryDataset(validation_arrays), batch_size=batch_size
    )
    test_loader = DataLoader(
        FeatureTrajectoryDataset(test_arrays), batch_size=batch_size
    )
    corrupted_loader = DataLoader(
        FeatureTrajectoryDataset(corrupted_arrays), batch_size=batch_size
    )

    results = {}
    for model_name in (
        "deterministic_recurrent",
        "monolithic_rssm",
        "object_centric_rssm",
    ):
        set_seed(seed)
        model, history = train_model(
            model_name,
            train_loader,
            validation_loader,
            config,
            epochs,
            learning_rate,
            device,
        )
        results[model_name] = {
            "parameters": parameter_count(model),
            "history": history,
            "filtering": asdict(evaluate_model(model, test_loader, config, device)),
            "open_loop_from_5": asdict(
                evaluate_model(
                    model, test_loader, config, device, open_loop_start=5
                )
            ),
            "handle_corruption_0_15": asdict(
                evaluate_model(model, corrupted_loader, corrupted_config, device)
            ),
        }

    return {
        "seed": seed,
        "device": str(device),
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "test_episodes": test_episodes,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "config": asdict(config),
        "latest_frame_baseline": latest_frame_baseline(test_arrays, config),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=800)
    parser.add_argument("--validation-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_experiment(
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        test_episodes=args.test_episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
