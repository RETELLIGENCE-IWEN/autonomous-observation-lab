from dataclasses import dataclass

import numpy as np

from .config import BenchmarkConfig
from .env import StagedEvidenceEnv


@dataclass(frozen=True)
class LeakageProbeResult:
    samples: int
    positive_rate: float
    nuisance_balanced_accuracy: float
    permutation_p95: float
    passed: bool


def collect_nuisance_dataset(
    config: BenchmarkConfig, seeds: range
) -> tuple[np.ndarray, np.ndarray]:
    """Collect non-evidence fields only.

    Appearance and motion cues are intentionally excluded because they are
    legitimate target evidence. The probe detects accidental shortcuts through
    handle, bbox geometry, confidence, quality, or scenario timing.
    """
    rows: list[np.ndarray] = []
    labels: list[float] = []
    env = StagedEvidenceEnv(config)
    for seed in seeds:
        observation, _ = env.reset(seed)
        for detection in observation.detections:
            rows.append(
                np.concatenate(
                    [
                        np.array(
                            [
                                detection.handle / max(1, config.num_objects - 1),
                                detection.confidence,
                                detection.quality,
                                observation.remaining_steps / config.horizon,
                            ],
                            dtype=np.float64,
                        ),
                        detection.bbox.astype(np.float64),
                    ]
                )
            )
            labels.append(float(detection.handle == env.target_id))
    return np.stack(rows), np.asarray(labels, dtype=np.float64)


def _fit_logistic(
    x: np.ndarray, y: np.ndarray, train_indices: np.ndarray, iterations: int = 250
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = x[train_indices]
    train_y = y[train_indices]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (train_x - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    positive_weight = 0.5 / max(train_y.mean(), 1e-6)
    negative_weight = 0.5 / max(1.0 - train_y.mean(), 1e-6)
    sample_weight = np.where(train_y > 0.5, positive_weight, negative_weight)

    for _ in range(iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weight * (probability - train_y)) / len(train_y)
        weights -= 0.15 * gradient
    return weights, mean, scale


def _balanced_accuracy(y: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    positive = y > 0.5
    negative = ~positive
    tpr = float(np.mean(prediction[positive])) if np.any(positive) else 0.5
    tnr = float(np.mean(~prediction[negative])) if np.any(negative) else 0.5
    return 0.5 * (tpr + tnr)


def _score_split(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> float:
    weights, mean, scale = _fit_logistic(x, y, train_indices)
    test_x = (x[test_indices] - mean) / scale
    design = np.column_stack([np.ones(len(test_x)), test_x])
    probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30.0, 30.0)))
    return _balanced_accuracy(y[test_indices], probability)


def run_leakage_probe(
    config: BenchmarkConfig | None = None,
    seed_start: int = 30_000,
    episodes: int = 2_000,
    permutations: int = 20,
) -> LeakageProbeResult:
    config = config or BenchmarkConfig()
    x, y = collect_nuisance_dataset(
        config, range(seed_start, seed_start + episodes)
    )
    rng = np.random.default_rng(90210)
    indices = rng.permutation(len(y))
    split = int(0.7 * len(indices))
    train_indices, test_indices = indices[:split], indices[split:]
    observed = _score_split(x, y, train_indices, test_indices)

    null_scores = []
    for _ in range(permutations):
        permuted_y = rng.permutation(y)
        null_scores.append(
            _score_split(x, permuted_y, train_indices, test_indices)
        )
    p95 = float(np.quantile(null_scores, 0.95))
    # A small tolerance prevents declaring leakage from finite-sample noise.
    passed = observed <= max(0.56, p95 + 0.02)
    return LeakageProbeResult(
        samples=len(y),
        positive_rate=float(y.mean()),
        nuisance_balanced_accuracy=observed,
        permutation_p95=p95,
        passed=passed,
    )

