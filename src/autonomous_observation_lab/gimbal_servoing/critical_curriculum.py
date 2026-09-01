"""Episode-level curriculum for controller-critical gimbal states."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .control_criticality import ControlCriticality
from .dataset import GimbalTargetStateDataset


@dataclass(frozen=True)
class CriticalEpisodeCurriculumConfig:
    """Concentrate batches without changing per-label loss geometry."""

    concentration_strength: float = 2.0
    maximum_episode_weight: float = 4.0
    normalize_mean_weight: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.concentration_strength) or (
            self.concentration_strength < 0.0
        ):
            raise ValueError(
                "concentration strength must be finite and non-negative"
            )
        if not math.isfinite(self.maximum_episode_weight) or (
            self.maximum_episode_weight < 1.0
        ):
            raise ValueError("maximum episode weight must be at least one")


@dataclass(frozen=True)
class CriticalEpisodeCurriculum:
    episode_weights: np.ndarray
    critical_label_fraction_by_episode: np.ndarray
    observed_critical_label_fraction: float
    expected_sampled_critical_label_fraction: float


def compute_critical_episode_curriculum(
    dataset: GimbalTargetStateDataset,
    criticality: ControlCriticality,
    *,
    config: CriticalEpisodeCurriculumConfig | None = None,
) -> CriticalEpisodeCurriculum:
    """Weight whole causal sequences by their critical-state concentration."""

    config = config or CriticalEpisodeCurriculumConfig()
    valid = dataset.target_mask & dataset.sequence_mask[:, :, None]
    if criticality.critical_mask.shape != valid.shape:
        raise ValueError("criticality mask shape does not match dataset")
    valid_counts = np.sum(valid, axis=(1, 2)).astype(np.float64)
    critical_counts = np.sum(
        criticality.critical_mask & valid,
        axis=(1, 2),
    ).astype(np.float64)
    fractions = np.divide(
        critical_counts,
        valid_counts,
        out=np.zeros_like(critical_counts),
        where=valid_counts > 0.0,
    )
    total_valid = float(np.sum(valid_counts))
    observed = (
        float(np.sum(critical_counts)) / total_valid if total_valid else 0.0
    )
    if observed > 0.0:
        relative_concentration = fractions / observed
    else:
        relative_concentration = np.ones_like(fractions)
    weights = 1.0 + config.concentration_strength * np.maximum(
        relative_concentration - 1.0,
        0.0,
    )
    weights = np.clip(weights, 1.0, config.maximum_episode_weight)
    if config.normalize_mean_weight and len(weights):
        weights /= float(np.mean(weights))
    sampling_mass = float(np.sum(weights))
    probabilities = (
        weights / sampling_mass
        if sampling_mass > 0.0
        else np.full_like(weights, 1.0 / max(1, len(weights)))
    )
    expected_critical = float(np.sum(probabilities * critical_counts))
    expected_valid = float(np.sum(probabilities * valid_counts))
    expected_fraction = (
        expected_critical / expected_valid if expected_valid > 0.0 else 0.0
    )
    return CriticalEpisodeCurriculum(
        episode_weights=weights.astype(np.float32),
        critical_label_fraction_by_episode=fractions.astype(np.float32),
        observed_critical_label_fraction=observed,
        expected_sampled_critical_label_fraction=expected_fraction,
    )


def critical_episode_curriculum_report(
    dataset: GimbalTargetStateDataset,
    curriculum: CriticalEpisodeCurriculum,
    *,
    config: CriticalEpisodeCurriculumConfig | None = None,
) -> dict[str, Any]:
    config = config or CriticalEpisodeCurriculumConfig()
    weights = curriculum.episode_weights.astype(np.float64)
    by_scenario = {}
    total_weight = float(np.sum(weights))
    for scenario_index, scenario_name in enumerate(
        dataset.manifest.scenario_names
    ):
        selected = dataset.scenario_index == scenario_index
        by_scenario[scenario_name] = {
            "episode_count": int(np.sum(selected)),
            "uniform_sampling_fraction": float(np.mean(selected)),
            "curriculum_sampling_fraction": (
                float(np.sum(weights[selected])) / total_weight
                if total_weight > 0.0
                else 0.0
            ),
            "mean_critical_label_fraction": (
                float(
                    np.mean(
                        curriculum.critical_label_fraction_by_episode[selected]
                    )
                )
                if np.any(selected)
                else 0.0
            ),
        }
    return {
        "configuration": asdict(config),
        "episode_count": dataset.episode_count,
        "observed_critical_label_fraction": (
            curriculum.observed_critical_label_fraction
        ),
        "expected_sampled_critical_label_fraction": (
            curriculum.expected_sampled_critical_label_fraction
        ),
        "minimum_episode_weight": float(np.min(weights)),
        "maximum_episode_weight": float(np.max(weights)),
        "mean_episode_weight": float(np.mean(weights)),
        "by_scenario": by_scenario,
    }
