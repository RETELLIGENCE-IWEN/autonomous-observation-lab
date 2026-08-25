from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import TensorSpec


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    kl: torch.Tensor


def _normal_parameters(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, raw_std = torch.chunk(tensor, 2, dim=-1)
    std = F.softplus(raw_std) + 0.1
    return mean, std


def _kl_normal(
    posterior_mean: torch.Tensor,
    posterior_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    variance_ratio = (posterior_std / prior_std).square()
    mean_term = ((posterior_mean - prior_mean) / prior_std).square()
    return 0.5 * (
        variance_ratio + mean_term - 1.0 + 2.0 * (prior_std.log() - posterior_std.log())
    )


class DeterministicRecurrentModel(nn.Module):
    name = "deterministic_recurrent"

    def __init__(self, spec: TensorSpec, hidden_dim: int = 96, embed_dim: int = 96):
        super().__init__()
        self.spec = spec
        observation_dim = spec.num_objects * (spec.detection_dim + 1)
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
        )
        self.transition = nn.GRUCell(embed_dim + spec.action_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, spec.num_objects * spec.target_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(
        self, batch: dict[str, torch.Tensor], open_loop_start: int | None = None
    ) -> ModelOutput:
        detections = batch["detections"]
        mask = batch["detection_mask"]
        actions = batch["actions"]
        batch_size, sequence_length = detections.shape[:2]
        hidden = detections.new_zeros(batch_size, self.hidden_dim)
        predictions = []
        for time in range(sequence_length):
            if open_loop_start is not None and time >= open_loop_start:
                encoded = detections.new_zeros(
                    batch_size, self.observation_encoder[0].out_features
                )
            else:
                observation = torch.cat(
                    [detections[:, time], mask[:, time, :, None]], dim=-1
                ).flatten(1)
                encoded = self.observation_encoder(observation)
            hidden = self.transition(
                torch.cat([encoded, actions[:, time]], dim=-1), hidden
            )
            predictions.append(
                self.head(hidden).view(
                    batch_size, self.spec.num_objects, self.spec.target_dim
                )
            )
        prediction = torch.stack(predictions, dim=1)
        return ModelOutput(prediction=prediction, kl=prediction.new_zeros(()))


class MonolithicRSSM(nn.Module):
    name = "monolithic_rssm"

    def __init__(
        self,
        spec: TensorSpec,
        hidden_dim: int = 80,
        stochastic_dim: int = 24,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.spec = spec
        observation_dim = spec.num_objects * (spec.detection_dim + 1)
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
        )
        self.transition = nn.GRUCell(stochastic_dim + spec.action_dim, hidden_dim)
        self.prior = nn.Linear(hidden_dim, 2 * stochastic_dim)
        self.posterior = nn.Linear(hidden_dim + embed_dim, 2 * stochastic_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, spec.num_objects * spec.target_dim),
        )
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim

    def forward(
        self, batch: dict[str, torch.Tensor], open_loop_start: int | None = None
    ) -> ModelOutput:
        detections = batch["detections"]
        mask = batch["detection_mask"]
        actions = batch["actions"]
        batch_size, sequence_length = detections.shape[:2]
        hidden = detections.new_zeros(batch_size, self.hidden_dim)
        stochastic = detections.new_zeros(batch_size, self.stochastic_dim)
        predictions, kls = [], []

        for time in range(sequence_length):
            hidden = self.transition(
                torch.cat([stochastic, actions[:, time]], dim=-1), hidden
            )
            prior_mean, prior_std = _normal_parameters(self.prior(hidden))
            use_observation = open_loop_start is None or time < open_loop_start
            if use_observation:
                observation = torch.cat(
                    [detections[:, time], mask[:, time, :, None]], dim=-1
                ).flatten(1)
                encoded = self.observation_encoder(observation)
                posterior_mean, posterior_std = _normal_parameters(
                    self.posterior(torch.cat([hidden, encoded], dim=-1))
                )
                if self.training:
                    stochastic = posterior_mean + posterior_std * torch.randn_like(
                        posterior_std
                    )
                else:
                    stochastic = posterior_mean
                kls.append(
                    _kl_normal(
                        posterior_mean,
                        posterior_std,
                        prior_mean,
                        prior_std,
                    ).mean()
                )
            else:
                stochastic = prior_mean
            predictions.append(
                self.head(torch.cat([hidden, stochastic], dim=-1)).view(
                    batch_size, self.spec.num_objects, self.spec.target_dim
                )
            )
        prediction = torch.stack(predictions, dim=1)
        kl = torch.stack(kls).mean() if kls else prediction.new_zeros(())
        return ModelOutput(prediction=prediction, kl=kl)


class ObjectCentricRSSM(nn.Module):
    name = "object_centric_rssm"

    def __init__(
        self,
        spec: TensorSpec,
        hidden_dim: int = 72,
        stochastic_dim: int = 24,
        embed_dim: int = 80,
    ):
        super().__init__()
        self.spec = spec
        self.object_encoder = nn.Sequential(
            nn.Linear(spec.detection_dim + 1, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
        )
        transition_input = stochastic_dim + hidden_dim + spec.action_dim
        self.transition = nn.GRUCell(transition_input, hidden_dim)
        self.prior = nn.Linear(hidden_dim, 2 * stochastic_dim)
        self.posterior = nn.Linear(hidden_dim + embed_dim, 2 * stochastic_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + stochastic_dim + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, spec.target_dim),
        )
        self.hidden_dim = hidden_dim
        self.stochastic_dim = stochastic_dim

    def forward(
        self, batch: dict[str, torch.Tensor], open_loop_start: int | None = None
    ) -> ModelOutput:
        detections = batch["detections"]
        mask = batch["detection_mask"]
        actions = batch["actions"]
        batch_size, sequence_length, num_objects = detections.shape[:3]
        hidden = detections.new_zeros(batch_size, num_objects, self.hidden_dim)
        stochastic = detections.new_zeros(
            batch_size, num_objects, self.stochastic_dim
        )
        predictions, kls = [], []

        for time in range(sequence_length):
            global_context = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            action = actions[:, time, None, :].expand(
                batch_size, num_objects, self.spec.action_dim
            )
            transition_input = torch.cat(
                [stochastic, global_context, action], dim=-1
            ).reshape(batch_size * num_objects, -1)
            hidden = self.transition(
                transition_input, hidden.reshape(batch_size * num_objects, -1)
            ).view(batch_size, num_objects, self.hidden_dim)
            prior_mean, prior_std = _normal_parameters(self.prior(hidden))
            use_observation = open_loop_start is None or time < open_loop_start
            if use_observation:
                object_input = torch.cat(
                    [detections[:, time], mask[:, time, :, None]], dim=-1
                )
                encoded = self.object_encoder(object_input)
                posterior_mean, posterior_std = _normal_parameters(
                    self.posterior(torch.cat([hidden, encoded], dim=-1))
                )
                if self.training:
                    posterior_sample = posterior_mean + posterior_std * torch.randn_like(
                        posterior_std
                    )
                else:
                    posterior_sample = posterior_mean
                observed = mask[:, time, :, None]
                stochastic = observed * posterior_sample + (1.0 - observed) * prior_mean
                slot_kl = _kl_normal(
                    posterior_mean, posterior_std, prior_mean, prior_std
                )
                kls.append(
                    (slot_kl * observed).sum()
                    / observed.expand_as(slot_kl).sum().clamp_min(1.0)
                )
            else:
                stochastic = prior_mean
            global_context = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            predictions.append(
                self.head(torch.cat([hidden, stochastic, global_context], dim=-1))
            )
        prediction = torch.stack(predictions, dim=1)
        kl = torch.stack(kls).mean() if kls else prediction.new_zeros(())
        return ModelOutput(prediction=prediction, kl=kl)


def make_model(name: str, spec: TensorSpec) -> nn.Module:
    factories = {
        "deterministic_recurrent": lambda: DeterministicRecurrentModel(spec),
        "monolithic_rssm": lambda: MonolithicRSSM(spec),
        "object_centric_rssm": lambda: ObjectCentricRSSM(spec),
    }
    try:
        return factories[name]()
    except KeyError as error:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(factories)}") from error


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
