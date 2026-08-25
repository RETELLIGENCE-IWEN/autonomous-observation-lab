from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .data import TensorSpec
from .models import ModelOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    signature: torch.Tensor
    motion: torch.Tensor
    position: torch.Tensor
    visibility: torch.Tensor
    quality: torch.Tensor
    target: torch.Tensor
    kl: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.expand_as(value).sum().clamp_min(1.0)


def world_model_loss(
    output: ModelOutput,
    batch: dict[str, torch.Tensor],
    spec: TensorSpec,
    kl_weight: float = 0.05,
    free_nats: float = 0.2,
) -> LossBreakdown:
    prediction = output.prediction
    target = batch["targets"]
    mask = batch["target_mask"]
    d = spec.signature_bits

    signature = _masked_mean(
        F.binary_cross_entropy_with_logits(
            prediction[..., :d], target[..., :d], reduction="none"
        ),
        mask,
    )
    motion = _masked_mean(
        F.binary_cross_entropy_with_logits(
            prediction[..., d], target[..., d], reduction="none"
        ),
        mask,
    )
    position = _masked_mean(
        (prediction[..., d + 1 : d + 3] - target[..., d + 1 : d + 3]).square(),
        mask,
    )
    visibility = _masked_mean(
        F.binary_cross_entropy_with_logits(
            prediction[..., d + 3], target[..., d + 3], reduction="none"
        ),
        mask,
    )
    quality = _masked_mean(
        (prediction[..., d + 4] - target[..., d + 4]).square(),
        mask,
    )
    target_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            prediction[..., d + 5],
            target[..., d + 5],
            reduction="none",
            pos_weight=prediction.new_tensor(4.0),
        ),
        mask,
    )
    kl = (
        torch.clamp(output.kl, min=free_nats)
        if output.kl.requires_grad
        else output.kl
    )
    total = (
        signature
        + motion
        + position
        + visibility
        + quality
        + 1.5 * target_loss
        + kl_weight * kl
    )
    return LossBreakdown(
        total=total,
        signature=signature,
        motion=motion,
        position=position,
        visibility=visibility,
        quality=quality,
        target=target_loss,
        kl=output.kl,
    )
