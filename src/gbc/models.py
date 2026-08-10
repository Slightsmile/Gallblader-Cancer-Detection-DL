"""Backbone construction and weight averaging."""

from __future__ import annotations

import copy

import timm
import torch
import torch.nn as nn

# Backbone zoo. Names are timm identifiers; the value is the default input size.
BACKBONES: dict[str, int] = {
    "efficientnet_b0": 224,
    "efficientnet_b2": 260,
    "resnet50": 224,
    "densenet121": 224,
    "convnext_tiny": 224,
    "deit3_small_patch16_224": 224,
    "swin_tiny_patch4_window7_224": 224,
    "mobilenetv3_large_100": 224,
}


def build_model(
    name: str,
    num_outputs: int,
    *,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    second_order: bool = False,
) -> nn.Module:
    """A timm backbone with a fresh ``num_outputs``-way head.

    ``second_order=True`` swaps global average pooling for covariance pooling
    (:mod:`gbc.sop`), the ingredient Basu et al. credit for most of GBCNet's
    margin over stock backbones on this data.
    """
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    if second_order:
        from .sop import SecondOrderNet

        return SecondOrderNet(name, num_outputs, pretrained=pretrained, drop_rate=drop_rate)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_outputs, drop_rate=drop_rate)


class ModelEMA:
    """Exponential moving average of weights; evaluated instead of the raw model.

    Stabilises the validation signal on a dataset this small, where single-epoch
    accuracy swings of several points are otherwise common.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v.detach(), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)
