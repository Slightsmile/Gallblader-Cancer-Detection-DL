"""Second-order pooling head.

Basu et al. attribute most of GBCNet's margin over stock backbones to a
multi-scale **second-order** pooling classifier: instead of global average
pooling, which discards all channel interactions, they pool the covariance of
the feature map. Malignancy on B-mode ultrasound presents as *texture* — wall
irregularity, heterogeneous echotexture — and texture is a second-order
statistic, which is why first-order pooling underperforms here.

This module supplies that head as a drop-in replacement for a timm backbone's
classifier, so the effect can be ablated against the same backbone with average
pooling and everything else held fixed.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CovariancePool(nn.Module):
    """Covariance pooling with iterative matrix square-root normalisation.

    Projects ``C`` channels down to ``dim`` (the covariance is ``dim x dim``, so
    this keeps the head affordable), forms the sample covariance over spatial
    positions, applies Newton–Schulz square-root normalisation — which is what
    makes covariance features trainable end to end — and returns the upper
    triangle.
    """

    def __init__(self, in_channels: int, dim: int = 128, iterations: int = 5):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.dim = dim
        self.iterations = iterations

    @property
    def out_features(self) -> int:
        return self.dim * (self.dim + 1) // 2

    def _sqrtm(self, matrix: torch.Tensor) -> torch.Tensor:
        """Newton–Schulz iteration for the matrix square root of an SPD batch."""
        batch, dim, _ = matrix.shape
        norm = matrix.norm(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        y = matrix / norm
        eye = torch.eye(dim, device=matrix.device, dtype=matrix.dtype).expand_as(matrix)
        z = eye.clone()
        for _ in range(self.iterations):
            t = 0.5 * (3.0 * eye - z @ y)
            y, z = y @ t, t @ z
        return y * norm.sqrt()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.reduce(features)
        batch, dim, height, width = x.shape
        x = x.reshape(batch, dim, height * width)
        x = x - x.mean(dim=2, keepdim=True)
        covariance = x @ x.transpose(1, 2) / (height * width)
        # Ridge term keeps the iteration stable when a batch is near rank-deficient.
        covariance = covariance + 1e-4 * torch.eye(dim, device=x.device, dtype=x.dtype)
        root = self._sqrtm(covariance)
        rows, cols = torch.triu_indices(dim, dim, device=x.device)
        return root[:, rows, cols]


class SecondOrderNet(nn.Module):
    """A timm backbone with its classifier replaced by covariance pooling."""

    def __init__(self, backbone: str, num_outputs: int, *, pretrained: bool = True,
                 dim: int = 128, drop_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          global_pool="", drop_rate=drop_rate)
        # Probe rather than trust `num_features`: for several timm families the
        # advertised width differs from what `forward_features` actually emits.
        with torch.no_grad():
            probe = self.backbone.forward_features(torch.zeros(1, 3, 224, 224))
        if probe.ndim == 3:  # transformers emit [B, tokens, C]
            raise ValueError(f"{backbone!r} emits token features; covariance pooling expects a "
                             "convolutional feature map")
        channels = probe.shape[1]
        self.pool = CovariancePool(channels, dim=dim)
        self.head = nn.Sequential(
            nn.Dropout(drop_rate), nn.Linear(self.pool.out_features, num_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone.forward_features(x)))
