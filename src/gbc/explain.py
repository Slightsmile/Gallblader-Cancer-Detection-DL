"""Grad-CAM saliency for qualitative review by radiologists."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import build_model
from .transforms import build_transforms


def _last_conv(model: torch.nn.Module) -> torch.nn.Module:
    """Deepest module producing a spatial feature map."""
    candidates = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not candidates:
        raise ValueError("model has no Conv2d layer; Grad-CAM needs a convolutional backbone")
    return candidates[-1]


def grad_cam(model: torch.nn.Module, image: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
    """Return a ``[H, W]`` saliency map in ``[0, 1]`` for a single ``[3,H,W]`` input."""
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = _last_conv(model)
    handles = [
        layer.register_forward_hook(lambda _m, _i, o: activations.append(o)),
        layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0])),
    ]
    try:
        logits = model(image.unsqueeze(0))
        target = logits[0, class_idx if class_idx is not None else int(logits.argmax())]
        model.zero_grad(set_to_none=True)
        target.backward()
    finally:
        for handle in handles:
            handle.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    span = cam.max() - cam.min()
    return (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)


def overlay(image_path: str | Path, cam: np.ndarray, size: int = 224, alpha: float = 0.45) -> Image.Image:
    """Blend a saliency map over the letterboxed input as a JET-style heatmap."""
    from matplotlib import colormaps

    base = build_transforms(size, train=False).transforms[0](Image.open(image_path))
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(heat), alpha)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Rebuild a model from a checkpoint written by :func:`gbc.engine.train_fold`."""
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["config"]
    n_out = 3  # both tasks in this study have three outputs
    model = build_model(cfg["backbone"], n_out, pretrained=False)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), cfg
