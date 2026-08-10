"""Preprocessing and augmentation for B-mode ultrasound ROI crops.

Ultrasound crops are grayscale, vary widely in aspect ratio (801-1556 x 564-947
in the source images, far more after ROI cropping) and are dominated by speckle.
The pipeline therefore (a) letterboxes rather than squashes, so lesion geometry
survives, (b) optionally applies CLAHE, the standard contrast step in the GBC
ultrasound literature, and (c) augments with speckle-aware noise/blur in
addition to the usual geometric jitter.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_clahe(image: Image.Image, clip_limit: float = 2.0, grid: int = 8) -> Image.Image:
    """Contrast-limited adaptive histogram equalisation, implemented with numpy.

    Avoids an OpenCV dependency: the image is tiled into ``grid x grid`` blocks,
    each block's histogram is clipped and redistributed, and the per-block
    transfer functions are bilinearly interpolated across the image.
    """
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = array.shape
    tile_h, tile_w = max(height // grid, 1), max(width // grid, 1)
    clip = max(int(clip_limit * tile_h * tile_w / 256), 1)

    luts = np.empty((grid, grid, 256), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            block = array[i * tile_h : (i + 1) * tile_h or None, j * tile_w : (j + 1) * tile_w or None]
            hist = np.bincount(block.ravel(), minlength=256).astype(np.float32)
            excess = np.maximum(hist - clip, 0).sum()
            hist = np.minimum(hist, clip) + excess / 256.0
            cdf = np.cumsum(hist)
            luts[i, j] = 255.0 * cdf / max(cdf[-1], 1e-6)

    # Bilinear interpolation of the per-tile LUTs over pixel positions.
    ys = np.clip((np.arange(height) + 0.5) / tile_h - 0.5, 0, grid - 1)
    xs = np.clip((np.arange(width) + 0.5) / tile_w - 0.5, 0, grid - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, grid - 1), np.minimum(x0 + 1, grid - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]

    idx = array
    top = luts[y0[:, None], x0[None, :], idx] * (1 - wx) + luts[y0[:, None], x1[None, :], idx] * wx
    bot = luts[y1[:, None], x0[None, :], idx] * (1 - wx) + luts[y1[:, None], x1[None, :], idx] * wx
    out = top * (1 - wy) + bot * wy
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


class LoadROI:
    """PIL loader: grayscale -> optional CLAHE -> letterbox to a square -> RGB."""

    def __init__(self, size: int, clahe: bool = True, pad_value: int = 0):
        self.size = size
        self.clahe = clahe
        self.pad_value = pad_value

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        if self.clahe:
            image = apply_clahe(image)
        width, height = image.size
        scale = self.size / max(width, height)
        resized = image.resize((max(round(width * scale), 1), max(round(height * scale), 1)), Image.BICUBIC)
        canvas = Image.new("L", (self.size, self.size), self.pad_value)
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return canvas.convert("RGB")


class SpeckleNoise(torch.nn.Module):
    """Multiplicative noise, the first-order model of ultrasound speckle."""

    def __init__(self, sigma: float = 0.08, p: float = 0.3):
        super().__init__()
        self.sigma, self.p = sigma, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        return (x * (1.0 + torch.randn_like(x) * self.sigma)).clamp_(0.0, 1.0)


def build_transforms(size: int = 224, *, train: bool, clahe: bool = True) -> v2.Compose:
    load = LoadROI(size, clahe=clahe)
    if not train:
        return v2.Compose([load, v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                           v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return v2.Compose([
        load,
        v2.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=12, translate=(0.05, 0.05), shear=5),
        v2.ColorJitter(brightness=0.25, contrast=0.25),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.25),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        SpeckleNoise(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
