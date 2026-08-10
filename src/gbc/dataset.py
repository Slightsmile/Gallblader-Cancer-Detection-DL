"""Torch datasets over the reconstructed GBCU index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import DIAGNOSIS_CLASSES, PATHOLOGY_CLASSES


class ROIDataset(Dataset):
    """Yields ``(image, target, image_id)`` for one row of the index per item.

    ``task='diagnosis'`` gives an integer class in ``DIAGNOSIS_CLASSES``;
    ``task='pathology'`` gives a 3-dim multi-hot vector over ``PATHOLOGY_CLASSES``.
    """

    def __init__(self, frame: pd.DataFrame, transform, task: str = "diagnosis"):
        if task not in {"diagnosis", "pathology"}:
            raise ValueError(f"unknown task {task!r}")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.task = task

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        image = self.transform(Image.open(row["roi_path"]))
        if self.task == "diagnosis":
            target = torch.tensor(int(row["diagnosis_idx"]), dtype=torch.long)
        else:
            target = torch.tensor([float(row[c]) for c in PATHOLOGY_CLASSES], dtype=torch.float32)
        return image, target, row["image_id"]


def class_weights(frame: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weights over ``DIAGNOSIS_CLASSES``, normalised to mean 1."""
    counts = np.array([(frame["diagnosis_idx"] == i).sum() for i in range(len(DIAGNOSIS_CLASSES))],
                      dtype=np.float64)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)
