"""Shortcut baselines: how much of the task is solvable without looking at tissue?

The parent gallbladder ROI is a radiologist-drawn box, and radiologists draw
bigger boxes around bigger lesions. If crop *geometry* alone predicts the
diagnosis, then part of any model's accuracy is annotation shortcut rather than
image understanding — a confound reviewers of ROI-based ultrasound papers ask
about, and one no prior GBCU paper reports.

Two baselines, both evaluated under the same grouped folds as the real models:

``geometry``
    Gradient boosting on width, height, area, aspect ratio and log-area.
``intensity``
    The above plus 8 global grey-level statistics (mean, std, percentiles).

Their accuracy is the floor a real model must clear to be interesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import DIAGNOSIS_CLASSES

GEOMETRY_FEATURES = ["width", "height", "area", "aspect", "log_area"]
INTENSITY_FEATURES = ["mean", "std", "p05", "p25", "p50", "p75", "p95", "entropy"]


def extract_features(index: pd.DataFrame) -> pd.DataFrame:
    """Cheap per-crop descriptors that involve no learned representation."""
    rows = []
    for path in index["roi_path"]:
        with Image.open(path) as handle:
            width, height = handle.size
            grey = np.asarray(handle.convert("L"), dtype=np.float32)
        histogram = np.bincount(grey.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
        histogram /= max(histogram.sum(), 1.0)
        nonzero = histogram[histogram > 0]
        rows.append({
            "width": width, "height": height, "area": width * height,
            "aspect": width / max(height, 1), "log_area": float(np.log(width * height)),
            "mean": float(grey.mean()), "std": float(grey.std()),
            **{f"p{q:02d}": float(np.percentile(grey, q)) for q in (5, 25, 50, 75, 95)},
            "entropy": float(-(nonzero * np.log2(nonzero)).sum()),
        })
    return pd.DataFrame(rows, index=index.index)


def run_shortcut_baselines(index: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold accuracy of each shortcut baseline, using ``index['fold']``."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    features = extract_features(index)
    y = index["diagnosis_idx"].to_numpy()
    folds = index["fold"].to_numpy()

    results = []
    for name, columns in [("geometry only", GEOMETRY_FEATURES),
                          ("geometry + global intensity", GEOMETRY_FEATURES + INTENSITY_FEATURES)]:
        x = features[columns].to_numpy()
        predictions = np.zeros(len(y), dtype=int)
        for fold in sorted(set(folds)):
            train, test = folds != fold, folds == fold
            model = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
            model.fit(x[train], y[train])
            predictions[test] = model.predict(x[test])
        results.append({
            "baseline": name,
            "n_features": len(columns),
            "accuracy": float(accuracy_score(y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        })

    majority = max(np.bincount(y)) / len(y)
    results.insert(0, {"baseline": "majority class", "n_features": 0,
                       "accuracy": float(majority),
                       "balanced_accuracy": 1.0 / len(DIAGNOSIS_CLASSES)})
    return pd.DataFrame(results)
