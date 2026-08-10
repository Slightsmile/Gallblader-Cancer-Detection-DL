"""Evaluation: point metrics, bootstrap confidence intervals, significance, calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .data import DIAGNOSIS_CLASSES

MALIGNANT = DIAGNOSIS_CLASSES.index("malignant")


@dataclass
class Metrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    macro_auc: float
    cohen_kappa: float
    malignant_sensitivity: float
    malignant_specificity: float
    expected_calibration_error: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Standard equal-width ECE over the top-1 confidence."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def compute_metrics(probs: np.ndarray, y_true: np.ndarray) -> Metrics:
    """All headline metrics for a 3-class diagnosis prediction."""
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(DIAGNOSIS_CLASSES)))
    tp = cm[MALIGNANT, MALIGNANT]
    fn = cm[MALIGNANT].sum() - tp
    fp = cm[:, MALIGNANT].sum() - tp
    tn = cm.sum() - tp - fn - fp
    try:
        macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:  # a class missing from a bootstrap resample
        macro_auc = float("nan")
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_auc=macro_auc,
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
        malignant_sensitivity=float(tp / max(tp + fn, 1)),
        malignant_specificity=float(tn / max(tn + fp, 1)),
        expected_calibration_error=expected_calibration_error(probs, y_true),
    )


def bootstrap_ci(
    probs: np.ndarray, y_true: np.ndarray, *, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs over cases, for every field of :class:`Metrics`."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    n = len(y_true)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        for key, value in compute_metrics(probs[idx], y_true[idx]).as_dict().items():
            samples.setdefault(key, []).append(value)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))) for k, v in samples.items()
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on the discordant pairs of two classifiers."""
    from scipy.stats import binomtest

    a_only = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    b_only = int(((pred_b == y_true) & (pred_a != y_true)).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"a_correct_b_wrong": a_only, "b_correct_a_wrong": b_only, "p_value": p}


def temperature_scale(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a single temperature by minimising NLL (Guo et al., 2017)."""
    import torch

    t = torch.ones(1, requires_grad=True)
    lg = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y_true, dtype=torch.long)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / t.clamp(min=1e-2), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=1e-2).item())
