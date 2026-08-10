"""Ensembling over out-of-fold predictions.

Every combiner is fitted **inside** the cross-validation folds it is scored on:
weights and stackers are learned on the four training folds of each split and
applied to the held-out fold, so the reported ensemble score is not optimistic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss


def load_oof(results_dir: str | Path, task: str = "diagnosis") -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load every ``oof_<backbone>_<task>.csv`` aligned on ``image_id``.

    Returns ``{backbone: probs[n, k]}`` and the shared metadata frame.
    """
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob(f"oof_*_{task}.csv"))
    if not paths:
        raise FileNotFoundError(f"no oof_*_{task}.csv under {results_dir}")

    probs, meta = {}, None
    for path in paths:
        frame = pd.read_csv(path, dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
        name = path.stem[len("oof_") : -len(f"_{task}")]
        logit_cols = [c for c in frame.columns if c.startswith("logit_")]
        if meta is None:
            meta = frame.drop(columns=logit_cols)
        elif not meta["image_id"].equals(frame["image_id"]):
            raise ValueError(f"{path} is not aligned with the other OOF files")
        probs[name] = softmax(frame[logit_cols].to_numpy(dtype=np.float64), axis=1)
    return probs, meta


def _weighted(probs: dict[str, np.ndarray], weights: np.ndarray, rows: np.ndarray) -> np.ndarray:
    stack = np.stack([p[rows] for p in probs.values()])
    return np.tensordot(weights, stack, axes=(0, 0))


def fit_weights(probs: dict[str, np.ndarray], y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Simplex-constrained weights minimising log-loss on ``rows``."""
    n = len(probs)
    x0 = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        w = np.abs(w)
        w = w / max(w.sum(), 1e-9)
        return log_loss(y[rows], np.clip(_weighted(probs, w, rows), 1e-9, 1), labels=range(probs[next(iter(probs))].shape[1]))

    best = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4})
    w = np.abs(best.x)
    return w / max(w.sum(), 1e-9)


def evaluate_combiners(
    probs: dict[str, np.ndarray], meta: pd.DataFrame, *, seed: int = 0
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Produce nested-CV probabilities for each combiner, plus a comparison table.

    Combiners: ``mean`` (equal soft vote), ``weighted`` (fold-fitted simplex
    weights) and ``stack`` (multinomial logistic regression on concatenated
    member probabilities).
    """
    y = meta["diagnosis_idx"].to_numpy()
    folds = meta["fold"].to_numpy()
    n_classes = next(iter(probs.values())).shape[1]
    names = list(probs)

    out = {
        "mean": np.zeros((len(y), n_classes)),
        "weighted": np.zeros((len(y), n_classes)),
        "stack": np.zeros((len(y), n_classes)),
    }
    fitted_weights = []
    features = np.concatenate([probs[n] for n in names], axis=1)

    for fold in sorted(set(folds)):
        train_rows = np.flatnonzero(folds != fold)
        test_rows = np.flatnonzero(folds == fold)

        out["mean"][test_rows] = _weighted(probs, np.full(len(names), 1 / len(names)), test_rows)

        w = fit_weights(probs, y, train_rows)
        fitted_weights.append(w)
        out["weighted"][test_rows] = _weighted(probs, w, test_rows)

        stacker = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        stacker.fit(features[train_rows], y[train_rows])
        out["stack"][test_rows] = stacker.predict_proba(features[test_rows])

    table = pd.DataFrame(
        [{"member": n, "mean_weight": float(np.mean([w[i] for w in fitted_weights])),
          "balanced_accuracy": balanced_accuracy_score(y, probs[n].argmax(1))}
         for i, n in enumerate(names)]
    ).sort_values("balanced_accuracy", ascending=False)
    return out, table


def greedy_selection(
    probs: dict[str, np.ndarray], meta: pd.DataFrame, *, max_members: int | None = None
) -> list[str]:
    """Caruana-style greedy forward selection with replacement, scored out-of-fold."""
    y = meta["diagnosis_idx"].to_numpy()
    names = list(probs)
    max_members = max_members or 2 * len(names)
    chosen: list[str] = []
    best_score = -np.inf
    while len(chosen) < max_members:
        candidates = []
        for name in names:
            trial = chosen + [name]
            blended = np.mean([probs[m] for m in trial], axis=0)
            candidates.append((balanced_accuracy_score(y, blended.argmax(1)), name))
        score, name = max(candidates)
        if score <= best_score:
            break
        best_score, _ = score, chosen.append(name)
    return chosen
