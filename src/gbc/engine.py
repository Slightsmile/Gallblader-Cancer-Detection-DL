"""Cross-validated training loop producing out-of-fold predictions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import DIAGNOSIS_CLASSES, PATHOLOGY_CLASSES
from .dataset import ROIDataset, class_weights
from .models import ModelEMA, build_model
from .transforms import build_transforms


@dataclass
class TrainConfig:
    backbone: str = "efficientnet_b0"
    task: str = "diagnosis"
    image_size: int = 224
    clahe: bool = True
    epochs: int = 30
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    ema_decay: float = 0.98
    patience: int = 8
    balance_classes: bool = True
    tta_hflip: bool = True
    num_workers: int = 4
    seed: int = 1337
    folds: tuple[int, ...] = ()

    @property
    def num_outputs(self) -> int:
        return len(DIAGNOSIS_CLASSES) if self.task == "diagnosis" else len(PATHOLOGY_CLASSES)


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    best_val_score: float
    seconds: float
    history: list[dict] = field(default_factory=list)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _drain(device: torch.device) -> None:
    """Bound the number of in-flight Metal command buffers.

    Without this the MPS backend can exhaust its command-buffer pool part way
    through a fold and block indefinitely in ``-[_MTLCommandBuffer initWithQueue:]``.
    """
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


def _mixup(x: torch.Tensor, y: torch.Tensor, alpha: float, num_classes: int):
    """Returns mixed inputs and *soft* targets, so it works for both tasks."""
    soft = F.one_hot(y, num_classes).float() if y.ndim == 1 else y
    if alpha <= 0:
        return x, soft
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * soft + (1 - lam) * soft[perm]


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device, *, tta_hflip: bool) -> np.ndarray:
    model.eval()
    logits = []
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        out = model(images).float()
        if tta_hflip:
            out = (out + model(torch.flip(images, dims=[3])).float()) / 2
        logits.append(out.cpu())
    _drain(device)
    return torch.cat(logits).numpy()


def train_fold(
    index: pd.DataFrame, fold: int, cfg: TrainConfig, device: torch.device, out_dir: Path
) -> tuple[FoldResult, np.ndarray, pd.DataFrame]:
    """Train one fold; returns its result, the OOF logits and the held-out rows."""
    _seed_everything(cfg.seed + fold)
    train_df = index[index["fold"] != fold]
    val_df = index[index["fold"] == fold]

    train_ds = ROIDataset(train_df, build_transforms(cfg.image_size, train=True, clahe=cfg.clahe), cfg.task)
    val_ds = ROIDataset(val_df, build_transforms(cfg.image_size, train=False, clahe=cfg.clahe), cfg.task)
    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **loader_kw)

    model = build_model(cfg.backbone, cfg.num_outputs).to(device)
    ema = ModelEMA(model, cfg.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weights = class_weights(train_df).to(device) if cfg.balance_classes else None

    def loss_fn(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        if cfg.task == "pathology":
            return F.binary_cross_entropy_with_logits(logits, soft_targets)
        # Soft-target cross-entropy with label smoothing and optional class weights.
        n = soft_targets.size(1)
        targets = soft_targets * (1 - cfg.label_smoothing) + cfg.label_smoothing / n
        per_class = -(targets * F.log_softmax(logits, dim=1))
        if weights is not None:
            per_class = per_class * weights
        return per_class.sum(dim=1).mean()

    y_val = np.stack(val_df[list(PATHOLOGY_CLASSES)].to_numpy()) if cfg.task == "pathology" \
        else val_df["diagnosis_idx"].to_numpy()

    def score(logits: np.ndarray) -> float:
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score

        if cfg.task == "pathology":
            return float(np.mean([roc_auc_score(y_val[:, i], logits[:, i]) for i in range(logits.shape[1])]))
        return float(balanced_accuracy_score(y_val, logits.argmax(1)))

    result = FoldResult(fold=fold, best_epoch=-1, best_val_score=-np.inf, seconds=0.0)
    best_logits = np.zeros((len(val_df), cfg.num_outputs), dtype=np.float32)
    ckpt_path = out_dir / f"{cfg.backbone}_{cfg.task}_fold{fold}.pt"
    start = time.time()
    steps_per_epoch = max(len(train_loader), 1)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for step, (images, targets, _) in enumerate(train_loader):
            lr_scale = _cosine_lr(epoch + step / steps_per_epoch, cfg.epochs, cfg.warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * lr_scale
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            images, soft = _mixup(images, targets, cfg.mixup_alpha, cfg.num_outputs)
            loss = loss_fn(model(images), soft)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ema.update(model)
            running += loss.item()
            if step % 10 == 0:
                _drain(device)

        # With ~30 steps per epoch the EMA can lag the raw weights badly early on,
        # so both are scored and the better one is taken as the epoch's candidate.
        candidates = {"raw": model, "ema": ema.module}
        scored = {k: _predict(m, val_loader, device, tta_hflip=cfg.tta_hflip) for k, m in candidates.items()}
        which = max(scored, key=lambda k: score(scored[k]))
        logits = scored[which]
        val_score = score(logits)
        result.history.append(
            {"epoch": epoch, "train_loss": running / steps_per_epoch,
             "val_score": val_score, "weights": which}
        )
        if val_score > result.best_val_score:
            result.best_val_score, result.best_epoch, best_logits = val_score, epoch, logits
            torch.save({"state_dict": candidates[which].state_dict(), "config": asdict(cfg)}, ckpt_path)
        elif epoch - result.best_epoch >= cfg.patience:
            break

    result.seconds = time.time() - start
    return result, best_logits, val_df


def run_cv(index: pd.DataFrame, cfg: TrainConfig, out_dir: str | Path) -> dict:
    """Train every fold and persist out-of-fold logits plus a run summary."""
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = pick_device()
    folds = cfg.folds or tuple(sorted(index["fold"].unique()))

    frames, results = [], []
    for fold in folds:
        result, logits, val_df = train_fold(index, int(fold), cfg, device, out_dir / "checkpoints")
        results.append(result)
        frame = val_df[["image_id", "fold", "diagnosis_idx", *PATHOLOGY_CLASSES]].copy()
        for i in range(cfg.num_outputs):
            frame[f"logit_{i}"] = logits[:, i]
        frames.append(frame)
        print(f"[{cfg.backbone}/{cfg.task}] fold {fold}: "
              f"score={result.best_val_score:.4f} @epoch {result.best_epoch} ({result.seconds:.0f}s)", flush=True)

    oof = pd.concat(frames).sort_values("image_id").reset_index(drop=True)
    tag = f"{cfg.backbone}_{cfg.task}"
    oof.to_csv(out_dir / f"oof_{tag}.csv", index=False)
    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "folds": [asdict(r) for r in results],
        "mean_val_score": float(np.mean([r.best_val_score for r in results])),
        "std_val_score": float(np.std([r.best_val_score for r in results])),
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary
