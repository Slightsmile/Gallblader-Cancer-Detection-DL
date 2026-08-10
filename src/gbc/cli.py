"""Command line entry points for the GBCU study.

Examples
--------
    python -m gbc.cli audit
    python -m gbc.cli train --backbone efficientnet_b0 --epochs 20
    python -m gbc.cli ensemble
    python -m gbc.cli cam --checkpoint results/checkpoints/... --n 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    DIAGNOSIS_CLASSES,
    assign_folds,
    assign_folds_ungrouped,
    audit_leakage,
    build_crop_index,
    build_index,
)
from .engine import TrainConfig, run_cv
from .ensemble import evaluate_combiners, greedy_selection, load_oof
from .metrics import bootstrap_ci, compute_metrics, mcnemar_test


def _index(args) -> pd.DataFrame:
    return assign_folds(build_index(args.data), n_splits=args.folds, seed=args.seed)


def cmd_audit(args) -> None:
    report = audit_leakage(args.data)
    index = _index(args)
    print(report.to_markdown())
    print("\nReconstructed GBCU diagnosis distribution:")
    print(index["diagnosis"].value_counts().reindex(DIAGNOSIS_CLASSES).to_string())
    out = Path(args.out) / "leakage_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.__dict__ | {"leaked_fraction": report.leaked_fraction}, indent=2, default=str))
    index.to_csv(Path(args.out) / "index.csv", index=False)
    print(f"\nwrote {out} and {Path(args.out) / 'index.csv'}")


def cmd_train(args) -> None:
    cfg = TrainConfig(
        backbone=args.backbone,
        task=args.task,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        clahe=not args.no_clahe,
        mixup_alpha=args.mixup,
        num_workers=args.workers,
        seed=args.seed,
    )
    summary = run_cv(_index(args), cfg, args.out)
    print(f"mean CV score: {summary['mean_val_score']:.4f} +/- {summary['std_val_score']:.4f}")


def cmd_ensemble(args) -> None:
    probs, meta = load_oof(args.out, task="diagnosis")
    y = meta["diagnosis_idx"].to_numpy()
    combined, members = evaluate_combiners(probs, meta)

    rows = []
    for name, p in {**probs, **combined}.items():
        m = compute_metrics(p, y).as_dict()
        ci = bootstrap_ci(p, y, n_resamples=args.bootstrap)
        rows.append({"model": name, **m,
                     "accuracy_ci": f"[{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}]"})
    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    table.to_csv(Path(args.out) / "results_table.csv", index=False)

    best_single = max(probs, key=lambda n: compute_metrics(probs[n], y).accuracy)
    best_ens = max(combined, key=lambda n: compute_metrics(combined[n], y).accuracy)
    test = mcnemar_test(y, combined[best_ens].argmax(1), probs[best_single].argmax(1))

    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nmember weights:\n{members.to_string(index=False)}")
    print(f"\ngreedy selection: {greedy_selection(probs, meta)}")
    print(f"\nMcNemar {best_ens} vs {best_single}: {test}")
    (Path(args.out) / "ensemble_report.json").write_text(
        json.dumps({"best_single": best_single, "best_ensemble": best_ens,
                    "mcnemar": test, "table": rows}, indent=2))
    np.save(Path(args.out) / "ensemble_probs.npy", combined[best_ens])


def cmd_leakage_ablation(args) -> None:
    """Quantify how much accuracy source-image leakage buys, all else held fixed."""
    crops = build_crop_index(args.data)
    schemes = {
        "grouped (leakage-free)": assign_folds(crops, n_splits=args.cv_folds, seed=args.seed),
        "ungrouped (reproduces shipped split)": assign_folds_ungrouped(
            crops, n_splits=args.cv_folds, seed=args.seed
        ),
    }
    out_dir = Path(args.out) / "leakage_ablation"
    rows = []
    for name, frame in schemes.items():
        slug = "grouped" if name.startswith("grouped") else "ungrouped"
        cfg = TrainConfig(backbone=args.backbone, epochs=args.epochs, num_workers=args.workers,
                          seed=args.seed)
        summary = run_cv(frame, cfg, out_dir / slug)
        oof = pd.read_csv(out_dir / slug / f"oof_{args.backbone}_diagnosis.csv")
        logits = oof[[c for c in oof.columns if c.startswith("logit_")]].to_numpy()
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        y = oof["diagnosis_idx"].to_numpy()
        metrics = compute_metrics(probs, y).as_dict()
        ci = bootstrap_ci(probs, y, n_resamples=1000)
        rows.append({"split_scheme": name, **metrics,
                     "accuracy_ci_low": ci["accuracy"][0], "accuracy_ci_high": ci["accuracy"][1],
                     "cv_std": summary["std_val_score"]})

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "leakage_ablation.csv", index=False)
    inflation = table.loc[1, "accuracy"] - table.loc[0, "accuracy"]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\naccuracy inflation attributable to source-image leakage: "
          f"{inflation:+.4f} ({inflation * 100:+.1f} points)")
    (out_dir / "leakage_ablation.json").write_text(
        json.dumps({"rows": rows, "inflation_accuracy": float(inflation)}, indent=2))


def cmd_cam(args) -> None:
    import torch
    from PIL import Image

    from .explain import grad_cam, load_checkpoint, overlay
    from .transforms import build_transforms

    model, cfg = load_checkpoint(args.checkpoint)
    index = _index(args)
    transform = build_transforms(cfg["image_size"], train=False, clahe=cfg["clahe"])
    out_dir = Path(args.out) / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    picks = (index.groupby("diagnosis", group_keys=False)
             .apply(lambda g: g.sample(min(args.n, len(g)), random_state=args.seed)))
    for _, row in picks.iterrows():
        tensor = transform(Image.open(row["roi_path"]))
        with torch.enable_grad():
            cam = grad_cam(model, tensor)
        overlay(row["roi_path"], cam, size=cfg["image_size"]).save(
            out_dir / f"{row['diagnosis']}_{row['image_id']}.png")
    print(f"wrote {len(picks)} Grad-CAM overlays to {out_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gbc", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit", help="report leakage in the shipped split").set_defaults(func=cmd_audit)

    ablate = sub.add_parser(
        "leakage-ablation",
        help="train the same model on crop-level folds with and without grouping on image_id",
    )
    ablate.add_argument("--backbone", default="efficientnet_b0")
    ablate.add_argument("--epochs", type=int, default=10)
    ablate.add_argument("--cv-folds", type=int, default=3)
    ablate.add_argument("--workers", type=int, default=0)
    ablate.set_defaults(func=cmd_leakage_ablation)

    train = sub.add_parser("train", help="cross-validated training of one backbone")
    train.add_argument("--backbone", default="efficientnet_b0")
    train.add_argument("--task", default="diagnosis", choices=["diagnosis", "pathology"])
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--mixup", type=float, default=0.2)
    train.add_argument("--no-clahe", action="store_true")
    train.add_argument("--workers", type=int, default=0)
    train.set_defaults(func=cmd_train)

    ens = sub.add_parser("ensemble", help="combine OOF predictions and report metrics")
    ens.add_argument("--bootstrap", type=int, default=2000)
    ens.set_defaults(func=cmd_ensemble)

    cam = sub.add_parser("cam", help="write Grad-CAM overlays")
    cam.add_argument("--checkpoint", required=True)
    cam.add_argument("--n", type=int, default=6)
    cam.set_defaults(func=cmd_cam)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
