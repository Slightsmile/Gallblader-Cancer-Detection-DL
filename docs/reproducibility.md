# Reproducing the study

## Environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
```

Reference environment: macOS 15 (Apple M2, 10-core GPU), Python 3.12,
PyTorch 2.13 on the MPS backend, timm 1.0.28. The device is chosen automatically
by `gbc.engine.pick_device` (CUDA → MPS → CPU); on this machine MPS is ~13×
faster than CPU for the training step, so a CPU-only run is not practical.

## The pipeline

```bash
python -m gbc.cli audit                     # dataset reconstruction + leakage report
python -m gbc.cli train --backbone efficientnet_b0 --epochs 14
python -m gbc.cli train --backbone resnet50 --epochs 14
python -m gbc.cli train --backbone deit3_small_patch16_224 --epochs 14
python -m gbc.cli ensemble                  # nested-CV combiners, CIs, McNemar
python -m gbc.cli leakage-ablation          # grouped vs ungrouped folds
python -m gbc.cli cam --checkpoint results/checkpoints/efficientnet_b0_diagnosis_fold0.pt
```

`./run_study.sh` runs the training and ensembling steps end to end.

## What is fixed and what is not

Seeded: `random`, `numpy`, `torch` (`seed + fold` per fold), the fold assignment
(`StratifiedGroupKFold(random_state=seed)`), and the stacker. Not bit-exact:
MPS kernels are not deterministic and cuDNN autotuning is not disabled, so
expect variation of a few tenths of a point between runs on the same seed. All
headline numbers are therefore reported with per-fold standard deviations and
bootstrap confidence intervals rather than as single values.

## Protocol summary for the Methods section

| item | value |
| --- | --- |
| unit of analysis | source ultrasound image (1,255), input = parent gallbladder ROI crop |
| task | 3-class normal / benign / malignant (`diagnosis`); 3-label stone / mural thickening / malignancy (`pathology`) |
| cross-validation | stratified 5-fold, **grouped on `image_id`**, seed 1337 |
| preprocessing | grayscale → CLAHE (clip 2.0, 8×8) → aspect-preserving letterbox to 224² → RGB → ImageNet normalisation |
| augmentation | RandomResizedCrop(0.7–1.0), hflip, affine(±12°, 5 % translate, 5° shear), brightness/contrast 0.25, Gaussian blur p=0.25, multiplicative speckle σ=0.08 p=0.3, RandomErasing p=0.25 |
| regularisation | label smoothing 0.1, mixup α=0.2, dropout 0.3, grad-norm clip 5.0 |
| class imbalance | inverse-frequency class weights, normalised to mean 1 |
| optimiser | AdamW, lr 2e-4, weight decay 1e-4 |
| schedule | 3-epoch linear warmup then cosine decay over 14 epochs |
| weight averaging | EMA (decay 0.98); raw and EMA weights both scored each epoch, better one kept |
| cross-validation | **nested**: outer = stratified 5-fold grouped on `image_id`; inner = a further grouped 80/20 split of the four training folds |
| model selection | best epoch, and raw vs. EMA weights, chosen on the **inner** split only (balanced accuracy; macro AUC for `pathology`), patience 8 |
| reporting | the outer fold is scored **exactly once**, with the model selected above, and never influences training or selection |
| inference | horizontal-flip TTA |
| ensembling | equal soft vote, simplex-weighted vote (Nelder–Mead on log-loss), multinomial logistic stacking — all fitted inside the folds |
| statistics | 2,000-resample percentile bootstrap CIs; exact McNemar between best ensemble and best single model; ECE (15 equal-width bins) + temperature scaling |

## Artefacts written to `results/`

| file | contents |
| --- | --- |
| `index.csv` | reconstructed per-image index with fold assignment |
| `leakage_report.json` | machine-readable version of the audit |
| `oof_<backbone>_diagnosis.csv` | out-of-fold logits, one row per source image |
| `summary_<backbone>_diagnosis.json` | config, per-fold best epoch, inner selection score, outer score, runtime, mean ± SD |
| `checkpoints/<backbone>_diagnosis_fold<k>.pt` | selected weights + config |
| `results_table.csv` | all models and combiners with metrics and accuracy CIs |
| `ensemble_report.json` | best single, best combiner, McNemar result |
| `leakage_ablation/leakage_ablation.csv` | grouped vs ungrouped comparison |
| `gradcam/` | saliency overlays |

## Compute budget

On the reference M2, one 5-fold `efficientnet_b0` run at 14 epochs takes roughly
an hour; `resnet50` and `deit3_small_patch16_224` take longer. The whole study
as scripted is a several-hour single-GPU job — deliberately modest, so the
results are reproducible by a reviewer without cluster access.
