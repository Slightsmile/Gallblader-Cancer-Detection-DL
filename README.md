# Gallbladder cancer classification from ultrasound — a leakage-aware study on GBCU

This repository contains a reproducible benchmark for classifying gallbladder
(GB) pathology from B-mode ultrasound ROI crops, together with a dataset audit
that explains why the results previously reported here were not valid.

> **Read this first.** The `data/` directory is a flattened re-encoding of the
> **GBCU** dataset (Basu et al., CVPR 2022). The `training/validation/test`
> split that ships with it leaks: **34.1 % of source images appear in more than
> one split**, and **209 source images are shared between train and test**. The
> previously reported 60.6–87.2 % accuracies were computed on that split and on
> an ill-posed 5-way label, and have been withdrawn. Details and evidence:
> [docs/data_leakage_audit.md](docs/data_leakage_audit.md).

## What the data actually is

2,294 PNGs are not 2,294 studies. They are **ROI crops from 1,255 source
ultrasound images of 218 patients** — the GBCU dataset, reconstructed exactly:

| class | reconstructed | published GBCU |
| --- | --- | --- |
| normal | 432 | 432 |
| benign | 558 | 558 |
| malignant | 265 | 265 |

The five directory names are **ROI kinds, not classes**: `nml`/`abn` are the
gallbladder region box, and `stn`/`bmt`/`malg` are stone / mural-thickening /
malignancy boxes drawn *inside* it (948 of 949 pairs are strictly nested). Two
well-posed tasks are recovered instead:

- **`diagnosis`** — 3-class normal / benign / malignant, one sample per source
  image, directly comparable to the published literature.
- **`pathology`** — 3-way **multi-label** stone / mural thickening / malignancy,
  which is what the nested boxes actually annotate. Not reported by any prior
  GBCU paper.

## Protocol

Stratified **5-fold cross-validation grouped on the source image**, so every
crop of an image shares a fold. Ensemble weights and stackers are fitted on the
four training folds and applied to the held-out fold. Every headline number
carries a 2,000-resample bootstrap CI and a per-fold standard deviation.
Full recipe: [docs/reproducibility.md](docs/reproducibility.md).

**Known limitation:** grouping on image ID removes source-image leakage but not
**patient** leakage — the flattened filenames carry no patient identifier, and
GBCU averages ~5.8 images per patient. Results here are therefore optimistic
relative to the patient-wise protocol used by Basu et al., and must be labelled
as grouped-by-image until the licensed GBCU release with patient IDs is
obtained. See [docs/data_leakage_audit.md](docs/data_leakage_audit.md) §5.

## Where the field stands

Under a **patient-wise** protocol on this same dataset:

| Method | 10-fold CV accuracy |
| --- | --- |
| Expert radiologists (test set) | 68.3–70.0 % |
| ResNet50 | 81.1 ± 3.1 % |
| InceptionV3 | 84.4 ± 3.9 % |
| GBCNet | 88.2 ± 5.1 % |
| **GBCNet + visual-acuity curriculum** | **92.1 ± 2.9 %** |

Recent papers reporting 96–98.5 % on GBCU use *random* image-level splits. See
[docs/literature_review.md](docs/literature_review.md).

Our controlled ablation puts the split defect at **+2.0 accuracy points**
(0.7293 grouped vs 0.7498 ungrouped, overlapping CIs) — smaller than the 5–30
points reported for duplicate-slice leakage elsewhere, because nested ROI crops
differ more from each other than duplicated frames do. The larger share of the
gap to the old 87.17 % came from the **ill-posed 5-way label**, not the split.

## Results

Out-of-fold, **nested** 5-fold CV grouped on the source image: the stopping epoch
and the raw/EMA choice are made on an inner split, and each outer fold is scored
exactly once. Full table with CIs, calibration and ablations:
[docs/results.md](docs/results.md).

| model | accuracy | 95 % CI | balanced acc. | macro AUC | malignant sens. |
| --- | --- | --- | --- | --- | --- |
| efficientnet_b0 | 0.7402 | [0.716, 0.764] | 0.7554 | 0.8976 | 0.8038 |
| resnet50 | 0.7474 | [0.723, 0.771] | 0.7793 | 0.9082 | 0.8679 |
| efficientnet_b0 **+ second-order pooling** | 0.8215 | [0.800, 0.842] | 0.8424 | 0.9465 | **0.9170** |
| **deit3_small** | 0.8821 | [0.864, 0.900] | **0.8873** | **0.9731** | 0.8981 |
| ensemble (equal soft vote) | 0.8550 | [0.835, 0.874] | 0.8686 | 0.9625 | 0.9094 |
| **ensemble (logistic stacking)** | **0.8853** | [0.868, 0.902] | 0.8805 | 0.9713 | 0.8453 |

Four findings matter more than the headline 88.5 %:

1. **The transformer beats both CNNs by 9–13 balanced points** (0.8873 vs 0.7793
   / 0.7554), consistent with FocusMAE's image baselines where DeiT and ViT
   outrank ResNet50.
2. **Second-order pooling is worth +8.7 balanced points** on the same backbone,
   same schedule, same everything (0.8424 vs 0.7554), in every fold. Average
   pooling discards the channel covariance, and malignancy on B-mode ultrasound
   *is* a texture statistic — an independent confirmation of the mechanism Basu
   et al. credit for GBCNet's margin, from a 4.2 M-parameter model.
3. **Ensembling did not help.** The stack ties the single best model
   (McNemar p = 0.69) and the fitted weights put **99.5 %** of the mass on
   `deit3_small`. Equal-weight averaging is *worse* (0.8550) than its own best
   member. A 16-model equal average, the method this repository used
   originally, cannot be expected to beat one good model.
4. **Model-selection bias is not a constant.** Selecting the epoch on the
   reported fold inflates the transformer by 0.4 points but the CNNs by 3–5.
   Under the wrong protocol the second-order model looks close to the
   transformer; under the right one the gap is real. See
   [docs/results.md](docs/results.md) and
   [docs/results_biased_v1.md](docs/results_biased_v1.md).

Reaching GBCNet's 92.1 % would need its remaining ingredient, the
visual-acuity curriculum, plus a patient-wise protocol to compare honestly.

## Usage

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src

python -m gbc.cli audit                 # reconstruct the dataset, report leakage
./run_study.sh                          # 5-fold CV for each backbone, then ensemble
python -m gbc.cli leakage-ablation      # measure what the leak is worth
python -m gbc.cli cam --checkpoint results/checkpoints/efficientnet_b0_diagnosis_fold0.pt
```

## Code map

| path | role |
| --- | --- |
| `src/gbc/data.py` | GBCU reconstruction, leakage audit, grouped/ungrouped folds |
| `src/gbc/transforms.py` | CLAHE, aspect-preserving letterbox, speckle-aware augmentation |
| `src/gbc/dataset.py` | torch datasets for both tasks, class weights |
| `src/gbc/models.py` | timm backbones, EMA |
| `src/gbc/engine.py` | cross-validated training loop, out-of-fold predictions |
| `src/gbc/ensemble.py` | in-fold soft voting, weighted voting, stacking, greedy selection |
| `src/gbc/metrics.py` | metrics, bootstrap CIs, McNemar, ECE, temperature scaling |
| `src/gbc/explain.py` | Grad-CAM overlays |
| `src/gbc/cli.py` | command line entry points |

The two original notebooks (`Gallblader_Cancer.ipynb`, `Ensemble_GBC_88_.ipynb`)
are retained for provenance. **Their reported accuracies are invalid** for the
reasons above; use the package instead.

## Publishing

[docs/paper_outline.md](docs/paper_outline.md) sets out the framing, section
plan, target venues, and a pre-submission checklist — including the licensing
and ethics obligations that come with GBCU.

## Dataset citation and licence

This work uses GBCU. Cite:

```bibtex
@inproceedings{basu2022surpassing,
  title     = {Surpassing the Human Accuracy: Detecting Gallbladder Cancer
               from USG Images with Curriculum Learning},
  author    = {Basu, Soumen and Gupta, Mayank and Rana, Pratyaksha and
               Gupta, Pankaj and Arora, Chetan},
  booktitle = {CVPR},
  pages     = {20886--20896},
  year      = {2022}
}
```

GBCU requires a signed licence agreement obtained from
<https://gbc-iitd.github.io/data/gbcu>. The data was collected at PGIMER
Chandigarh under institutional ethics approval with written informed consent;
any publication must cite that provenance rather than claim its own.

---

> Developed as part of the Research and Innovation Project at Daffodil
> International University.
