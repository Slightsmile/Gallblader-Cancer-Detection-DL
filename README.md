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

Recent papers reporting 96–98.5 % on GBCU use *random* image-level splits.
Comparable inflation from improper splits has been measured at 5–30 accuracy
points elsewhere in medical imaging. See
[docs/literature_review.md](docs/literature_review.md).

## Results

Generated numbers live in `results/` (`results_table.csv`,
`ensemble_report.json`, `leakage_ablation/`). See
[docs/results.md](docs/results.md).

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
