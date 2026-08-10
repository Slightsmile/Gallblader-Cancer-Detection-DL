# Data audit: what `data/` actually contains, and why the 87.17 % result is not reportable

Run `python -m gbc.cli audit` to regenerate every number on this page.

## 1. The directory is a flattened re-encoding of GBCU

`data/` holds **2,294 PNG files** whose names follow
`<row>_im<image_id>_<roi_kind>_<k>.png`. Parsing them (`gbc.data.scan_crops`)
shows they are not 2,294 independent studies. They are **ROI crops taken from
1,255 source ultrasound images**, and those 1,255 images reconstruct the
published GBCU dataset exactly:

| class | reconstructed | published GBCU |
| --- | --- | --- |
| normal | 432 | 432 |
| benign | 558 | 558 |
| malignant | 265 | 265 |

`gbc.data.build_index` asserts this equality, so the reconstruction cannot
silently drift.

The five folder names are **ROI kinds, not classes**:

| kind | what the box is | count |
| --- | --- | --- |
| `nml` | gallbladder ROI of a normal study | 432 |
| `abn` | gallbladder ROI of an abnormal study — the *parent* box | 823 |
| `stn` | stone box | 502 |
| `bmt` | benign mural-thickening box | 182 |
| `malg` | malignancy box | 265 |

Label co-occurrence over source images is perfectly hierarchical: 432 images
carry `nml` alone, and every other image carries `abn` plus at least one of
`{stn, bmt, malg}` — `(abn, stn)` 376, `(abn, malg)` 176, `(abn, bmt)` 145,
`(abn, malg, stn)` 89, `(abn, bmt, stn)` 37. `bmt` and `malg` never co-occur.

## 2. The three defects

### 2.1 Cross-split contamination

| quantity | value |
| --- | --- |
| ROI crops on disk | 2,294 |
| distinct source images | 1,255 |
| crops in `training/` / `validation/` / `test/` | 1,605 / 346 / 343 |
| source images appearing in **more than one split** | **428 (34.1 %)** |
| source images shared by `training` **and** `test` | **209** |
| source images shared by `training` and `validation` | 216 |
| source images shared by `validation` and `test` | 53 |

For 209 of the 314 source images represented in `test/`, the model has already
seen another crop of the very same ultrasound frame during training.

### 2.2 The crops are nested, so the leak is spatial, not merely statistical

Of 949 (pathology box, parent box) pairs from the same source image, **948
(99.9 %)** have the pathology box strictly smaller in area than the parent `abn`
box. The `stn`/`bmt`/`malg` crops are sub-regions *inside* the `abn` crop. A
`malg` crop in the training set and the `abn` crop of the same image in the test
set are overlapping views of the same pixels.

### 2.3 The 5-way label is ill-posed

**823 source images carry crops with mutually contradictory 5-way labels** —
e.g. image `00086` contributes `1074_im00086_malg_0.png` (label `malg`,
`training/`) and `1075_im00086_abn_1.png` (label `abn`, `validation/`). A 5-way
softmax over `{nml, abn, stn, bmt, malg}` is being asked to separate a region
from its own sub-region, with no consistent target. Accuracy against such a
target is not a meaningful quantity regardless of the split.

## 3. Consequence for the previously reported results

The README's 60.64 %–87.17 % table was produced on the split described above.
Those numbers are **jointly** affected by (2.1) and (2.3) and should not be
submitted anywhere. The published patient-wise state of the art on this same
data is GBCNet at **92.1 ± 2.9 %** 10-fold CV accuracy (Basu et al., CVPR 2022),
against expert radiologists at 68–70 %.

### How much of the gap is leakage? We measured it.

`python -m gbc.cli leakage-ablation` trains one backbone twice on the identical
2,294 crops with identical hyperparameters, varying only whether folds are
grouped on `image_id`:

| split scheme | accuracy | 95 % CI | balanced acc. | malignant sens. |
| --- | --- | --- | --- | --- |
| grouped (leakage-free) | 0.7293 | [0.711, 0.747] | 0.7708 | 0.7532 |
| ungrouped (reproduces the shipped split) | 0.7498 | [0.733, 0.767] | 0.7878 | 0.7611 |

**Source-image leakage is worth +2.0 accuracy points** — real in direction and
consistent across every metric, but modest, and the two confidence intervals
overlap. This is *smaller* than the 5–30 point inflation Tampu et al. measured
for improper splits in OCT classification (*Scientific Data* 2022), and the
difference is instructive: their leak duplicated near-identical slices, whereas
the crops here are nested sub-regions that look substantially different from
their parent box, so a model gains less from having seen one of them.

**Do not therefore conclude that the shipped results were only 2 points
optimistic.** The ablation isolates defect 2.1 alone, holding the task fixed at
the well-posed 3-class label. The larger part of the gap between the previously
reported 87.17 % and the 78.3 % measured here comes from defect 2.3 — the
5-way label — and the two cannot be decomposed further, because the label
spaces differ and the numbers are not measuring the same quantity. That is
precisely why the old figure is uninterpretable rather than merely inflated.

## 4. The corrected protocol used in this repository

1. **Reconstruct source images.** One sample per source image
   (`gbc.data.build_index`), input = the parent gallbladder ROI (`abn` or `nml`
   crop) — the same region GBCNet classifies.
2. **Two well-posed targets.**
   - `diagnosis`: 3-class normal / benign / malignant, matching GBCU and
     directly comparable to published numbers.
   - `pathology`: 3-way *multi-label* stone / mural thickening / malignancy,
     which is what the nested boxes actually annotate.
3. **Stratified grouped 5-fold CV** on `image_id`
   (`gbc.data.assign_folds`), so all crops of a source image share a fold.
4. **Every combiner fitted inside the folds** — ensemble weights and stackers
   are learned on the four training folds and applied to the held-out fold
   (`gbc.ensemble.evaluate_combiners`).
5. **Uncertainty always reported**: 2,000-resample percentile bootstrap CIs,
   per-fold standard deviations, and exact McNemar tests between models.

## 5. The residual limitation — state this in the paper

Grouping on `image_id` removes source-image leakage. It does **not** remove
**patient** leakage. GBCU's 1,255 images come from 218 patients (≈ 5.8 images
per patient), and the flattened filenames shipped here carry no patient
identifier, so patient-wise folds cannot be reconstructed from this directory.

Basu et al. split patient-wise. Our grouped-by-image results are therefore
**still optimistic relative to a patient-wise protocol** and are not strictly
comparable to the 92.1 % figure. Two honest responses, in order of preference:

1. **Obtain the licensed GBCU release** from
   <https://gbc-iitd.github.io/data/gbcu>, which carries patient IDs, and rerun
   `assign_folds(..., group_col="patient_id")` — the code already takes the
   column as a parameter. Note the license: requests are accepted from permanent
   faculty/staff, not students directly.
2. If (1) is not possible before submission, report the grouped-by-image numbers
   **labelled as such**, state this limitation in the Limitations section, and
   do not claim parity with patient-wise results.

Submitting grouped-by-image numbers as if they were patient-wise would repeat,
in a smaller way, the error this audit documents.
