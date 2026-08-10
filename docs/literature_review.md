# Related work: deep learning for gallbladder cancer on B-mode ultrasound

*Compiled August 2026. Every number below is quoted from the cited source; the
protocol column is the part reviewers care about most and is reported verbatim
where the source states it.*

## 1. The benchmark this repository actually uses

The `data/` directory of this repository is a re-encoding of **GBCU** (Basu et
al., CVPR 2022) — the first public GBC ultrasound dataset. Our reconstruction in
`gbc.data.build_index` recovers its published composition exactly:

| | images | patients |
| --- | --- | --- |
| normal | 432 | 71 |
| benign | 558 | 100 |
| malignant | 265 | 47 |
| **total** | **1,255** | **218** |

Images are grayscale B-mode scans (801–1,556 × 564–947 px) acquired on a Logiq
S8 at PGIMER Chandigarh, with **biopsy-proven** labels and radiologist-drawn
axis-aligned boxes for the gallbladder plus adjacent liver parenchyma, and
further boxes for stones, mural thickening and malignancy. The provenance
matters: because the labels are biopsy-confirmed, GBCU is one of very few
ultrasound benchmarks where a 90 %+ claim is meaningful rather than an artefact
of noisy reference standards.

Any submission using this data **must** cite Basu et al. and comply with the
GBCU license (signed agreement, requests accepted only from permanent
faculty/staff). See `docs/data_leakage_audit.md` §5.

## 2. Published results on GBCU

Basu et al. report a patient-wise 1,133/122 train/test split *and* 10-fold
patient-wise cross-validation. This is the protocol our study matches as closely
as the redistributed crops allow.

| Method | Test acc. | Test spec. | Test sens. | 10-fold CV acc. | CV spec. | CV sens. |
| --- | --- | --- | --- | --- | --- | --- |
| Radiologist A | 70.0 | 87.3 | 70.7 | – | – | – |
| Radiologist B | 68.3 | 81.1 | 73.2 | – | – | – |
| VGG16 | 62.3 | 90.0 | 38.1 | 69.3 ± 3.6 | 96.0 ± 4.6 | 49.5 ± 23.4 |
| ResNet50 | 76.2 | 87.5 | 61.9 | 81.1 ± 3.1 | 92.6 ± 6.9 | 67.2 ± 14.7 |
| InceptionV3 | 77.9 | 87.5 | 80.1 | 84.4 ± 3.9 | 95.3 ± 2.9 | 80.7 ± 9.7 |
| Faster-RCNN | 71.3 | 76.2 | 81.0 | 75.7 ± 5.3 | 84.0 ± 4.6 | 80.8 ± 10.4 |
| RetinaNet | 75.4 | 86.3 | 78.6 | 74.9 ± 7.3 | 86.7 ± 7.8 | 79.1 ± 8.9 |
| EfficientDet | 58.2 | 86.3 | 62.0 | 73.9 ± 8.4 | 88.1 ± 9.9 | 85.8 ± 6.1 |
| GBCNet (ROI + MS-SoP) | 87.7 | 90.0 | 92.9 | 88.2 ± 5.1 | 94.2 ± 3.7 | 92.3 ± 7.1 |
| **GBCNet + visual-acuity curriculum** | **91.0** | **95.0** | **97.6** | **92.1 ± 2.9** | 96.7 ± 2.3 | 91.9 ± 6.3 |

Source: Basu et al., *Surpassing the Human Accuracy: Detecting Gallbladder
Cancer from USG Images with Curriculum Learning*, CVPR 2022, Table 1.

**Two conclusions that should shape any new GBCU paper.** First, expert
radiologists score 68–70 % on this data, so the clinically interesting
comparison is against ~70 %, not against 50 %. Second, a plain ResNet50 gets
81.1 % under patient-wise CV — a paper reporting ~87 % from a bag of stock
ImageNet backbones is *below* the 2022 state of the art and will be desk
rejected unless it contributes something else.

## 3. The wider GBC-on-ultrasound literature

| Work | Venue / year | Data | Protocol | Headline result |
| --- | --- | --- | --- | --- |
| GBCNet (Basu et al.) | CVPR 2022 | GBCU, 1,255 img / 218 pt | patient-wise split + 10-fold patient-wise CV | 92.1 ± 2.9 % CV acc. |
| RadFormer (Basu et al.) | Medical Image Analysis 2023 | GBCU | patient-wise | global–local transformer attention; ~92 % acc., interpretable |
| FocusMAE (Basu et al.) | CVPR 2024 | GBUSV + 27 extra videos | 5-fold **patient-wise** CV | 96.4 % acc. (video); image models on the same data: GBCNet 84.0 ± 10.5, RadFormer 84.0 ± 10.5, DeiT 82.9 ± 3.4, ViT 79.6 ± 6.8 |
| GBCHV (Sci. Rep. 2025) | Scientific Reports 2025 | GBCU, 1,005/126/125 | **random** split, single run | 96.21 % acc., 99.68 % AUC |
| GallNet (preprint 2025) | preprint | multi-class GB disease | not patient-wise per the report | 98.50 ± 0.05 % acc. |
| MSFE-GallNet-X | BMC Med. Imaging 2025 | grayscale GB ultrasound | — | multi-scale features + XAI |
| Attention-driven detection | Eng. Appl. AI 2026 | GB ultrasound | — | 92.62 % (3-class); 98.36 % normal vs. abnormal |

### Reading the table critically

The apparent progression 92 % → 96 % → 98.5 % is **not** a progression in
capability. FocusMAE's own image-model baselines drop to ~84 % once evaluation
is patient-wise and variance is reported, while the papers reporting 96–98.5 %
use a *random* image-level split on a dataset with ~5.8 images per patient. With
multiple scans per patient in both train and test, the classifier can match on
patient-specific speckle, probe settings and body habitus rather than pathology.

This is a documented, quantified failure mode, not a hypothetical one: in
OCT-based classification, improper splits inflate performance by **5–30
percentage points of accuracy** (Tampu et al., *Scientific Data* 2022). The
inflation is of exactly the magnitude that separates the 92 % patient-wise state
of the art from the 96–98.5 % random-split claims.

**Implication for this project:** the honest target is 90 %+ *under a
leakage-controlled protocol with confidence intervals*, positioned against
GBCNet's 92.1 %. Reporting 96 %+ from a random split would place this work in
the group of papers a careful reviewer discounts.

## 4. Methods that reliably help on this data

Drawn from the works above and reproduced in `src/gbc/`:

- **ROI cropping over full-frame input.** GBCNet's central finding is that
  classifying the gallbladder + adjacent liver region beats classifying the
  whole frame, because full frames let the model latch onto acquisition
  artefacts. The crops shipped here are already these ROIs.
- **CLAHE + speckle-aware denoising.** Standard in the GBC ultrasound pipeline
  (GBCHV uses median filtering + CLAHE). Implemented in `gbc.transforms`.
- **Texture-bias control.** GBCNet's visual-acuity curriculum blurs early and
  sharpens later, worth +3.9 points of CV accuracy in their ablation
  (88.2 → 92.1). Our blur/speckle augmentation is a cheaper stand-in; a genuine
  curriculum is the most promising unexplored direction here.
- **Transformer/CNN ensembling.** DeiT and ViT reach 80–83 % where ResNet50
  reaches 71 % on the FocusMAE video benchmark, i.e. the two families make
  different errors — the precondition for ensembling to pay.
- **Sensitivity-weighted reporting.** Malignancy sensitivity is the clinically
  decisive metric; several baselines above pair high specificity with
  catastrophic sensitivity (VGG16: 96.0 % spec., 49.5 % sens.).

## 5. Where a new contribution can plausibly sit

Given that GBCNet already occupies "best accuracy on GBCU", a publishable
contribution from this repository is not another backbone leaderboard. The
defensible angles, in descending order of strength:

1. **A leakage audit of the redistributed GBCU derivatives** now circulating on
   dataset-sharing sites, with a measurement of the induced inflation. Section 2
   of `docs/data_leakage_audit.md` shows 34.1 % of source images appear in more
   than one split of the copy shipped here.
2. **Multi-label pathology recognition** (stone / mural thickening /
   malignancy), which uses the nested box annotations as they were actually
   drawn instead of collapsing them into a mutually exclusive 5-way label. No
   prior GBCU work reports this task.
3. **Calibration and decision-curve analysis**, absent from every paper above,
   and a prerequisite for the "clinical decision support" claim they all make.

## Sources

- [Basu et al., Surpassing the Human Accuracy (GBCNet), CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Basu_Surpassing_the_Human_Accuracy_Detecting_Gallbladder_Cancer_From_USG_Images_CVPR_2022_paper.html) · [arXiv](https://arxiv.org/abs/2204.11433)
- [GBCU dataset page](https://gbc-iitd.github.io/data/gbcu)
- [Basu et al., RadFormer, Medical Image Analysis 2023](https://www.sciencedirect.com/science/article/abs/pii/S1361841522003048)
- [Basu et al., FocusMAE, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Basu_FocusMAE_Gallbladder_Cancer_Detection_from_Ultrasound_Videos_with_Focused_Masked_CVPR_2024_paper.pdf)
- [GBCHV, Scientific Reports 2025](https://www.nature.com/articles/s41598-025-89232-5)
- [GallNet preprint, 2025](https://labs.sciety.org/articles/by?article_doi=10.21203/rs.3.rs-7625399/v1)
- [MSFE-GallNet-X, BMC Medical Imaging 2025](https://link.springer.com/article/10.1186/s12880-025-01902-y)
- [Attention-driven deep object detection for GBC, Eng. Appl. of AI 2026](https://www.sciencedirect.com/science/article/pii/S0952197626010390)
- [Tampu et al., Inflation of test accuracy due to data leakage, Scientific Data 2022](https://www.nature.com/articles/s41597-022-01618-6)
