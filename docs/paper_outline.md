# Submission plan

## Positioning

The obvious framing — "we ensembled 16 ImageNet backbones and got 87 %" — is not
publishable. GBCNet reached **92.1 ± 2.9 %** on this exact dataset under a
*patient-wise* protocol in CVPR 2022, and a plain ResNet50 reached 81.1 % there.
An accuracy-only contribution below the 2022 state of the art, computed on a
contaminated split, has no path through review.

What this repository *does* have that the literature does not:

1. **A quantified leakage audit of a circulating GBCU derivative.** 34.1 % of
   source images appear in more than one split; the 5-way label is ill-posed for
   823 of 1,255 images. The ablation in `gbc.cli leakage-ablation` measures the
   accuracy this buys, with everything else held fixed.
2. **The multi-label pathology task** (stone / mural thickening / malignancy)
   that the nested box annotations actually encode. No prior GBCU paper reports
   it.
3. **Calibration and uncertainty**, absent from every GBC ultrasound paper
   surveyed, despite all of them claiming clinical decision support.

## Title

*Leakage-aware benchmarking of gallbladder cancer classification from
ultrasound: what a contaminated split is worth, and what remains after removing
it.*

## Section plan

1. **Introduction.** GBC's late presentation and 5 % five-year survival; the
   recent run of 96–98.5 % claims; the observation that these coincide with
   random image-level splits on a dataset with ~5.8 images per patient.
2. **Related work.** `docs/literature_review.md`. The key table contrasts
   patient-wise results (GBCNet 92.1 %, FocusMAE's image baselines 79–84 %) with
   random-split results (96.21 %, 98.50 %).
3. **Dataset forensics.** `docs/data_leakage_audit.md`. Reconstruction of GBCU
   from the flattened crops, verified against published counts; the three
   defects; the nesting geometry.
4. **Method.** Corrected task definitions, grouped CV, the training recipe
   (`docs/reproducibility.md`), in-fold ensembling.
5. **Experiments.**
   - E1 3-class diagnosis under grouped 5-fold CV — per-model and ensemble,
     with bootstrap CIs and McNemar tests.
   - E2 leakage ablation — grouped vs ungrouped folds, same model, same data.
   - E3 multi-label pathology — per-label AUC and mAP.
   - E4 calibration — ECE before and after temperature scaling, reliability
     diagrams.
   - E5 ablations — CLAHE, mixup, EMA, TTA, letterbox vs. squash.
   - E6 qualitative — Grad-CAM overlays, reviewed against the radiologist boxes.
6. **Discussion and limitations.** Residual **patient-level** leakage
   (`docs/data_leakage_audit.md` §5) is the first limitation and must be stated
   plainly; single-centre data; no external validation; ROI boxes are assumed
   given at test time, as in GBCNet.
7. **Conclusion.**

## Before submitting — hard requirements

- [ ] **Obtain the licensed GBCU release with patient IDs** and rerun with
      `assign_folds(..., group_col="patient_id")`. Until this is done, do not
      claim comparability with GBCNet's 92.1 %.
- [ ] Cite Basu et al. (CVPR 2022) as the dataset source and comply with the
      GBCU license terms; state the license in a Data Availability section.
- [ ] Ethics statement: GBCU was collected under PGIMER ethics approval with
      written informed consent — cite this rather than claiming your own.
- [ ] Report an external or at least a second-centre validation, or state its
      absence as a limitation.
- [ ] Follow **CLAIM 2024** (Checklist for AI in Medical Imaging) and
      **TRIPOD+AI** reporting checklists; both are commonly required by
      clinical-imaging journals.
- [ ] Remove every accuracy figure computed on the shipped split from the
      manuscript, slides and README.

## Venue options

| venue | fit |
| --- | --- |
| *Computers in Biology and Medicine* / *Computer Methods and Programs in Biomedicine* | good fit for a leakage-audit + benchmark paper with released code |
| *Scientific Reports* | precedent for GBC ultrasound work (GBCHV); broad scope |
| *Journal of Imaging* / *Diagnostics* (MDPI) | fastest route; lower prestige |
| MICCAI / IEEE ISBI | possible if the multi-label pathology task is developed into a genuine method contribution rather than a benchmark |

A benchmarking-and-reproducibility paper is the strongest honest framing
available from this data without new patient recruitment.
