# GATES-ACT vision validation protocol

Locked before accessing the protected-test pixels or training any vision model.

## Scientific boundary

The microscopy studies are independent of the neural-MEA studies. They do not contain paired
imaging and electrophysiology. Imaging can therefore support an upstream quality-control or
measurement-validity gate, but it cannot be used to claim improved MEA effect prediction.

## OOC quality classification

- Source: Zenodo record `10203721`, CC BY 4.0 in the repository metadata. The associated paper
  describes the dataset as CC BY-SA; GATES-ACT redistributes neither images nor labels and treats
  attribution and share-alike as applicable to derived dataset material.
- Unit of independence: acquisition date, parsed from the first six characters of `imageID`.
  The date is a conservative experiment/batch proxy; the source does not publish chip identifiers.
- Split: sort the 59 unique acquisition dates. The first 39 dates are training, the next 10 are
  calibration, and the final 10 are the protected chronological test set.
- Primary endpoint: recall for expert-labelled bad images. Secondary endpoints: balanced accuracy,
  AUROC, AUPRC, Brier score, expected calibration error, automatic-pass coverage, and the rate at
  which bad images are incorrectly passed.
- Baselines: majority class, metadata-only logistic regression, classical image features, and an
  ImageNet-pretrained frozen representation. Fine-tuning is retained only if calibration results
  justify it.
- Decision rule: `RETAKE/REVIEW`, `PASS`, or `ABSTAIN`. Thresholds and any probability calibration
  are chosen using training and calibration dates only. The protected test is evaluated once.

## Brain-organoid segmentation and cross-laboratory measurement

- Source: Zenodo record `10301912`, CC BY 4.0; 1,407 images from 64 longitudinally tracked
  organoids, four clones, ten days, and two laboratories.
- Split: within each clone, sort organoid IDs by SHA-256 of
  `gates-act-vision-v1:{clone}:{org_id}`. The first 9 are training, the next 3 calibration, and the
  final 4 protected test (36/12/16 organoids total).
- Training and model selection use Lab A only. Lab A images from protected organoids form an
  internal specimen-held-out test. Lab B images from the same protected organoids form the locked
  unseen-laboratory test; no Lab B pixels are used for fitting, threshold selection, or calibration.
- Primary endpoint: per-image Dice on Lab B. Decision endpoint: absolute organoid-area error among
  automatically accepted segmentations, reported together with coverage and failures.
- Baselines: global Otsu/morphology, the source paper's public CellProfiler/OrganoSeg/MOrgAna
  results, and the source paper's SegFormer result. A new model is not called superior unless a
  paired organoid-level comparison supports it.
- Uncertainty/abstention thresholds are locked on Lab A calibration organoids and transferred
  unchanged to Lab A and Lab B protected organoids.

## Statistics and release rules

- All confidence intervals and paired comparisons resample or permute independent experiment/date
  or organoid groups, never individual images as if independent.
- The protected sets are not used for hyperparameter selection. Failed methods and negative results
  remain in the machine-readable audit evidence.
- A microscopy capability enters the canonical GATES-ACT workflow only if it improves a predefined
  laboratory decision without hiding coverage, false negatives, or external-lab failures.
- Existing frozen GATES and GATES-ACT evidence is immutable.

## Integrity anchors

- `OOC_datasheet.xlsx` SHA-256:
  `863fc133f8be825c010c41ea26168fedb75af8d2722646f61402779583073eba`
- OOC image archive MD5 expected:
  `8f7e058996203d48eb03b2d86c0a2e4d`
- Brain-organoid archive MD5 verified:
  `53e6b41af957ac33c03b927e0fbe69aa`
- `dataset_overview.csv` SHA-256:
  `95cde068f6bedf34b3a0c96467ff6b81fbb9c8ce6844e3ef60d3c450c2b7c0a9`

