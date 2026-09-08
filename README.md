# CAMELYON16 Attention-MIL: Multi-Modal Weakly-Supervised Breast Cancer Metastasis Detection

Attention-based Multiple Instance Learning (MIL) for whole-slide breast cancer histopathology
classification, using the official [CAMELYON16](https://camelyon16.grand-challenge.org/) dataset
of gigapixel H&E-stained lymph node whole-slide images (WSIs).

## Overview

Given a whole-slide image (up to ~220,000 × 97,000 pixels) and only a slide-level label
(tumor / normal — no pixel-level supervision at training time), the model must predict
whether the slide contains metastatic tissue **and** learn to localize it via attention,
using gated Attention MIL (Ilse et al., 2018).

Three architectures are implemented and compared:

| Model | Patch Encoder | Feature Dim |
|---|---|---|
| CNN-MIL | EfficientNet-B0 (ImageNet-pretrained, frozen) | 1280 |
| Foundation-MIL | Phikon-v2 (pathology-pretrained ViT, frozen) | 1024 |
| Fusion-MIL | Both encoders, projected + concatenated | 512+512 |

All three share the same gated-attention aggregation head (`src/models/attention_mil.py`).

## Dataset

- **Source:** official CAMELYON16 training set, downloaded from the public S3 mirror
  (`s3://camelyon-dataset/CAMELYON16/`, no-sign-request, CC0-licensed per the bucket's `license.txt`).
- **Scope used:** 150 of the 270 official training slides (62 tumor + 88 normal) for
  training/validation, plus a separate 20-slide held-out test set (8 tumor + 12 normal) —
  170 slides total, a documented subset selected for time/compute reasons, not the full
  official set. See `configs/phase3_plan.yaml` for the exact slide list and selection rationale.
- **Patching:** tissue regions detected via Otsu thresholding on the saturation channel at a
  low-resolution pyramid level; 256×256 patches extracted at level-0 (full resolution) from
  tissue-passing coordinates, capped at 4,000 patches/slide (random uniform subsample, seed=42)
  to keep the cached-embedding budget under ~15GB. See `configs/patching.yaml` for the full
  storage/cap rationale.
- **Lifecycle:** raw WSIs are downloaded in small batches, immediately converted to cached
  CNN + Phikon-v2 patch embeddings, then deleted — raw gigapixel images are never persisted to
  Drive/GitHub, only embeddings, coordinates, and checkpoints.

## Results

### 4-fold slide-level stratified cross-validation (150 slides)

| Model | Mean Val AUROC | Std | Per-fold |
|---|---|---|---|
| CNN-MIL | 0.893 | ±0.094 | 0.730, 0.949, 0.945, 0.948 |
| Foundation-MIL | 0.962 | ±0.045 | 0.889, 1.000, 1.000, 0.958 |
| **Fusion-MIL** | **0.966** | **±0.041** | 0.901, 1.000, 1.000, 0.964 |

**Finding:** Foundation-MIL and Fusion-MIL are statistically close to each other (overlapping
variance) but both are confidently and consistently ahead of CNN-MIL alone, especially on the
harder fold (Fold 1: 0.730 vs 0.889/0.901). This is the primary, defensible result of this
project: pathology-pretrained foundation features materially help patch-level MIL classification
over an ImageNet-pretrained CNN, and fusing the two pathways offers a small additional edge.

*Earlier note on methodology:* an initial 20-slide pilot run of this same pipeline showed
apparent AUROC=1.0 for all three models on a single train/val split — 4-fold CV on that smaller
scale revealed this was a lucky partition (true means were 0.625–0.708 with high variance). The
150-slide result above only became trustworthy once cross-validated at this larger scale; this
is documented as part of the project's methodology, not hidden.

### Held-out test evaluation (final result)

After all model selection, cross-validation, and hyperparameter decisions were finalized, the
frozen Fusion-MIL checkpoint (trained on 127/150 slides, val_auroc=0.9154) was evaluated **once**
on 20 slides (8 tumor + 12 normal) that had never been seen at any prior stage — not in training,
validation, architecture selection, or the ensemble/explainability analyses below.

| Metric | Value |
|---|---|
| AUROC | 0.927 |
| AUPRC | 0.931 |
| Accuracy | 0.900 |
| Precision | 1.000 |
| Specificity | 1.000 |
| Sensitivity (Recall) | 0.750 |
| F1 | 0.857 |

Confusion matrix: `[[12, 0], [2, 6]]` (12/12 normal slides correct, 6/8 tumor slides correct, 2 missed).

**Finding:** the held-out AUROC (0.927) falls within — in fact slightly above — the 4-fold CV
range for Fusion-MIL (0.901–1.000, mean 0.966), confirming the cross-validation results
generalize and were not an artifact of the validation folds. The model achieved zero false
positives (perfect precision and specificity) but missed 2 of 8 tumor slides (sensitivity
0.75). This is reported as an honest limitation rather than smoothed over: in a clinical
context, missed malignancies are more costly than false alarms, so this precision/sensitivity
trade-off would need to be addressed (e.g. via threshold tuning or more training data) before
any deployment-oriented claim.

*Note on sample size:* with only 8 tumor slides in the test set, each misclassification shifts
sensitivity by 12.5 percentage points — this is a directional confirmation at the current data
scale, not a large-scale clinical validation. Full results: `results/heldout_test_results.json`.

### Explainability: attention vs. expert annotations

CAMELYON16 provides pixel-level tumor boundary annotations (unlike patch-classification datasets
such as BreaKHis), enabling a direct check of whether the model's learned attention actually
falls on annotated malignant tissue.

Across the 10 tumor slides in the validation split, the trained Fusion-MIL model's attention
mass was compared against expert-annotated tumor polygons (patch-center point-in-polygon test):

- **8 of 10 slides:** 65–100% of total attention mass fell inside expert-annotated tumor regions,
  despite those regions covering as little as 0.1–40% of the slide's sampled patches — strong
  evidence of weakly-supervised localization with no pixel-level training signal.
- **1 slide (tumor_043):** 0% attention-in-tumor was a sampling artifact — the annotated region
  was small enough (~3×3 patches) that the random 4,000-patch cap missed it by chance, not a
  model failure.
- **1 slide (tumor_040):** a genuine false negative — the model predicted benign (prob=0.002)
  and attention was diffuse with no localized signal, an honest documented limitation.

Full results: `results/attention_annotation_analysis.json`.

### Ensemble analysis

A validation-only weighted ensemble search across all three models (CNN-MIL, Foundation-MIL,
Fusion-MIL — same 127/23 split as above) was also run:

| | Val AUROC |
|---|---|
| CNN-MIL (standalone) | 0.800 |
| Foundation-MIL (standalone) | 0.885 |
| Fusion-MIL (standalone) | 0.938 |
| **Best weighted ensemble** | **0.938** (weights: 0% CNN, 0% Foundation, 100% Fusion) |

**Finding:** the ensemble search collapsed to 100% weight on Fusion-MIL, giving zero improvement
over Fusion-MIL alone — mirroring the same result seen in the earlier BreaKHis (V1) project. The
interpretation is that Fusion-MIL, having already internally combined CNN and foundation-model
features, leaves no complementary signal in the standalone models for an outer ensemble to
exploit. (Note: weights were searched and evaluated on the same validation set, so this is a
directional finding, not an unbiased AUROC estimate — see `results/ensemble_analysis.json`.)

## Repository structure

  configs/ pilot/phase2/phase3 slide-selection configs, patching parameters
src/data/ WSI tissue detection, patch coordinate extraction, MIL bag construction, splits
src/features/ CNN and Phikon-v2 patch encoders, embedding cache I/O
src/models/ Gated Attention MIL (single-branch and fusion variants)
src/training/ Training loop (early stopping + LR scheduling), k-fold CV driver, ensemble search
src/explainability/ Annotation XML parsing, attention-vs-annotation comparison
results/ All metrics, CV summaries, held-out test, and explainability results as JSON
docs/ Session resume guides for multi-session Kaggle extraction runs



## Reproducing this work

Raw WSIs and cached embeddings are not stored in this repo (by design — see Lifecycle above).
To reproduce:

1. Environment: `pip install openslide-python timm transformers`, `apt-get install openslide-tools`
2. Download slides per `configs/phase3_plan.yaml` from the public S3 bucket (no AWS credentials needed)
3. Run tissue detection + patching (`src/data/`) and feature extraction (`src/features/`) to
   rebuild the embedding cache
4. Train with `src/training/train_mil.py` / `src/training/kfold_cv.py`

Extraction was run across multiple Kaggle sessions in small batches (5–20 slides at a time,
with immediate raw-file deletion after caching) to stay within Kaggle's working-storage limits;
see `docs/SESSION_RESUME.md` for the exact multi-session protocol used.

## Known limitations

- 150/270 slides used for training/validation (documented subset, not the full official
  training set); held-out test set is a further 20-slide subset
- Held-out test sensitivity (0.75, 2/8 tumor slides missed) is a genuine limitation at the
  current data scale — see Held-out test evaluation above
- Grad-CAM / pixel-level saliency not implemented for this project (attention-weight analysis
  only); attention reflects the model's internal weighting, not confirmed biological importance
  beyond the tumor/normal correlation shown above
- Patch capping (4,000/slide) can occasionally miss small annotated regions entirely by chance,
  as documented in the explainability analysis
- Ensemble weights were searched and evaluated on the same validation set (see caveat above)

## Acknowledgments

CAMELYON16 dataset: Bejnordi et al., *Diagnostic Assessment of Deep Learning Algorithms for
Detection of Lymph Node Metastases in Women With Breast Cancer*, JAMA 2017.
Gated Attention MIL: Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML 2018.
Phikon-v2: Owkin, pathology foundation model (`owkin/phikon-v2`).
