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
- **Scope used:** 150 of the 270 official training slides (62 tumor + 88 normal), a documented
  subset selected for time/compute reasons — **not** the full official set. See
  `configs/phase3_plan.yaml` for the exact slide list and selection rationale.
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

## Repository structure
configs/ pilot/phase2/phase3 slide-selection configs, patching parameters
src/data/ WSI tissue detection, patch coordinate extraction, MIL bag construction, splits
src/features/ CNN and Phikon-v2 patch encoders, embedding cache I/O
src/models/ Gated Attention MIL (single-branch and fusion variants)
src/training/ Training loop (early stopping + LR scheduling), k-fold CV driver
src/explainability/ Annotation XML parsing, attention-vs-annotation comparison
results/ All metrics, CV summaries, and explainability results as JSON
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

- 150/270 slides used (documented subset, not the full official training set)
- No held-out test-set evaluation yet — all reported numbers are cross-validated, not from a
  final untouched test split
- Grad-CAM / pixel-level saliency not implemented for this project (attention-weight analysis
  only); attention reflects the model's internal weighting, not confirmed biological importance
  beyond the tumor/normal correlation shown above
- Patch capping (4,000/slide) can occasionally miss small annotated regions entirely by chance,
  as documented in the explainability analysis

## Acknowledgments

CAMELYON16 dataset: Bejnordi et al., *Diagnostic Assessment of Deep Learning Algorithms for
Detection of Lymph Node Metastases in Women With Breast Cancer*, JAMA 2017.
Gated Attention MIL: Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML 2018.
Phikon-v2: Owkin, pathology foundation model (`owkin/phikon-v2`).
