# GRAND_PLAN.md — EVE-RAG Full Project Guide

**Project:** EVE-RAG
**Research topic:** Out-of-context image detection using entity verification, visual grounding, CLIP-based representation learning, NLI evidence stance reasoning, and gated multi-stream fusion.
**Repository root:** `eve-rag/`
**Current phase:** ✅ All three streams complete. ✅ Fusion complete (GSAF trained, ablations done). 🔴 MLLM explanation pending. 🔴 Formal eval notebooks pending.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Plan summary](#2-plan-summary)
3. [Final directory structure](#3-final-directory-structure)
4. [Dataset details and gains](#4-dataset-details-and-gains)
5. [Dataset preprocessing and processing](#5-dataset-preprocessing-and-processing)
6. [Notebook names and completion status](#6-notebook-names-and-completion-status)
7. [Notebook summaries](#7-notebook-summaries)
8. [Output tables and figures](#8-output-tables-and-figures)
9. [Output summary so far](#9-output-summary-so-far)
10. [Issues faced and how we solved them](#10-issues-faced-and-how-we-solved-them)
11. [Important implementation notes](#11-important-implementation-notes)
12. [How to run the project](#12-how-to-run-the-project)
13. [Current limitations and next research fixes](#13-current-limitations-and-next-research-fixes)
14. [Glossary](#14-glossary)
15. [Beginner mental model](#15-beginner-mental-model)

---

# 1. Project overview

EVE-RAG is a research project for **out-of-context (OOC) image detection** — detecting when an image is paired with a misleading or false caption/context.

## The core task

> Given an image and a text caption, decide: does this image truly belong with this text, or has the image been recycled in a misleading way?

Each data sample is an **image-text pair** with a binary label:

| Label | Name | Meaning |
|---:|---|---|
| `0` | match / genuine | The image and text correctly belong together. |
| `1` | mismatch / OOC | The image is used out of context or misleadingly. |

## Why is this hard?

Out-of-context misinformation is subtle. The image and caption often look plausible together. The mismatch comes from:

- **Person mismatch** — wrong person in the image for the caption.
- **Location mismatch** — image from a different place.
- **Event mismatch** — image from a different event.
- **Date mismatch** — image reused from a different time.
- **Source mismatch** — image from a conflicting source.
- **Evidence contradiction** — external evidence contradicts the claim.

No single signal is enough. That is why EVE-RAG uses multiple complementary **streams**.

## The three streams

| Stream | Short name | What it asks |
|---|---|---|
| Stream A | Entity + visual grounding | "Do the named entities in the caption (people, places, organizations) appear visually or evidentially supported?" |
| Stream B | CLIP representation | "Do the image and text embeddings look compatible in a shared vector space?" |
| Stream C | Evidence + stance reasoning | "Does external evidence about this image/caption support or contradict the claim?" |

These streams are then combined by a **fusion model** that learns which stream to trust for each sample.

---

# 2. Plan summary

## What changed and why

The original project plan broke down because:

- Dataset folders were messy and inconsistent.
- Stream B used fake (hash-generated) image vectors for NewsCLIPpings instead of real embeddings.
- Stream C evidence was circular (it included the caption in its own evidence text).
- Notebooks hardcoded old paths.

The project was restructured into a clean, reproducible layout with a strict separation between raw data, processed data, caches, and paper outputs.

## Final plan

1. Keep raw datasets untouched under `data/raw/`.
2. Build a clean, normalized processed dataset under `data/processed/ooc_pairs/`.
3. Store all local images in one flat folder: `data/processed/ooc_pairs/images/`.
4. Store all metadata in normalized manifest CSVs: `data/processed/ooc_pairs/manifests/`.
5. Store reproducible split ID files in `data/splits/`.
6. Update all notebooks to use manifest CSVs instead of raw folder scanning.
7. Store all intermediate outputs in `cache/`.
8. Store all paper outputs in `paper/figures/` and `paper/tables/`.
9. Use GPU for encoding, training, and inference.
10. Run fusion using combined Stream A / B / C features.
11. Add MLLM explanation layer after fusion.

## Current status at a glance

| Component | Status | Notes |
|---|---|---|
| Directory cleanup | ✅ Complete | |
| Flat processed image folder | ✅ Complete | 11,495 images |
| Manifest CSVs | ✅ Complete | 97,361 rows |
| Split files | ✅ Complete | |
| Setup notebooks | ✅ Complete | |
| EDA notebook | ✅ Complete | |
| Stream A | ✅ Complete | Weak alone; feeds fusion |
| Stream B | ✅ Complete | Strongest individual stream |
| Stream C v3 | ✅ Complete | NLI stance scoring, decircularized evidence |
| Fusion (GSAF) | ✅ Complete | Mixed-domain training; strong OOD results |
| MLLM explanation | 🔴 Pending | After fusion |
| Formal eval notebooks | 🔴 Pending | `07_eval/` |

---

# 3. Final directory structure

```text
eve-rag/
├── .env.example
├── .gitignore
├── README.md
├── GRAND_PLAN.md
├── requirements.txt
├── eve-rag.code-workspace
│
├── configs/
│   ├── eve_rag.yaml
│   ├── ablation_frozen_clip.yaml
│   ├── ablation_mlp_fusion.yaml
│   ├── ablation_no_mllm.yaml
│   ├── ablation_no_stream_a.yaml
│   └── ablation_no_stream_c.yaml
│
├── data/
│   ├── raw/
│   │   ├── newsclippings/
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   │   ├── test.json
│   │   │   ├── metadata/
│   │   │   └── embeddings/
│   │   │       ├── clip_image_embeddings/
│   │   │       ├── clip_text_embeddings/
│   │   │       ├── facenet_embeddings/
│   │   │       ├── places_resnet50/
│   │   │       └── sbert_embeddings/
│   │   │
│   │   ├── verite/
│   │   │   └── image-text-verification/
│   │   │       └── VERITE/
│   │   │           ├── VERITE.csv
│   │   │           ├── VERITE_articles.csv
│   │   │           ├── VERITE_clip_image_embeddings_ViTL14.npy
│   │   │           ├── VERITE_clip_text_embeddings_ViTL14.npy
│   │   │           └── images/
│   │   │
│   │   └── mmfakebench/
│   │       ├── MMFakeBench_val.json
│   │       ├── MMFakeBench_test.json
│   │       ├── MMFakeBench_val/
│   │       └── MMFakeBench_test/
│   │
│   ├── processed/
│   │   └── ooc_pairs/
│   │       ├── images/
│   │       │   ├── verite_true_0.jpg
│   │       │   ├── verite_false_0.jpg
│   │       │   ├── mmfakebench_test_real_bbc_...0.png
│   │       │   └── ...
│   │       │
│   │       ├── manifests/
│   │       │   ├── train.csv
│   │       │   ├── val.csv
│   │       │   ├── test.csv
│   │       │   ├── verite_test.csv
│   │       │   ├── mmfakebench_val.csv
│   │       │   ├── mmfakebench_test.csv
│   │       │   └── all.csv
│   │       │
│   │       └── reports/
│   │           └── manifest_summary.csv
│   │
│   ├── splits/
│   │   ├── train_pair_ids.txt
│   │   ├── val_pair_ids.txt
│   │   ├── test_pair_ids.txt
│   │   ├── verite_test_pair_ids.txt
│   │   ├── mmfakebench_val_pair_ids.txt
│   │   ├── mmfakebench_test_pair_ids.txt
│   │   ├── all_pair_ids.txt
│   │   ├── local_image_pair_ids.txt
│   │   ├── newsclippings_pair_ids.txt
│   │   └── ood_pair_ids.txt
│   │
│   └── external/
│       └── pretrained/
│
├── cache/
│   ├── stream_a/
│   │   ├── ner/
│   │   ├── visual_grounding/
│   │   ├── entity_scorer/
│   │   └── results/
│   │
│   ├── stream_b/
│   │   ├── clip_embeddings/
│   │   ├── features/
│   │   ├── lora_stage1/
│   │   ├── lora_stage2/
│   │   └── final/
│   │
│   ├── stream_c/
│   │   ├── evidence/
│   │   ├── stance_scores/
│   │   └── results/
│   │
│   └── fusion/
│       ├── fusion_features_all.csv
│       ├── feature_schema.json
│       ├── feature_scaler.json
│       ├── fusion_model.pt
│       ├── fusion_predictions.csv
│       ├── fusion_thresholds.json
│       └── fusion_train_history.csv
│
├── notebooks/
│   ├── 00_make_flat_processed_dataset.ipynb
│   │
│   ├── 01_setup/
│   │   ├── env_check.ipynb
│   │   ├── data_setup.ipynb
│   │   └── eda.ipynb
│   │
│   ├── 02_stream_a/
│   │   ├── ner_pipeline_v2_updated.ipynb
│   │   ├── visual_grounding_v2_updated.ipynb
│   │   ├── entity_scorer_v2_updated.ipynb
│   │   └── stream_a_results_v2_updated.ipynb
│   │
│   ├── 03_stream_b/
│   │   ├── clip_features_updated.ipynb
│   │   ├── lora_stage1_updated.ipynb
│   │   ├── lora_stage2_fixed_updated.ipynb
│   │   └── stream_b_results_fixed_updated.ipynb
│   │
│   ├── 04_stream_c/
│   │   ├── evidence_retrieval_final.ipynb
│   │   ├── evidence_val_final.ipynb
│   │   ├── stance_scoring_final.ipynb
│   │   └── stream_c_results_final.ipynb
│   │
│   ├── 05_fusion/
│   │   ├── baselines_final.ipynb
│   │   ├── fusion_module.ipynb
│   │   └── fusion_training.ipynb
│   │
│   ├── 06_mllm/
│   │   ├── qwen_finetune.ipynb
│   │   ├── mllm_inference.ipynb
│   │   └── chain_of_thought.ipynb
│   │
│   ├── 07_eval/
│   │   ├── newsclippings_eval.ipynb
│   │   ├── cross_dataset_eval.ipynb
│   │   ├── ablations.ipynb
│   │   └── qualitative_analysis.ipynb
│   │
│   └── 08_paper/
│       ├── figures.ipynb
│       ├── tables.ipynb
│       └── explanation_eval.ipynb
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── fusion.py          ← auto-generated by fusion_module.ipynb
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── paths.py
│
├── scripts/
│   ├── build_dataset.py
│   └── inspect_dataset_shapes.py
│
├── experiments/
│
└── paper/
    ├── figures/
    └── tables/
```

## Core path constants (`src/utils/paths.py`)

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"

RAW = DATA / "raw"
PROCESSED = DATA / "processed" / "ooc_pairs"
MANIFESTS = PROCESSED / "manifests"
PROCESSED_IMAGES = PROCESSED / "images"
REPORTS = PROCESSED / "reports"

SPLITS = DATA / "splits"
EXTERNAL = DATA / "external"
PRETRAINED = EXTERNAL / "pretrained"

NEWSCLIPPINGS_RAW = RAW / "newsclippings"
MMFAKEBENCH_RAW = RAW / "mmfakebench"

VERITE_RAW = RAW / "verite"
VERITE_CANON = VERITE_RAW / "image-text-verification" / "VERITE"
VERITE_CSV = VERITE_CANON / "VERITE.csv"
VERITE_ARTICLES = VERITE_CANON / "VERITE_articles.csv"
VERITE_IMAGES = VERITE_CANON / "images"

TRAIN_MANIFEST = MANIFESTS / "train.csv"
VAL_MANIFEST = MANIFESTS / "val.csv"
TEST_MANIFEST = MANIFESTS / "test.csv"
VERITE_TEST_MANIFEST = MANIFESTS / "verite_test.csv"
MMFAKEBENCH_VAL_MANIFEST = MANIFESTS / "mmfakebench_val.csv"
MMFAKEBENCH_TEST_MANIFEST = MANIFESTS / "mmfakebench_test.csv"
ALL_MANIFEST = MANIFESTS / "all.csv"

MANIFEST_SUMMARY = REPORTS / "manifest_summary.csv"

CACHE = ROOT / "cache"
STREAM_A_CACHE = CACHE / "stream_a"
STREAM_B_CACHE = CACHE / "stream_b"
STREAM_C_CACHE = CACHE / "stream_c"
FUSION_CACHE = CACHE / "fusion"

PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
```

---

# 4. Dataset details and gains

EVE-RAG uses three public datasets:

| Dataset | Type | Size | Images available? |
|---|---|---|---|
| **NewsCLIPpings** | News image-text pairs | 85,360 rows | ❌ No local images; real CLIP embeddings available |
| **VERITE** | Real-world OOC pairs | 1,001 test rows | ✅ Partial (621 rows / 495 unique files) |
| **MMFakeBench** | Mixed-source fake detection | 11,000 rows | ✅ Full coverage |

## Where we got each dataset

| Dataset | Source | Notes |
|---|---|---|
| NewsCLIPpings | [GitHub: S-Abdelnabi/NewsCLIPpings](https://github.com/S-Abdelnabi/NewsCLIPpings) | Includes pre-extracted CLIP/SBERT/ResNet embeddings. VisualNews local images are not redistributed. |
| VERITE | [GitHub: stevejpapad/image-text-verification](https://github.com/stevejpapad/image-text-verification) | Includes `VERITE.csv`, `VERITE_articles.csv`, and a subset of local images. |
| MMFakeBench | [GitHub: poloclub/mmfakebench](https://github.com/poloclub/mmfakebench) or Hugging Face | Includes val/test JSON + image folders. |

## Final manifest counts

| Manifest | Dataset | Rows | Local image rows | Unique image files | Notes |
|---|---|---:|---:|---:|---|
| `train.csv` | NewsCLIPpings | 71,072 | 0 | 59,876 emb keys | Balanced; real CLIP embeddings |
| `val.csv` | NewsCLIPpings | 7,024 | 0 | 5,943 emb keys | Balanced |
| `test.csv` | NewsCLIPpings | 7,264 | 0 | 6,147 emb keys | Balanced |
| `verite_test.csv` | VERITE | 1,001 | 621 | 495 | Imbalanced (33% match / 66% OOC) |
| `mmfakebench_val.csv` | MMFakeBench | 1,000 | 1,000 | 1,000 | Imbalanced (30% / 70%) |
| `mmfakebench_test.csv` | MMFakeBench | 10,000 | 10,000 | 10,000 | Imbalanced (30% / 70%) |
| **`all.csv`** | **Combined** | **97,361** | **11,621** | **11,495 files** | Unified manifest |

## Label distribution

| Dataset / Split | Label 0: match | Label 1: OOC | Positive rate |
|---|---:|---:|---:|
| NewsCLIPpings train | 35,536 | 35,536 | 50.0% |
| NewsCLIPpings val | 3,512 | 3,512 | 50.0% |
| NewsCLIPpings test | 3,632 | 3,632 | 50.0% |
| VERITE test | 338 | 663 | 66.2% |
| MMFakeBench val | 300 | 700 | 70.0% |
| MMFakeBench test | 3,000 | 7,000 | 70.0% |
| **All combined** | **46,318** | **51,043** | **52.4%** |

> **Why does label imbalance matter?**
> On MMFakeBench and VERITE, the majority class is "OOC" (label 1). A naive classifier that always predicts OOC would score 70% accuracy on MMFakeBench without learning anything. This is why **AUROC** (which is threshold-independent) is the primary metric, and why threshold tuning is necessary for accuracy to be meaningful.

## What we gained through cleanup

**Before cleanup:**

- Dataset folders were inconsistent; notebooks scanned raw paths manually.
- VERITE had duplicated image folders (two overlapping copies).
- MMFakeBench images were buried in nested source-specific subfolders.
- NewsCLIPpings local images were unavailable; this wasn't handled cleanly.
- Stream B used fake hash-generated image vectors for NewsCLIPpings → weak results.
- Stream C evidence was circular (it included the caption in its own evidence text).

**After cleanup:**

| Gain | Details |
|---|---|
| Unified manifests | Every dataset uses the same schema. All notebooks read `all.csv`. |
| Flat image folder | All local images copied to `data/processed/ooc_pairs/images/` with unique names. |
| 97,361 pair rows | Combined normalized dataset. |
| 11,621 rows with local images | VERITE + MMFakeBench combined. |
| 11,495 unique image files | After deduplication. |
| Real NewsCLIP embeddings | NewsCLIPpings now uses real PKL CLIP embeddings (85,360 rows, 0 missing). |
| 0 broken image paths | All processed manifests validate cleanly. |
| Reproducible splits | `data/splits/` contains row ID lists. |
| Decircularized Stream C | Caption never in evidence text; NLI model instead of cosine similarity. |
| Fusion complete | GSAF trained; MMFakeBench AUROC 61 → 88 after fix. |

---

# 5. Dataset preprocessing and processing

## Why preprocessing was needed

Each dataset comes in a different raw format:

| Dataset | Raw format problem | Solution |
|---|---|---|
| NewsCLIPpings | JSON + metadata + PKL embeddings; no local images | Use real pre-extracted CLIP PKL embeddings; map `clip_image_key` / `clip_text_key` |
| VERITE | CSV with image references like `images/true_0.jpg`; only partial images downloaded | Resolve paths, copy to flat processed folder |
| MMFakeBench | JSON + nested source-specific image subfolders | Copy all images to flat folder with unique names |

## Processed manifest schema

Every manifest CSV uses this normalized schema:

| Column | Type | Meaning |
|---|---|---|
| `pair_id` | string | Unique ID for this image-text pair row |
| `dataset` | string | `newsclippings`, `verite`, or `mmfakebench` |
| `split` | string | `train`, `val`, or `test` |
| `image_id` | string | Unique ID for the image file or embedding |
| `image_path` | string | Processed local image path (if available) |
| `raw_image_path` | string | Original dataset image path/reference |
| `has_image` | int | `1` if a local processed image file exists, else `0` |
| `caption` | string | Text/caption/context for this pair |
| `label` | int | `0` = match, `1` = mismatch/OOC |
| `label_name` | string | Human-readable label |
| `source_image_id` | string | Source dataset image ID |
| `source_text_id` | string | Source dataset text ID |
| `original_id` | string | Original row/item ID in source dataset |
| `clip_image_key` | string | Key for looking up CLIP image embeddings |
| `clip_text_key` | string | Key for looking up CLIP text embeddings |
| `text_source` | string | MMFakeBench text source (e.g. BBC, CNN) |
| `image_source` | string | MMFakeBench image source |
| `fake_cls` | string | MMFakeBench fake subtype |
| `topic` | string | NewsCLIPpings topic |
| `source` | string | NewsCLIPpings news source |
| `timestamp` | string | NewsCLIPpings timestamp |
| `title` | string | NewsCLIPpings article title |
| `raw_label` | string | Original source label format |
| `notes` | string | Notes about special row handling |

## Why the flat image folder?

Instead of nested folders like:

```text
images/mmfakebench/test/fake/antifact_image_generation/...
images/verite/...
```

we use a single flat folder:

```text
data/processed/ooc_pairs/images/
```

with unique names that encode source info:

```text
verite_true_0.jpg
verite_false_0.jpg
mmfakebench_test_real_bbc_test_500_bbc_test_0.png
mmfakebench_test_fake_antifact_image_generation_...1.png
```

This works because the manifest stores the exact path for each row. Notebooks never need to scan folder structures — they just read the manifest.

## How to rebuild processed data from scratch

Run in order:

```text
notebooks/01_setup/data_setup.ipynb
```

It does:

1. Reads raw NewsCLIPpings JSON + metadata.
2. Reads raw VERITE CSV + image folder.
3. Reads raw MMFakeBench JSON + image folders.
4. Normalizes all rows into the manifest schema.
5. Copies local images to the flat processed folder.
6. Writes all manifest CSVs.
7. Writes split ID files.
8. Validates image paths.
9. Writes `manifest_summary.csv`.

---

# 6. Notebook names and completion status

**Status legend:**

| Symbol | Meaning |
|---|---|
| ✅ Complete | Ran successfully; all outputs exist |
| 🟡 Weak | Technically valid but scientifically limited (used in fusion) |
| 🟢 Done (this phase) | Newly completed in the fusion phase |
| 🔴 Pending | Not yet implemented |

## Setup

| Notebook | Status | Notes |
|---|---|---|
| `01_setup/env_check.ipynb` | ✅ | Checks Python, GPU, paths, manifests |
| `01_setup/data_setup.ipynb` | ✅ | Builds manifests, flat images, splits |
| `01_setup/eda.ipynb` | ✅ | EDA figures and tables |

## Stream A

| Notebook | Status | Notes |
|---|---|---|
| `02_stream_a/ner_pipeline_v2_updated.ipynb` | ✅ | NER cached for all 97,361 rows |
| `02_stream_a/visual_grounding_v2_updated.ipynb` | ✅ | CLIP grounding for 11,621 local-image rows |
| `02_stream_a/entity_scorer_v2_updated.ipynb` | ✅ | 12-feature table built |
| `02_stream_a/stream_a_results_v2_updated.ipynb` | 🟡 | Weak standalone; contributes to fusion |

## Stream B

| Notebook | Status | Notes |
|---|---|---|
| `03_stream_b/clip_features_updated.ipynb` | ✅ | Uses real NewsCLIPpings PKL embeddings |
| `03_stream_b/lora_stage1_updated.ipynb` | ✅ | Stage 1 MLP training |
| `03_stream_b/lora_stage2_fixed_updated.ipynb` | ✅ | Stage 2 classifier; predictions saved |
| `03_stream_b/stream_b_results_fixed_updated.ipynb` | ✅ | Best individual stream |

## Stream C

| Notebook | Status | Notes |
|---|---|---|
| `04_stream_c/evidence_retrieval_final.ipynb` | ✅ | Decircularized; caption never in evidence |
| `04_stream_c/evidence_val_final.ipynb` | ✅ | Repair fallback; labeled-leak assert passes |
| `04_stream_c/stance_scoring_final.ipynb` | ✅ | NLI DeBERTa; 97,361 rows in ~2.5 min |
| `04_stream_c/stream_c_results_final.ipynb` | ✅ | Per-dataset threshold tuning |

## Fusion

| Notebook | Status | Notes |
|---|---|---|
| `05_fusion/baselines_final.ipynb` | 🟢 Done | Baselines + combined streams summary; CLIP AUROC on VERITE = 48.54 (confirms OOD problem) |
| `05_fusion/fusion_module.ipynb` | 🟢 Done | GSAF architecture; 30 features (A=12, B=11, C=7); writes `src/models/fusion.py` |
| `05_fusion/fusion_training.ipynb` | 🟢 Done | Mixed-domain training; MMFakeBench AUROC 61→88; all 4 ablations |

## Pending

| Notebook | Status | Notes |
|---|---|---|
| `07_eval/newsclippings_eval.ipynb` | 🔴 | Formal in-distribution eval |
| `07_eval/cross_dataset_eval.ipynb` | 🔴 | Formal OOD eval for VERITE/MMFakeBench |
| `07_eval/ablations.ipynb` | 🔴 | Full ablation configs |
| `07_eval/qualitative_analysis.ipynb` | 🔴 | Human-readable examples |
| `06_mllm/qwen_finetune.ipynb` | 🔴 | MLLM fine-tuning |
| `06_mllm/mllm_inference.ipynb` | 🔴 | Explanation generation |
| `06_mllm/chain_of_thought.ipynb` | 🔴 | Chain-of-thought explanation format |

---

# 7. Notebook summaries

## `env_check.ipynb`

Verifies the environment before running anything else. Checks Python version, CUDA, GPU name and VRAM, root path detection, `src.utils.paths` imports, manifest existence, flat image folder, split files, and cache/paper folder structure.

**Run this first and after any environment change.**

---

## `data_setup.ipynb`

Builds the entire canonical dataset layer from raw data. This is the most important setup notebook.

**Outputs:**

```text
data/processed/ooc_pairs/images/          ← 11,495 image files
data/processed/ooc_pairs/manifests/       ← 7 CSV files
data/processed/ooc_pairs/reports/         ← manifest_summary.csv
data/splits/                              ← 10 split ID txt files
```

---

## `eda.ipynb`

Reads processed manifests and produces exploratory data analysis figures and tables:

- Dataset count tables, class balance tables, image coverage summary.
- Caption length analysis, fake class distribution (MMFakeBench).
- Paper-ready EDA figures saved to `paper/figures/`.

---

## `ner_pipeline_v2_updated.ipynb`

Extracts named entities (people, places, organizations) from all captions using spaCy (or a regex fallback).

**Outputs:**

```text
cache/stream_a/ner/entities_all.pkl
cache/stream_a/ner/entities_newsclippings_{train,val,test}.pkl
cache/stream_a/ner/entities_verite_test.pkl
cache/stream_a/ner/entities_mmfakebench_{val,test}.pkl
```

---

## `visual_grounding_v2_updated.ipynb`

For rows that have a local processed image, computes CLIP-based visual grounding scores: how well does the image match each named entity mentioned in the caption?

**Outputs:**

```text
cache/stream_a/visual_grounding/vg_scores_all.pkl
cache/stream_a/visual_grounding/caption_clip_sim_all.pkl
cache/stream_a/visual_grounding/clip_image_embs_processed.pkl
cache/stream_a/visual_grounding/clip_entity_prompt_embs.pkl
```

---

## `entity_scorer_v2_updated.ipynb`

Converts NER and visual grounding outputs into a flat numeric feature table (12 columns) for every row.

**Outputs:**

```text
cache/stream_a/entity_scorer/features_all.csv
cache/stream_a/entity_scorer/features_all.pkl
```

---

## `stream_a_results_v2_updated.ipynb`

Trains and evaluates a Stream A classifier from entity/grounding features. Produces Stream A result tables and figures.

Stream A is weak standalone. Its value is in contributing entity-grounding features to the fusion model.

---

## `clip_features_updated.ipynb`

Builds Stream B CLIP-derived features for every row.

**Key fix:** NewsCLIPpings uses real pre-extracted PKL embeddings from `data/raw/newsclippings/embeddings/`. VERITE and MMFakeBench use OpenCLIP encodings from local processed images.

**Outputs:**

```text
cache/stream_b/clip_embeddings/
cache/stream_b/features/clip_features_all.csv    ← 97,361 rows × 13 features
```

---

## `lora_stage1_updated.ipynb`

Trains a Stage 1 MLP-style representation/classifier on Stream B features.

**Outputs:**

```text
cache/stream_b/lora_stage1/stage1_model.pt
cache/stream_b/lora_stage1/stage1_scores_all.csv
cache/stream_b/lora_stage1/stage1_history.csv
```

---

## `lora_stage2_fixed_updated.ipynb`

Trains and evaluates a Stage 2 classifier on top of Stage 1 features.

**Outputs:**

```text
cache/stream_b/lora_stage2/stage2_model.pkl
cache/stream_b/final/stream_b_predictions.csv
cache/stream_b/final/stream_b_final_features.csv    ← 97,361 rows × 11 features
paper/tables/stream_b_results.csv
```

---

## `stream_b_results_fixed_updated.ipynb`

Generates Stream B figures, summary tables, and score distributions. Documents Stream B as the currently strongest individual stream.

---

## `evidence_retrieval_final.ipynb`

Builds decircularized offline evidence text for every row from independent sources:

- **VERITE:** Uses the `query` field from `VERITE_articles.csv` (32.47% coverage with high-quality evidence).
- **NewsCLIPpings:** Uses article title + topic + source + timestamp (99.97–99.99% coverage).
- **MMFakeBench:** Uses `text_source` + `image_source` + `fake_cls` metadata.

**Critical rule:** the caption is **never** included in the evidence text. An assertion enforces this (0 "Caption:" fields must appear in any `evidence_text` value).

**Outputs:**

```text
cache/stream_c/evidence/raw_evidence_all.csv
```

---

## `evidence_val_final.ipynb`

Validates and repairs missing or low-quality evidence rows. The repair fallback is dataset+split metadata only (no caption). The labeled-leak assertion confirms 0 caption leaks.

---

## `stance_scoring_final.ipynb`

Frames OOC detection as Natural Language Inference (NLI):

> **Premise** = independent evidence text  
> **Hypothesis** = caption  
> **Question** = does the evidence *entail*, *contradict*, or stay *neutral* toward the caption?

Uses `cross-encoder/nli-deberta-v3-base` for entailment/neutral/contradiction probabilities. Adds SBERT cosine (`all-MiniLM-L6-v2`) as a complementary feature.

Full 97,361 rows scored in ~2.5 minutes on RTX 4060 Ti.

**Outputs:**

```text
cache/stream_c/stance_scores/stream_c_stance_scores_all.csv    ← 7 features per row
```

---

## `stream_c_results_final.ipynb`

Trains a LogisticRegression classifier on the 7 NLI + SBERT features. Performs per-dataset threshold tuning on validation splits. Saves `stream_c_thresholds.json` for use in the fusion model.

---

## `baselines_final.ipynb`

Computes all paper baselines and combines stream results into one summary table.

**Baselines computed:**

- Random uniform predictor (5-seed average for stability).
- Majority class predictor (per-dataset).
- Always-match (always predict label 0).
- Always-mismatch (always predict label 1).
- CLIP cosine threshold (raw similarity ≥ median → match).
- CLIP cosine rank (percentile-rank score; AUROC reflects ranking quality).

**Key finding:** CLIP cosine AUROC on VERITE = **48.54** (barely above random). This confirms that pure similarity-based methods fail on real-world OOC detection. Fusion is necessary.

**Outputs:**

```text
paper/tables/baseline_results.csv
paper/tables/clip_baseline_results.csv
paper/tables/all_streams_and_baselines_summary.csv
paper/figures/all_streams_summary_bar.png
```

---

## `fusion_module.ipynb`

Defines the **Gated Stream Attention Fusion (GSAF)** architecture and prepares the feature matrix.

**What it does:**

1. Loads Stream A / B / C feature tables and merges on `pair_id`.
2. Identifies 30 numeric features (A=12, B=11, C=7).
3. Drops constant / all-NaN columns; documents coverage per stream.
4. Saves merged feature matrix to `cache/fusion/fusion_features_all.csv`.
5. Saves feature schema to `cache/fusion/feature_schema.json`.
6. Writes the GSAF architecture to `src/models/fusion.py`.
7. Smoke-tests all 5 model variants (default + 4 ablations).

**Features used:**

| Stream | Feature count | Example features |
|---|---|---|
| Stream A | 12 | `n_entities`, `n_person`, `n_gpe`, `max_grounding`, `mean_grounding`, `entity_match_ratio` |
| Stream B | 11 | `clip_cosine`, `l2_dist`, `absdiff_mean`, `text_norm`, `has_real_image_emb` |
| Stream C | 7 | `support_score`, `contradiction_score`, `neutral_score`, `stance_similarity`, `len_ratio` |

**Outputs:**

```text
cache/fusion/fusion_features_all.csv     ← 97,361 rows × 35 cols
cache/fusion/feature_schema.json
src/models/fusion.py                     ← auto-generated GSAF class
```

---

## `fusion_training.ipynb`

Trains the GSAF model and all ablations.

**Training setup:**

- Train on NewsCLIPpings train (71,072 rows) + half of MMFakeBench val (500 rows, stratified).
- Validate (early stopping) on NewsCLIPpings val (7,024 rows).
- Evaluate on: NewsCLIPpings test, MMFakeBench val_tune (held-out 500), MMFakeBench test, VERITE test.
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4.
- Batch size: 1024. Early stop patience: 8 epochs on val AUROC.
- Per-dataset threshold tuning on validation splits.

**Outputs:**

```text
cache/fusion/fusion_model.pt
cache/fusion/fusion_predictions.csv      ← 25,789 rows with per-row gate weights
cache/fusion/fusion_thresholds.json
cache/fusion/fusion_train_history.csv
paper/tables/fusion_results.csv
paper/tables/ablation_results.csv
paper/figures/fusion_summary_bar.png
paper/figures/fusion_gate_distribution.png
paper/figures/fusion_confusion_{dataset}.png   ← 3 files
paper/figures/fusion_roc_{dataset}.png         ← 3 files
```

---

# 8. Output tables and figures

All paper outputs live in:

```text
paper/figures/
paper/tables/
```

When accessed from GitHub, use full raw URLs. The base for this repo is:

```
https://github.com/Sefayet-Alam/eve-rag/blob/main/
```

---

## EDA figures

| Figure | GitHub link |
|---|---|
| Class balance | [eda_class_balance_processed.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/eda_class_balance_processed.png) |
| Caption length | [eda_caption_length_processed.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/eda_caption_length_processed.png) |
| Image coverage | [eda_image_coverage_processed.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/eda_image_coverage_processed.png) |

![EDA Class Balance](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/eda_class_balance_processed.png)
![EDA Caption Length](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/eda_caption_length_processed.png)
![EDA Image Coverage](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/eda_image_coverage_processed.png)

---

## Stream A figures

| Figure | GitHub link |
|---|---|
| Summary bar | [stream_a_summary_bar.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_a_summary_bar.png) |
| MMFakeBench confusion | [stream_a_confusion_mmfakebench_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_a_confusion_mmfakebench_test.png) |
| VERITE confusion | [stream_a_confusion_verite_test_local.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_a_confusion_verite_test_local.png) |
| MMFakeBench ROC | [stream_a_roc_mmfakebench_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_a_roc_mmfakebench_test.png) |
| VERITE ROC | [stream_a_roc_verite_test_local.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_a_roc_verite_test_local.png) |

![Stream A Summary](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_a_summary_bar.png)

---

## Stream B figures

| Figure | GitHub link |
|---|---|
| Summary bar | [stream_b_summary_bar.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_b_summary_bar.png) |
| NewsCLIP ROC | [stream_b_roc_newsclippings_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_b_roc_newsclippings_test.png) |
| MMFakeBench ROC | [stream_b_roc_mmfakebench_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_b_roc_mmfakebench_test.png) |
| VERITE ROC | [stream_b_roc_verite_test_local.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_b_roc_verite_test_local.png) |
| Score distribution | [stream_b_score_distribution_newsclippings_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_b_score_distribution_newsclippings_test.png) |

![Stream B Summary](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_b_summary_bar.png)

---

## Stream C figures

| Figure | GitHub link |
|---|---|
| Summary bar | [stream_c_summary_bar.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_c_summary_bar.png) |
| NewsCLIP confusion | [stream_c_confusion_newsclippings_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_c_confusion_newsclippings_test.png) |
| NewsCLIP ROC | [stream_c_roc_newsclippings_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_c_roc_newsclippings_test.png) |

![Stream C Summary](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/stream_c_summary_bar.png)

---

## All-streams baseline summary

| Figure | GitHub link |
|---|---|
| All streams + baselines bar | [all_streams_summary_bar.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/all_streams_summary_bar.png) |

![All Streams Summary](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/all_streams_summary_bar.png)

---

## Fusion figures

| Figure | GitHub link |
|---|---|
| Fusion summary bar (default vs ablations) | [fusion_summary_bar.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_summary_bar.png) |
| Gate distribution (interpretability) | [fusion_gate_distribution.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_gate_distribution.png) |
| NewsCLIP confusion | [fusion_confusion_newsclippings_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_confusion_newsclippings_test.png) |
| NewsCLIP ROC | [fusion_roc_newsclippings_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_roc_newsclippings_test.png) |
| MMFakeBench confusion | [fusion_confusion_mmfakebench_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_confusion_mmfakebench_test.png) |
| MMFakeBench ROC | [fusion_roc_mmfakebench_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_roc_mmfakebench_test.png) |
| VERITE confusion | [fusion_confusion_verite_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_confusion_verite_test.png) |
| VERITE ROC | [fusion_roc_verite_test.png](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_roc_verite_test.png) |

![Fusion Summary Bar](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_summary_bar.png)
![Fusion Gate Distribution](https://github.com/Sefayet-Alam/eve-rag/blob/main/paper/figures/fusion_gate_distribution.png)

---

## Key tables

| Table | Path | Description |
|---|---|---|
| Manifest summary | `paper/tables/manifest_summary.csv` | Dataset row/image summary |
| EDA dataset summary | `paper/tables/eda_dataset_summary.csv` | Processed manifest stats |
| Class balance | `paper/tables/class_balance.csv` | Label counts per split |
| NER summary | `paper/tables/ner_manifest_summary.csv` | NER entity stats |
| Visual grounding summary | `paper/tables/visual_grounding_summary.csv` | Stream A grounding stats |
| Entity scorer features | `paper/tables/entity_scorer_feature_summary.csv` | Stream A feature stats |
| Stream A results | `paper/tables/stream_a_results.csv` | Stream A metrics |
| Stream B feature summary | `paper/tables/stream_b_clip_feature_summary.csv` | Stream B feature stats |
| Stream B results | `paper/tables/stream_b_results.csv` | Stream B metrics |
| Stream C evidence summary | `paper/tables/stream_c_evidence_summary.csv` | Evidence text stats |
| Stream C stance summary | `paper/tables/stream_c_stance_summary.csv` | NLI/SBERT feature stats |
| Stream C results | `paper/tables/stream_c_results.csv` | Stream C metrics |
| Baseline results | `paper/tables/baseline_results.csv` | Majority/constant baselines |
| CLIP baseline | `paper/tables/clip_baseline_results.csv` | CLIP cosine baseline |
| All streams + baselines | `paper/tables/all_streams_and_baselines_summary.csv` | Combined summary pivot |
| Fusion results | `paper/tables/fusion_results.csv` | GSAF default results |
| Ablation results | `paper/tables/ablation_results.csv` | All 5 ablation configs |

---

# 9. Output summary so far

## Dataset outputs

| Check | Result |
|---|---:|
| Total manifest rows | 97,361 |
| NewsCLIPpings rows | 85,360 |
| VERITE test rows | 1,001 |
| MMFakeBench rows | 11,000 |
| Local-image rows | 11,621 |
| Flat image files | 11,495 |
| Unique image IDs | 83,461 |
| Broken image paths | 0 |
| Empty captions | 0 |

---

## Stream A result summary

Stream A is weak standalone but valid as a fusion feature source.

| Eval set | Accuracy | Interpretation |
|---|---:|---|
| MMFakeBench test | 62.56% | Below majority baseline (70%) |
| VERITE local | 53.30% | Near random |

Stream A features used in fusion: 12 numeric features including entity counts, visual grounding scores, person/location/org counts.

---

## Stream B result summary

Stream B is the strongest individual stream. The key fix was switching from fake hash-generated image vectors to real pre-extracted NewsCLIPpings PKL embeddings.

| Eval set | Accuracy | AUROC | Interpretation |
|---|---:|---:|---|
| NewsCLIP test | 65.87% | 73.91% | Strong in-distribution signal |
| MMFakeBench test | 46.95% | 63.35% | Some OOD signal; threshold wrong |
| VERITE local | 63.77% | 49.12% | AUROC near random |

Stream B features used in fusion: 11 features including `clip_cosine`, `l2_dist`, `absdiff_mean`, `text_norm`, `has_real_image_emb`.

---

## Stream C result summary

Stream C v3 uses NLI-based stance scoring with decircularized evidence.

| Dataset | Threshold | Accuracy | Macro-F1 | AUROC |
|---|---|---:|---:|---:|
| NewsCLIP val | default (0.5) | 50.73% | 50.77% | — |
| NewsCLIP test | default (0.5) | 51.53% | 51.01% | — |
| MMFakeBench test | default (0.5) | 34.78% | 32.32% | 70.22% |
| **MMFakeBench test** | **tuned (0.47)** | **73.65%** | **63.85%** | 70.22% |
| VERITE test | default (0.5) | 36.16% | 30.70% | 43.25% |

Stream C features used in fusion: 7 features including `support_score`, `contradiction_score`, `neutral_score`, `stance_similarity`, `caption_len`, `evidence_len`, `len_ratio`.

> **Key insight:** Stream C's AUROC on MMFakeBench is 70.22% (strong ranking signal) but the default threshold (0.5) gives only 34.78% accuracy. This is because MMFakeBench has a 70/30 OOC/match imbalance — the right decision boundary is much lower than 0.5. After threshold tuning to 0.47, accuracy jumps to 73.65%.

---

## Baseline summary

| Baseline | NC test Acc | MMFB test Acc | VERITE Acc | NC AUROC | MMFB AUROC | VERITE AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Random uniform | ~50% | ~50% | ~50% | 50.00 | 50.00 | 50.00 |
| Majority class | 50.00% | 70.00% | 66.23% | — | — | — |
| Always OOC (label 1) | 50.00% | 70.00% | 66.23% | — | — | — |
| Always match (label 0) | 50.00% | 30.00% | 33.77% | — | — | — |
| CLIP cosine threshold | 65.75% | 58.12% | 48.35% | 71.97 | 64.13 | 48.54 |
| CLIP cosine rank | 65.74% | 58.13% | 48.25% | 71.97 | 64.13 | 48.54 |

> **What the baselines tell us:**
> - On NewsCLIP, CLIP cosine (71.97 AUROC) is a competitive baseline. Stream B's 73.91 AUROC is meaningfully above it.
> - On VERITE, CLIP cosine AUROC = 48.54 ≈ random. Pure similarity fails on real-world OOC. This is the **similarity-vs-factuality problem** from the literature.
> - MMFakeBench's majority baseline is 70% accuracy (just predict OOC always). Any model must beat this clearly.

---

## Fusion (GSAF) result summary

### What GSAF is

GSAF stands for **Gated Stream Attention Fusion**. The core idea: each stream's reliability varies per sample. Stream B is very reliable for NewsCLIPpings in-distribution but less reliable OOD. Stream C is weak on VERITE (32% evidence coverage) but strong on MMFakeBench. Instead of a fixed combination, GSAF learns a **per-sample softmax gate** over the three streams.

### GSAF architecture

```
Stream A features (12)  ─── Linear(12→32) + LayerNorm + ReLU + Dropout ─┐
Stream B features (11)  ─── Linear(11→32) + LayerNorm + ReLU + Dropout ─┤── concat(3×32=96)
Stream C features  (7)  ─── Linear( 7→32) + LayerNorm + ReLU + Dropout ─┘
                                                                          │
                                             Gate MLP: Linear(96→32)→ReLU→Linear(32→3)→Softmax
                                                                          │
                                             gates = [g_A, g_B, g_C]  (sum = 1.0 per sample)
                                                                          │
                                         weighted sum = g_A·h_A + g_B·h_B + g_C·h_C  (32-dim)
                                                                          │
                                    Classifier: Linear(32→16)→ReLU→Dropout→Linear(16→1)→logit
                                                                          │
                                                         sigmoid(logit) = OOC probability
```

Total parameters: **8,164** (deliberately lightweight — the power is in the features, not the network size).

### Training setup

| Setting | Value |
|---|---|
| Training data | NC train (71,072) + MMFB val half (500) = 71,572 rows |
| Validation (early stop) | NC val (7,024 rows) |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Batch size | 1,024 |
| Early stop | 8 epochs no val AUROC improvement |
| Epochs trained | 37 (stopped at best val AUROC=76.01) |
| Wall time | 38.6 seconds on RTX 4060 Ti |

> **Why add 500 MMFakeBench rows to training?**
> In the first training run (NC only), the gate learned to always trust Stream B because that is the dominant reliable stream in the NewsCLIPpings distribution. Stream C is weak on NewsCLIPpings (near-random NLI signal from short article titles). So the gating contributed nothing over a flat MLP. Adding 500 MMFB rows gives the gate examples where Stream C is the dominant signal. After this fix, the gate genuinely learns to switch. See ablation results below.

### Per-dataset threshold tuning

The model outputs a probability score between 0 and 1. The default threshold is 0.5. Because VERITE and MMFakeBench have class imbalance, a single 0.5 threshold is often wrong.

| Dataset | Tuned threshold | Why |
|---|---|---|
| NewsCLIPpings val | 0.482 | Slightly below 0.5; balanced data; minor bias correction |
| MMFakeBench val_tune | 0.486 | Close to 0.5 — model now well-calibrated on MMFB after mixed training |
| VERITE (no val) | falls back to NC val threshold | No VERITE val split available |

> **Why was the MMFakeBench threshold 0.333 in Run 1 but 0.486 in Run 2?**
> In Run 1, the model trained only on NewsCLIPpings did not generalize well to MMFakeBench's distribution. Its raw scores were miscalibrated — the model under-predicted OOC probability for MMFakeBench samples. Pushing the threshold down to 0.333 compensated for this. In Run 2, with 500 MMFB rows in training, the model genuinely understood MMFakeBench's distribution, so its raw scores were already well-calibrated and the natural threshold (0.486 ≈ 0.5) worked.

### GSAF final results (tuned thresholds)

| Dataset | Accuracy | F1 | F1-macro | AUROC |
|---|---:|---:|---:|---:|
| NewsCLIPpings val | 69.15% | 68.47% | 69.13% | 76.01% |
| **NewsCLIPpings test** | **69.01%** | **68.34%** | **69.00%** | **76.32%** |
| MMFakeBench val_tune | 82.00% | 86.84% | 79.18% | 89.92% |
| **MMFakeBench test** | **78.27%** | **84.41%** | **74.28%** | **87.83%** |
| **VERITE test** | **62.54%** | **76.34%** | **43.21%** | **48.13%** |

> **VERITE note:** AUROC 48.13% means the model's ranking is barely better than random on VERITE. This is a known open limitation — VERITE has no training split, low evidence coverage (32%), and a distribution shift unlike any training data. We report the full VERITE test and the `has_image=1` subset separately (Option A, per the paper convention).

### Fusion gains over individual streams

| Dataset | Metric | Stream B alone | Stream C alone | **GSAF fusion** | Gain vs best stream |
|---|---|---:|---:|---:|---:|
| NewsCLIP test | AUROC | 73.91% | ~52% | **76.32%** | +2.4 pp |
| NewsCLIP test | Acc | 65.87% | ~51% | **69.01%** | +3.1 pp |
| MMFakeBench test | AUROC | 63.35% | 70.22% | **87.83%** | +17.6 pp |
| MMFakeBench test | Acc | 46.95% | 73.65% | **78.27%** | +4.6 pp |

### Ablation results

Four ablation configs were trained under identical conditions:

| Config | What it removes | NC test AUROC | MMFB test AUROC | Conclusion |
|---|---|---:|---:|---|
| `default` (GSAF) | nothing | **76.32%** | **87.83%** | Best overall |
| `no_a` | Stream A (entity grounding) | 75.23% | 65.24% | A contributes +22 pp OOD |
| `no_b` | Stream B (CLIP) | 52.49% | 65.27% | B is the in-distribution workhorse |
| `no_c` | Stream C (NLI evidence) | 75.47% | 62.87% | C contributes +25 pp OOD |
| `mlp_fusion` | Gating mechanism | 76.26% | 65.96% | Gating contributes +22 pp OOD |

> **The smoking gun:** On NewsCLIPpings test, default GSAF (76.32%) and mlp_fusion (76.26%) are nearly identical — both handle in-distribution well. But on MMFakeBench test, GSAF (87.83%) crushes mlp_fusion (65.96%) by **+21.9 pp AUROC**. This proves the gating mechanism is not just cosmetic — it genuinely improves OOD generalization by learning to down-weight the unreliable stream (B) and trust Stream C on distributions where evidence signals matter.

### How the gate weights are calculated — beginner explanation

Think of the gate as the model asking itself, *"for this particular image-text pair, which of my three advisors should I trust most?"*

Here is what happens inside the model step by step:

**Step 1: Project each stream to a common language (32 numbers)**

Each stream gives us features in its own "language" — Stream A has 12 entity numbers, Stream B has 11 CLIP numbers, Stream C has 7 NLI numbers. These can't be directly compared. So each stream's features are passed through a small neural network (a "projector") that converts them all to the same-sized representation of 32 numbers.

$$h_A = \text{ReLU}(\text{LayerNorm}(W_A \cdot x_A + b_A))$$
$$h_B = \text{ReLU}(\text{LayerNorm}(W_B \cdot x_B + b_B))$$
$$h_C = \text{ReLU}(\text{LayerNorm}(W_C \cdot x_C + b_C))$$

**Step 2: Ask "who should I trust?"**

The three 32-number representations are concatenated (joined end to end) to make one 96-number vector. This is passed through a "gate network" — another small neural network — that outputs 3 scores, one for each stream. These 3 scores are converted to probabilities using a **softmax** function, which makes them sum to 1.0:

$$\text{gate\_logits} = W_g \cdot [h_A; h_B; h_C] + b_g \quad \text{(three numbers)}$$
$$[g_A, g_B, g_C] = \text{softmax}(\text{gate\_logits}) \quad \text{so that } g_A + g_B + g_C = 1.0$$

For example, on a typical NewsCLIPpings sample: [g_A=0.22, g_B=0.51, g_C=0.27]. On a typical MMFakeBench sample after training: Stream C gets a higher weight because the gate has learned that C is more reliable there.

**Step 3: Combine using the gate weights**

The final fused representation is a weighted average of the three stream representations:

$$z = g_A \cdot h_A + g_B \cdot h_B + g_C \cdot h_C$$

**Step 4: Classify**

The fused vector z is passed through a small 2-layer classifier that outputs one logit (a raw score). This is converted to a probability with sigmoid:

$$P(\text{OOC}) = \sigma(\text{classifier}(z)) = \frac{1}{1 + e^{-\text{logit}}}$$

If P(OOC) ≥ threshold → predict mismatch (label 1). Otherwise → predict match (label 0).

**How training works:**

During training, all of these weights (projectors, gate network, classifier) are adjusted together by comparing predictions to the true labels and computing a **binary cross-entropy loss**:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

where $y_i \in \{0, 1\}$ is the true label and $\hat{p}_i$ is the predicted OOC probability. The optimizer (Adam) adjusts all weights to minimize this loss over 37 epochs.

---

# 10. Issues faced and how we solved them

## Issue 1: Raw dataset layout was messy

**Problem:** Old folders, duplicated image folders, inconsistent paths. Notebooks scanned raw data manually.

**Solution:** Created canonical `data/raw/` and `data/processed/ooc_pairs/` structures. Updated `src/utils/paths.py`. All notebooks now read `ALL_MANIFEST`.

---

## Issue 2: VERITE had duplicate image folders

**Problem:** Two overlapping VERITE image folders existed:

```text
data/verite/image-text-verification/VERITE/images/
data/verite/images/
```

**Solution:** Canonical raw path is `data/raw/verite/image-text-verification/VERITE/images/`. Processed images are copied into the flat processed folder.

---

## Issue 3: VERITE `true_*.jpg` / `false_*.jpg` naming

**Problem:** VERITE CSV directly references `images/true_0.jpg` etc. Should we rename?

**Solution:** Do **not** rename or delete raw files. The processed folder uses normalized unique names. Raw files stay as-is.

---

## Issue 4: VERITE row count vs image file count mismatch

**Problem:** 621 local-image rows but only 495 unique image files.

**Solution:** VERITE intentionally reuses the same image across multiple rows (different captions paired with the same image). Row count ≠ unique image count. This is expected and correct.

---

## Issue 5: NewsCLIPpings local images unavailable

**Problem:** NewsCLIPpings uses VisualNews images, which are not publicly redistributed. No local image files.

**Solution:** Use `has_image = 0` for all NewsCLIPpings rows. Preserve `clip_image_key` / `clip_text_key`. Use real pre-extracted PKL embeddings from `data/raw/newsclippings/embeddings/`.

---

## Issue 6: Stream B used fake hash-generated image vectors

**Problem:** The first Stream B notebook used deterministic hash vectors for NewsCLIPpings image embeddings:

```python
ivec = stable_hash_vec(clip_image_key)   # WRONG — produces garbage vectors
```

These vectors look like real CLIP embeddings but contain no actual image information.

**Solution:** Load real NewsCLIPpings PKL embeddings:

```python
clip_image_embeddings_train.pkl
clip_image_embeddings_val.pkl
clip_image_embeddings_test.pkl
clip_text_embeddings_train.pkl
clip_text_embeddings_val.pkl
clip_text_embeddings_test.pkl
```

**Result:**

```text
Used real NewsCLIP embeddings: 85,360
Missing image/text embeddings: 0
Fallback (for 380 rows): computed embeddings
```

Stream B NewsCLIP AUROC improved from ~55% → 73.91%.

---

## Issue 7: PowerShell heredoc syntax failed

**Problem:** PowerShell does not support Bash-style inline Python:

```bash
python - <<'PY'   # does not work in PowerShell
```

**Solution:** Use PowerShell-safe script creation:

```powershell
@'
python code here
'@ | Set-Content script.py
python script.py
```

---

## Issue 8: Python backslash escaping in notebook cells

**Problem:** Cells contained unescaped backslashes:

```python
path.replace('\', '/')   # SyntaxError
```

**Solution:**

```python
path.replace("\\", "/")
```

---

## Issue 9: EDA notebook duplicate `savefig()` definition

**Problem:** A later cell overwrote a flexible `savefig()` helper with a stricter one, causing `savefig("figure_name")` calls to fail.

**Solution:** Restored the flexible helper:

```python
def savefig(fig_or_name, name=None, dpi=200):
    if name is None:
        fig = plt.gcf()
        name = fig_or_name
    else:
        fig = fig_or_name
```

---

## Issue 10: Stream C evidence was circular

**Problem:** Stream C evidence text included the caption itself. Measuring similarity between caption and evidence-that-contains-caption is circular — it measures nothing meaningful.

**Solution (v3):** Caption fully removed from all evidence text. VERITE uses the `query` field from `VERITE_articles.csv`. NewsCLIPpings uses article title + topic + source + timestamp. An assertion confirms 0 "Caption:" fields in any `evidence_text`. NLI model replaces cosine similarity.

---

## Issue 11: Old caches contaminating results

**Problem:** Old caches built with fake embeddings or wrong paths would be silently reused.

**Solution:** Delete the relevant cache folder before a major re-run:

```powershell
# For Stream B after fake embedding fix:
Remove-Item cache/stream_b -Recurse -Force

# For Stream C after decircularization:
Remove-Item cache/stream_c/evidence -Recurse -Force
Remove-Item cache/stream_c/stance_scores -Recurse -Force
Remove-Item cache/stream_c/results -Recurse -Force
```

---

## Issue 12: Fusion gating collapsed to uniform weights (Run 1)

**Problem:** After the first fusion training run, the gating mechanism produced nearly uniform gate weights regardless of the dataset. `default` GSAF (76.11 AUROC) was virtually identical to `mlp_fusion` (76.12 AUROC). This meant the gating did not help — the model was just a fancy MLP.

**Root cause:** The model was trained only on NewsCLIPpings. In that distribution, Stream B dominates. Stream C is weak (short article-title evidence → near-random NLI scores). So the gate learned to always trust B, and had no signal to learn switching behavior. When applied to MMFakeBench, this prior was wrong (Stream C is strong there), and the model failed: MMFakeBench test AUROC = 61.29%.

**Solution (Fix A — mixed-domain training):** Added a stratified random half of `mmfakebench_val` (500 rows) to the training pool. This gave the gate examples where Stream C is the reliable signal. The gate then learned to genuinely switch between streams.

**Result:**

| | Before (NC only) | After (mixed) |
|---|---|---|
| MMFakeBench test AUROC | 61.29% | **87.83%** |
| default vs mlp_fusion gap (MMFB AUROC) | 0.00% | **+21.9 pp** |

---

## Issue 13: `sklearn.train_test_split` crashed with Pandas 3.0 PyArrow-backed strings

**Problem:** Python 3.13 + Pandas 3.0 uses PyArrow-backed string columns by default. Sklearn's `train_test_split` internally uses NumPy-style integer indexing which is incompatible with PyArrow arrays:

```
TypeError: only integer scalar arrays can be converted to a scalar index
```

**Solution:** Replaced `train_test_split` with a manual NumPy stratified split:

```python
rng = np.random.default_rng(SEED)
for lbl in sorted(mmfb_val_df["label"].unique().tolist()):
    pids = mmfb_val_df.loc[mmfb_val_df["label"] == lbl, "pair_id"].astype(str).tolist()
    rng.shuffle(pids)
    half = len(pids) // 2
    mmfb_train_set.update(pids[:half])
    mmfb_tune_set.update(pids[half:])
```

Casting to `.tolist()` materializes plain Python strings, avoiding the PyArrow indexing issue.

---

# 11. Important implementation notes

## GPU usage

Always use GPU for:

- CLIP image/text encoding.
- SentenceTransformer encoding.
- Neural model training (GSAF, Stage 1/2 MLP).
- MLLM inference (future).

Target machine:

```text
RTX 4060 Ti, 8.6 GB VRAM
Python 3.13, CUDA 12.4
PyTorch 2.6.0+cu124
```

Recommended batch sizes:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_BATCH_SIZE = 48
TEXT_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 1024   # fusion MLP is tiny; can handle large batches
```

If CUDA OOM:

```python
IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 512
```

---

## Always read from manifests

```python
from src.utils.paths import ALL_MANIFEST
df = pd.read_csv(ALL_MANIFEST, low_memory=False)
```

Never scan `data/raw/` manually in notebooks.

---

## `pair_id` vs `image_id`

- Use `pair_id` for row-level outputs: predictions, labels, features, evidence, stance scores.
- Use `image_id` for image-level outputs: CLIP embeddings, visual grounding vectors.

---

## Keep raw data unchanged

Never delete raw dataset files. They cannot be regenerated without re-downloading.

---

## `src/models/fusion.py` is auto-generated

`fusion_module.ipynb` writes this file. Do not manually edit it. If you change the architecture, edit the notebook cell and re-run.

---

## GitHub note

The repo can contain code, configs, notebooks, README, GRAND_PLAN, small result tables, and final figures. Large datasets, PKL embedding files, and `cache/` outputs exceed GitHub limits and should be in `.gitignore`.

```gitignore
data/raw/
data/processed/ooc_pairs/images/
cache/
*.pkl
*.pt
```


---

# 12. How to run the project

## Full first-time setup

```text
1. notebooks/01_setup/env_check.ipynb       ← verify environment
2. notebooks/01_setup/data_setup.ipynb      ← build manifests, flat images, splits
3. notebooks/01_setup/env_check.ipynb       ← verify again after setup
4. notebooks/01_setup/eda.ipynb             ← EDA figures and tables
```

## Stream A run

```text
1. notebooks/02_stream_a/ner_pipeline_v2_updated.ipynb
2. notebooks/02_stream_a/visual_grounding_v2_updated.ipynb
3. notebooks/02_stream_a/entity_scorer_v2_updated.ipynb
4. notebooks/02_stream_a/stream_a_results_v2_updated.ipynb
```

## Stream B run

```text
1. notebooks/03_stream_b/clip_features_updated.ipynb
2. notebooks/03_stream_b/lora_stage1_updated.ipynb
3. notebooks/03_stream_b/lora_stage2_fixed_updated.ipynb
4. notebooks/03_stream_b/stream_b_results_fixed_updated.ipynb
```

## Stream C run

```text
1. notebooks/04_stream_c/evidence_retrieval_final.ipynb
2. notebooks/04_stream_c/evidence_val_final.ipynb
3. notebooks/04_stream_c/stance_scoring_final.ipynb       ← set DEBUG=False for full run
4. notebooks/04_stream_c/stream_c_results_final.ipynb
```

> If re-running Stream C from scratch, delete caches first:
> ```powershell
> Remove-Item cache/stream_c -Recurse -Force
> ```

## Fusion run

```text
1. notebooks/05_fusion/baselines_final.ipynb
2. notebooks/05_fusion/fusion_module.ipynb
3. notebooks/05_fusion/fusion_training.ipynb              ← restart kernel; run top to bottom
```

> `fusion_module.ipynb` must run before `fusion_training.ipynb` — it generates `src/models/fusion.py` and `cache/fusion/fusion_features_all.csv`.

## Planned next runs (pending)

```text
1. notebooks/07_eval/newsclippings_eval.ipynb
2. notebooks/07_eval/cross_dataset_eval.ipynb
3. notebooks/07_eval/ablations.ipynb
4. notebooks/08_paper/tables.ipynb
5. notebooks/08_paper/figures.ipynb
6. notebooks/06_mllm/qwen_finetune.ipynb
7. notebooks/06_mllm/mllm_inference.ipynb
```

---

# 13. Current limitations and next research fixes

## ~~Limitation 1: Stream C evidence was circular~~ — ✅ RESOLVED (v3)

Stream C v3 uses independent evidence. Caption never appears in `evidence_text`. Assert confirms 0 leaks. NLI replaces cosine similarity.

## ~~Limitation 2: MMFakeBench OOD transfer was weak~~ — ✅ RESOLVED

Mixed-domain training (adding 500 MMFB val rows) + threshold tuning resolved this. MMFakeBench test AUROC went from 61.29% → 87.83%.

## ~~Limitation 3: Gating mechanism was ineffective~~ — ✅ RESOLVED

Run 1 GSAF ≈ mlp_fusion (no gating benefit). After mixed-domain training, GSAF outperforms mlp_fusion by +21.9 AUROC on MMFakeBench test. Gating is now empirically justified.

## ~~Limitation 4: Stream B used fake embeddings~~ — ✅ RESOLVED

Real PKL embeddings now used. 0 missing embeddings.

## Limitation 5: VERITE AUROC remains near random

VERITE test AUROC = 48.13%. This is a known open problem:

- VERITE has no training or validation split — no in-distribution signal to learn from.
- Only 32.47% of VERITE rows have high-quality evidence (the `query` field from `VERITE_articles.csv`).
- The distribution shift from NewsCLIPpings to VERITE is large.

**Mitigation for the paper:** Report VERITE `has_image=1` subset separately. Report AUROC alongside accuracy (since accuracy can be inflated by majority-class prediction). Acknowledge as a limitation and future work direction.

**Possible future fix:** Incorporate real-time web retrieval or use VERITE-specific evidence sources. This is out of scope for the current submission.

## Limitation 6: MLLM explanation not implemented

After fusion, the project plan includes a multimodal LLM explanation layer (Qwen or similar) to generate human-readable explanations of why a pair is classified as OOC.

**Plan:**
- Example selection from fusion predictions.
- Qwen/MLLM inference with chain-of-thought prompting.
- Structured explanation format (entity mismatch / evidence contradiction / visual grounding failure).
- Explanation quality evaluation.

## Limitation 7: This is not true RAG

Stream C currently uses offline metadata evidence, not real-time web retrieval. The "RAG" in EVE-RAG refers to the design intent (Retrieval-Augmented Grounding), not the current implementation. The current Stream C should be described in the paper as an **offline evidence baseline**.

---

# 14. Glossary

| Term | Definition |
|---|---|
| **OOC** | Out-of-context. An image used in a misleading context (label 1). |
| **Match** | A correct image-text pair (label 0). |
| **Mismatch** | An incorrect/misleading image-text pair (label 1). |
| **Manifest** | A normalized CSV file storing all metadata for one dataset split. |
| **pair_id** | Unique string ID for one image-text pair row. |
| **image_id** | Unique ID for one image file or embedding. |
| **has_image** | 1 if a local processed image file exists for this row, else 0. |
| **clip_image_key** | Key used to look up pre-extracted CLIP image embeddings. |
| **clip_text_key** | Key used to look up pre-extracted CLIP text embeddings. |
| **CLIP** | Contrastive Language–Image Pretraining. A model by OpenAI that maps images and text into the same vector space so they can be compared. |
| **Cosine similarity** | A number between -1 and 1 measuring how aligned two vectors are. CLIP image and text embeddings should have high cosine similarity if they "match". |
| **AUROC** | Area Under the Receiver Operating Characteristic curve. Measures ranking quality independent of any fixed decision threshold. 1.0 = perfect, 0.5 = random. The primary metric in EVE-RAG. |
| **Threshold** | The probability cutoff above which the model predicts OOC (label 1). Default is 0.5. Per-dataset tuning finds the threshold that maximizes macro-F1 on a validation split. |
| **Macro-F1** | F1 score averaged equally across both classes (match and OOC). Better than plain accuracy when classes are imbalanced. |
| **OOD** | Out-of-domain. VERITE and MMFakeBench are OOD relative to NewsCLIPpings training data. |
| **NER** | Named Entity Recognition. Extracting mentions of people, places, organizations from text. |
| **Visual grounding** | Checking whether a named entity mentioned in text is visually supported by the image. |
| **LoRA** | Low-Rank Adaptation. Used as naming convention in Stream B notebooks. Current updated notebooks use lightweight MLP classifiers rather than full LoRA fine-tuning. |
| **NLI** | Natural Language Inference. Given a premise and a hypothesis, classify the relationship as: entailment (premise supports hypothesis), contradiction (premise contradicts), or neutral. Used in Stream C. |
| **DeBERTa** | A transformer model (`cross-encoder/nli-deberta-v3-base`) used for NLI stance scoring in Stream C v3. |
| **SBERT** | Sentence-BERT. Used for sentence-level semantic similarity (`all-MiniLM-L6-v2`). Used as a complementary feature in Stream C. |
| **RAG** | Retrieval-Augmented Generation/Grounding. The design intent of Stream C: retrieve external evidence about a claim and use it to verify or refute the image-text pair. Currently implemented as offline metadata. |
| **Stance scoring** | Classifying whether evidence supports, contradicts, or is neutral toward a claim. The output of Stream C. |
| **GSAF** | Gated Stream Attention Fusion. The EVE-RAG fusion architecture. Learns a per-sample softmax gate over three streams, weighting them by their reliability for that sample. |
| **Gate** | In GSAF, a learned weight in [0,1] for each stream, summing to 1.0 per sample. Tells the model how much to trust each stream for each prediction. |
| **Softmax** | A function that converts raw scores into probabilities (all positive, sum to 1.0). Used for the gate. |
| **Ablation** | A controlled experiment where one component is removed to measure its contribution. EVE-RAG has 4 ablations: no_a, no_b, no_c, mlp_fusion. |
| **Mixed-domain training** | Training on samples from more than one dataset distribution to prevent the model from over-specializing on one distribution. In EVE-RAG: NC train + 500 MMFB val rows. |
| **Calibration** | How well a model's predicted probabilities match actual outcomes. A well-calibrated model with threshold 0.5 should not need aggressive threshold correction. |
| **Feature standardization** | Subtracting the mean and dividing by standard deviation, computed on the training split only, to bring all features to a common scale. Applied to all fusion features. |
| **Zero-imputation** | Filling missing feature values with 0. After standardization, 0 equals the training mean — a sensible default. |
| **Early stopping** | Stopping training when validation performance has not improved for N epochs (patience). Prevents overfitting. EVE-RAG uses patience=8 on val AUROC. |
| **MLLM** | Multimodal Large Language Model. A language model that can process both images and text. Used in the future explanation layer. |

---

# 15. Beginner mental model

Read this section if you're new to the project and want to understand everything from scratch.

## Step 1: What is the task?

We have a database of **image-text pairs**. Each pair has a label:

- `0` = the image genuinely belongs with the text (real news).
- `1` = the image was taken out of context or used misleadingly (fake/OOC).

Our job is to build a model that looks at each pair and predicts 0 or 1.

## Step 2: Why not just use one model?

No single signal catches all types of OOC. Consider:

- A photo of a protest can look visually correct with any political caption. You need **external evidence** to know if the date/location is right.
- A doctored portrait might fool CLIP similarity but fail **entity grounding** (the person named in the caption doesn't match the face in the image).
- A real historical photo reused with a new caption might fool any image-text model but fail on **NLI evidence** from news articles.

So we build three independent detectors (streams) and fuse them.

## Step 3: Where does the data live?

```text
data/processed/ooc_pairs/manifests/all.csv     ← one row per image-text pair
data/processed/ooc_pairs/images/               ← local image files (VERITE + MMFakeBench only)
data/raw/newsclippings/embeddings/             ← pre-extracted CLIP vectors for NewsCLIPpings
```

Every notebook starts by loading `all.csv`. From there, you can reach any image, embedding, or metadata.

## Step 4: Stream A — "Who is in this image?"

Stream A asks: "Do the named people, places, and organizations mentioned in the caption appear to be visually or evidentially supported?"

1. **NER:** Extract entity names from the caption (e.g., "President Biden", "Paris", "Apple Inc.").
2. **Visual grounding:** For images we have locally, compute how well each entity name "matches" the image using CLIP.
3. **Scoring:** Build 12 numeric features (entity count, grounding scores, person/org/location counts, etc.).
4. **Classifier:** Train a simple logistic regression on these 12 features.

Stream A alone is weak — visual grounding is noisy and many caption entities don't have clear visual correlates. But these 12 features are useful to the fusion model.

## Step 5: Stream B — "Do the image and text look compatible?"

Stream B asks: "If I map the image and the text into the same vector space (using CLIP), are they pointing in the same direction?"

1. **Embeddings:** For NewsCLIPpings, load pre-extracted PKL CLIP vectors. For VERITE/MMFakeBench, encode local images with OpenCLIP.
2. **Features:** Compute cosine similarity, L2 distance, element-wise differences, norms — 11 numeric features.
3. **Stage 1 / Stage 2 MLP:** Train a 2-stage neural classifier.

Stream B is the strongest individual stream (AUROC 73.91% on NewsCLIPpings). It captures the visual-semantic compatibility signal directly.

## Step 6: Stream C — "Does the news evidence agree with this caption?"

Stream C v3 asks: "Given independent evidence about this image/story, does that evidence logically entail, contradict, or remain neutral toward the caption?"

This is Natural Language Inference (NLI) applied to fact-checking.

- **Evidence:** Not the caption itself, but independent metadata — article titles, publication source, VERITE article query field.
- **NLI model:** `cross-encoder/nli-deberta-v3-base` reads (evidence, caption) pairs and outputs 3 probabilities: P(entailment), P(neutral), P(contradiction).
- **Output features:** 7 numeric features including the 3 NLI probabilities, SBERT cosine similarity, text length features.

Stream C is weak on NewsCLIPpings (near-random) because article title evidence is very short and vague. But on MMFakeBench (where evidence encodes source type and fake class metadata), it reaches 70.22% AUROC.

## Step 7: Fusion — "Which stream should I trust right now?"

The fusion model (GSAF) combines all three streams:

1. Each stream's features are projected to a common 32-dimensional space.
2. A **gate network** computes 3 weights (summing to 1.0) — one per stream.
3. The streams are combined as a weighted average using these gates.
4. A classifier predicts OOC probability from the combined representation.

The key insight: the gate is **per-sample**, not global. For a NewsCLIPpings sample, the gate might say "trust Stream B (0.51), ignore Stream A (0.22), be a little skeptical of Stream C (0.27)." For an MMFakeBench sample, after seeing 500 MMFB training examples, the gate might redistribute weight toward Stream C.

**Why this matters:** On MMFakeBench test, GSAF achieves 87.83% AUROC vs 65.96% for a flat MLP fusion — a +21.9 pp gap. The gating genuinely generalizes OOD in a way that a fixed-weight combination cannot.

## Step 8: What comes next?

The fusion model is complete. The remaining work is:

1. **Formal evaluation notebooks** (`07_eval/`) — deeper analysis, confusion matrices, qualitative examples.
2. **MLLM explanation** (`06_mllm/`) — use a multimodal LLM to generate human-readable explanations of each prediction.
3. **Paper writing** (`08_paper/`) — assemble tables and figures for submission.

Target venues: **IEEE Transactions on Information Forensics & Security** (Q1 journal, rolling deadline) or **WACV 2026** (top conference, ~July 2025 deadline).

---

# End of GRAND_PLAN.md
