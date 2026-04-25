# EVE-RAG — Grand Plan (Updated)

**Last updated after Stream A completion.**

---

## Project Overview

**Task:** Out-of-context (OOC) multimodal misinformation detection — given an image-caption pair, determine whether the caption accurately describes the image.

**Architecture:** Three-stream ensemble (EVE-RAG):
- **Stream A** — Entity mismatch scoring (NER + visual grounding + CLIP similarity)
- **Stream B** — LoRA fine-tuned CLIP (load pretrained `best.ckpt`, adapt with LoRA)
- **Stream C** — Evidence retrieval + stance scoring (reverse image search)
- **Fusion** — Learned combination of all stream features
- **MLLM** — Qwen2-VL 7B for explanation generation + CoT reasoning

---

## Critical Findings & Data Notes

### 1. NEWSCLIP path points to ROOT (not merged_balanced)
```
NEWSCLIP = data/newsclippings/          # ROOT level
           ├── test.json   (7,264 pairs)   # both genuine+OOC via composite keys
           ├── train.json  (71,072 pairs)
```
**NOT** `data/newsclippings/data/merged_balanced/` (3,632 / 35,536 pairs).

### 2. Embedding keys are INT (not str)
All pre-extracted embeddings in `cache/clip_embeddings/` use **integer** image_ids as keys.
```python
# CORRECT:
img_embs[row['image_id']]        # int key
txt_embs[row['id']]              # int key

# WRONG (causes 0 scores):
img_embs[str(row['image_id'])]   # str key — never matches
```

### 3. NC id-collision fix — composite key
merged_balanced JSON has same `id` for genuine and OOC pair of each image.
Fix: use `f"{id}_{int(falsified)}"` as composite pair key throughout all caches.

### 4. VisualNews captions — GATED (not downloadable)
`data/visualnews/origin/data.json` does not exist. Captions for NC come from:
```
data/newsclippings/metadata/{train,val,test}.json
```
Fields: `caption`, `caption_entities_spacy`, `caption_entities_rel`, `image_path`, `image_has_person`, `topic`, `source`.

### 5. Pretrained NC models available — Stream B shortcut
```
data/newsclippings/models/vitb32/
  merged_balanced_vitb32/            best.ckpt + config.yaml
  person_sbert_text_text_vitb32/     best.ckpt + config.yaml
  scene_resnet_place_vitb32/         best.ckpt + config.yaml
  semantics_clip_text_image_vitb32/  best.ckpt + config.yaml
  semantics_clip_text_text_vitb32/   best.ckpt + config.yaml
```
**Stream B loads `merged_balanced_vitb32/best.ckpt` as backbone** — no training from scratch.

### 6. Multiple embedding types available
```
cache/clip_embeddings/
  clip_image_embeddings/  {test, train, val}.pkl   ViT-B/32, 50K entries, int keys
  clip_text_embeddings/   {test, train, val}.pkl   ViT-B/32, int keys
  facenet_embeddings/     {test, train, val}.pkl   face identity embeddings
  places_resnet50/        {test, val}.pkl           scene classification (NO train!)
  sbert_embeddings/       {test, train, val}.pkl   SBERT sentence embeddings
  verite/                 image_embeddings.npy      ViT-L/14 (768-dim)
                          text_embeddings.npy       ViT-L/14 (768-dim)
```

### 7. VERITE images — two locations
```
data/verite/images/                                   # ~43/1001 images present
data/verite/image-text-verification/VERITE/images/   # ALL 1,001 images present ✓
```
**Fix for VG:** use `data/verite/image-text-verification/VERITE/images/` path.
Also: `VERITE_clip_image_embeddings_ViTL14.npy` at same location — pre-extracted.

### 8. MMFakeBench image paths
Root: `data/mmfakebench/MMFakeBench_test/` (not `data/mmfakebench/`)
Structure: `MMFakeBench_test/{fake,real}/{subcategory}/{filename}.png`

---

## Directory Structure

```
F:\GitHub\eve-rag\
├── cache\
│   ├── clip_embeddings\
│   │   ├── clip_image_embeddings\  test.pkl  train.pkl  val.pkl    [int keys, 50K]
│   │   ├── clip_text_embeddings\   test.pkl  train.pkl  val.pkl    [int keys]
│   │   ├── facenet_embeddings\     test.pkl  train.pkl  val.pkl
│   │   ├── places_resnet50\        test.pkl  val.pkl               [NO train.pkl]
│   │   ├── sbert_embeddings\       test.pkl  train.pkl  val.pkl
│   │   └── verite\                 image_embeddings.npy  text_embeddings.npy  [ViT-L/14]
│   ├── entity_scorer\              features_{verite,mmfake,newsclippings_*,all}.pkl
│   ├── evidence\                   (empty — Stream C)
│   ├── ner\                        entities_{all,mmfake,newsclippings_*,verite}.pkl
│   ├── ner_annotations\            (empty)
│   ├── stance_scores\              (empty — Stream C)
│   ├── visual_entities\            (empty)
│   └── visual_grounding\           vg_{all,mmfake,newsclippings_*,verite}.pkl
│
├── data\
│   ├── mmfakebench\
│   │   ├── MMFakeBench_test.json                 10,000 pairs  [text, image_path, gt_answers]
│   │   ├── MMFakeBench_val.json                  ~1,000 pairs
│   │   └── MMFakeBench_test\
│   │       ├── fake\  {14 subcategories}
│   │       └── real\  {6 subcategories × 500 images}
│   │
│   ├── newsclippings\
│   │   ├── test.json                ROOT 7,264 pairs  [id, image_id, falsified — int keys]
│   │   ├── train.json               ROOT 71,072 pairs
│   │   ├── data\
│   │   │   ├── merged_balanced\    test 3,632  train 35,536  val 3,513  [id collision!]
│   │   │   ├── full\               train 71,072
│   │   │   ├── person_sbert_text_text\
│   │   │   ├── scene_resnet_place\
│   │   │   ├── semantics_clip_text_image\
│   │   │   └── semantics_clip_text_text\
│   │   ├── embeddings\             [same data as cache/, different filename format]
│   │   │   ├── clip_image_embeddings\  clip_image_embeddings_{split}.pkl
│   │   │   ├── clip_text_embeddings\
│   │   │   ├── facenet_embeddings\
│   │   │   ├── places_resnet50\
│   │   │   └── sbert_embeddings\
│   │   ├── metadata\
│   │   │   ├── test.json            7,264 entries  [caption, caption_entities_spacy,
│   │   │   ├── train.json                           caption_entities_rel, image_path,
│   │   │   └── val.json                             image_has_person, topic, source]
│   │   └── models\
│   │       ├── generic_classifier\  best.ckpt  config.yaml  data.csv
│   │       └── vitb32\
│   │           ├── merged_balanced_vitb32\            best.ckpt  config.yaml  ← USE THIS
│   │           ├── person_sbert_text_text_vitb32\     best.ckpt  config.yaml
│   │           ├── scene_resnet_place_vitb32\         best.ckpt  config.yaml
│   │           ├── semantics_clip_text_image_vitb32\  best.ckpt  config.yaml
│   │           └── semantics_clip_text_text_vitb32\   best.ckpt  config.yaml
│   │
│   └── verite\
│       ├── VERITE.csv                     1,001 rows  [caption, label, image_path]
│       ├── VERITE_articles.csv
│       ├── images\                       ~43/1001 images  ← INCOMPLETE
│       └── image-text-verification\
│           └── VERITE\
│               ├── VERITE.csv
│               ├── VERITE_clip_image_embeddings_ViTL14.npy  ← USE THIS for VG
│               └── images\              ALL 1,001 images   ← USE THIS path
│
├── notebooks\
│   ├── 01_setup\
│   ├── 02_stream_a\
│   ├── 03_stream_b\
│   ├── 04_stream_c\
│   ├── 05_fusion\
│   ├── 06_mllm\
│   ├── 07_eval\
│   └── 08_paper\
│
├── paper\
│   ├── figures\   [all PNG outputs]
│   └── tables\    [all CSV outputs]
│
└── src\
    └── utils\
        └── paths.py   NEWSCLIP = data/newsclippings/  CLIP_EMB = cache/clip_embeddings/
```

---

## Notebook Run Status

### ✅ 01_setup
| Notebook | Status | Notes |
|---|---|---|
| `env_check.ipynb` | ✅ Done | en_core_web_sm loaded (trf not installed) |
| `data_download.ipynb` | ✅ Done | All datasets present |
| `eda.ipynb` | ✅ Done | PNG figs + CSV tables auto-saved |

### ✅ 02_stream_a
| Notebook | Version | Status | Notes |
|---|---|---|---|
| `ner_pipeline.ipynb` | v2 | ✅ Done | 85,360 NC pairs, 100% entity coverage from metadata |
| `visual_grounding.ipynb` | v2 | ✅ Done | NC VG real scores; VERITE path fix needed for rerun |
| `entity_scorer.ipynb` | v2 | ✅ Done | Composite keys; NC clip_sim valid |
| `stream_a_results.ipynb` | v2 | ✅ Done | All 3 datasets evaluated |

### ⏳ 03_stream_b
| Notebook | Status | Notes |
|---|---|---|
| `clip_features.ipynb` | ❌ Not started | Load pretrained `best.ckpt`; extract features; fix composite keys |
| `lora_stage1.ipynb` | ❌ Not started | LoRA on top of pretrained backbone |
| `lora_stage2.ipynb` | ❌ Not started | OOC-specific fine-tuning stage |
| `stream_b_results.ipynb` | ❌ Not started | |

### ⏳ 04_stream_c
| Notebook | Status | Notes |
|---|---|---|
| `evidence_retrieval.ipynb` | ❌ Not started | Google Vision API / reverse image search |
| `stance_scoring.ipynb` | ❌ Not started | Reranking retrieved evidence |
| `stream_c_results.ipynb` | ❌ Not started | |

### ⏳ 05_fusion
| Notebook | Status | Notes |
|---|---|---|
| `baselines.ipynb` | ❌ Not started | |
| `fusion_module.ipynb` | ❌ Not started | |
| `fusion_training.ipynb` | ❌ Not started | |

### ⏳ 06_mllm
| Notebook | Status | Notes |
|---|---|---|
| `qwen_finetune.ipynb` | ❌ Not started | Qwen2-VL 7B LoRA |
| `mllm_inference.ipynb` | ❌ Not started | |
| `chain_of_thought.ipynb` | ❌ Not started | CoT prompting |

### ⏳ 07_eval
| Notebook | Status | Notes |
|---|---|---|
| `newsclippings_eval.ipynb` | ❌ Not started | |
| `cross_dataset_eval.ipynb` | ❌ Not started | VERITE + MMFakeBench |
| `ablations.ipynb` | ❌ Not started | |
| `qualitative_analysis.ipynb` | ❌ Not started | |

### ⏳ 08_paper
| Notebook | Status | Notes |
|---|---|---|
| `figures.ipynb` | ❌ Not started | |
| `tables.ipynb` | ❌ Not started | |
| `explanation_eval.ipynb` | ❌ Not started | |

---

## Stream A — Final Results

| Dataset | Majority | Stream A LR | Δ | AUROC | Signal |
|---|---|---|---|---|---|
| NewsCLIPpings test | 50.0% | **66.0%** | **+16.0%** | **73.2%** | Strong (in-dist) |
| VERITE | 66.2% | 65.7% | -0.6% | 55.3% | Weak (OOD) |
| MMFakeBench | 70.0% | 70.6% | +0.6% | 58.1% | Weak (diverse fakes) |

**Best single feature:** `clip_sim` (corr = -0.236 with OOC label)
- Lower image-text CLIP similarity → more likely OOC

**Paper narrative:**
Stream A achieves AUROC 73% in-distribution (NC) but drops to ~55-58% on OOD datasets.
This motivates Stream B (LoRA fine-tuning for cross-domain generalisation) and
Stream C (evidence retrieval for cases where visual similarity alone is insufficient).

---

## Stream B — Plan

**Key insight from data discovery:** 5 pretrained ViT-B/32 classifiers already exist.
`clip_features.ipynb` will:
1. Load `data/newsclippings/models/vitb32/merged_balanced_vitb32/best.ckpt`
2. Extract penultimate-layer features for all NC + VERITE + MMFakeBench pairs
3. Use composite key `f"{id}_{int(falsified)}"` for NC (same as Stream A)
4. Also extract FaceNet + Places + SBERT features (all available in cache/)
5. Save feature matrix for fusion

`lora_stage1.ipynb` will apply LoRA adapters to the pretrained backbone for
cross-domain adaptation (target: VERITE + MMFakeBench generalisation).

---

## Open Issues / TODOs

| # | Issue | Fix | Affects |
|---|---|---|---|
| 1 | VERITE VG only 43/1001 images loaded | Use `data/verite/image-text-verification/VERITE/images/` | visual_grounding_v2 |
| 2 | places_resnet50 has no train.pkl | Use test+val only; skip for train features | clip_features |
| 3 | VERITE VG loaded from old cache | Delete vg_verite.pkl, rerun with correct image path | visual_grounding_v2 |
| 4 | MMFake AUROC < 0.5 for entity features | Expected — entity grounding has no signal for non-OOC fakes | Known limitation |
| 5 | NC images not available | No visual_news folder; use pre-extracted embeddings only | All notebooks |
