# MAGE / EVE-RAG — Complete Beginner-to-Seminar Manual

**Paper title:** *MAGE: Multi-Stream Adaptive Gating for Cross-Dataset Out-of-Context Image Detection*  
**Project codename:** EVE-RAG  
**Repository:** `github.com/Sefayet-Alam/eve-rag`  
**Purpose of this file:** A single compact but complete manual that explains the paper from beginner level to seminar / conference defense level. It includes the main project explanation, dataset details, preprocessing, notebooks, outputs, issues, remaining tasks, glossary, and professor-style Q&A.

---

## How to Use This Manual

Read it in this order:

1. **Sections 1–3** to understand the problem and project story.
2. **Sections 4–8** to understand datasets, preprocessing, notebooks, outputs, and results.
3. **Sections 9–12** to understand issues, limitations, remaining work, and implementation details.
4. **Sections 13–15** before a seminar or conference Q&A.

---

# 1. Summary

## 1.1 One-page summary

This project studies **out-of-context image detection**. The task is:

> Given an image and a caption, decide whether the image truly belongs with that caption or whether the image has been reused in a misleading context.

This is different from ordinary fake-image detection. In out-of-context misinformation, the image may be real and the caption may sound plausible, but the **pairing** is false. For example, an old protest image from one country can be reused with a new caption about a different event.

MAGE / EVE-RAG solves this using three complementary streams:

| Stream | Name | Main question |
|---|---|---|
| **Stream A** | Entity + visual grounding | Do people, places, and organizations mentioned in the caption appear visually or evidentially supported? |
| **Stream B** | CLIP representation | Do the image and text look compatible in a shared image-text embedding space? |
| **Stream C** | Evidence + stance reasoning | Does independent evidence support, contradict, or remain neutral toward the caption? |

The final model is **GSAF: Gated Stream Attention Fusion**. Instead of treating all streams equally, GSAF learns a **per-sample gate** that decides how much to trust Stream A, Stream B, and Stream C for each image-text pair.

## 1.2 Headline results

| Dataset | Setting | Accuracy | Macro-F1 | AUROC |
|---|---|---:|---:|---:|
| NewsCLIPpings test | In-distribution | **69.01%** | **69.00%** | **76.32%** |
| MMFakeBench test | Out-of-distribution | **78.27%** | **74.28%** | **87.83%** |
| VERITE test | Real-world fact-checking | **62.54%** | **43.21%** | **48.13%** |

The strongest result is on **MMFakeBench**, where GSAF reaches **87.83% AUROC** and strongly beats non-gated fusion. The main limitation is **VERITE**, where AUROC remains near random because there is no training split, evidence coverage is limited, and the distribution is very different from the training data.

## 1.3 Core contribution

The main contribution is not just using CLIP, NLI, or metadata. The contribution is the **multi-stream adaptive fusion design**:

1. It separates the problem into entity grounding, visual-semantic compatibility, and evidence stance.
2. It uses a lightweight gated fusion model to combine these signals.
3. It evaluates across datasets instead of only reporting one in-distribution score.
4. It includes ablations showing that adaptive gating matters, especially on MMFakeBench.

---

# 2. Title Explained Simply

**MAGE: Multi-Stream Adaptive Gating for Cross-Dataset Out-of-Context Image Detection**

| Term | Meaning |
|---|---|
| **MAGE** | The method name used for the paper. |
| **Multi-Stream** | The system uses three different evidence streams instead of one model only. |
| **Adaptive Gating** | The model learns how much to trust each stream for each example. |
| **Cross-Dataset** | The model is tested on multiple datasets, including datasets different from training. |
| **Out-of-Context Image Detection** | Detecting whether an image-caption pair is misleading because the image is used in the wrong context. |

---

# 3. Problem Background and Motivation

## 3.1 What is out-of-context image detection?

Out-of-context image detection asks whether an image and its caption truly belong together. It is not enough to ask whether the image is real. The image can be real but still misleading when paired with the wrong caption.

## 3.2 Why is this hard?

Out-of-context misinformation is subtle because:

- The image and text can look visually compatible.
- The image may be genuine, not AI-generated or edited.
- CLIP similarity can be high even when the factual context is wrong.
- The mismatch may depend on person, location, event, time, or evidence source.
- A single model may overfit to one dataset and fail on another.

## 3.3 Why use three streams?

No single signal is enough:

| Failure case | Needed signal |
|---|---|
| Caption names a person not shown in the image | Entity / visual grounding |
| Image and caption are semantically similar but factually wrong | Evidence stance reasoning |
| Caption generally matches visual content | CLIP compatibility |
| Dataset distribution changes | Adaptive fusion / gating |

---

# 4. Dataset Details, Sources, and Gains

## 4.1 Datasets used

EVE-RAG uses three public datasets:

| Dataset | Type | Size | Images available? | Role in project |
|---|---|---:|---|---|
| **NewsCLIPpings** | News image-text pairs | 85,360 rows | No local images; real CLIP embeddings available | Main training / in-distribution dataset |
| **VERITE** | Real-world OOC pairs | 1,001 rows | Partial: 621 rows / 495 unique files | Hard real-world test set |
| **MMFakeBench** | Mixed-source fake / OOC-style benchmark | 11,000 rows | Full local image coverage | Important OOD evaluation dataset |

## 4.2 Where the datasets came from

| Dataset | Source | Notes |
|---|---|---|
| NewsCLIPpings | `S-Abdelnabi/NewsCLIPpings` GitHub repository | Provides pre-extracted CLIP/SBERT/ResNet embeddings. Local VisualNews images are not redistributed. |
| VERITE | `stevejpapad/image-text-verification` GitHub repository | Provides CSV files, article metadata, and a subset of images. |
| MMFakeBench | `poloclub/mmfakebench` GitHub / Hugging Face | Provides validation/test JSON files and image folders. |

## 4.3 Final manifest counts

| Manifest | Dataset | Rows | Local image rows | Unique image files | Notes |
|---|---|---:|---:|---:|---|
| `train.csv` | NewsCLIPpings | 71,072 | 0 | 59,876 embedding keys | Balanced; real CLIP embeddings |
| `val.csv` | NewsCLIPpings | 7,024 | 0 | 5,943 embedding keys | Balanced |
| `test.csv` | NewsCLIPpings | 7,264 | 0 | 6,147 embedding keys | Balanced |
| `verite_test.csv` | VERITE | 1,001 | 621 | 495 | Imbalanced; hard OOD |
| `mmfakebench_val.csv` | MMFakeBench | 1,000 | 1,000 | 1,000 | 30/70 match/OOC |
| `mmfakebench_test.csv` | MMFakeBench | 10,000 | 10,000 | 10,000 | 30/70 match/OOC |
| **`all.csv`** | Combined | **97,361** | **11,621** | **11,495** | Unified manifest |

## 4.4 Label distribution

| Dataset / Split | Label 0: match | Label 1: OOC | Positive rate |
|---|---:|---:|---:|
| NewsCLIPpings train | 35,536 | 35,536 | 50.0% |
| NewsCLIPpings val | 3,512 | 3,512 | 50.0% |
| NewsCLIPpings test | 3,632 | 3,632 | 50.0% |
| VERITE test | 338 | 663 | 66.2% |
| MMFakeBench val | 300 | 700 | 70.0% |
| MMFakeBench test | 3,000 | 7,000 | 70.0% |
| **All combined** | **46,318** | **51,043** | **52.4%** |

**Beginner explanation:** label imbalance matters because a naive model that always predicts OOC already gets 70% accuracy on MMFakeBench. That is why AUROC and Macro-F1 are important.

## 4.5 What we gained through dataset cleanup

Before cleanup:

- Dataset folders were inconsistent.
- VERITE images existed in overlapping folders.
- MMFakeBench images were buried in nested folders.
- NewsCLIPpings had no local images, but this was not handled cleanly.
- Stream B used fake hash-generated vectors for NewsCLIPpings image embeddings.
- Stream C evidence was circular because the caption appeared inside the evidence text.

After cleanup:

| Gain | Details |
|---|---|
| Unified manifests | Every dataset uses the same schema. |
| Flat image folder | All local images copied into one processed image directory. |
| 97,361 pair rows | Combined normalized dataset. |
| 11,621 rows with local images | VERITE + MMFakeBench combined. |
| 11,495 unique image files | After processed-image normalization. |
| Real NewsCLIP embeddings | NewsCLIPpings uses real PKL CLIP embeddings. |
| 0 broken image paths | Processed manifests validate cleanly. |
| Reproducible splits | Split ID files stored in `data/splits/`. |
| Decircularized Stream C | Caption removed from evidence text. |
| Fusion complete | GSAF trained with all ablations. |

## 4.6 Dataset figures

### Figure — Class balance

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/eda_class_balance_processed.png" alt="Class balance" width="650"/>


**Beginner summary:** This figure shows how match and OOC labels are distributed across the processed datasets. NewsCLIPpings is balanced, while MMFakeBench and VERITE contain more OOC samples.

### Figure — Caption length distribution

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/eda_caption_length_distribution.png" alt="Caption length distribution" width="650"/>


**Beginner summary:** This figure helps show whether caption length differs across datasets. Caption length matters because very short or very long captions may affect text embedding and evidence reasoning quality.

### Figure — Image coverage

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/eda_image_coverage_processed.png" alt="Image coverage" width="650"/>


**Beginner summary:** NewsCLIPpings has no local images but has pre-extracted embeddings. VERITE has partial image coverage. MMFakeBench has full local image coverage.

---

# 5. Dataset Preprocessing and Processing

## 5.1 Why preprocessing was needed

Each dataset had a different raw format:

| Dataset | Raw format problem | Solution |
|---|---|---|
| NewsCLIPpings | JSON + metadata + PKL embeddings; no local images | Use real pre-extracted embeddings and preserve CLIP keys. |
| VERITE | CSV with image references like `images/true_0.jpg`; partial downloaded images | Resolve paths and copy available images to flat processed folder. |
| MMFakeBench | JSON plus nested source-specific image folders | Copy all images to a flat folder with unique names. |

## 5.2 Processed manifest schema

Every processed manifest uses a normalized schema. Important columns include:

| Column | Meaning |
|---|---|
| `pair_id` | Unique row ID for one image-text pair. |
| `dataset` | Dataset name: `newsclippings`, `verite`, or `mmfakebench`. |
| `split` | Train, validation, or test split. |
| `image_id` | Unique ID for image file or embedding. |
| `image_path` | Processed local image path, if available. |
| `has_image` | 1 if local image exists, otherwise 0. |
| `caption` | Text/caption/context. |
| `label` | 0 = match, 1 = mismatch/OOC. |
| `label_name` | Human-readable label. |
| `clip_image_key` | Key for CLIP image embedding lookup. |
| `clip_text_key` | Key for CLIP text embedding lookup. |
| `text_source` / `image_source` | MMFakeBench source metadata. |
| `topic`, `source`, `timestamp`, `title` | NewsCLIPpings metadata. |

## 5.3 Why use a flat image folder?

Instead of forcing notebooks to scan nested dataset folders, all local images are copied into:

```text
data/processed/ooc_pairs/images/
```

The manifest stores the exact image path, so every notebook reads the manifest instead of guessing folder structure.

## 5.4 How processed data is rebuilt

The core setup notebook is:

```text
notebooks/01_setup/data_setup.ipynb
```

It:

1. Reads raw NewsCLIPpings JSON + metadata.
2. Reads raw VERITE CSV + image folder.
3. Reads raw MMFakeBench JSON + image folders.
4. Normalizes all rows into one schema.
5. Copies local images to a flat processed folder.
6. Writes manifest CSV files.
7. Writes split ID files.
8. Validates image paths.
9. Writes `manifest_summary.csv`.

---

# 6. Notebook Names and Completion Status

Status legend:

| Symbol | Meaning |
|---|---|
| ✅ Complete | Ran successfully and outputs exist. |
| 🟡 Weak | Technically valid but scientifically limited alone. |
| 🟢 Done | Completed in the final fusion/eval phase. |
| ❌ Cancelled | Cancelled due to feasibility or rate limits. |

## 6.1 Setup

| Notebook | Status | Notes |
|---|---|---|
| `01_setup/env_check.ipynb` | ✅ | Checks Python, GPU, paths, manifests, folders. |
| `01_setup/data_setup.ipynb` | ✅ | Builds manifests, flat images, and split files. |
| `01_setup/eda.ipynb` | ✅ | Produces EDA figures and tables. |

## 6.2 Stream A

| Notebook | Status | Notes |
|---|---|---|
| `02_stream_a/ner_pipeline_v2_updated.ipynb` | ✅ | NER cached for all 97,361 rows. |
| `02_stream_a/visual_grounding_v2_updated.ipynb` | ✅ | CLIP grounding for 11,621 local-image rows. |
| `02_stream_a/entity_scorer_v2_updated.ipynb` | ✅ | Builds 12-feature table. |
| `02_stream_a/stream_a_results_v2_updated.ipynb` | 🟡 | Weak standalone but useful in fusion. |

## 6.3 Stream B

| Notebook | Status | Notes |
|---|---|---|
| `03_stream_b/clip_features_updated.ipynb` | ✅ | Uses real NewsCLIPpings PKL embeddings. |
| `03_stream_b/lora_stage1_updated.ipynb` | ✅ | Stage 1 MLP training. |
| `03_stream_b/lora_stage2_fixed_updated.ipynb` | ✅ | Stage 2 classifier and predictions. |
| `03_stream_b/stream_b_results_fixed_updated.ipynb` | ✅ | Best individual stream summary. |

## 6.4 Stream C

| Notebook | Status | Notes |
|---|---|---|
| `04_stream_c/evidence_retrieval_final.ipynb` | ✅ | Decircularized evidence; caption never included as evidence. |
| `04_stream_c/evidence_val_final.ipynb` | ✅ | Validates/repairs evidence rows; leak assertion passes. |
| `04_stream_c/stance_scoring_final.ipynb` | ✅ | NLI DeBERTa stance scoring for all rows. |
| `04_stream_c/stream_c_results_final.ipynb` | ✅ | Stream C results and threshold tuning. |

## 6.5 Fusion

| Notebook | Status | Notes |
|---|---|---|
| `05_fusion/baselines_final.ipynb` | 🟢 | Baselines and combined stream summary. |
| `05_fusion/fusion_module.ipynb` | 🟢 | Defines GSAF and writes `src/models/fusion.py`. |
| `05_fusion/fusion_training.ipynb` | 🟢 | Trains GSAF and all 4 ablations. |

## 6.6 Evaluation and paper outputs

| Notebook | Status | Notes |
|---|---|---|
| `07_eval/newsclippings_eval.ipynb` | ✅ | In-distribution evaluation. |
| `07_eval/cross_dataset_eval.ipynb` | ✅ | OOD evaluation on MMFakeBench and VERITE. |
| `07_eval/ablations.ipynb` | ✅ | Ablation grid and delta analysis. |
| `07_eval/qualitative_analysis.ipynb` | ✅ | Example cases and gate attribution table. |
| `08_paper/tables.ipynb` | ✅ | Final CSV and LaTeX tables. |
| `08_paper/figures.ipynb` | ✅ | Final paper figures and hero figures. |

## 6.7 Cancelled

| Notebook | Status | Why |
|---|---|---|
| `06_mllm/gemini_setup.ipynb` | ❌ | MLLM explanation layer cancelled. |
| `06_mllm/mllm_inference.ipynb` | ❌ | Free-tier rate limits and slow inference made it infeasible. |
| `06_mllm/chain_of_thought.ipynb` | ❌ | Cancelled with MLLM explanation layer. |

---

# 7. Notebook Summaries: Which Notebook Does What

## 7.1 Setup notebooks

### `env_check.ipynb`
Checks the environment before running anything. Verifies Python version, CUDA/GPU availability, root path, imports, manifests, flat image folder, split files, and cache/paper folders.

### `data_setup.ipynb`
Builds the canonical dataset layer from raw datasets. It creates processed images, manifest CSVs, reports, and split files.

### `eda.ipynb`
Produces exploratory dataset tables and figures: dataset counts, class balance, image coverage, caption lengths, and fake-class summaries.

## 7.2 Stream A notebooks

### `ner_pipeline_v2_updated.ipynb`
Extracts named entities such as people, places, and organizations from captions using spaCy or a fallback method.

### `visual_grounding_v2_updated.ipynb`
Computes CLIP-based visual grounding scores for rows with local images. It checks how well entity names match the image.

### `entity_scorer_v2_updated.ipynb`
Aggregates NER and grounding outputs into 12 numeric features per row.

### `stream_a_results_v2_updated.ipynb`
Trains/evaluates Stream A. Stream A is weak alone but contributes useful features to fusion.

## 7.3 Stream B notebooks

### `clip_features_updated.ipynb`
Builds CLIP-derived features. The important fix is that NewsCLIPpings uses real PKL embeddings instead of fake hash-generated vectors.

### `lora_stage1_updated.ipynb`
Trains a first-stage lightweight neural classifier/representation model on Stream B features.

### `lora_stage2_fixed_updated.ipynb`
Trains the second-stage classifier and saves final Stream B predictions/features.

### `stream_b_results_fixed_updated.ipynb`
Summarizes Stream B results and figures. Stream B is the strongest individual stream on NewsCLIPpings.

## 7.4 Stream C notebooks

### `evidence_retrieval_final.ipynb`
Builds independent evidence text. The caption is not included in evidence. VERITE uses article query fields, NewsCLIPpings uses article metadata, and MMFakeBench uses source/fake-class metadata.

### `evidence_val_final.ipynb`
Validates evidence quality and repairs missing rows using safe metadata-only fallback.

### `stance_scoring_final.ipynb`
Uses an NLI model to score whether evidence entails, contradicts, or is neutral toward the caption. Also adds SBERT similarity and length features.

### `stream_c_results_final.ipynb`
Trains/evaluates a classifier on Stream C features and performs threshold tuning.

## 7.5 Fusion/evaluation notebooks

### `baselines_final.ipynb`
Computes majority, random, always-match, always-OOC, CLIP cosine threshold, and CLIP rank baselines.

### `fusion_module.ipynb`
Merges Stream A/B/C features, defines the GSAF architecture, saves the feature schema, and writes `src/models/fusion.py`.

### `fusion_training.ipynb`
Trains GSAF using NewsCLIPpings train plus 500 MMFakeBench validation rows. Also trains ablations: `no_a`, `no_b`, `no_c`, and `mlp_fusion`.

### `newsclippings_eval.ipynb`
Evaluates in-distribution performance on NewsCLIPpings.

### `cross_dataset_eval.ipynb`
Evaluates cross-dataset performance on MMFakeBench and VERITE.

### `ablations.ipynb`
Generates ablation result tables and delta-AUROC heatmaps.

### `qualitative_analysis.ipynb`
Saves qualitative examples and gate attribution outputs.

### `tables.ipynb`
Creates paper-ready CSV and LaTeX tables.

### `figures.ipynb`
Creates final paper figures, including the hero AUROC, gate attribution, and ablation figures.

---

# 8. Output Tables and Figures

## 8.1 Main figure locations

All paper figures live mainly in:

```text
paper/figures/
finalized_outputs/figures/
```

## 8.2 EDA figures

| Figure | File |
|---|---|
| Class balance | `paper/figures/eda_class_balance_processed.png` |
| Caption length | `paper/figures/eda_caption_length_distribution.png` |
| Image coverage | `paper/figures/eda_image_coverage_processed.png` |

## 8.3 Stream figures

| Area | Important figures |
|---|---|
| Stream A | `stream_a_summary_bar.png`, Stream A confusion/ROC figures |
| Stream B | `stream_b_summary_bar.png`, Stream B ROC figures, score distribution |
| Stream C | `stream_c_summary_bar.png`, Stream C confusion/ROC figures |
| Baselines | `all_streams_summary_bar.png` |
| Fusion | `fusion_summary_bar.png`, `fusion_gate_distribution.png`, fusion confusion/ROC figures |

## 8.4 Hero figures

### Hero AUROC comparison

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/finalized_outputs/figures/00_HERO_auroc_comparison.png" alt="Hero AUROC comparison" width="720"/>


**Beginner summary:** This is the main result figure. It shows how the full GSAF model compares with streams and ablations across datasets.

### Hero gate attribution

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/finalized_outputs/figures/00_HERO_gate_attribution.png" alt="Hero gate attribution" width="720"/>


**Beginner summary:** This figure explains interpretability. It shows how the gate distributes attention among the three streams.

### Hero ablation delta

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/finalized_outputs/figures/00_HERO_ablation_delta.png" alt="Hero ablation delta" width="720"/>


**Beginner summary:** This figure shows how performance changes when components are removed. Large drops prove that those components matter.

## 8.5 Fusion figures

### Fusion summary

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_summary_bar.png" alt="Fusion summary" width="650"/>


### Gate distribution

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_gate_distribution.png" alt="Fusion gate distribution" width="650"/>


### Fusion ROC curves

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_roc_newsclippings_test.png" alt="NewsCLIPpings ROC" width="620"/>


<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_roc_mmfakebench_test.png" alt="MMFakeBench ROC" width="620"/>


<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_roc_verite_test.png" alt="VERITE ROC" width="620"/>


## 8.6 Key paper tables

Raw CSV tables are in `paper/tables/`. Finalized tables are in `finalized_outputs/tables/`.

| Table | Description |
|---|---|
| `manifest_summary.csv` | Dataset row/image summary. |
| `eda_dataset_summary.csv` | Processed manifest stats. |
| `class_balance.csv` | Label counts per split. |
| `stream_a_results.csv` | Stream A metrics. |
| `stream_b_results.csv` | Stream B metrics. |
| `stream_c_results.csv` | Stream C metrics. |
| `baseline_results.csv` | Majority/random/constant baselines. |
| `clip_baseline_results.csv` | CLIP cosine baseline. |
| `all_streams_and_baselines_summary.csv` | Combined summary table. |
| `fusion_results.csv` | Main GSAF results. |
| `ablation_results.csv` | Ablation comparison. |
| `01_headline_cross_dataset.csv` | Final headline results across datasets. |
| `02_ablation_delta_auroc.csv` | Delta-AUROC from ablations. |
| `03_dataset_summary.csv` | Dataset summary for paper. |
| `04_per_stream_results.csv` | Per-stream results. |
| `05_per_class_newsclippings.csv` | Per-class NewsCLIPpings metrics. |

---

# 9. Output Summary So Far

## 9.1 Dataset outputs

| Check | Result |
|---|---:|
| Total manifest rows | 97,361 |
| NewsCLIPpings rows | 85,360 |
| VERITE rows | 1,001 |
| MMFakeBench rows | 11,000 |
| Local-image rows | 11,621 |
| Flat image files | 11,495 |
| Unique image IDs | 83,461 |
| Broken image paths | 0 |
| Empty captions | 0 |

## 9.2 Stream A summary

Stream A uses entity recognition and visual grounding. It is weak alone but useful in fusion.

| Eval set | Accuracy | Interpretation |
|---|---:|---|
| MMFakeBench test | 62.56% | Below majority baseline, but gives useful grounding features. |
| VERITE local | 53.30% | Near random alone. |

## 9.3 Stream B summary

Stream B is the strongest individual stream. The key fix was using real NewsCLIPpings PKL embeddings instead of fake hash vectors.

| Eval set | Accuracy | AUROC | Interpretation |
|---|---:|---:|---|
| NewsCLIPpings test | 65.87% | 73.91% | Strong in-distribution signal. |
| MMFakeBench test | 46.95% | 63.35% | Some OOD signal, but threshold weak. |
| VERITE local | 63.77% | 49.12% | Ranking near random. |

## 9.4 Stream C summary

Stream C uses decircularized evidence plus NLI stance scoring.

| Dataset | Threshold | Accuracy | Macro-F1 | AUROC |
|---|---|---:|---:|---:|
| NewsCLIP test | default 0.5 | 51.53% | 51.01% | near-random |
| MMFakeBench test | default 0.5 | 34.78% | 32.32% | 70.22% |
| MMFakeBench test | tuned 0.47 | 73.65% | 63.85% | 70.22% |
| VERITE test | default 0.5 | 36.16% | 30.70% | 43.25% |

## 9.5 Baseline summary

| Baseline | NC test Acc | MMFB test Acc | VERITE Acc | NC AUROC | MMFB AUROC | VERITE AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Random uniform | ~50% | ~50% | ~50% | 50.00 | 50.00 | 50.00 |
| Majority class | 50.00% | 70.00% | 66.23% | — | — | — |
| Always OOC | 50.00% | 70.00% | 66.23% | — | — | — |
| Always match | 50.00% | 30.00% | 33.77% | — | — | — |
| CLIP cosine threshold | 65.75% | 58.12% | 48.35% | 71.97 | 64.13 | 48.54 |
| CLIP cosine rank | 65.74% | 58.13% | 48.25% | 71.97 | 64.13 | 48.54 |

## 9.6 GSAF final results

| Dataset | Accuracy | F1 | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| NewsCLIPpings val | 69.15% | 68.47% | 69.13% | 76.01% |
| NewsCLIPpings test | **69.01%** | **68.34%** | **69.00%** | **76.32%** |
| MMFakeBench val_tune | 82.00% | 86.84% | 79.18% | 89.92% |
| MMFakeBench test | **78.27%** | **84.41%** | **74.28%** | **87.83%** |
| VERITE test | **62.54%** | **76.34%** | **43.21%** | **48.13%** |

## 9.7 Fusion gains over individual streams

| Dataset | Metric | Best individual stream | GSAF fusion | Gain |
|---|---|---:|---:|---:|
| NewsCLIPpings test | AUROC | 73.91% | 76.32% | +2.41 pp |
| NewsCLIPpings test | Accuracy | 65.87% | 69.01% | +3.14 pp |
| MMFakeBench test | AUROC | 70.22% | 87.83% | +17.61 pp |
| MMFakeBench test | Accuracy | 73.65% | 78.27% | +4.62 pp |

## 9.8 Ablation results

| Config | Removed | NC test AUROC | MMFB test AUROC | Main conclusion |
|---|---|---:|---:|---|
| `default` | Nothing | **76.32%** | **87.83%** | Best overall. |
| `no_a` | Stream A | 75.23% | 65.24% | A helps OOD fusion. |
| `no_b` | Stream B | 52.49% | 65.27% | B is crucial in-distribution. |
| `no_c` | Stream C | 75.47% | 62.87% | C is crucial OOD. |
| `mlp_fusion` | Gating mechanism | 76.26% | 65.96% | Gating gives large OOD gain. |

**Key seminar line:** On NewsCLIPpings, GSAF and MLP fusion are almost equal, but on MMFakeBench, GSAF beats MLP fusion by about **+21.9 AUROC points**. This is the strongest evidence that adaptive gating matters.

---

# 10. Issues Faced and How We Solved Them

## Issue 1: Raw dataset layout was messy

**Problem:** Old folders, duplicated image folders, inconsistent paths, and notebooks scanning raw data manually.

**Solution:** Created canonical `data/raw/` and `data/processed/ooc_pairs/` structures. Updated path constants. All notebooks now read manifests.

## Issue 2: VERITE had duplicate image folders

**Problem:** Two overlapping VERITE image folders existed.

**Solution:** Defined one canonical raw path and copied processed images into the flat folder.

## Issue 3: VERITE image naming looked confusing

**Problem:** VERITE uses `true_*.jpg` and `false_*.jpg` naming. Renaming raw files could break reproducibility.

**Solution:** Keep raw files unchanged. Use normalized processed names only in the processed image folder.

## Issue 4: VERITE row count and image count mismatch

**Problem:** 621 local-image rows but only 495 unique image files.

**Solution:** Recognized this is expected because VERITE reuses images with different captions. Row count and unique image count are not the same.

## Issue 5: NewsCLIPpings local images unavailable

**Problem:** VisualNews images are not redistributed.

**Solution:** Set `has_image=0` for NewsCLIPpings and use real pre-extracted CLIP embeddings through `clip_image_key` and `clip_text_key`.

## Issue 6: Stream B originally used fake hash-generated vectors

**Problem:** Early Stream B used deterministic hash vectors instead of real image embeddings for NewsCLIPpings. This looked valid technically but contained no visual information.

**Solution:** Load real NewsCLIPpings PKL embeddings. Result: Stream B NewsCLIP AUROC improved from roughly 55% to 73.91%.

## Issue 7: PowerShell heredoc syntax failed

**Problem:** Bash-style heredoc code did not work in PowerShell.

**Solution:** Use PowerShell-safe script creation or run commands in a compatible shell environment.

## Issue 8: Python backslash escaping errors

**Problem:** Unescaped backslashes caused syntax errors.

**Solution:** Use escaped backslashes, e.g. `path.replace("\", "/")`.

## Issue 9: EDA notebook helper was overwritten

**Problem:** A later cell overwrote the flexible `savefig()` helper.

**Solution:** Restored a flexible helper that accepts either current figure or explicit figure object.

## Issue 10: Stream C evidence was circular

**Problem:** The evidence text included the caption itself, making the similarity/evidence signal circular.

**Solution:** Removed captions from evidence. Stream C now uses independent metadata/evidence and NLI stance scoring. Leak assertion confirms no caption leakage.

## Issue 11: Old caches contaminated results

**Problem:** Old cache files built with wrong embeddings or old evidence could be reused silently.

**Solution:** Delete relevant cache folders before major reruns, especially Stream B and Stream C caches.

## Issue 12: Fusion gating collapsed in Run 1

**Problem:** When trained only on NewsCLIPpings, the gate learned to mostly trust Stream B and did not learn real stream switching. MMFakeBench AUROC was only 61.29%.

**Solution:** Add 500 stratified MMFakeBench validation rows to training. Result: MMFakeBench AUROC improved to 87.83% and gating became meaningful.

## Issue 13: Pandas / sklearn split crash

**Problem:** Python 3.13 + Pandas 3.0 PyArrow strings caused `train_test_split` indexing issues.

**Solution:** Replaced `train_test_split` with manual NumPy stratified splitting using Python string lists.

---

# 11. Remaining Tasks

## 11.1 Completed

| Component | Status |
|---|---|
| Dataset processing | ✅ Complete |
| Stream A | ✅ Complete |
| Stream B | ✅ Complete |
| Stream C | ✅ Complete |
| Fusion model | ✅ Complete |
| Ablations | ✅ Complete |
| Formal evaluation | ✅ Complete |
| Paper tables | ✅ Complete |
| Paper figures | ✅ Complete |
| Draw.io diagrams | ✅ Complete according to project notes |

## 11.2 Remaining or paper-writing tasks

| Task | Status | Notes |
|---|---|---|
| Final paper writing | Pending | Write LaTeX paper using finalized tables/figures. |
| Final proofreading | Pending | Check claims, limitations, and figure references. |
| Final bibliography formatting | Pending | Add dataset and model citations in paper. |
| Optional VERITE analysis | Future work | Explain why VERITE remains hard. |
| Optional real-time retrieval | Future work | Would make Stream C closer to true RAG. |
| Optional MLLM explanation layer | Future work | Cancelled for current phase due to rate limits. |

## 11.3 Run order already completed

```text
1. notebooks/07_eval/newsclippings_eval.ipynb
2. notebooks/07_eval/cross_dataset_eval.ipynb
3. notebooks/07_eval/ablations.ipynb
4. notebooks/07_eval/qualitative_analysis.ipynb
5. notebooks/05_fusion/fusion_training.ipynb
6. notebooks/08_paper/tables.ipynb
7. notebooks/08_paper/figures.ipynb
```

---

# 12. Other Important Information

## 12.1 GSAF architecture

GSAF has three stream-specific projectors and a gate:

```text
Stream A features (12) → projector → h_A (32)
Stream B features (11) → projector → h_B (32)
Stream C features (7)  → projector → h_C (32)

concat(h_A, h_B, h_C) → Gate MLP → [g_A, g_B, g_C]

z = g_A*h_A + g_B*h_B + g_C*h_C
z → classifier → sigmoid → P(OOC)
```

Total parameters: about **8,164**.

## 12.2 Training setup

| Setting | Value |
|---|---|
| Training data | NewsCLIPpings train + 500 MMFakeBench validation rows |
| Validation | NewsCLIPpings val |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 1024 |
| Early stopping | Patience 8 on validation AUROC |
| Epochs trained | 37 |
| Best validation AUROC | 76.01% |

## 12.3 Threshold tuning

The model outputs an OOC probability. A threshold decides the final class:

```text
If P(OOC) >= threshold → predict OOC
else → predict match
```

Threshold tuning is important because MMFakeBench and VERITE are imbalanced. AUROC remains useful because it measures ranking quality independent of a fixed threshold.

## 12.4 Safe claims

You can safely claim:

- The system detects out-of-context image-caption pairs.
- It combines entity grounding, CLIP compatibility, and evidence stance features.
- GSAF improves MMFakeBench OOD AUROC substantially over flat MLP fusion.
- Stream B is strongest in-distribution.
- Stream C helps more under MMFakeBench-style OOD shift.
- VERITE remains an open limitation.

## 12.5 Claims to avoid

Do not claim:

- The system is a complete fact-checker.
- The system solves VERITE.
- This is full real-time web RAG.
- MLLM explanations were completed.
- The system detects all deepfakes.
- Accuracy alone proves robustness.

---

# 13. Glossary

| Term | Definition |
|---|---|
| **OOC** | Out-of-context. An image is used with the wrong or misleading caption/context. |
| **Match** | Correct image-text pair; label 0. |
| **Mismatch** | Incorrect or misleading image-text pair; label 1. |
| **Manifest** | A normalized CSV file storing one row per image-text pair. |
| **pair_id** | Unique ID for one image-text pair row. |
| **image_id** | Unique ID for one image file or embedding. |
| **has_image** | Whether a local processed image file exists for that row. |
| **CLIP** | A vision-language model that maps images and text into a shared vector space. |
| **Embedding** | A numeric vector representation of image or text. |
| **Cosine similarity** | Measures how aligned two vectors are. Higher means more similar. |
| **AUROC** | Measures ranking quality across thresholds. 1.0 is perfect, 0.5 is random. |
| **Macro-F1** | F1 averaged equally across classes, useful for imbalanced datasets. |
| **OOD** | Out-of-domain. A test dataset differs from the training distribution. |
| **NER** | Named Entity Recognition: extracting people, places, organizations, etc. |
| **Visual grounding** | Checking whether something mentioned in text is visually supported by the image. |
| **NLI** | Natural Language Inference: entailment, contradiction, or neutral relation between evidence and claim. |
| **DeBERTa** | Transformer model used for NLI stance scoring. |
| **SBERT** | Sentence-BERT model used for sentence-level semantic similarity. |
| **RAG** | Retrieval-Augmented Generation/Grounding. In this project, Stream C is an offline evidence baseline, not full live RAG. |
| **GSAF** | Gated Stream Attention Fusion, the final fusion architecture. |
| **Gate** | Learned weights over Stream A/B/C that sum to 1 for each sample. |
| **Softmax** | Function that converts raw scores into probabilities summing to 1. |
| **Ablation** | Removing one component to measure its importance. |
| **Mixed-domain training** | Training with examples from more than one dataset distribution. |
| **Calibration** | How well predicted probabilities match true outcome frequencies. |
| **Threshold** | Cutoff used to convert probability into class prediction. |
| **MLLM** | Multimodal Large Language Model; planned for explanations but cancelled in current phase. |

---

# 14. Professor / Seminar Q&A

## Q1. What problem does your paper solve?

It solves out-of-context image detection: deciding whether an image truly belongs with a given caption or whether it is being reused misleadingly.

## Q2. How is OOC detection different from fake image detection?

Fake image detection asks whether the image itself is manipulated or generated. OOC detection asks whether the image-caption pairing is truthful. The image can be completely real but still misleading.

## Q3. Why is this problem important?

Many misinformation posts reuse real images with false captions. This is dangerous because real images look trustworthy and are harder to detect with simple manipulation detectors.

## Q4. What is the main idea of MAGE?

MAGE combines three signals: entity grounding, CLIP image-text compatibility, and evidence stance reasoning. Then GSAF adaptively gates these streams per sample.

## Q5. What is the main novelty?

The novelty is the adaptive multi-stream fusion design and the cross-dataset evaluation showing that gating improves OOD robustness, especially on MMFakeBench.

## Q6. Why not use only CLIP?

CLIP measures semantic compatibility, not factual correctness. A misleading caption can still be visually compatible with an image. That is why evidence stance and entity grounding are needed.

## Q7. Why is Stream B strongest on NewsCLIPpings?

NewsCLIPpings is close to image-text semantic mismatch detection, so CLIP-style representation features are highly useful.

## Q8. Why does Stream C help on MMFakeBench?

MMFakeBench contains metadata/evidence patterns where source and fake-class information can help. NLI stance features provide useful ranking signal there.

## Q9. Why is VERITE hard?

VERITE has no training split, limited evidence coverage, and a real-world distribution shift. The model has no in-domain validation data to learn or tune from.

## Q10. Is VERITE performance a failure?

It is a limitation, but it is honestly reported. The paper does not claim to solve VERITE. It shows strong MMFakeBench OOD improvement while identifying VERITE as future work.

## Q11. Why use AUROC?

AUROC measures how well the model ranks OOC examples above match examples across all thresholds. This is important when datasets are imbalanced and threshold choice can distort accuracy.

## Q12. Why use Macro-F1?

Macro-F1 gives equal importance to both classes. It prevents majority-class performance from hiding poor minority-class behavior.

## Q13. Why is accuracy not enough?

On MMFakeBench, always predicting OOC gives 70% accuracy. So a model can look good by accuracy without learning meaningful detection.

## Q14. What does the ablation study prove?

It shows which components matter. Removing Stream B hurts NewsCLIPpings heavily. Removing Stream C or the gate hurts MMFakeBench heavily. This supports the claim that different streams matter under different dataset shifts.

## Q15. Why does GSAF beat MLP fusion on MMFakeBench?

MLP fusion combines features without explicit per-stream trust weighting. GSAF learns per-sample stream weights, so it can down-weight unreliable streams and rely more on useful OOD signals.

## Q16. Why did you add 500 MMFakeBench validation rows to training?

Training only on NewsCLIPpings made the gate over-specialize to Stream B. Adding 500 MMFakeBench rows gave the gate examples where evidence/stance signals are useful, enabling real stream switching.

## Q17. Is the model lightweight?

Yes. The fusion model has only about 8K parameters. Most heavy computation happens in precomputed feature extraction; fusion itself is efficient.

## Q18. Is EVE-RAG actually real RAG?

Not fully. Stream C follows the retrieval/evidence-grounding idea but uses offline metadata evidence rather than live web retrieval. The safe phrase is: **offline evidence-based stance reasoning**, not full real-time RAG.

## Q19. Does the system explain predictions?

It provides partial interpretability through gate weights and stream attribution. It does not generate complete natural-language explanations in the final implemented system.

## Q20. Can this detect deepfakes?

Not directly. The task is image-context mismatch, not image manipulation or generation detection. A deepfake detector would require different signals.

## Q21. What is the biggest limitation?

The biggest limitation is VERITE generalization. Another limitation is that Stream C uses offline evidence rather than live retrieval.

## Q22. What would you improve next?

The best future improvements are real-time retrieval for Stream C, better entity grounding, VERITE-specific adaptation, stronger calibration, and an MLLM explanation layer.

## Q23. What is the one-sentence takeaway?

Out-of-context image detection needs more than image-text similarity; adaptive fusion of visual, entity, and evidence signals improves robustness when the test distribution changes.

---

# 15. Dangerous Questions and Safe Answers

## 15.1 “Is your system a fact-checker?”

**Weak answer:** Yes, it fact-checks images.  
**Safe answer:** It is not a full fact-checker. It detects whether an image-caption pair appears contextually mismatched using visual, entity, and evidence-based signals.

## 15.2 “Did you solve VERITE?”

**Weak answer:** Yes, our model works on all datasets.  
**Safe answer:** No. VERITE remains challenging and is reported honestly as a limitation. The model performs strongly on MMFakeBench but needs better evidence retrieval or adaptation for VERITE.

## 15.3 “Is EVE-RAG truly RAG?”

**Weak answer:** Yes, it is full RAG.  
**Safe answer:** The current Stream C is an offline evidence stance module inspired by retrieval-augmented grounding. Full real-time retrieval is future work.

## 15.4 “Does GSAF explain why an image is fake?”

**Weak answer:** Yes, it gives explanations.  
**Safe answer:** It provides stream-level interpretability through gate weights, but not full human-readable explanations in the final implemented version.

## 15.5 “Why does MMFakeBench improve so much?”

**Safe answer:** Because mixed-domain training gave the gate examples where Stream C and grounding signals were more useful than CLIP alone. The ablation results show that the gate and Stream C are crucial on MMFakeBench.

---

# 16. Ultra-Short Cheat Sheet

## Problem

Detect whether an image-caption pair is genuinely matched or out-of-context.

## Labels

- 0 = match / genuine
- 1 = mismatch / OOC

## Datasets

- NewsCLIPpings: 85,360 rows; main training; no local images but real embeddings.
- VERITE: 1,001 rows; hard real-world test; partial images.
- MMFakeBench: 11,000 rows; OOD benchmark; full images.

## Method

- Stream A: entity + visual grounding.
- Stream B: CLIP representation.
- Stream C: evidence + NLI stance.
- Fusion: GSAF adaptive gate.

## Best result

MMFakeBench test AUROC = **87.83%**.

## Strongest evidence

GSAF beats MLP fusion by about **+21.9 AUROC points** on MMFakeBench.

## Biggest limitation

VERITE AUROC = **48.13%**, near random, due to no train split, limited evidence coverage, and domain shift.

## What not to overclaim

Not a full fact-checker. Not full live RAG. Not a deepfake detector. Not a solved VERITE system.
