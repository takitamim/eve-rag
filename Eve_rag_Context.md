# MAGE / EVE-RAG — Beginner-to-Advanced Manual

**Project codename:** EVE-RAG  
**Paper title:** MAGE: Multi-Stream Adaptive Gating for Cross-Dataset Out-of-Context Image Detection  
**Research topic:** Out-of-context image detection using entity verification, visual grounding, CLIP representation learning, NLI evidence reasoning, and gated multi-stream fusion.  
**Repository:** `github.com/Sefayet-Alam/eve-rag`  
**Status:** Core pipeline complete: dataset processing, Stream A, Stream B, Stream C, GSAF fusion, ablations, formal evaluation, paper tables, and figures.

---

## 0. One-Page Summary

This project studies **out-of-context image detection**. The task is simple to state but difficult to solve:

> Given an image and a caption, decide whether the image truly belongs with that caption or whether it has been reused in a misleading context.

A key point is that an out-of-context image is often **not edited or fake**. The image may be real. The caption may also be real. The misinformation comes from pairing them incorrectly. For example, an old protest photo from one country may be reused with a caption about a new protest in another country.

To solve this, MAGE / EVE-RAG uses three complementary streams:

| Stream | Name | Main question |
|---|---|---|
| Stream A | Entity + visual grounding | Do people, places, and organizations mentioned in the caption appear visually or evidentially supported? |
| Stream B | CLIP representation | Do the image and text look compatible in a shared embedding space? |
| Stream C | Evidence + stance reasoning | Does external evidence support, contradict, or remain neutral toward the caption? |

The final model is called **GSAF: Gated Stream Attention Fusion**. Instead of treating the three streams equally, GSAF learns how much to trust each stream for each individual image-text pair.

Final headline results:

| Dataset | Setting | Accuracy | Macro-F1 | AUROC |
|---|---|---:|---:|---:|
| NewsCLIPpings test | In-distribution | 69.01% | 69.00% | 76.32% |
| MMFakeBench test | Out-of-distribution | 78.27% | 74.28% | 87.83% |
| VERITE test | Real-world fact-checking | 62.54% | 43.21% | 48.13% |

The strongest result is on **MMFakeBench**, where GSAF reaches **87.83% AUROC** and clearly beats non-gated fusion. The main limitation is **VERITE**, where AUROC remains near random because the dataset has no training split, limited evidence coverage, and a strong distribution shift from the training data.

---

## 1. Title Explained in Simple Words

### Full title

**MAGE: Multi-Stream Adaptive Gating for Cross-Dataset Out-of-Context Image Detection**

### What each part means

| Term | Simple meaning |
|---|---|
| **MAGE** | The paper name for the final method. |
| **Multi-Stream** | The system does not rely on one model only. It uses three different evidence streams. |
| **Adaptive Gating** | The model learns how much to trust each stream for each example. |
| **Cross-Dataset** | The model is tested on multiple datasets, including datasets different from the training distribution. |
| **Out-of-Context Image Detection** | The task is to detect image-caption pairs where the image is real but used with the wrong context. |

### Why this title is appropriate

The paper is both:

1. a **system paper**, because it builds a full three-stream detection pipeline; and  
2. a **method paper**, because it introduces and evaluates the GSAF gated fusion mechanism.

---

## 2. Beginner Glossary

| Term | Plain English definition |
|---|---|
| **Image-text pair** | One image and one caption/context that belong together or do not belong together. |
| **Match / genuine** | Label `0`. The image and caption correctly belong together. |
| **Mismatch / OOC** | Label `1`. The image is used in a misleading or out-of-context way. |
| **Out-of-context image** | A real image paired with the wrong caption, time, place, event, or claim. |
| **Dataset** | A collection of examples used for training or testing a model. |
| **Benchmark** | A standard dataset or evaluation setup used to compare methods fairly. |
| **Train set** | Data used to teach the model. |
| **Validation set** | Data used to tune thresholds and choose settings. |
| **Test set** | Final unseen data used for reporting results. |
| **In-distribution** | Test data that comes from the same general distribution as the training data. |
| **Out-of-distribution (OOD)** | Test data that differs from the training data. OOD testing is harder and more realistic. |
| **CLIP** | A vision-language model that maps images and text into the same vector space. Matching image-text pairs should be close together. |
| **Embedding / vector** | A list of numbers representing the meaning of an image or text. |
| **Cosine similarity** | A score measuring how aligned two vectors are. Higher means more similar. |
| **NER** | Named Entity Recognition. It extracts names of people, places, organizations, and similar entities from text. |
| **Visual grounding** | Checking whether an entity mentioned in text is visually supported by the image. |
| **NLI** | Natural Language Inference. It checks whether evidence supports, contradicts, or is neutral toward a claim. |
| **DeBERTa** | A transformer model used here for NLI stance scoring. |
| **SBERT** | Sentence-BERT. Used for sentence-level semantic similarity. |
| **Fusion** | Combining several streams or models into one final prediction. |
| **Gate** | A learned mechanism that decides how much to trust each stream. |
| **Softmax** | A function that converts raw scores into probabilities that sum to 1. Used for gate weights. |
| **Sigmoid** | A function that converts a raw score into a probability between 0 and 1. |
| **Threshold** | The cutoff probability above which the model predicts OOC. |
| **Accuracy** | Percentage of predictions that are correct. Can be misleading on imbalanced datasets. |
| **Macro-F1** | Average F1 across classes, treating both small and large classes equally. |
| **AUROC** | Measures how well the model separates classes across thresholds. 0.5 = random, 1.0 = perfect. |
| **Ablation study** | Removing one component at a time to see whether it really matters. |

---

## 3. Research Problem and Motivation

### 3.1 What problem are we solving?

Out-of-context image misinformation happens when a real image is attached to the wrong story. This is dangerous because images make posts feel more trustworthy. A reader may believe a false claim because the image looks authentic.

Example:

- Image: a real photo of a flood from 2019.
- Caption: “Current flood in City X after yesterday’s storm.”
- Problem: the image is real, but it is from another time and place.

This is different from deepfake detection. In deepfake detection, the image itself may be manipulated. In OOC detection, the image can be completely real, but the **relationship between image and caption is false**.

### 3.2 Why is this hard?

OOC detection is hard because the image and caption often look plausible together. A photo of a crowd can match many protest captions. A picture of a politician can match many political stories. Basic visual similarity is not enough.

Common mismatch types include:

1. **Person mismatch** — the caption names one person, but the image shows another.
2. **Location mismatch** — the caption says one city or country, but the image is from elsewhere.
3. **Event mismatch** — the image belongs to a different event.
4. **Date mismatch** — an old image is reused as if it were new.
5. **Source mismatch** — metadata or source information conflicts.
6. **Evidence contradiction** — external evidence contradicts the caption.

### 3.3 Why use multiple streams?

No single signal catches all OOC cases.

- CLIP may say an image and caption are semantically similar, even when the location is wrong.
- Entity grounding may help when the caption mentions people or places, but it is weak when the caption is vague.
- Evidence reasoning may detect contradiction, but only if useful evidence exists.

So MAGE combines all three.

---

## 4. Dataset Overview

MAGE uses three public datasets. Together they create both in-distribution and out-of-distribution evaluation.

| Dataset | Rows | Role in project | Main difficulty |
|---|---:|---|---|
| **NewsCLIPpings** | 85,360 | Main training and standard benchmark | Machine-generated mismatches; good for controlled training. |
| **MMFakeBench** | 11,000 | Main OOD benchmark | Diverse fake types and strong class imbalance. |
| **VERITE** | 1,001 | Real-world fact-checking benchmark | No training split, limited evidence coverage, hardest transfer setting. |
| **Total** | **97,361** | Unified dataset | Mixed sources and distributions. |

### 4.1 Label meaning

| Label | Name | Meaning |
|---:|---|---|
| `0` | Match / genuine | The image and caption correctly belong together. |
| `1` | Mismatch / OOC | The image is misleadingly paired with the caption. |

### 4.2 Final manifest counts

| Split / Manifest | Dataset | Rows | Notes |
|---|---|---:|---|
| `train.csv` | NewsCLIPpings | 71,072 | Balanced train split. |
| `val.csv` | NewsCLIPpings | 7,024 | Balanced validation split. |
| `test.csv` | NewsCLIPpings | 7,264 | Balanced test split. |
| `verite_test.csv` | VERITE | 1,001 | Real-world test only. |
| `mmfakebench_val.csv` | MMFakeBench | 1,000 | Used partly for mixed-domain training and threshold tuning. |
| `mmfakebench_test.csv` | MMFakeBench | 10,000 | Main OOD test. |
| `all.csv` | Combined | 97,361 | Unified manifest for all rows. |

### Figure 1 — Class balance across processed datasets

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/eda_class_balance_processed.png" width="650"/>

**Beginner summary:** This figure shows how many examples belong to each label. NewsCLIPpings is balanced, while MMFakeBench and VERITE contain more OOC examples. This matters because accuracy can look high if a model simply predicts the majority class.

### Figure 2 — Caption length distribution

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/eda_caption_length_distribution.png" width="650"/>

**Beginner summary:** Caption length helps us understand whether the task could be solved using shallow clues. If both classes have overlapping text-length patterns, the model must learn image-text meaning, not just count words.

### Figure 3 — Image coverage summary

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/eda_image_coverage_processed.png" width="650"/>

**Beginner summary:** Some datasets provide local image files, while NewsCLIPpings mainly provides pre-extracted embeddings. This is why the project handles local images and precomputed CLIP embeddings separately.

---

## 5. Dataset Processing in Simple Words

The original datasets did not use the same file structure or schema. One dataset used JSON, another used CSV, and image paths were stored differently. The project therefore created a clean processed layer.

### 5.1 What the processed layer does

The processed layer makes every dataset look the same to the notebooks.

Important outputs:

```text
data/processed/ooc_pairs/manifests/all.csv
data/processed/ooc_pairs/images/
data/splits/
```

### 5.2 Unified manifest schema

Every row in the final manifest contains fields such as:

| Column | Meaning |
|---|---|
| `pair_id` | Unique ID for one image-text pair. |
| `dataset` | Source dataset name. |
| `split` | Train, validation, or test split. |
| `image_id` | Image identifier. |
| `image_path` | Local image path if available. |
| `has_image` | Whether the local image exists. |
| `caption` | Text paired with the image. |
| `label` | `0` for match, `1` for OOC. |
| `clip_image_key` | Key for loading pre-extracted image embeddings. |
| `clip_text_key` | Key for loading pre-extracted text embeddings. |

### 5.3 Why this matters

Without a unified schema:

1. every notebook would need dataset-specific code;
2. paths would break easily;
3. train/test splits would become inconsistent;
4. features from different streams would be hard to merge;
5. experiments would be difficult to reproduce.

The final setup solves this by forcing every experiment to start from the manifest.

---

## 6. Method Overview

The final method follows this pipeline:

```text
Image + Caption
      |
      |--> Stream A: Entity + visual grounding features
      |--> Stream B: CLIP representation features
      |--> Stream C: Evidence + stance reasoning features
      |
      --> GSAF gated fusion
      --> OOC probability
      --> Final prediction
```

### 6.1 Why three streams?

Each stream captures a different kind of signal.

| Stream | Strength | Weakness |
|---|---|---|
| Stream A | Useful for named entities, people, places, organizations. | Weak when captions do not mention visually groundable entities. |
| Stream B | Strong at image-text semantic compatibility. | Can be fooled when an image is visually related but factually wrong. |
| Stream C | Useful when evidence supports or contradicts the caption. | Depends on evidence quality and coverage. |

---

## 7. Stream A — Entity and Visual Grounding

### 7.1 Beginner idea

Stream A asks:

> “The caption mentions certain people, places, or organizations. Does the image visually support those mentions?”

Example:

- Caption: “President Macron visits Berlin.”
- Entities: `President Macron`, `Berlin`.
- Stream A checks whether the image visually aligns with those entity mentions.

### 7.2 What Stream A does

1. **Named Entity Recognition (NER)** extracts entities from the caption.
2. **Visual grounding** uses CLIP-style image-text similarity to compare entities with the image.
3. **Entity scoring** turns the results into 12 numeric features.
4. A simple classifier can be trained on those features.

Example features:

| Feature | Meaning |
|---|---|
| `n_entities` | Number of extracted entities. |
| `n_person` | Number of person mentions. |
| `n_gpe` | Number of geopolitical/location mentions. |
| `max_grounding` | Strongest entity-image grounding score. |
| `mean_grounding` | Average grounding score. |
| `entity_match_ratio` | Fraction of entities that appear visually supported. |

### Figure 4 — Stream A summary

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/stream_a_summary_bar.png" width="650"/>

**Beginner summary:** Stream A is not strong enough as a standalone model, but it contributes useful entity-level clues to the fusion system. It helps the final model notice when named entities are suspicious or weakly grounded.

---

## 8. Stream B — CLIP Representation Learning

### 8.1 Beginner idea

Stream B asks:

> “Do the image and caption look like they belong together in CLIP embedding space?”

CLIP maps both images and text into vectors. If an image and caption match, their vectors should usually be close. If they do not match, their vectors may be farther apart.

### 8.2 What Stream B computes

Stream B uses image and text embeddings to produce 11 features. Examples include:

| Feature | Meaning |
|---|---|
| `clip_cosine` | Cosine similarity between image and text embeddings. |
| `l2_dist` | Distance between image and text embeddings. |
| `absdiff_mean` | Average absolute difference between vectors. |
| `text_norm` | Magnitude of the text vector. |
| `has_real_image_emb` | Whether the image embedding is real or computed/available. |

### 8.3 Important fix

Earlier versions used fake hash-generated image vectors for NewsCLIPpings. That was scientifically invalid because those vectors did not contain real image information.

The corrected Stream B uses real pre-extracted NewsCLIPpings CLIP embeddings and computed OpenCLIP embeddings for local-image datasets.

### Figure 5 — Stream B summary

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/stream_b_summary_bar.png" width="650"/>

**Beginner summary:** Stream B is the strongest individual stream on NewsCLIPpings because CLIP is good at measuring image-text compatibility. However, CLIP alone can fail when the image and caption are visually related but factually mismatched.

### Figure 6 — Stream B score distribution

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/stream_b_score_distribution_newsclippings_test.png" width="650"/>

**Beginner summary:** This plot shows how Stream B scores are distributed on NewsCLIPpings. Separation between match and OOC examples indicates that CLIP features contain useful signal, but overlap between classes explains why CLIP alone is not perfect.

---

## 9. Stream C — Evidence and Stance Reasoning

### 9.1 Beginner idea

Stream C asks:

> “Given independent evidence, does that evidence support or contradict the caption?”

This is treated as a Natural Language Inference problem.

- **Premise:** independent evidence text.
- **Hypothesis:** the caption.
- **Output:** entailment, contradiction, or neutral.

### 9.2 Why decircularization matters

A previous version accidentally included the caption inside the evidence text. That is circular: the model could appear to find support simply because the caption was repeated in the evidence.

The corrected Stream C removes the caption from evidence and uses only independent metadata or article fields.

### 9.3 Stream C features

Stream C produces 7 features:

| Feature | Meaning |
|---|---|
| `support_score` | How strongly evidence supports the caption. |
| `contradiction_score` | How strongly evidence contradicts the caption. |
| `neutral_score` | How unrelated or uncertain the evidence is. |
| `stance_similarity` | SBERT similarity between evidence and caption. |
| `caption_len` | Caption length. |
| `evidence_len` | Evidence length. |
| `len_ratio` | Ratio of caption length to evidence length. |

### Figure 7 — Stream C summary

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/stream_c_summary_bar.png" width="650"/>

**Beginner summary:** Stream C is not equally strong on every dataset. It is weak when evidence is short or incomplete, but it becomes useful when metadata or evidence provides contradiction signals.

---

## 10. Fusion: Gated Stream Attention Fusion (GSAF)

### 10.1 Why fusion is needed

Each stream is useful but imperfect. A fixed average of streams is also limited because the best stream changes by dataset and by example.

For example:

- On NewsCLIPpings, Stream B is often reliable.
- On MMFakeBench, Stream C can become more useful.
- On some entity-heavy examples, Stream A may provide important clues.

GSAF solves this by learning a **per-sample gate**.

### 10.2 Architecture in easy words

The model takes 30 features:

| Source | Feature count |
|---|---:|
| Stream A | 12 |
| Stream B | 11 |
| Stream C | 7 |
| **Total** | **30** |

Each stream is first projected into a 32-dimensional representation:

```text
Stream A features (12) -> projector -> h_A (32)
Stream B features (11) -> projector -> h_B (32)
Stream C features (7)  -> projector -> h_C (32)
```

Then a gate network outputs three trust weights:

```text
[g_A, g_B, g_C] = softmax(gate_network([h_A; h_B; h_C]))
```

The final fused representation is:

```text
z = g_A * h_A + g_B * h_B + g_C * h_C
```

A classifier converts `z` into an OOC probability.

### 10.3 Architecture details

| Component | Setup |
|---|---|
| Stream A projector | Linear(12 -> 32) + LayerNorm + ReLU + Dropout |
| Stream B projector | Linear(11 -> 32) + LayerNorm + ReLU + Dropout |
| Stream C projector | Linear(7 -> 32) + LayerNorm + ReLU + Dropout |
| Gate MLP | Linear(96 -> 32) -> ReLU -> Linear(32 -> 3) -> Softmax |
| Fused vector | Weighted sum of stream representations |
| Classifier | Linear(32 -> 16) -> ReLU -> Dropout -> Linear(16 -> 1) |
| Output | Sigmoid probability of OOC |
| Parameters | 8,164 |

### Figure 8 — Fusion summary

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_summary_bar.png" width="650"/>

**Beginner summary:** This figure compares fusion results across datasets. The key result is that GSAF improves over individual streams, especially on MMFakeBench, where adaptive gating helps the model generalize better.

### Figure 9 — Gate distribution

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_gate_distribution.png" width="650"/>

**Beginner summary:** This figure shows how much the model trusts each stream. The important idea is not that one stream always wins, but that the gate can change trust depending on the example or dataset.

---

## 11. Training Setup

### 11.1 Fusion training data

GSAF was trained using mixed-domain training:

| Training source | Rows | Purpose |
|---|---:|---|
| NewsCLIPpings train | 71,072 | Main training distribution. |
| Half of MMFakeBench validation | 500 | Helps the gate learn OOD behavior. |
| **Total** | **71,572** | Final fusion training pool. |

### 11.2 Validation and evaluation

| Split | Purpose |
|---|---|
| NewsCLIPpings validation | Early stopping and threshold tuning. |
| MMFakeBench validation half | Threshold tuning for MMFakeBench. |
| NewsCLIPpings test | In-distribution final test. |
| MMFakeBench test | Main OOD final test. |
| VERITE test | Real-world fact-checking final test. |

### 11.3 Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 1024 |
| Early stopping patience | 8 epochs |
| Monitor metric | Validation AUROC |
| Best validation AUROC | 76.01% |
| Training time | About 38.6 seconds on RTX 4060 Ti |

### 11.4 Why mixed-domain training mattered

Training only on NewsCLIPpings made the gate over-trust Stream B. That worked in-distribution but failed to generalize to MMFakeBench.

Adding 500 MMFakeBench validation rows gave the gate examples where Stream C and other signals mattered. This improved MMFakeBench test AUROC from about 61% to **87.83%**.

---

## 12. Baselines and Individual Stream Results

Baselines are necessary because they show whether the proposed method is actually better than simple alternatives.

### 12.1 Baselines used

| Baseline | What it does |
|---|---|
| Random uniform | Randomly predicts match or OOC. |
| Majority class | Always predicts the most common class. |
| Always match | Always predicts label `0`. |
| Always OOC | Always predicts label `1`. |
| CLIP cosine threshold | Uses raw CLIP similarity with a threshold. |
| CLIP cosine rank | Uses CLIP similarity ranking as the prediction score. |

### 12.2 Why accuracy alone is dangerous

MMFakeBench has 70% OOC examples. A model that always predicts OOC gets 70% accuracy without understanding anything. That is why AUROC and Macro-F1 are important.

### Figure 10 — All streams and baselines

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/all_streams_summary_bar.png" width="650"/>

**Beginner summary:** This figure compares simple baselines, individual streams, and fusion. It shows why the final model must be judged against both naive baselines and stronger CLIP-based baselines.

---

## 13. Final Results

### 13.1 GSAF final results

| Dataset | Accuracy | F1 | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| NewsCLIPpings validation | 69.15% | 68.47% | 69.13% | 76.01% |
| NewsCLIPpings test | 69.01% | 68.34% | 69.00% | 76.32% |
| MMFakeBench validation tune | 82.00% | 86.84% | 79.18% | 89.92% |
| MMFakeBench test | 78.27% | 84.41% | 74.28% | 87.83% |
| VERITE test | 62.54% | 76.34% | 43.21% | 48.13% |

### 13.2 Main takeaways

1. **NewsCLIPpings:** GSAF improves over individual streams and gives stable in-distribution performance.
2. **MMFakeBench:** This is the strongest success case. GSAF achieves 87.83% AUROC and clearly beats non-gated fusion.
3. **VERITE:** The model struggles. Accuracy is above random, but AUROC is near random, meaning ranking quality is weak.
4. **Gating is useful:** The large MMFakeBench gain shows that adaptive stream weighting is not just cosmetic.

### Figure 11 — NewsCLIPpings confusion matrix

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_confusion_newsclippings_test.png" width="600"/>

**Beginner summary:** The confusion matrix shows where the model is correct and where it confuses match vs OOC. Diagonal cells are correct predictions; off-diagonal cells are mistakes.

### Figure 12 — NewsCLIPpings ROC curve

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_roc_newsclippings_test.png" width="600"/>

**Beginner summary:** The ROC curve shows how well the model separates match and OOC examples across many thresholds. A curve farther from the diagonal is better.

### Figure 13 — MMFakeBench confusion matrix

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_confusion_mmfakebench_test.png" width="600"/>

**Beginner summary:** MMFakeBench is imbalanced, so the confusion matrix helps reveal whether the model is doing more than simply predicting the majority class.

### Figure 14 — MMFakeBench ROC curve

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_roc_mmfakebench_test.png" width="600"/>

**Beginner summary:** This is one of the most important plots. The high AUROC shows that GSAF ranks OOC examples above match examples well on the main OOD benchmark.

### Figure 15 — VERITE confusion matrix

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_confusion_verite_test.png" width="600"/>

**Beginner summary:** VERITE is difficult because it contains real-world cases and no training split. Accuracy alone should not be overclaimed here.

### Figure 16 — VERITE ROC curve

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/fusion_roc_verite_test.png" width="600"/>

**Beginner summary:** The near-random VERITE AUROC is an honest limitation. It means the model does not rank VERITE examples reliably, even if thresholded accuracy looks acceptable.

---

## 14. Ablation Study

### 14.1 What is an ablation?

An ablation removes one component from the system to measure how much that component matters.

MAGE evaluates four ablations:

| Ablation | What is removed or changed? |
|---|---|
| `no_a` | Removes Stream A. |
| `no_b` | Removes Stream B. |
| `no_c` | Removes Stream C. |
| `mlp_fusion` | Removes the adaptive gate and uses ordinary MLP fusion. |

### 14.2 Ablation results

| Config | NewsCLIPpings AUROC | MMFakeBench AUROC | Interpretation |
|---|---:|---:|---|
| Full GSAF | 76.32% | 87.83% | Best overall. |
| `no_a` | 75.23% | 65.24% | Stream A helps OOD fusion. |
| `no_b` | 52.49% | 65.27% | Stream B is critical for in-distribution performance. |
| `no_c` | 75.47% | 62.87% | Stream C is critical for MMFakeBench OOD transfer. |
| `mlp_fusion` | 76.26% | 65.96% | Gating gives the major MMFakeBench gain. |

### 14.3 Main lesson

On NewsCLIPpings, full GSAF and ordinary MLP fusion are almost tied. But on MMFakeBench, GSAF beats MLP fusion by about **21.9 percentage points AUROC**. This proves the gate matters most when the distribution changes.

### Figure 17 — Ablation comparison

<img src="https://raw.githubusercontent.com/Sefayet-Alam/eve-rag/main/paper/figures/ablations_auroc_grouped.png" width="650"/>

**Beginner summary:** This figure shows what happens when components are removed. The large drops on MMFakeBench show that Stream C and the gating mechanism are especially important for OOD generalization.

---

## 15. How to Interpret the Gate

The gate is like a small decision-maker inside the model. It asks:

> “For this specific image-caption pair, which stream should I trust most?”

### 15.1 Example intuition

Suppose the image and caption both mention a political rally. CLIP may think they match because both are semantically about politics. But if the caption says Berlin and the evidence says the event was in Munich, Stream C may detect contradiction.

In that case, the gate should down-weight Stream B and trust Stream C more.

### 15.2 Gate weights

The gate outputs three numbers:

```text
g_A = trust in Stream A
g_B = trust in Stream B
g_C = trust in Stream C
```

These always sum to 1:

```text
g_A + g_B + g_C = 1.0
```

Example:

```text
g_A = 0.22
g_B = 0.51
g_C = 0.27
```

This means the model currently trusts Stream B most.

Another example:

```text
g_A = 0.38
g_B = 0.28
g_C = 0.34
```

This means the model trusts entity grounding and evidence more than CLIP similarity.

---

## 16. Important Design Decisions

| Decision | Why it was made |
|---|---|
| Use three streams | OOC misinformation requires visual, textual, and evidence-based reasoning. |
| Use CLIP features | CLIP provides strong image-text compatibility signal. |
| Use NLI in Stream C | OOC detection often requires support vs contradiction reasoning. |
| Decircularize evidence | Prevents the evidence stream from cheating by repeating the caption. |
| Use AUROC as primary metric | Accuracy is misleading on imbalanced datasets like MMFakeBench and VERITE. |
| Use mixed-domain training | Helps the gate learn beyond NewsCLIPpings. |
| Use per-dataset thresholds | Different datasets have different class balances and score calibration. |
| Keep GSAF lightweight | The model has only 8,164 parameters; the strength comes from meaningful features and gating. |

---

## 17. Issues Fixed During the Project

| Issue | Why it mattered | Fix |
|---|---|---|
| Messy raw folders | Made experiments fragile. | Built unified manifests and flat image folder. |
| Fake Stream B embeddings | Produced invalid image vectors. | Replaced with real NewsCLIPpings CLIP embeddings. |
| Circular Stream C evidence | Evidence contained the caption and could self-confirm. | Removed caption from evidence; used independent metadata/evidence. |
| Old caches | Could silently reuse wrong results. | Deleted and rebuilt relevant cache folders. |
| Gating collapse | Gate learned to mostly trust NewsCLIPpings patterns. | Added 500 MMFakeBench validation rows to training. |
| Threshold mismatch | Default 0.5 threshold was poor on imbalanced data. | Tuned thresholds per dataset. |
| VERITE transfer weakness | Real-world cases differ strongly from training data. | Reported honestly as limitation. |

---

## 18. Limitations and Honest Claims

### 18.1 VERITE remains hard

VERITE AUROC is 48.13%, which is close to random ranking. This should be reported honestly. Possible reasons:

1. VERITE has no train or validation split.
2. Only part of VERITE has high-quality evidence coverage.
3. VERITE contains real-world misinformation patterns unlike the training data.
4. A better solution may require real-time web retrieval or stronger fact-checking evidence.

### 18.2 Stream C is not full real-time RAG

The project name EVE-RAG reflects the design goal of retrieval-augmented grounding, but the current Stream C uses **offline metadata/evidence**, not live web retrieval. Therefore, the paper should describe Stream C as an **offline evidence-based stance stream**, not a full real-time RAG system.

### 18.3 MLLM explanation layer was cancelled

A multimodal large language model explanation layer was planned but cancelled because free-tier throughput and rate limits made large-scale inference infeasible. This should be framed as future work, not as part of the final implemented system.

### 18.4 Do not overclaim

Do not claim:

1. that the model solves real-world OOC detection completely;
2. that VERITE performance is strong;
3. that Stream C is true live web RAG;
4. that CLIP alone is enough;
5. that the gate always chooses the correct stream;
6. that the model explains misinformation like a human fact-checker.

Safe claim:

> MAGE shows that lightweight adaptive gating over complementary entity, CLIP, and evidence-based streams improves cross-dataset OOC detection, especially on MMFakeBench, while VERITE remains an open challenge.

---

## 19. Viva / Seminar Questions and Straight Answers

### Q1. What is out-of-context image detection?

It is the task of detecting whether an image is paired with the correct caption or reused in a misleading context.

### Q2. How is OOC different from fake image detection?

In OOC detection, the image can be real. The problem is that the image and caption do not belong together.

### Q3. Why is CLIP not enough?

CLIP measures visual-semantic similarity. It can say a protest image matches a protest caption, but it may miss that the protest happened in the wrong country or year.

### Q4. Why use Stream A?

Stream A captures entity-level clues such as people, places, and organizations mentioned in the caption. It is weak alone but useful in fusion.

### Q5. Why use Stream C?

Stream C checks whether independent evidence supports or contradicts the caption. This is important when image-text similarity is misleading.

### Q6. What is GSAF?

GSAF means Gated Stream Attention Fusion. It learns per-sample trust weights over Stream A, Stream B, and Stream C.

### Q7. Why is the gate better than ordinary MLP fusion?

Ordinary MLP fusion combines all streams without explicit adaptive trust weights. GSAF learns which stream to emphasize for each sample. This gives a large AUROC gain on MMFakeBench.

### Q8. Why is MMFakeBench important?

It tests out-of-distribution generalization and contains diverse fake types. A model that only works on NewsCLIPpings may fail on MMFakeBench.

### Q9. Why is AUROC the primary metric?

Because datasets like MMFakeBench and VERITE are imbalanced. Accuracy can be inflated by predicting the majority class.

### Q10. Why is VERITE performance weak?

VERITE has no training split, limited evidence coverage, and real-world cases that differ from the training data. It remains an open limitation.

### Q11. What was the most important bug fixed?

The fake hash-generated Stream B embeddings were replaced with real CLIP embeddings. Also, Stream C evidence was decircularized so it no longer contained the caption.

### Q12. What is the main contribution?

The main contribution is a lightweight three-stream adaptive gating framework that improves cross-dataset OOC detection, especially on MMFakeBench.

---

## 20. Recommended Paper Framing

### 20.1 Strong contribution statement

> We propose MAGE, a lightweight multi-stream adaptive gating framework for out-of-context image detection. MAGE combines entity grounding, CLIP-based representation features, and NLI evidence stance features through a per-sample gated fusion module. Experiments across NewsCLIPpings, MMFakeBench, and VERITE show that adaptive gating improves cross-dataset generalization, especially on MMFakeBench, where GSAF reaches 87.83% AUROC and substantially outperforms non-gated fusion.

### 20.2 Safe limitation statement

> Although MAGE improves cross-dataset performance on MMFakeBench, VERITE remains challenging. The model obtains near-random AUROC on VERITE, likely due to the absence of a training split, limited evidence coverage, and large distribution shift. Future work should incorporate stronger retrieval, real-time evidence collection, and multimodal explanation models.

---

## 21. Minimal Run Guide

Use this section only as a compact operational reminder. The full project plan contains more detailed notebook-level instructions.

### 21.1 Setup

```text
1. notebooks/01_setup/env_check.ipynb
2. notebooks/01_setup/data_setup.ipynb
3. notebooks/01_setup/eda.ipynb
```

### 21.2 Stream A

```text
1. notebooks/02_stream_a/ner_pipeline_v2_updated.ipynb
2. notebooks/02_stream_a/visual_grounding_v2_updated.ipynb
3. notebooks/02_stream_a/entity_scorer_v2_updated.ipynb
4. notebooks/02_stream_a/stream_a_results_v2_updated.ipynb
```

### 21.3 Stream B

```text
1. notebooks/03_stream_b/clip_features_updated.ipynb
2. notebooks/03_stream_b/lora_stage1_updated.ipynb
3. notebooks/03_stream_b/lora_stage2_fixed_updated.ipynb
4. notebooks/03_stream_b/stream_b_results_fixed_updated.ipynb
```

### 21.4 Stream C

```text
1. notebooks/04_stream_c/evidence_retrieval_final.ipynb
2. notebooks/04_stream_c/evidence_val_final.ipynb
3. notebooks/04_stream_c/stance_scoring_final.ipynb
4. notebooks/04_stream_c/stream_c_results_final.ipynb
```

### 21.5 Fusion and evaluation

```text
1. notebooks/05_fusion/baselines_final.ipynb
2. notebooks/05_fusion/fusion_module.ipynb
3. notebooks/05_fusion/fusion_training.ipynb
4. notebooks/07_eval/newsclippings_eval.ipynb
5. notebooks/07_eval/cross_dataset_eval.ipynb
6. notebooks/07_eval/ablations.ipynb
7. notebooks/07_eval/qualitative_analysis.ipynb
8. notebooks/08_paper/tables.ipynb
9. notebooks/08_paper/figures.ipynb
```

---

## 22. Final Beginner Mental Model

Imagine the model has three advisors.

1. **Advisor A** looks for names, places, and organizations in the caption and checks whether the image supports them.
2. **Advisor B** compares the image and caption using CLIP embeddings.
3. **Advisor C** reads independent evidence and checks whether it supports or contradicts the caption.

A simple model would average all advisors. MAGE does something smarter. It asks:

> “For this example, which advisor should I trust most?”

That is the purpose of the GSAF gate.

The final prediction is not just based on visual similarity, not just based on entities, and not just based on evidence. It is based on a learned mixture of all three.

This is why the project is useful: out-of-context misinformation is not one problem with one signal. It is a combination of visual compatibility, factual grounding, entity consistency, and evidence reasoning.

---

## 23. Compact Figure Index

All figures below are embedded from the public GitHub repository using raw image URLs.

| Figure | File |
|---|---|
| Dataset class balance | `paper/figures/eda_class_balance_processed.png` |
| Caption length distribution | `paper/figures/eda_caption_length_distribution.png` |
| Image coverage | `paper/figures/eda_image_coverage_processed.png` |
| Stream A summary | `paper/figures/stream_a_summary_bar.png` |
| Stream B summary | `paper/figures/stream_b_summary_bar.png` |
| Stream C summary | `paper/figures/stream_c_summary_bar.png` |
| All streams + baselines | `paper/figures/all_streams_summary_bar.png` |
| Fusion summary | `paper/figures/fusion_summary_bar.png` |
| Gate distribution | `paper/figures/fusion_gate_distribution.png` |
| NewsCLIP fusion confusion | `paper/figures/fusion_confusion_newsclippings_test.png` |
| NewsCLIP fusion ROC | `paper/figures/fusion_roc_newsclippings_test.png` |
| MMFakeBench fusion confusion | `paper/figures/fusion_confusion_mmfakebench_test.png` |
| MMFakeBench fusion ROC | `paper/figures/fusion_roc_mmfakebench_test.png` |
| VERITE fusion confusion | `paper/figures/fusion_confusion_verite_test.png` |
| VERITE fusion ROC | `paper/figures/fusion_roc_verite_test.png` |
| Ablation AUROC grouped | `paper/figures/ablations_auroc_grouped.png` |

---

# End of Eve_rag_Context.md
