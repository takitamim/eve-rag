# EVE-RAG: Entity Verification + Evidence RAG for OOC Detection

**Paper title:** EVE-RAG: Entity Verification and Evidence Retrieval-Augmented Generation
for Robust Out-of-Context Image Detection

## Quick start

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_trf

# 3. Open in VS Code
open eve-rag.code-workspace
```

## Notebook order

| Folder | Purpose |
|--------|---------|
| `01_setup/` | Env check, dataset download, EDA |
| `02_stream_a/` | Entity verification pipeline (NER + visual grounding) |
| `03_stream_b/` | LoRA-CLIP feature extraction and fine-tuning |
| `04_stream_c/` | Evidence retrieval + stance reranking |
| `05_fusion/` | Cross-attention fusion + all baselines |
| `06_mllm/` | Qwen2-VL 7B LoRA fine-tuning + CoT inference |
| `07_eval/` | Full evaluation: NewsCLIPpings, VERITE, COSMOS, ablations |
| `08_paper/` | Paper figures, LaTeX tables, explanation eval |

## Src helpers (minimal — don't grow these)

- `src/utils/paths.py` — all paths in one place, import in every notebook
- `src/utils/io.py` — save/load JSON and .npy

## What goes in git

Only `notebooks/`, `src/`, `configs/`, `paper/`, `scripts/`, `requirements.txt`.
`data/`, `cache/`, `experiments/` are git-ignored (too large).
