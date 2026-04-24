"""Central path registry — import this in every notebook."""
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
DATA       = ROOT / "data"
CACHE      = ROOT / "cache"
EXPERIMENTS= ROOT / "experiments"
CONFIGS    = ROOT / "configs"

NEWSCLIP   = DATA / "newsclippings"
VERITE     = DATA / "verite"
COSMOS     = DATA / "cosmos"

CLIP_EMB   = CACHE / "clip_embeddings"
NER_ANN    = CACHE / "ner_annotations"
VIS_ENT    = CACHE / "visual_entities"
EVIDENCE   = CACHE / "evidence"
STANCE     = CACHE / "stance_scores"
