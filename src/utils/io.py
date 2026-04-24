"""Tiny load/save helpers — keeps notebooks clean."""
import json, numpy as np
from pathlib import Path

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_npy(arr, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def load_npy(path):
    return np.load(path, allow_pickle=True)
