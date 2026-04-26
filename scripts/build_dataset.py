from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "ooc_pairs"
MANIFESTS = PROCESSED / "manifests"

VERITE_ROOT = RAW / "verite" / "image-text-verification" / "VERITE"
VERITE_CSV = VERITE_ROOT / "VERITE.csv"
VERITE_IMG_RAW = VERITE_ROOT / "images"
VERITE_IMG_PROC = PROCESSED / "images" / "verite"

MMFAKE_ROOT = RAW / "mmfakebench"
NEWS_ROOT = RAW / "newsclippings"

MANIFESTS.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_records(path: Path):
    obj = load_json(path)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # NewsCLIPpings root JSON uses this.
        if "annotations" in obj:
            ann = obj["annotations"]
            if isinstance(ann, list):
                return ann
            if isinstance(ann, dict):
                return list(ann.values())

        # Other common wrappers.
        for key in ["data", "items", "samples", "examples", "pairs"]:
            if key in obj:
                val = obj[key]
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    return list(val.values())

        # Metadata JSON keyed by id.
        if all(isinstance(v, dict) for v in obj.values()):
            rows = []
            for k, v in obj.items():
                row = dict(v)
                row.setdefault("_key", k)
                rows.append(row)
            return rows

        # Dict of column lists.
        if all(isinstance(v, list) for v in obj.values()):
            return pd.DataFrame(obj).to_dict("records")

    raise ValueError(f"Unsupported JSON structure: {path}")


def norm_label(value):
    s = str(value).strip().lower()

    if s in {"0", "true", "real", "matched", "match", "pristine", "truthful", "yes", "original"}:
        return 0

    if s in {"1", "false", "fake", "falsified", "ooc", "out-of-context", "mismatch", "miscaptioned", "no"}:
        return 1

    return None


def relpath(p: Path):
    return str(p.relative_to(ROOT)).replace("\\", "/")


def find_by_name_or_stem(folder: Path, value):
    if value is None:
        return None

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None

    name = Path(s).name
    stem = Path(name).stem

    candidates = [folder / name]

    for ext in IMAGE_EXTS:
        candidates.append(folder / f"{stem}{ext}")

    try:
        n = int(float(stem))
        for width in [1, 4, 5, 6]:
            for ext in IMAGE_EXTS:
                candidates.append(folder / f"{n:0{width}d}{ext}")
    except Exception:
        pass

    for p in candidates:
        if p.exists():
            return p

    return None


def build_news_metadata(split_name):
    meta_path = NEWS_ROOT / "metadata" / f"{split_name}.json"
    if not meta_path.exists():
        return {}

    raw = load_json(meta_path)
    index = {}

    if isinstance(raw, dict):
        for key, row in raw.items():
            if not isinstance(row, dict):
                continue

            keys = {
                str(key),
                str(row.get("id", "")),
                str(row.get("image_id", "")),
                Path(str(row.get("image_path", ""))).name,
                Path(str(row.get("image_path", ""))).stem,
            }

            for k in keys:
                if k and k != "nan":
                    index[k] = row
                    try:
                        index[str(int(float(k)))] = row
                    except Exception:
                        pass

    else:
        for row in load_records(meta_path):
            keys = {
                str(row.get("id", "")),
                str(row.get("image_id", "")),
                Path(str(row.get("image_path", ""))).name,
                Path(str(row.get("image_path", ""))).stem,
            }

            for k in keys:
                if k and k != "nan":
                    index[k] = row
                    try:
                        index[str(int(float(k)))] = row
                    except Exception:
                        pass

    return index


def build_news(split_name):
    pairs_path = NEWS_ROOT / f"{split_name}.json"
    if not pairs_path.exists():
        print(f"[WARN] Missing NewsCLIPpings pairs: {pairs_path}")
        return pd.DataFrame()

    pairs = load_records(pairs_path)
    meta = build_news_metadata(split_name)

    rows = []

    for idx, r in enumerate(pairs):
        pair_id_raw = r.get("id", idx)
        image_id = r.get("image_id", r.get("image", r.get("image_path", "")))
        falsified = int(r.get("falsified", r.get("label", 0)))

        m = (
            meta.get(str(image_id))
            or meta.get(str(pair_id_raw))
            or meta.get(Path(str(image_id)).name)
            or meta.get(Path(str(image_id)).stem)
        )

        caption = ""
        if m:
            for col in ["caption", "text", "title", "sentence"]:
                val = m.get(col, "")
                if str(val).strip():
                    caption = str(val).strip()
                    break

        rows.append({
            "pair_id": f"nc_{pair_id_raw}_{falsified}",
            "dataset": "newsclippings",
            "split": split_name,
            "image_path": "",
            "caption": caption,
            "label": falsified,
            "source_image_id": image_id,
            "original_id": pair_id_raw,
            "has_image": 0,
            "notes": "uses pre-extracted embeddings; local VisualNews image unavailable",
        })

    out = pd.DataFrame(rows)

    out_name = {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
    }.get(split_name, f"newsclippings_{split_name}.csv")

    out.to_csv(MANIFESTS / out_name, index=False)
    return out


def build_verite():
    if not VERITE_CSV.exists():
        print(f"[WARN] Missing VERITE CSV: {VERITE_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(VERITE_CSV)
    rows = []

    for i, r in df.iterrows():
        raw_image_path = str(r.get("image_path", "")).strip()
        caption = str(r.get("caption", r.get("text", ""))).strip()
        label_raw = r.get("label", "")
        label = norm_label(label_raw)

        candidates = []

        # First: exact CSV path/name, because VERITE uses images/true_0.jpg etc.
        for folder in [VERITE_IMG_PROC, VERITE_IMG_RAW]:
            p = find_by_name_or_stem(folder, raw_image_path)
            if p:
                candidates.append(p)

        # Second: numbered downloaded filenames.
        for folder in [VERITE_IMG_PROC, VERITE_IMG_RAW]:
            for stem in [str(i), f"{i:04d}", f"{i+1:04d}", f"{i:05d}", f"{i+1:05d}"]:
                p = find_by_name_or_stem(folder, stem)
                if p:
                    candidates.append(p)

        found = candidates[0] if candidates else None

        rows.append({
            "pair_id": f"verite_{i:04d}",
            "dataset": "verite",
            "split": "test",
            "image_path": relpath(found) if found else "",
            "caption": caption,
            "label": label,
            "source_image_id": "",
            "original_id": int(r.get("Unnamed: 0", i)),
            "has_image": int(found is not None),
            "notes": f"raw_label={label_raw}; raw_image_path={raw_image_path}",
        })

    out = pd.DataFrame(rows)
    out.to_csv(MANIFESTS / "verite_test.csv", index=False)
    return out


def resolve_mmfake_image(img_rel, split_name):
    s = str(img_rel).strip()
    if not s or s.lower() == "nan":
        return None

    s_clean = s.replace("\\", "/").lstrip("/")
    name = Path(s_clean).name

    candidates = [
        MMFAKE_ROOT / s_clean,
        MMFAKE_ROOT / f"MMFakeBench_{split_name}" / s_clean,
        MMFAKE_ROOT / f"MMFakeBench_{split_name}" / name,
    ]

    for p in candidates:
        if p.exists():
            return p

    split_dir = MMFAKE_ROOT / f"MMFakeBench_{split_name}"
    if split_dir.exists() and name:
        matches = list(split_dir.rglob(name))
        if matches:
            return matches[0]

    return None


def build_mmfake(split_name):
    json_path = MMFAKE_ROOT / f"MMFakeBench_{split_name}.json"
    if not json_path.exists():
        print(f"[WARN] Missing MMFakeBench JSON: {json_path}")
        return pd.DataFrame()

    records = load_records(json_path)
    rows = []

    for i, r in enumerate(records):
        text = str(r.get("text", r.get("caption", r.get("claim", "")))).strip()
        img_rel = str(r.get("image_path", r.get("image", r.get("img_path", "")))).strip()
        label_raw = r.get("gt_answers", r.get("label", ""))

        label = norm_label(label_raw)
        if label is None:
            label = 1 if "/fake/" in img_rel.replace("\\", "/").lower() else 0

        img_path = resolve_mmfake_image(img_rel, split_name)

        rows.append({
            "pair_id": f"mmfakebench_{split_name}_{i:05d}",
            "dataset": "mmfakebench",
            "split": split_name,
            "image_path": relpath(img_path) if img_path else "",
            "caption": text,
            "label": label,
            "source_image_id": "",
            "original_id": i,
            "has_image": int(img_path is not None),
            "notes": f"raw_label={label_raw}; raw_image_path={img_rel}",
        })

    out = pd.DataFrame(rows)
    out.to_csv(MANIFESTS / f"mmfakebench_{split_name}.csv", index=False)
    return out


def main():
    outputs = []

    print("Building NewsCLIPpings manifests...")
    for split in ["train", "val", "test"]:
        df = build_news(split)
        if len(df):
            outputs.append(df)
            print(f"  {split}: {len(df)} rows, captions={(df['caption'].astype(str).str.len() > 0).sum()}")

    print("Building VERITE manifest...")
    verite = build_verite()
    if len(verite):
        outputs.append(verite)
        print(f"  verite_test: {len(verite)} rows, has_image={verite['has_image'].sum()}")

    print("Building MMFakeBench manifests...")
    for split in ["val", "test"]:
        df = build_mmfake(split)
        if len(df):
            outputs.append(df)
            print(f"  mmfakebench_{split}: {len(df)} rows, has_image={df['has_image'].sum()}")

    if outputs:
        all_df = pd.concat(outputs, ignore_index=True)
        all_df.to_csv(MANIFESTS / "all.csv", index=False)
        print(f"Saved all.csv: {len(all_df)} rows")

    print(f"Manifests written to: {MANIFESTS}")


if __name__ == "__main__":
    main()
