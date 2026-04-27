import pickle
from pathlib import Path

files = [
    "data/raw/newsclippings/embeddings/clip_image_embeddings/clip_image_embeddings_train.pkl",
    "data/raw/newsclippings/embeddings/clip_text_embeddings/clip_text_embeddings_train.pkl",
]

for f in files:
    p = Path(f)
    print("\nFILE:", p)
    obj = pickle.load(open(p, "rb"))
    print("type:", type(obj))

    if isinstance(obj, dict):
        print("dict keys sample:", list(obj.keys())[:5])
        first_key = list(obj.keys())[0]
        first_val = obj[first_key]
        print("first key:", first_key)
        print("first value type:", type(first_val))
        try:
            print("first value shape:", first_val.shape)
        except Exception:
            print("first value sample:", str(first_val)[:300])

    elif isinstance(obj, list):
        print("list len:", len(obj))
        print("first item type:", type(obj[0]))
        print("first item:", str(obj[0])[:500])

    else:
        try:
            print("shape:", obj.shape)
        except Exception:
            print("sample:", str(obj)[:500])
