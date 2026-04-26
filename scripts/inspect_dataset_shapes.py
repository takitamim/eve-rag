from pathlib import Path
import json
import pandas as pd

ROOT = Path(".")
NEWS = ROOT / "data/raw/newsclippings"
MM = ROOT / "data/raw/mmfakebench"
VER = ROOT / "data/raw/verite/image-text-verification/VERITE"

print("News files:")
for p in [NEWS/"train.json", NEWS/"metadata/train.json"]:
    print("\n", p, p.exists())
    if p.exists():
        obj = json.load(open(p, encoding="utf-8"))
        print("type:", type(obj))
        if isinstance(obj, dict):
            print("keys:", list(obj.keys())[:20])
            first = next(iter(obj.values()))
            print("first value type:", type(first))
            print("first value sample:", str(first)[:800])
        elif isinstance(obj, list):
            print("first item:", obj[0])

print("\nMMFake first sample:")
for name in ["MMFakeBench_val.json", "MMFakeBench_test.json"]:
    p = MM / name
    obj = json.load(open(p, encoding="utf-8"))
    print("\n", name, type(obj))
    item = next(iter(obj.values())) if isinstance(obj, dict) else obj[0]
    print(item)

print("\nVERITE columns/sample:")
df = pd.read_csv(VER / "VERITE.csv")
print(df.columns.tolist())
print(df.head())
