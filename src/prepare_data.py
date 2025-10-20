# src/prepare_data.py
import hashlib
from pathlib import Path
import pandas as pd
from .config import (
    RAW_CSV, TRAIN_PCT, VAL_PCT, TEXT_COL_CANDIDATES,
    PRODUCT_COL_CANDIDATES, ID_COL_CANDIDATES, map_product_to_label, PROC_DIR
)

pd.options.mode.chained_assignment = None

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)[:30]}")

def stable_bucket(key_str: str) -> float:
    h = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    return (int(h, 16) % 10_000) / 10_000.0

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv(RAW_CSV, chunksize=200_000, low_memory=False)

    text_col = prod_col = id_col = None
    agg = {"train": [], "val": [], "test": []}

    for i, chunk in enumerate(reader):
        if text_col is None:
            text_col = pick_column(chunk, TEXT_COL_CANDIDATES)
            prod_col = pick_column(chunk, PRODUCT_COL_CANDIDATES)
            id_col   = pick_column(chunk, ID_COL_CANDIDATES)
            print(f"[INFO] Using columns — text: {text_col} | product: {prod_col} | id: {id_col}")

        df = chunk[[id_col, prod_col, text_col]].copy()
        df["label"] = df[prod_col].map(map_product_to_label)
        df = df[df["label"].notna()]
        df[text_col] = df[text_col].fillna("").astype(str).str.strip()
        df = df[df[text_col].str.len() > 0]

        def choose_split(row):
            key = str(row[id_col]) if pd.notna(row[id_col]) else row[text_col][:200]
            r = stable_bucket(key)
            if r < TRAIN_PCT: return "train"
            if r < TRAIN_PCT + VAL_PCT: return "val"
            return "test"

        df["split"] = df.apply(choose_split, axis=1)
        for split in ("train", "val", "test"):
            part = df[df["split"] == split][[text_col, "label"]]
            if not part.empty:
                part.columns = ["text", "label"]
                agg[split].append(part)

        total = {k: sum(len(x) for x in v) for k, v in agg.items()}
        print(f"[CHUNK {i}] train={total['train']} val={total['val']} test={total['test']}")

    for split in ("train", "val", "test"):
        if agg[split]:
            out = pd.concat(agg[split], ignore_index=True)
            (PROC_DIR / f"{split}.parquet").unlink(missing_ok=True)
            out.to_parquet(PROC_DIR / f"{split}.parquet",
                           engine="pyarrow", compression="zstd", compression_level=12, index=False)
            print(f"[WRITE] {split}: {len(out)} rows -> {PROC_DIR / f'{split}.parquet'}")
    print("[DONE] Compressed splits written.")

if __name__ == "__main__":
    main()
