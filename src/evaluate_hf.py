# src/evaluate_hf.py
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
BEST = ROOT / "models" / "hf_best_model"
OUT_IMG = ROOT / "models" / "confusion_matrix.png"

NUM_LABELS = 4
MAX_LEN = 192
BATCH = 64        # increase if you have VRAM

def read_parquet_all(path: Path):
    """Load 'text' and 'label' from a compressed Parquet split."""
    pf = pq.ParquetFile(path)
    texts, labels = [], []
    for batch in pf.iter_batches(batch_size=8192, columns=["text", "label"]):
        tb = batch.to_pydict()
        texts.extend("" if v is None else str(v) for v in tb["text"])
        labels.extend(int(v) for v in tb["label"])
    return texts, labels

class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_len)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

def main():
    tok = AutoTokenizer.from_pretrained(str(BEST), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(BEST), num_labels=NUM_LABELS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("Device:", device)

    texts, labels = read_parquet_all(PROC_DIR / "test.parquet")
    ds = TextClsDataset(texts, labels, tok, MAX_LEN)
    collate = DataCollatorWithPadding(tokenizer=tok)

    # IMPORTANT: num_workers=0 on Windows to avoid pickling issues
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=collate)

    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            labels_t = batch.pop("labels")
            for k in batch:
                batch[k] = batch[k].to(device)
            logits = model(**batch).logits
            preds.append(logits.argmax(-1).cpu().numpy())
            gts.append(labels_t.numpy())

    preds = np.concatenate(preds); gts = np.concatenate(gts)
    print(classification_report(gts, preds, digits=4))

    cm = confusion_matrix(gts, preds, normalize="true")
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix (normalized)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_IMG)
    print("[SAVED]", OUT_IMG.resolve())

if __name__ == "__main__":
    main()
