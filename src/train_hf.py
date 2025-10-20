# src/train_hf.py
from pathlib import Path
import random
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers as tf

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
MAX_LEN = 192            # shorter seq -> faster
BATCH = 16               # drop to 8/4 if OOM
EPOCHS = 2               # reduce for faster training
LR = 2e-5
SEED = 42

# Downsample caps (lower these to speed up time even more)
MAX_TRAIN_SAMPLES = 300_000   # set to e.g. 150_000 or 100_000 for faster runs
MAX_VAL_SAMPLES   = 60_000

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def read_parquet_sample(path: Path, max_samples=None):
    pf = pq.ParquetFile(path)
    texts, labels = [], []
    for batch in pf.iter_batches(batch_size=8192, columns=["text", "label"]):
        tb = batch.to_pydict()
        t, y = tb["text"], tb["label"]
        texts.extend("" if v is None else str(v) for v in t)
        labels.extend(int(v) for v in y)
        if max_samples is not None and len(texts) >= max_samples:
            texts = texts[:max_samples]; labels = labels[:max_samples]
            break
    return texts, labels

class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_len)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "precision_macro": pr, "recall_macro": rc}

def main():
    set_seed()
    print("CUDA:", torch.cuda.is_available(),
          "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print("transformers:", tf.__version__)

    # Load compact splits (no HuggingFace dataset cache)
    train_texts, train_labels = read_parquet_sample(PROC_DIR / "train.parquet", MAX_TRAIN_SAMPLES)
    val_texts,   val_labels   = read_parquet_sample(PROC_DIR / "val.parquet",   MAX_VAL_SAMPLES)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds = TextClsDataset(train_texts, train_labels, tok, MAX_LEN)
    val_ds   = TextClsDataset(val_texts,   val_labels,   tok, MAX_LEN)

    data_collator = DataCollatorWithPadding(tokenizer=tok)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.config.problem_type = "single_label_classification"

    # IMPORTANT: Only pass universally-supported args (no evaluation_strategy/save_strategy/etc.)
    args = TrainingArguments(
        output_dir=str(MODELS_DIR / "hf_runs"),
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=100,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,   # faster loading
        report_to="none" if "none" in str(TrainingArguments.__init__) else None,  # harmless if ignored
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,   # we won't eval during training; use evaluate_hf.py after
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save only the final model (~250MB)
    out_dir = MODELS_DIR / "hf_best_model"
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print("[DONE] Saved:", out_dir)

    # Clean transient run dir to free space
    import shutil
    shutil.rmtree(MODELS_DIR / "hf_runs", ignore_errors=True)

if __name__ == "__main__":
    main()
