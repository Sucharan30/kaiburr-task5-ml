
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
BEST = ROOT / "models" / "hf_best_model"

LABELS = {
    0: "Credit reporting/credit repair/other",
    1: "Debt collection",
    2: "Consumer loan",
    3: "Mortgage",
}
IDX2LABEL = LABELS  # alias for clarity

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(str(BEST), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(BEST))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tok, model, device

def predict(texts, tok, model, device, max_len=192, batch_size=32):
    """Return NxC probability array for a list of texts."""
    if isinstance(texts, str):
        texts = [texts]
    # tokenize in batches to avoid VRAM spikes
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
    return np.vstack(probs_all)

def print_text_output(texts, probs, topk, device):
    """Pretty, sorted, per-sample output."""
    for idx, (t, p) in enumerate(zip(texts, probs)):
        order = np.argsort(-p)[:topk]
        pred = int(order[0])
        print(f"\nSample {idx+1}:")
        print(f"  Text: {t[:120].strip()}{'...' if len(t)>120 else ''}")
        print(f"  Top prediction: {IDX2LABEL[pred]} ({pred})  —  confidence: {p[pred]*100:.2f}%  [device={device}]")
        print("  Top classes:")
        for j in order:
            print(f"    {j}: {IDX2LABEL[j]:40s}  {p[j]*100:6.2f}%")

def print_json_output(texts, probs, topk, device):
    out = []
    for t, p in zip(texts, probs):
        order = np.argsort(-p)[:topk]
        pred = int(order[0])
        out.append({
            "text": t,
            "device": device,
            "top_prediction": {"id": pred, "label": IDX2LABEL[pred], "confidence": float(p[pred])},
            "topk": [
                {"id": int(j), "label": IDX2LABEL[int(j)], "confidence": float(p[int(j)])}
                for j in order
            ],
            "all_probs": {str(i): float(pi) for i, pi in enumerate(p)}
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))

def read_texts_from_file(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Single complaint text")
    g.add_argument("--file", type=str, help="Path to a .txt file (one complaint per line)")
    ap.add_argument("--topk", type=int, default=4, help="How many top classes to show")
    ap.add_argument("--max_len", type=int, default=192, help="Max sequence length for tokenization")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction")
    ap.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = ap.parse_args()

    # Load model/tokenizer
    tok, model, device = load_model_and_tokenizer()

    # Gather inputs
    if args.file:
        texts = read_texts_from_file(args.file)
        if not texts:
            print("File is empty after trimming lines.")
            return
    else:
        texts = [args.text]

    # Predict
    probs = predict(texts, tok, model, device, max_len=args.max_len, batch_size=args.batch_size)

    # Output
    if args.format == "json":
        print_json_output(texts, probs, args.topk, device)
    else:
        print_text_output(texts, probs, args.topk, device)

if __name__ == "__main__":
    main()
