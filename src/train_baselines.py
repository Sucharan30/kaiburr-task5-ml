from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import dump
import json

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"; MODELS.mkdir(parents=True, exist_ok=True)
DOCS = ROOT / "docs"; DOCS.mkdir(parents=True, exist_ok=True)

MAX_TRAIN = 200_000   # adjust for speed/memory
MAX_VAL   = 40_000
MAX_TEST  = 40_000

def load_parquet_sample(path, limit=None):
    pf = pq.ParquetFile(path)
    texts, labels = [], []
    for b in pf.iter_batches(batch_size=20000, columns=["text","label"]):
        d = b.to_pydict()
        texts += ["" if t is None else str(t) for t in d["text"]]
        labels += [int(v) for v in d["label"]]
        if limit and len(texts) >= limit:
            texts, labels = texts[:limit], labels[:limit]
            break
    return texts, labels

def eval_model(name, pipe, Xtr, ytr, Xva, yva):
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xva)
    return {
        "model": name,
        "val_accuracy": accuracy_score(yva, p),
        "val_macro_f1": f1_score(yva, p, average="macro")
    }, pipe

def main():
    Xtr, ytr = load_parquet_sample(PROC / "train.parquet", MAX_TRAIN)
    Xva, yva = load_parquet_sample(PROC / "val.parquet", MAX_VAL)
    Xte, yte = load_parquet_sample(PROC / "test.parquet", MAX_TEST)

    # shared TF-IDF (char+word via union)
    word = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                           ngram_range=(1,2), min_df=3, max_features=200_000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3, max_features=200_000)

    # three compact pipelines
    models = [
        ("LinearSVC", make_pipeline(word, LinearSVC(C=1.0, class_weight="balanced"))),
        ("LogReg",    make_pipeline(word, LogisticRegression(max_iter=2000, n_jobs=2, class_weight="balanced"))),
        ("NB",        make_pipeline(word, MultinomialNB(alpha=1.0))),
    ]

    results = []
    best = None
    for name, pipe in models:
        r, fitted = eval_model(name, pipe, Xtr, ytr, Xva, yva)
        results.append(r)
        if best is None or r["val_macro_f1"] > best[0]["val_macro_f1"]:
            best = (r, fitted)

    # test evaluation on best baseline
    best_name = best[0]["model"]
    p = best[1].predict(Xte)
    report = classification_report(yte, p, digits=4)
    (MODELS / "baseline_best.txt").write_text(best_name)
    (DOCS / "baseline_report.txt").write_text(report)

    # save table
    df = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
    df.to_csv(DOCS / "model_comparison.csv", index=False)

    print(df)
    print("\nBest baseline:", best_name)
    print("\nTest report written to docs/baseline_report.txt")

if __name__ == "__main__":
    main()
