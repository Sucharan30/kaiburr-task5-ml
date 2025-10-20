from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)
IMG = DOCS / "eda_hist_lengths.png"
OUT = DOCS / "eda_summary.md"

def read(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    texts, labels = [], []
    for b in pf.iter_batches(batch_size=20_000, columns=["text", "label"]):
        d = b.to_pydict()
        texts += ["" if t is None else str(t) for t in d["text"]]
        labels += [int(v) for v in d["label"]]
    return pd.DataFrame({"text": texts, "label": labels})

def df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except Exception:
        return "```\n" + df.to_string() + "\n```"

def main():
    dftr = read(PROC / "train.parquet")
    dfva = read(PROC / "val.parquet")
    dfte = read(PROC / "test.parquet")
    df = pd.concat(
        [dftr.assign(split="train"), dfva.assign(split="val"), dfte.assign(split="test")],
        ignore_index=True,
    )

    # Class balance
    balance = df.groupby(["split", "label"]).size().unstack(fill_value=0)

    # Text length and robust stats
    df["len"] = df["text"].str.len()
    grp = df.groupby("split")["len"]
    stats = pd.DataFrame({
        "count": grp.count(),
        "mean": grp.mean().round(2),
        "median": grp.median(),
        "p90": grp.quantile(0.9),
        "max": grp.max(),
    }).astype({"count": int})

    # Length histogram (clipped)
    plt.figure(figsize=(7, 4))
    df["len"].clip(upper=2000).hist(bins=50)
    plt.title("Complaint length distribution (chars, clipped @2000)")
    plt.xlabel("length")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(IMG)
    plt.close()

    # Sample rows (first 3 per class)
    samples = []
    for k in [0, 1, 2, 3]:
        ex = df[df["label"] == k].head(3)["text"].to_list()
        samples.append((k, ex))

    # Write summary markdown
    with OUT.open("w", encoding="utf-8") as f:
        f.write("# EDA Summary\n\n")
        f.write("## Class balance (rows per split)\n\n")
        f.write(df_to_md(balance) + "\n\n")
        f.write("## Text length stats (chars)\n\n")
        f.write(df_to_md(stats) + "\n\n")
        f.write(f"![Lengths]({IMG.name})\n\n")
        f.write("## Example texts (first 3 per class)\n")
        for k, ex_list in samples:
            f.write(f"\n### Class {k}\n")
            for i, t in enumerate(ex_list, 1):
                safe_text = (t or "").replace("\n", " ")
                preview = safe_text[:300]
                suffix = "..." if len(safe_text) > 300 else ""
                f.write(f"{i}. {preview}{suffix}\n")

    print("[WROTE]", OUT, "\n[IMG]", IMG)

if __name__ == "__main__":
    main()
