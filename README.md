# Kaiburr Task 5 — Consumer Complaint Text Classification

Classify CFPB consumer complaints into 4 categories:

- **0** = Credit reporting/repair/other  
- **1** = Debt collection  
- **2** = Consumer loan  
- **3** = Mortgage

## Project Structure

```
kaiburr-task5-ml/
├─ README.md
├─ requirements.txt
├─ data/
│ ├─ complaints.csv                    # (not committed)
│ └─ processed/                        # auto-created Parquet splits
├─ notebooks/
│ ├─ 01_eda.ipynb
│ ├─ 02_model_baselines.ipynb
│ └─ 03_model_final_eval.ipynb
├─ src/
│ ├─ prepare_data.py                   # CSV -> compressed Parquet (train/val/test)
│ ├─ train_hf.py                       # GPU fine-tuning (DistilBERT)
│ ├─ evaluate_hf.py                    # test report + confusion matrix
│ ├─ predict_hf.py                     # single/batch prediction CLI
│ ├─ train_baselines.py                # fast TF-IDF baselines (LinearSVC/LogReg/NB)
│ └─ eda_quick.py                      # quick EDA artifacts
├─ models/
│ └─ hf_best_model/                    # final DistilBERT model (optional via LFS/release)
└─ docs/
   ├─ eda_summary.md
   ├─ eda_hist_lengths.png
   ├─ results.md
   ├─ model_comparison.csv
   └─ screenshots/                     # include your name + system clock in each screenshot
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python -m src.prepare_data
```

## Usage

### Training Models

#### Fine-tune DistilBERT (GPU)
```bash
python -m src.train_hf
```

#### Train Baselines (fast, CPU)
```bash
# quick LinearSVC baseline on a subset
python -m src.train_baselines --max-train 100000 --max-val 25000 --max-test 25000 --features word --max-features 100000 --models svc
```

### Evaluation
```bash
python -m src.evaluate_hf
# saves models/confusion_matrix.png and prints a classification report
```

### Prediction

#### Single Prediction
```bash
python -m src.predict_hf --text "My mortgage servicer misapplied my payment and added late fees."
```

#### Batch Prediction
```bash
python -m src.predict_hf --file data/complaints.txt --topk 3
```

## Results Summary

- **Final model**: DistilBERT (fine-tuned)
- **Test Accuracy**: 0.9461
- **Test Macro-F1**: 0.8055
- **Best baseline (val)**: LinearSVC — Acc 0.9394, Macro-F1 0.7833

### EDA Highlights
- **Test split class counts**: 0: 342,682, 1: 55,884, 2: 1,423, 3: 20,289
- **Median text length**: ≈ 627–628 chars across splits; p90 ≈ ~1990–2001 chars
- See `docs/eda_summary.md` for detailed analysis
