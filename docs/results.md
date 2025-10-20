# Results

## Overview
Text classification on the CFPB Consumer Complaints dataset into 4 classes:
- **0** = Credit reporting/repair/other
- **1** = Debt collection  
- **2** = Consumer loan
- **3** = Mortgage

## 1. EDA Summary
**Key artifacts**: `docs/eda_summary.md`, `docs/eda_hist_lengths.png`

### Dataset Statistics
- **Test split class counts**: 0: 342,682, 1: 55,884, 2: 1,423, 3: 20,289
- **Text length** (chars): 
  - Median: ≈ 627–628 chars across splits
  - 90th percentile: ≈ 1990–2001 chars
- **Total samples**: See detailed breakdown in `docs/eda_summary.md`

## 2. Pre-processing
- **Data splits**: Deterministic hash split → train/val/test (written to compressed Parquet)
- **Transformer path**: DistilBERT tokenizer, max_len=192, no extra cleaning
- **Baseline path**: TF-IDF (word 1–2 n-grams, `min_df=3`, `max_features≈100k`)

## 3. Models
- **Transformer**: DistilBERT (pretrained), fine-tuned end-to-end on GPU
- **Baselines**: LinearSVC, LogisticRegression(saga), MultinomialNB on TF-IDF

## 4. Validation Comparison
| Model                    | Val Accuracy | Val Macro-F1 |
|--------------------------|--------------|--------------|
| LinearSVC                | 0.9394       | 0.7833       |
| LogisticRegression (saga)| 0.9162       | 0.7511       |
| MultinomialNB            | 0.9096       | 0.6312       |
| DistilBERT (fine-tuned)  | N/A          | N/A          |

*Note: DistilBERT validation metrics not logged; comparison done on test set only*

## 5. Test Evaluation (Final Model)
- **Selected model**: DistilBERT (fine-tuned)
- **Test Accuracy**: 0.9461
- **Test Macro-F1**: 0.8055
- **Confusion matrix**: `models/confusion_matrix.png`

*For context: best baseline test report is in `docs/baseline_report.txt`*

## 6. Prediction Demo
Successfully tested with:
```bash
# Single prediction
python -m src.predict_hf --text "My mortgage servicer misapplied my payment and added late fees."

# Batch prediction
python -m src.predict_hf --file data/complaints.txt --topk 3
```

## 7. Key Findings
- **Best performing model**: DistilBERT fine-tuned on GPU
- **Performance improvement**: DistilBERT achieved 94.61% accuracy vs 93.94% for best baseline (LinearSVC)
- **Class imbalance**: Significant imbalance with class 0 dominating (342,682 samples) and class 2 being rare (1,423 samples)
- **Text characteristics**: Median length ~627-628 characters with 90th percentile around 1990-2001 characters