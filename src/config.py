from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "complaints.csv"   # your ~6GB file
PROC_DIR = DATA_DIR / "processed"       # parquet shards will go here
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility & split ratios
SEED = 42
TRAIN_PCT, VAL_PCT = 0.70, 0.15  # test = 0.15

# Column name candidates (CFPB exports can vary slightly)
TEXT_COL_CANDIDATES    = ["Consumer complaint narrative", "consumer_complaint_narrative", "Complaint narrative"]
PRODUCT_COL_CANDIDATES = ["Product", "product"]
ID_COL_CANDIDATES      = ["Complaint ID", "complaint_id", "Complaint ID#", "complaint_id_"]

# 4-class mapping per assessment
# 0: Credit reporting/credit repair/other, 1: Debt collection, 2: Consumer loan, 3: Mortgage
def map_product_to_label(prod: str):
    if not isinstance(prod, str):
        return None
    s = prod.strip().lower()
    if "credit reporting" in s:   return 0
    if "debt collection" in s:    return 1
    if "consumer loan" in s:      return 2
    if "mortgage" in s:           return 3
    return None
