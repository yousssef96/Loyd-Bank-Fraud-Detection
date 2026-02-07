import os, sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),".." )))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/LBG Step Up Data Set.xlsx"
OUT = "data/processed/LGB_loan_processed.xlsx"

# load raw
df = pd.read_excel(RAW)

# Preprocess (drop id, loan_status to 0/1, etc.)
df = preprocess_data(df, target_col="loan_status")

# ensure target is 0/1 only if still object
if "loan_status" in df.columns and df['loan_status'].dtype == "object":
    df['loan_status'] = df["loan_status"].str.strip().map({"Fully Paid" : 0, "Charged Off" : 1})

# sanity checks
assert df["loan_status"].isna().sum() == 0, "loan_status has NaNs after preprocess"
assert set(df['loan_status'].unique()) <= {0, 1}, "loan_status not 0/1 after preprocess"

# features
df_processed, high_card_cols = build_features(df, target_col="loan_status")

# save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_excel(OUT, index=False)
print(f"Processed dataset saved to {OUT} | Shape: {df_processed.shape}")